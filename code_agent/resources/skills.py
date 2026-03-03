"""
Skills System - Skill metadata and script discovery

Skills are reusable capability units containing:
- SKILL.md: Metadata and description
- script.py OR scripts/*.py: Executable scripts that can be run via bash

This module uses WriteTool from pi_coding_agent.tools to create skill files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent_core import AgentTool, AgentToolResult, TextContent

load_dotenv()
INFO_DIR = os.getenv("INFO_DIR", "./.personal")
SKILLS_DIR = os.path.join(INFO_DIR, "skills")


@dataclass
class SkillMetadata:
    """Skill metadata"""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    disable_model_invocation: bool = False


@dataclass
class Skill:
    """Skill definition"""

    name: str
    description: str
    metadata: SkillMetadata
    script_path: str | None = None
    base_dir: str = ""


class SkillTool(AgentTool):
    """
    Skill tool - Wraps Python script as Agent tool
    """

    def __init__(self, skill: Skill):
        self.skill = skill
        self.name = skill.name
        self.label = skill.metadata.name
        self.description = skill.metadata.description
        self.parameters = skill.metadata.parameters or {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input data (JSON string or plain text)"}},
            "required": [],
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """Execute skill script"""

        if not self.skill.script_path:
            return AgentToolResult(
                content=[
                    TextContent(text="Error: Skill has no executable script.py. This is an instruction-only skill.")
                ],
                details={"error": "missing_script", "skill": self.skill.name},
            )

        # Prepare input data
        input_data = json.dumps(params)

        # Execute Python script
        try:
            result = subprocess.run(
                [sys.executable, self.skill.script_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.skill.base_dir,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Script execution failed"
                return AgentToolResult(
                    content=[TextContent(text=f"Error: {error_msg}")],
                    details={"error": error_msg, "returncode": result.returncode},
                )

            # Parse output
            output = result.stdout.strip()

            # Try to parse as JSON
            try:
                output_data = json.loads(output)
                return AgentToolResult(
                    content=[TextContent(text=json.dumps(output_data, ensure_ascii=False, indent=2))],
                    details=output_data,
                )
            except json.JSONDecodeError:
                # Plain text output
                return AgentToolResult(content=[TextContent(text=output)], details={"output": output})

        except subprocess.TimeoutExpired:
            return AgentToolResult(
                content=[TextContent(text="Error: Script timeout (30s)")], details={"error": "timeout"}
            )
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {str(e)}")], details={"error": str(e)})


class SkillManager:
    """
    Skill Manager

    Manages Skills - loading metadata, resolving script paths, and creating skills.
    Uses WriteTool to create skill files.
    """

    def __init__(self, skills_dir: str | None = None, write_tool=None):
        """
        Initialize SkillManager

        Args:
            skills_dir: Skills directory path (default: .skills/)
            write_tool: WriteTool instance for creating files (optional, for creating skills)
        """
        if skills_dir is None:
            skills_dir = SKILLS_DIR

        self.skills_dir = Path(skills_dir)
        self.write_tool = write_tool
        self.skills: Dict[str, Skill] = {}
        self.tools: Dict[str, SkillTool] = {}

    def load_skills(self) -> List[str]:
        """
        Load all skills from directory

        Returns:
            List of loaded skill names
        """
        self.skills.clear()
        self.tools.clear()

        if not self.skills_dir.exists():
            return []

        loaded_skills = []

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue

            try:
                skill = self._load_skill(skill_dir)
                if skill:
                    self.skills[skill.name] = skill
                    loaded_skills.append(skill.name)
            except Exception as e:
                print(f"Warning: Failed to load skill {skill_dir.name}: {e}")

        return loaded_skills

    def _load_skill(self, skill_dir: Path) -> Optional[Skill]:
        """Load single skill from directory"""
        skill_md = skill_dir / "SKILL.md"

        if not skill_md.exists():
            return None

        metadata = self._parse_skill_md(skill_md)
        if not metadata:
            return None

        script_path = self._resolve_script_path(skill_dir)

        return Skill(
            name=skill_dir.name,
            description=metadata.description,
            metadata=metadata,
            script_path=script_path,
            base_dir=str(skill_dir),
        )

    def _resolve_script_path(self, skill_dir: Path) -> str | None:
        """Resolve executable script path for a skill directory."""
        root_script = skill_dir / "script.py"
        if root_script.exists():
            return str(root_script)

        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            return None

        candidates = sorted(scripts_dir.glob("*.py"))
        if not candidates:
            return None

        if len(candidates) == 1:
            return str(candidates[0])

        normalized_skill_name = skill_dir.name.replace("-", "_").lower()
        preferred = [p for p in candidates if p.stem.lower() == normalized_skill_name]
        if preferred:
            return str(preferred[0])

        return str(candidates[0])

    def _parse_skill_md(self, skill_md_path: Path) -> Optional[SkillMetadata]:
        """Parse SKILL.md file"""
        try:
            content = skill_md_path.read_text(encoding="utf-8")

            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            frontmatter = parts[1].strip()
            metadata_dict = self._parse_simple_yaml(frontmatter)

            name = metadata_dict.get("name", "")
            description = metadata_dict.get("description", "")

            if not name or not description:
                return None

            return SkillMetadata(
                name=name,
                description=description,
                parameters=metadata_dict.get("parameters", {}),
                disable_model_invocation=metadata_dict.get("disable-model-invocation", False),
            )

        except Exception as e:
            print(f"Error parsing {skill_md_path}: {e}")
            return None

    def _parse_simple_yaml(self, yaml_str: str) -> Dict[str, Any]:
        """Simple YAML parser for basic key-value pairs"""
        result = {}
        lines = yaml_str.split("\n")
        current_key = None
        current_value = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if ":" in stripped and len(line) - len(line.lstrip()) == 0:
                if current_key:
                    result[current_key] = "\n".join(current_value).strip()

                key, value = stripped.split(":", 1)
                current_key = key.strip()
                current_value = [value.strip()] if value.strip() else []
            elif current_key:
                current_value.append(stripped)

        if current_key:
            result[current_key] = "\n".join(current_value).strip()

        # Convert booleans
        for key, value in result.items():
            if isinstance(value, str):
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False

        return result

    def get_tools(self) -> List[AgentTool]:
        """Get all skill tools"""
        return list(self.tools.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get specific skill"""
        return self.skills.get(name)

    async def create_skill(
        self,
        name: str,
        description: str,
        script_content: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create new skill using WriteTool

        Args:
            name: Skill name
            description: Skill description
            script_content: Python script content
            parameters: JSON Schema parameters

        Returns:
            Success status
        """
        if not self.write_tool:
            print("Error: WriteTool not available. Cannot create skill.")
            return False

        if not self._validate_skill_name(name):
            print(f"Error: Invalid skill name: {name}")
            return False

        skill_dir = self.skills_dir / name
        if skill_dir.exists():
            print(f"Error: Skill {name} already exists")
            return False

        try:
            # Create SKILL.md
            frontmatter = f"""---
name: {name}
description: {description}
---

# {name}

{description}
"""

            skill_md_path = skill_dir / "SKILL.md"
            await self.write_tool.execute("create_skill_md", {"path": str(skill_md_path), "content": frontmatter})

            # Create script.py
            full_script = f'''#!/usr/bin/env python3
"""
{name} - {description}

Auto-generated skill script.
Input: JSON params via stdin
Output: JSON result via stdout
"""

import sys
import json

def main():
    try:
        params = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        params = {{}}

    # Script logic
{self._indent_code(script_content, 4)}

if __name__ == "__main__":
    main()
'''

            script_py_path = skill_dir / "script.py"
            await self.write_tool.execute("create_skill_script", {"path": str(script_py_path), "content": full_script})

            # Reload skills
            self.load_skills()

            print(f"✅ Created skill: {name}")
            return True

        except Exception as e:
            print(f"Error creating skill {name}: {e}")
            return False

    def _validate_skill_name(self, name: str) -> bool:
        """Validate skill name"""
        import re

        return bool(re.match(r"^[a-z0-9-]+$", name))

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code"""
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)
