"""
Resource Management Tools - Tools for creating Skills and Agent configs

These tools allow agents to create and manage persistent resources.
They depend on SkillManager and AgentConfigManager which use WriteTool.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from agent_core import AgentToolResult, TextContent

load_dotenv()
INFO_DIR = os.getenv("INFO_DIR", "./.personal")
SKILLS_DIR = os.path.join(INFO_DIR, "skills")
MEMORY_DIR = os.path.join(INFO_DIR, "memory")


class CreateSkillTool:
    """Tool for creating Skills"""

    def __init__(self, skill_manager):
        self.skill_manager = skill_manager
        self.name = "create_skill"
        self.label = "Create Skill"
        self.description = (
            "Create a new executable Python skill script. "
            "Skills are saved to .pi/skills/ and can be executed via bash scripts when needed."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (lowercase, numbers, hyphens, e.g. 'calculate-stats')",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed skill description",
                },
                "script_content": {
                    "type": "string",
                    "description": (
                        "Python script main logic. "
                        "Script reads JSON params from stdin, outputs JSON to stdout. "
                        "Example: result = {'answer': 42}; print(json.dumps(result))"
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": "Skill parameters (JSON Schema format, optional)",
                },
            },
            "required": ["name", "description", "script_content"],
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """Execute skill creation"""

        name = params["name"]
        description = params["description"]
        script_content = params["script_content"]
        parameters = params.get("parameters")

        success = await self.skill_manager.create_skill(
            name=name,
            description=description,
            script_content=script_content,
            parameters=parameters,
        )

        if success:
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Successfully created skill '{name}'.\n"
                        f"Location: .pi/skills/{name}/\n"
                        f"Run the script from .pi/skills/{name}/ via bash when needed."
                    )
                ],
                details={
                    "success": True,
                    "skill_name": name,
                    "path": f".pi/skills/{name}/",
                },
            )
        else:
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Failed to create skill '{name}'. " "Check if name is valid or skill already exists."
                    )
                ],
                details={"success": False, "skill_name": name},
            )


class CreateAgentConfigTool:
    """Tool for creating Agent configs"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.name = "create_agent_config"
        self.label = "Create Agent Config"
        self.description = (
            "Create a new agent configuration. " "Configs are saved to .pi/agents/ for creating specialized sub-agents."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Config name (lowercase, numbers, hyphens, e.g. 'math-expert')",
                },
                "description": {
                    "type": "string",
                    "description": "Brief agent description",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Agent system prompt defining role and capabilities",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names array (optional)",
                },
                "thinking_level": {
                    "type": "string",
                    "enum": ["off", "minimal", "low", "medium", "high", "xhigh"],
                    "description": "Thinking level (default: 'off')",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature (0.0-2.0, optional)",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max tokens (optional)",
                },
            },
            "required": ["name", "description", "system_prompt"],
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """Execute config creation"""

        name = params["name"]
        description = params["description"]
        system_prompt = params["system_prompt"]
        tools = params.get("tools")
        thinking_level = params.get("thinking_level", "off")
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")

        success = await self.config_manager.create_config(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            thinking_level=thinking_level,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if success:
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Successfully created agent config '{name}'.\n"
                        f"Location: {INFO_DIR}/agents/{name}/\n"
                        f"Config can now be used to create specialized sub-agents."
                    )
                ],
                details={
                    "success": True,
                    "config_name": name,
                    "path": f"{INFO_DIR}/agents/{name}/",
                },
            )
        else:
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Failed to create agent config '{name}'. "
                        "Check if name is valid or config already exists."
                    )
                ],
                details={"success": False, "config_name": name},
            )


class ListResourcesTool:
    """Tool for listing available resources"""

    def __init__(self, skill_manager, config_manager):
        self.skill_manager = skill_manager
        self.config_manager = config_manager
        self.name = "list_resources"
        self.label = "List Resources"
        self.description = "List all available Skills and Agent configs"
        self.parameters = {
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "enum": ["skills", "agents", "all"],
                    "description": "Resource type to list (default: 'all')",
                }
            },
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """Execute resource listing"""

        resource_type = params.get("resource_type", "all")
        result_parts = []

        if resource_type in ["skills", "all"]:
            skills = list(self.skill_manager.skills.values())
            if skills:
                result_parts.append("### Available Skills:\n")
                for skill in skills:
                    result_parts.append(f"- **{skill.name}**: {skill.description}")
            else:
                result_parts.append("### Skills: None")

        if resource_type in ["agents", "all"]:
            if resource_type == "all":
                result_parts.append("\n")

            configs = list(self.config_manager.configs.values())
            if configs:
                result_parts.append("### Available Agent Configs:\n")
                for config in configs:
                    result_parts.append(f"- **{config.name}**: {config.description}")
            else:
                result_parts.append("### Agent Configs: None")

        result_text = "\n".join(result_parts)

        return AgentToolResult(
            content=[TextContent(text=result_text)],
            details={
                "skills": [s.name for s in self.skill_manager.skills.values()],
                "agent_configs": list(self.config_manager.configs.keys()),
            },
        )
