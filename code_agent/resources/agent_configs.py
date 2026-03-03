"""
Agent Configs System - Reusable agent configurations

Agent configs define pre-configured agents with:
- system prompt
- tools
- thinking level
- other settings

This module uses WriteTool to create AGENT.yaml files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent_core import Model, ThinkingLevel


@dataclass
class AgentConfig:
    """Agent configuration"""

    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    thinking_level: str = "off"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentConfigManager:
    """
    Agent Config Manager

    Manages agent configurations using WriteTool.
    """

    def __init__(self, agents_dir: str | None = None, write_tool=None):
        """
        Initialize AgentConfigManager

        Args:
            agents_dir: Agent configs directory (default: .pi/agents/)
            write_tool: WriteTool instance for creating files
        """
        if agents_dir is None:
            agents_dir = os.path.join(os.getcwd(), ".pi", "agents")

        self.agents_dir = Path(agents_dir)
        self.write_tool = write_tool
        self.configs: Dict[str, AgentConfig] = {}

    def load_configs(self) -> List[str]:
        """Load all agent configs"""
        self.configs.clear()

        if not self.agents_dir.exists():
            return []

        loaded_configs = []

        for config_dir in self.agents_dir.iterdir():
            if not config_dir.is_dir() or config_dir.name.startswith("."):
                continue

            config_file = None
            for ext in ["yaml", "yml", "json"]:
                candidate = config_dir / f"AGENT.{ext}"
                if candidate.exists():
                    config_file = candidate
                    break

            if not config_file:
                continue

            try:
                config = self._load_config(config_file, config_dir.name)
                if config:
                    self.configs[config.name] = config
                    loaded_configs.append(config.name)
            except Exception as e:
                print(f"Warning: Failed to load config {config_dir.name}: {e}")

        return loaded_configs

    def _load_config(self, config_file: Path, dir_name: str) -> Optional[AgentConfig]:
        """Load single config from file"""
        try:
            content = config_file.read_text(encoding="utf-8")

            if config_file.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
            elif config_file.suffix == ".json":
                data = json.loads(content)
            else:
                return None

            name = data.get("name", dir_name)
            description = data.get("description", "")
            system_prompt = data.get("system_prompt", data.get("systemPrompt", ""))

            if not name or not system_prompt:
                return None

            return AgentConfig(
                name=name,
                description=description,
                system_prompt=system_prompt,
                tools=data.get("tools", []),
                thinking_level=data.get("thinking_level", data.get("thinkingLevel", "off")),
                temperature=data.get("temperature"),
                max_tokens=data.get("max_tokens", data.get("maxTokens")),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            print(f"Error parsing {config_file}: {e}")
            return None

    def get_config(self, name: str) -> Optional[AgentConfig]:
        """Get specific config"""
        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        """List all config names"""
        return list(self.configs.keys())

    async def create_config(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: Optional[List[str]] = None,
        thinking_level: str = "off",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create new agent config using WriteTool

        Args:
            name: Config name
            description: Config description
            system_prompt: System prompt
            tools: Tool list
            thinking_level: Thinking level
            temperature: Temperature
            max_tokens: Max tokens
            metadata: Additional metadata

        Returns:
            Success status
        """
        if not self.write_tool:
            print("Error: WriteTool not available. Cannot create config.")
            return False

        if not self._validate_config_name(name):
            print(f"Error: Invalid config name: {name}")
            return False

        config_dir = self.agents_dir / name
        if config_dir.exists():
            print(f"Error: Agent config {name} already exists")
            return False

        try:
            config_data = {
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "tools": tools or [],
                "thinking_level": thinking_level,
            }

            if temperature is not None:
                config_data["temperature"] = temperature

            if max_tokens is not None:
                config_data["max_tokens"] = max_tokens

            if metadata:
                config_data["metadata"] = metadata

            # Write AGENT.yaml
            config_file = config_dir / "AGENT.yaml"
            yaml_content = yaml.dump(config_data, allow_unicode=True, default_flow_style=False)

            await self.write_tool.execute("create_agent_config", {"path": str(config_file), "content": yaml_content})

            # Reload configs
            self.load_configs()

            print(f"✅ Created agent config: {name}")
            return True

        except Exception as e:
            print(f"Error creating config {name}: {e}")
            return False

    def _validate_config_name(self, name: str) -> bool:
        """Validate config name"""
        import re

        return bool(re.match(r"^[a-z0-9-]+$", name))
