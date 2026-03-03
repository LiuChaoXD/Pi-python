"""
Resource management for coding agents

Includes Skills and Agent Configs systems that use file tools
to create and manage persistent resources.
"""

from .agent_configs import AgentConfig, AgentConfigManager
from .memory import MemoryManager
from .memory_tools import MemoryReadTool, MemoryWriteTool
from .resource_tools import CreateAgentConfigTool, CreateSkillTool, ListResourcesTool
from .skills import Skill, SkillManager, SkillMetadata, SkillTool

__all__ = [
    # Skills
    "Skill",
    "SkillManager",
    "SkillMetadata",
    "SkillTool",
    # Agent Configs
    "AgentConfig",
    "AgentConfigManager",
    # Resource Tools
    "CreateSkillTool",
    "CreateAgentConfigTool",
    "ListResourcesTool",
    # Memory
    "MemoryManager",
    "MemoryReadTool",
    "MemoryWriteTool",
]
