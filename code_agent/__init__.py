"""
Pi Coding Agent - File system tools and coding utilities

This package extends pi_agent_core with file system operations, command execution,
and resource management capabilities for coding tasks.
"""

__version__ = "0.1.0"

# Tools
from .tools import (
    # Individual tools
    WriteTool, create_write_tool, write_tool,
    ReadTool, create_read_tool, read_tool,
    EditTool, create_edit_tool, edit_tool,
    BashTool, create_bash_tool, bash_tool,
    GrepTool, create_grep_tool, grep_tool,
    FindTool, create_find_tool, find_tool,
    LsTool, create_ls_tool, ls_tool,
    # Tool combinations
    get_coding_tools,
    get_readonly_tools,
    get_all_tools,
)

# Resources
from .resources import (
    Skill,
    SkillManager,
    SkillMetadata,
    SkillTool,
    AgentConfig,
    AgentConfigManager,
    CreateSkillTool,
    CreateAgentConfigTool,
    ListResourcesTool,
)

# High-level Agent
from .coding_agent import CodingAgent, create_coding_agent

# Session Management
from .session import SessionManager, SessionMetadata, CodingSession

__all__ = [
    # Tools - Individual
    "WriteTool", "create_write_tool", "write_tool",
    "ReadTool", "create_read_tool", "read_tool",
    "EditTool", "create_edit_tool", "edit_tool",
    "BashTool", "create_bash_tool", "bash_tool",
    "GrepTool", "create_grep_tool", "grep_tool",
    "FindTool", "create_find_tool", "find_tool",
    "LsTool", "create_ls_tool", "ls_tool",
    # Tools - Combinations
    "get_coding_tools",
    "get_readonly_tools",
    "get_all_tools",
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
    # High-level Agent
    "CodingAgent",
    "create_coding_agent",
    # Session Management
    "SessionManager",
    "SessionMetadata",
    "CodingSession",
]
