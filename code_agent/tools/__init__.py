"""
File system tools for coding agents

These tools provide file system operations like read, write, edit, etc.
All tools follow the AgentTool protocol from pi_agent_core.
"""

from .write import WriteTool, create_write_tool, write_tool
from .read import ReadTool, create_read_tool, read_tool
from .edit import EditTool, create_edit_tool, edit_tool
from .bash import BashTool, create_bash_tool, bash_tool
from .grep import GrepTool, create_grep_tool, grep_tool
from .find import FindTool, create_find_tool, find_tool
from .ls import LsTool, create_ls_tool, ls_tool

__all__ = [
    # Write
    "WriteTool",
    "create_write_tool",
    "write_tool",
    # Read
    "ReadTool",
    "create_read_tool",
    "read_tool",
    # Edit
    "EditTool",
    "create_edit_tool",
    "edit_tool",
    # Bash
    "BashTool",
    "create_bash_tool",
    "bash_tool",
    # Grep
    "GrepTool",
    "create_grep_tool",
    "grep_tool",
    # Find
    "FindTool",
    "create_find_tool",
    "find_tool",
    # Ls
    "LsTool",
    "create_ls_tool",
    "ls_tool",
]

# Tool combinations
def get_coding_tools(cwd: str | None = None):
    """
    Get all coding tools (read, write, edit, bash)

    These tools provide full file system access for coding tasks.

    Args:
        cwd: Working directory (default: current directory)

    Returns:
        List of coding tools
    """
    if cwd:
        return [
            create_read_tool(cwd),
            create_write_tool(cwd),
            create_edit_tool(cwd),
            create_bash_tool(cwd),
        ]
    else:
        return [read_tool, write_tool, edit_tool, bash_tool]


def get_readonly_tools(cwd: str | None = None):
    """
    Get read-only tools (read, grep, find, ls)

    These tools are for exploring and understanding the codebase without modifications.

    Args:
        cwd: Working directory (default: current directory)

    Returns:
        List of read-only tools
    """
    if cwd:
        return [
            create_read_tool(cwd),
            create_grep_tool(cwd),
            create_find_tool(cwd),
            create_ls_tool(cwd),
        ]
    else:
        return [read_tool, grep_tool, find_tool, ls_tool]


def get_all_tools(cwd: str | None = None):
    """
    Get all file system tools

    Args:
        cwd: Working directory (default: current directory)

    Returns:
        List of all tools
    """
    if cwd:
        return [
            create_read_tool(cwd),
            create_write_tool(cwd),
            create_edit_tool(cwd),
            create_bash_tool(cwd),
            create_grep_tool(cwd),
            create_find_tool(cwd),
            create_ls_tool(cwd),
        ]
    else:
        return [read_tool, write_tool, edit_tool, bash_tool, grep_tool, find_tool, ls_tool]


__all__ += [
    "get_coding_tools",
    "get_readonly_tools",
    "get_all_tools",
]
