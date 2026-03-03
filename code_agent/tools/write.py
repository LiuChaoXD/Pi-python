"""
Write Tool - Create or overwrite files

This is the foundational tool that enables all resource management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class WriteTool:
    """
    Write tool - Create or overwrite files

    This tool is the foundation for:
    - Creating Skills (script.py, SKILL.md)
    - Creating Agent configs (AGENT.yaml)
    - Any file creation operations
    """

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Write tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "write"
        self.label = "Write File"
        self.description = (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve path relative to working directory

        Args:
            path: File path (relative or absolute)

        Returns:
            Absolute Path object
        """
        p = Path(path)

        if p.is_absolute():
            return p

        return self.cwd / p

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute file write operation

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing 'path' and 'content'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with success message
        """
        path = params["path"]
        content = params["content"]

        try:
            # Resolve absolute path
            absolute_path = self._resolve_path(path)

            # Create parent directories if needed
            absolute_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            absolute_path.write_text(content, encoding="utf-8")

            # Get file size
            size = absolute_path.stat().st_size

            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {size} bytes to {path}")],
                details={
                    "path": str(absolute_path),
                    "size": size,
                    "created_dirs": not absolute_path.parent.exists(),
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error writing file {path}: {str(e)}")],
                details={
                    "error": str(e),
                    "path": path,
                },
            )


def create_write_tool(cwd: str) -> WriteTool:
    """
    Factory function to create a write tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        WriteTool instance
    """
    return WriteTool(cwd=cwd)


# Default instance using current directory
write_tool = WriteTool()
