"""
Ls Tool - List directory contents

Lists files and directories with proper formatting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class LsTool:
    """
    Ls tool - List directory contents

    Features:
    - Lists all files and directories
    - Includes hidden files (dotfiles)
    - Directories marked with trailing /
    - Alphabetically sorted
    """

    MAX_ENTRIES = 500
    MAX_BYTES = 2 * 1024 * 1024  # 2MB

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Ls tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "ls"
        self.label = "List Directory"
        self.description = (
            "List directory contents. Shows files and directories, "
            "including hidden files. Directories are marked with trailing /."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: current directory)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries (default: 500)",
                    "minimum": 1,
                    "maximum": 10000,
                },
            },
            "required": [],
        }

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working directory"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.cwd / p

    def _format_entry(self, path: Path, base_path: Path) -> str:
        """
        Format directory entry

        Args:
            path: Path to entry
            base_path: Base directory path

        Returns:
            Formatted entry string
        """
        # Get relative name
        try:
            name = path.relative_to(base_path).name
        except ValueError:
            name = path.name

        # Add trailing / for directories
        if path.is_dir():
            return f"{name}/"
        else:
            return name

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute directory listing

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing optional 'path' and 'limit'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with directory contents
        """
        dir_path = params.get("path", ".")
        limit = params.get("limit", self.MAX_ENTRIES)

        # Resolve path
        abs_path = self._resolve_path(dir_path)

        # Check if path exists
        if not abs_path.exists():
            return AgentToolResult(
                content=[TextContent(text=f"Directory not found: {dir_path}")],
                details={
                    "error": "Directory not found",
                    "path": dir_path,
                },
            )

        # Check if it's a directory
        if not abs_path.is_dir():
            return AgentToolResult(
                content=[TextContent(text=f"Path is not a directory: {dir_path}")],
                details={
                    "error": "Not a directory",
                    "path": dir_path,
                },
            )

        try:
            # List directory contents
            entries = []

            for entry in abs_path.iterdir():
                entries.append(entry)

                # Check limit
                if len(entries) >= limit:
                    break

            # Sort entries (directories first, then files)
            def sort_key(p: Path) -> tuple[int, str]:
                # 0 for directories, 1 for files
                return (0 if p.is_dir() else 1, p.name.lower())

            entries.sort(key=sort_key)

            # Format entries
            formatted_entries = [self._format_entry(entry, abs_path) for entry in entries]

            # Count directories and files
            dir_count = sum(1 for e in entries if e.is_dir())
            file_count = len(entries) - dir_count

            # Check truncation
            was_truncated = len(entries) >= limit

            # Format output
            result_text = f"Contents of {abs_path}:\n"
            result_text += f"({dir_count} directories, {file_count} files"
            if was_truncated:
                result_text += f", truncated to {limit}"
            result_text += ")\n\n"

            result_text += "\n".join(formatted_entries)

            return AgentToolResult(
                content=[TextContent(text=result_text)],
                details={
                    "path": str(abs_path),
                    "total_entries": len(entries),
                    "directories": dir_count,
                    "files": file_count,
                    "truncated": was_truncated,
                    "entries": formatted_entries,
                },
            )

        except PermissionError:
            return AgentToolResult(
                content=[TextContent(text=f"Permission denied: {dir_path}")],
                details={
                    "error": "Permission denied",
                    "path": dir_path,
                },
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error listing directory: {str(e)}")],
                details={
                    "error": str(e),
                    "path": dir_path,
                },
            )


def create_ls_tool(cwd: str) -> LsTool:
    """
    Factory function to create an ls tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        LsTool instance
    """
    return LsTool(cwd=cwd)


# Default instance using current directory
ls_tool = LsTool()
