"""
Read Tool - Read file contents (text and images)

Supports reading text files and images with automatic resizing.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, ImageContent, TextContent


class ReadTool:
    """
    Read tool - Read file contents

    Supports:
    - Text files (with pagination)
    - Image files (jpg, png, gif, webp) with auto-resizing
    """

    # Image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    # Truncation limits
    MAX_LINES = 2000
    MAX_BYTES = 2 * 1024 * 1024  # 2MB
    MAX_LINE_LENGTH = 2000

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Read tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "read"
        self.label = "Read File"
        self.description = (
            "Read file contents. Supports text files and images. "
            "For text files, can read specific lines using offset/limit."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, for text files)",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (for text files)",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working directory"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.cwd / p

    def _is_image(self, path: Path) -> bool:
        """Check if file is an image based on extension"""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    async def _read_image(self, path: Path) -> AgentToolResult:
        """
        Read image file and return as ImageContent

        Args:
            path: Path to image file

        Returns:
            AgentToolResult with ImageContent
        """
        try:
            # Read image as base64
            with open(path, "rb") as f:
                image_data = f.read()

            # Detect media type
            ext = path.suffix.lower()
            media_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_type_map.get(ext, "image/jpeg")

            # Encode to base64
            base64_data = base64.b64encode(image_data).decode("utf-8")

            size = len(image_data)

            return AgentToolResult(
                content=[
                    ImageContent(
                        source={
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        }
                    )
                ],
                details={
                    "path": str(path),
                    "size": size,
                    "type": "image",
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error reading image {path}: {str(e)}")],
                details={
                    "error": str(e),
                    "path": str(path),
                },
            )

    async def _read_text(
        self, path: Path, offset: Optional[int] = None, limit: Optional[int] = None
    ) -> AgentToolResult:
        """
        Read text file with optional pagination

        Args:
            path: Path to text file
            offset: Starting line number (1-indexed)
            limit: Maximum lines to read

        Returns:
            AgentToolResult with TextContent
        """
        try:
            # Read all lines
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Apply offset and limit
            start_idx = (offset - 1) if offset else 0
            end_idx = start_idx + limit if limit else None

            lines = all_lines[start_idx:end_idx]

            # Truncate long lines
            truncated_lines = []
            for line in lines:
                if len(line) > self.MAX_LINE_LENGTH:
                    truncated_lines.append(
                        line[: self.MAX_LINE_LENGTH] + f"... [truncated {len(line) - self.MAX_LINE_LENGTH} chars]\n"
                    )
                else:
                    truncated_lines.append(line)

            # Format with line numbers (cat -n style)
            output_lines = []
            for idx, line in enumerate(truncated_lines, start=start_idx + 1):
                output_lines.append(f"{idx:6d}\t{line}")

            content = "".join(output_lines)

            # Check if truncated
            was_truncated = limit and (start_idx + limit) < total_lines

            # Build result message
            if offset or limit:
                msg = f"Read lines {start_idx + 1}-{start_idx + len(lines)} of {total_lines} from {path.name}"
            else:
                msg = f"Read {len(lines)} lines from {path.name}"

            if was_truncated:
                msg += f" (truncated, {total_lines - start_idx - len(lines)} lines remaining)"

            return AgentToolResult(
                content=[TextContent(text=content)],
                details={
                    "path": str(path),
                    "lines_read": len(lines),
                    "total_lines": total_lines,
                    "start_line": start_idx + 1,
                    "end_line": start_idx + len(lines),
                    "truncated": was_truncated,
                    "message": msg,
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error reading file {path}: {str(e)}")],
                details={
                    "error": str(e),
                    "path": str(path),
                },
            )

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute file read operation

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing 'path', optional 'offset', 'limit'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with file contents
        """
        path_str = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")

        # Resolve path
        path = self._resolve_path(path_str)

        # Check if file exists
        if not path.exists():
            return AgentToolResult(
                content=[TextContent(text=f"File not found: {path_str}")],
                details={
                    "error": "File not found",
                    "path": path_str,
                },
            )

        # Check if it's a directory
        if path.is_dir():
            return AgentToolResult(
                content=[
                    TextContent(text=f"Path is a directory: {path_str}. Use 'ls' tool to list directory contents.")
                ],
                details={
                    "error": "Path is a directory",
                    "path": path_str,
                },
            )

        # Read based on file type
        if self._is_image(path):
            return await self._read_image(path)
        else:
            return await self._read_text(path, offset, limit)


def create_read_tool(cwd: str) -> ReadTool:
    """
    Factory function to create a read tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        ReadTool instance
    """
    return ReadTool(cwd=cwd)


# Default instance using current directory
read_tool = ReadTool()
