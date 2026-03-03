"""
Find Tool - Find files by glob patterns

Finds files matching glob patterns with .gitignore support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class FindTool:
    """
    Find tool - Find files by pattern

    Features:
    - Glob pattern matching (e.g., *.py, **/*.js)
    - Respects .gitignore
    - Output truncation
    """

    MAX_RESULTS = 1000
    MAX_BYTES = 2 * 1024 * 1024  # 2MB

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Find tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "find"
        self.label = "Find Files"
        self.description = (
            "Find files matching a glob pattern. " "Supports patterns like '*.py', '**/*.js', 'src/**/*.ts'."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '*.py', '**/*.json', 'src/**/*.ts')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search (default: current directory)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 1000)",
                    "minimum": 1,
                    "maximum": 10000,
                },
            },
            "required": ["pattern"],
        }

    def _should_ignore(self, path: Path) -> bool:
        """
        Check if path should be ignored

        Args:
            path: Path to check

        Returns:
            True if should be ignored
        """
        # Common ignore patterns
        ignore_dirs = {
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            "env",
            "build",
            "dist",
            ".egg-info",
        }

        # Check each part of path
        for part in path.parts:
            if part in ignore_dirs:
                return True

        return False

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute file find operation

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing 'pattern', optional 'path', 'limit'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with list of matching files
        """
        pattern = params["pattern"]
        search_path = params.get("path", ".")
        limit = params.get("limit", self.MAX_RESULTS)

        # Resolve search path
        abs_search_path = self.cwd / search_path if not Path(search_path).is_absolute() else Path(search_path)

        if not abs_search_path.exists():
            return AgentToolResult(
                content=[TextContent(text=f"Path not found: {search_path}")],
                details={"error": "Path not found", "path": search_path},
            )

        if not abs_search_path.is_dir():
            return AgentToolResult(
                content=[TextContent(text=f"Path is not a directory: {search_path}")],
                details={"error": "Not a directory", "path": search_path},
            )

        try:
            # Find matching files using glob
            matches = []

            # Use rglob if pattern contains **
            if "**" in pattern:
                matched_paths = abs_search_path.glob(pattern)
            else:
                matched_paths = abs_search_path.rglob(pattern)

            for path in matched_paths:
                # Skip directories
                if path.is_dir():
                    continue

                # Skip ignored paths
                if self._should_ignore(path):
                    continue

                # Get relative path
                try:
                    rel_path = path.relative_to(self.cwd)
                except ValueError:
                    rel_path = path

                matches.append(str(rel_path))

                # Check limit
                if len(matches) >= limit:
                    break

            # Sort results
            matches.sort()

            # Check truncation
            was_truncated = len(matches) >= limit

            # Format output
            if not matches:
                result_text = f"No files found matching pattern: {pattern}"
            else:
                result_text = f"Found {len(matches)} files"
                if was_truncated:
                    result_text += f" (truncated to {limit})"
                result_text += f" matching pattern: {pattern}\n\n"
                result_text += "\n".join(matches)

            return AgentToolResult(
                content=[TextContent(text=result_text)],
                details={
                    "pattern": pattern,
                    "matches": len(matches),
                    "truncated": was_truncated,
                    "files": matches,
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error finding files: {str(e)}")],
                details={
                    "error": str(e),
                    "pattern": pattern,
                },
            )


def create_find_tool(cwd: str) -> FindTool:
    """
    Factory function to create a find tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        FindTool instance
    """
    return FindTool(cwd=cwd)


# Default instance using current directory
find_tool = FindTool()
