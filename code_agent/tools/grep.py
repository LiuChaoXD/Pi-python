"""
Grep Tool - Search file contents

Uses Python's built-in search with regex support (simpler alternative to ripgrep).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class GrepTool:
    """
    Grep tool - Search file contents

    Features:
    - Search with regex or literal patterns
    - File filtering with glob patterns
    - Case-sensitive or case-insensitive search
    - Context lines (before/after matches)
    - Respects .gitignore (basic support)
    """

    MAX_MATCHES = 100
    MAX_BYTES = 2 * 1024 * 1024  # 2MB
    MAX_LINE_LENGTH = 5000

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Grep tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "grep"
        self.label = "Search Content"
        self.description = (
            "Search for patterns in files. Supports regex patterns and glob filtering. "
            "Returns matching lines with file paths and line numbers."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "File pattern filter (e.g., '*.py', '*.{js,ts}')",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Ignore case in pattern matching (default: false)",
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string, not regex (default: false)",
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after matches (default: 0)",
                    "minimum": 0,
                    "maximum": 10,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)",
                    "minimum": 1,
                    "maximum": 1000,
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
        ignore_patterns = {
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            "env",
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "*.pyo",
            "*.so",
            "*.dylib",
        }

        # Check each part of path
        for part in path.parts:
            if part in ignore_patterns:
                return True

        return False

    def _match_glob(self, path: Path, glob_pattern: Optional[str]) -> bool:
        """
        Check if path matches glob pattern

        Args:
            path: File path
            glob_pattern: Glob pattern (e.g., '*.py')

        Returns:
            True if matches
        """
        if not glob_pattern:
            return True

        # Handle {ext1,ext2} pattern
        if "{" in glob_pattern and "}" in glob_pattern:
            base = glob_pattern[: glob_pattern.index("{")]
            exts = glob_pattern[glob_pattern.index("{") + 1 : glob_pattern.index("}")]
            ext_list = exts.split(",")
            for ext in ext_list:
                if path.match(base + ext):
                    return True
            return False

        return path.match(glob_pattern)

    def _search_file(
        self,
        file_path: Path,
        pattern: re.Pattern,
        context: int,
    ) -> list[dict[str, Any]]:
        """
        Search a single file

        Args:
            file_path: Path to file
            pattern: Compiled regex pattern
            context: Number of context lines

        Returns:
            List of match results
        """
        matches = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                if pattern.search(line):
                    # Get context lines
                    start_line = max(0, line_num - 1 - context)
                    end_line = min(len(lines), line_num + context)
                    context_lines = lines[start_line:end_line]

                    # Truncate long lines
                    truncated_lines = []
                    for idx, ctx_line in enumerate(context_lines, start=start_line + 1):
                        if len(ctx_line) > self.MAX_LINE_LENGTH:
                            ctx_line = ctx_line[: self.MAX_LINE_LENGTH] + "... [truncated]\n"
                        truncated_lines.append((idx, ctx_line))

                    # Get relative path
                    try:
                        rel_path = file_path.relative_to(self.cwd)
                    except ValueError:
                        rel_path = file_path

                    matches.append(
                        {
                            "file": str(rel_path),
                            "line": line_num,
                            "content": line.rstrip("\n"),
                            "context_lines": truncated_lines if context > 0 else None,
                        }
                    )

        except Exception as e:
            # Skip files that can't be read
            pass

        return matches

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute grep search

        Args:
            tool_call_id: Tool call identifier
            params: Search parameters
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with search results
        """
        pattern_str = params["pattern"]
        search_path = params.get("path", ".")
        glob_pattern = params.get("glob")
        ignore_case = params.get("ignore_case", False)
        literal = params.get("literal", False)
        context = params.get("context", 0)
        limit = params.get("limit", self.MAX_MATCHES)

        # Resolve search path
        abs_search_path = self.cwd / search_path if not Path(search_path).is_absolute() else Path(search_path)

        if not abs_search_path.exists():
            return AgentToolResult(
                content=[TextContent(text=f"Path not found: {search_path}")],
                details={"error": "Path not found", "path": search_path},
            )

        # Compile pattern
        try:
            if literal:
                pattern_str = re.escape(pattern_str)

            flags = re.IGNORECASE if ignore_case else 0
            pattern = re.compile(pattern_str, flags)
        except re.error as e:
            return AgentToolResult(
                content=[TextContent(text=f"Invalid regex pattern: {str(e)}")],
                details={"error": "Invalid regex", "pattern": pattern_str},
            )

        # Search files
        all_matches = []
        files_searched = 0

        try:
            # Walk directory tree
            for root, dirs, files in os.walk(abs_search_path):
                root_path = Path(root)

                # Filter directories (modify in-place to prune tree)
                dirs[:] = [d for d in dirs if not self._should_ignore(root_path / d)]

                # Search files
                for filename in files:
                    file_path = root_path / filename

                    # Skip ignored files
                    if self._should_ignore(file_path):
                        continue

                    # Check glob pattern
                    if not self._match_glob(file_path, glob_pattern):
                        continue

                    # Search file
                    matches = self._search_file(file_path, pattern, context)
                    all_matches.extend(matches)
                    files_searched += 1

                    # Check limit
                    if len(all_matches) >= limit:
                        break

                if len(all_matches) >= limit:
                    break

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error during search: {str(e)}")], details={"error": str(e)}
            )

        # Truncate results
        was_truncated = len(all_matches) > limit
        all_matches = all_matches[:limit]

        # Format output
        if not all_matches:
            result_text = f"No matches found for pattern: {pattern_str}"
        else:
            result_text = f"Found {len(all_matches)} matches"
            if was_truncated:
                result_text += f" (truncated, showing first {limit})"
            result_text += f" in {files_searched} files\n\n"

            for match in all_matches:
                result_text += f"{match['file']}:{match['line']}: {match['content']}\n"

                # Add context lines if requested
                if match["context_lines"]:
                    for ctx_line_num, ctx_line in match["context_lines"]:
                        if ctx_line_num != match["line"]:
                            result_text += f"  {ctx_line_num}: {ctx_line.rstrip()}\n"
                    result_text += "\n"

        return AgentToolResult(
            content=[TextContent(text=result_text)],
            details={
                "pattern": pattern_str,
                "matches": len(all_matches),
                "files_searched": files_searched,
                "truncated": was_truncated,
            },
        )


def create_grep_tool(cwd: str) -> GrepTool:
    """
    Factory function to create a grep tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        GrepTool instance
    """
    return GrepTool(cwd=cwd)


# Default instance using current directory
grep_tool = GrepTool()
