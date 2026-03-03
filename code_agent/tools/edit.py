"""
Edit Tool - Precise file editing with find and replace

Performs exact text replacement with fuzzy matching fallback.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class EditTool:
    """
    Edit tool - Precise file editing

    Features:
    - Exact text replacement (including whitespace and indentation)
    - Automatic line ending detection (CRLF/LF)
    - Fuzzy matching fallback
    - Unified diff generation
    """

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Edit tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "edit"
        self.label = "Edit File"
        self.description = (
            "Edit a file by finding and replacing exact text. "
            "The old_text must match exactly (including whitespace). "
            "If exact match fails, fuzzy matching will be attempted."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find (must match exactly including whitespace)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working directory"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.cwd / p

    def _detect_line_ending(self, content: str) -> str:
        """
        Detect line ending style (CRLF or LF)

        Args:
            content: File content

        Returns:
            Line ending string ('\r\n' or '\n')
        """
        if "\r\n" in content:
            return "\r\n"
        return "\n"

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (remove BOM, normalize line endings)

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Remove BOM
        if text.startswith("\ufeff"):
            text = text[1:]

        # Normalize line endings to \n
        text = text.replace("\r\n", "\n")

        return text

    def _find_fuzzy_match(self, content: str, old_text: str) -> Optional[tuple[int, int]]:
        """
        Find fuzzy match for old_text in content

        Args:
            content: File content
            old_text: Text to find

        Returns:
            Tuple of (start_pos, end_pos) if found, None otherwise
        """
        # Split into lines for better matching
        content_lines = content.split("\n")
        old_lines = old_text.split("\n")

        if not old_lines:
            return None

        # Try to find a sequence of lines with high similarity
        best_match = None
        best_ratio = 0.0
        threshold = 0.8  # 80% similarity

        for i in range(len(content_lines) - len(old_lines) + 1):
            window = content_lines[i : i + len(old_lines)]
            window_text = "\n".join(window)

            # Calculate similarity
            ratio = difflib.SequenceMatcher(None, window_text, old_text).ratio()

            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                # Calculate position in original content
                start_pos = len("\n".join(content_lines[:i]))
                if i > 0:
                    start_pos += 1  # Add newline
                end_pos = start_pos + len(window_text)
                best_match = (start_pos, end_pos)

        return best_match

    def _generate_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """
        Generate unified diff

        Args:
            old_content: Original content
            new_content: New content
            filename: File name for diff header

        Returns:
            Unified diff string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )

        return "".join(diff)

    def _find_first_changed_line(self, old_content: str, new_content: str) -> int:
        """
        Find the first line number that changed

        Args:
            old_content: Original content
            new_content: New content

        Returns:
            Line number (1-indexed) of first change
        """
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines), start=1):
            if old_line != new_line:
                return i

        # If all common lines are same, change is at the end
        return min(len(old_lines), len(new_lines)) + 1

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute file edit operation

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing 'path', 'old_text', 'new_text'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with edit status and diff
        """
        path_str = params["path"]
        old_text = params["old_text"]
        new_text = params["new_text"]

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

        try:
            # Read file
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                original_content = f.read()

            # Detect line ending
            line_ending = self._detect_line_ending(original_content)

            # Normalize content for matching
            normalized_content = self._normalize_text(original_content)
            normalized_old = self._normalize_text(old_text)
            normalized_new = self._normalize_text(new_text)

            # Try exact match first
            if normalized_old in normalized_content:
                new_content = normalized_content.replace(normalized_old, normalized_new, 1)
                match_type = "exact"
            else:
                # Try fuzzy match
                fuzzy_result = self._find_fuzzy_match(normalized_content, normalized_old)
                if fuzzy_result:
                    start_pos, end_pos = fuzzy_result
                    new_content = normalized_content[:start_pos] + normalized_new + normalized_content[end_pos:]
                    match_type = "fuzzy"
                else:
                    # No match found
                    return AgentToolResult(
                        content=[
                            TextContent(
                                text=f"Could not find text to replace in {path_str}. "
                                f"Make sure the old_text matches exactly (including whitespace)."
                            )
                        ],
                        details={
                            "error": "Text not found",
                            "path": path_str,
                            "old_text_length": len(old_text),
                        },
                    )

            # Restore original line endings
            if line_ending == "\r\n":
                new_content = new_content.replace("\n", "\r\n")

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Generate diff
            diff = self._generate_diff(original_content, new_content, path.name)

            # Find first changed line
            first_changed_line = self._find_first_changed_line(original_content, new_content)

            # Success message
            msg = f"Successfully edited {path.name}"
            if match_type == "fuzzy":
                msg += " (used fuzzy matching)"

            return AgentToolResult(
                content=[TextContent(text=f"{msg}\n\nDiff:\n{diff}")],
                details={
                    "path": str(path),
                    "match_type": match_type,
                    "first_changed_line": first_changed_line,
                    "diff": diff,
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error editing file {path_str}: {str(e)}")],
                details={
                    "error": str(e),
                    "path": path_str,
                },
            )


def create_edit_tool(cwd: str) -> EditTool:
    """
    Factory function to create an edit tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        EditTool instance
    """
    return EditTool(cwd=cwd)


# Default instance using current directory
edit_tool = EditTool()
