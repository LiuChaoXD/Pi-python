"""
Memory tools based on per-session markdown files.
"""

from __future__ import annotations

from typing import Any

from agent_core import AgentToolResult, TextContent

from .memory import MemoryManager


class MemoryWriteTool:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.name = "memory_write"
        self.label = "Memory Write"
        self.description = "Write content to memory markdown file for a session."
        self.parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID. If omitted, uses default session_id.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["append", "overwrite"],
                    "description": "Write mode (default: append)",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        try:
            result = await self.memory_manager.write(
                content=params["content"],
                session_id=params.get("session_id"),
                mode=params.get("mode", "append"),
            )
            return AgentToolResult(
                content=[TextContent(text=f"Memory updated: {result['path']}")],
                details={"success": True, **result},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Failed to write memory: {e}")],
                details={"success": False, "error": str(e)},
            )


class MemoryReadTool:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.name = "memory_read"
        self.label = "Memory Read"
        self.description = "Read memory file for a session or search by keyword."
        self.parameters = {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID. If omitted, uses default session_id.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword for line-based search",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Max results when keyword is provided (default: 20)",
                },
            },
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        try:
            result = await self.memory_manager.read(
                session_id=params.get("session_id"),
                keyword=params.get("keyword"),
                limit=int(params.get("limit", 20)),
            )

            if not result.get("keyword"):
                content = result["content"]
                if len(content) > 8000:
                    content = content[:8000] + "\n...\n[truncated]"
                return AgentToolResult(
                    content=[TextContent(text=content)],
                    details={"success": True, **result},
                )

            matches = result["matches"]
            if not matches:
                text = f"No matches for keyword: {result['keyword']}"
            else:
                text = "\n".join([f"{m['line']}: {m['text']}" for m in matches])
                if result["total"] > len(matches):
                    text += f"\n... and {result['total'] - len(matches)} more matches"

            return AgentToolResult(
                content=[TextContent(text=text)],
                details={"success": True, **result},
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Failed to read memory: {e}")],
                details={"success": False, "error": str(e)},
            )
