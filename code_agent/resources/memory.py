"""
Simple Memory Manager

Storage layout:
- .personal/memory/<session_id>.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
INFO_DIR = os.getenv("INFO_DIR", "./.personal")
SKILLS_DIR = os.path.join(INFO_DIR, "skills")
MEMORY_DIR = os.path.join(INFO_DIR, "memory")


class MemoryManager:
    SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")

    def __init__(self, memory_dir: Optional[str] = None, default_session_id: Optional[str] = None):
        if memory_dir is None:
            memory_dir = MEMORY_DIR
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.default_session_id = (default_session_id or "default").strip()

    def _normalize_session_id(self, session_id: Optional[str]) -> str:
        sid = (session_id or self.default_session_id).strip()
        if not sid:
            sid = "default"
        if not self.SESSION_ID_PATTERN.fullmatch(sid):
            raise ValueError("Invalid session_id. Use letters/numbers/._- only.")
        return sid

    def _memory_path(self, session_id: str) -> Path:
        return self.memory_dir / f"{session_id}.md"

    def _ensure_file(self, path: Path, session_id: str) -> None:
        if not path.exists():
            path.write_text(f"# Memory ({session_id})\n\n", encoding="utf-8")

    async def write(self, content: str, session_id: Optional[str] = None, mode: str = "append") -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be 'append' or 'overwrite'")

        sid = self._normalize_session_id(session_id)
        path = self._memory_path(sid)
        self._ensure_file(path, sid)

        if mode == "overwrite":
            new_text = content
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
        else:
            old_text = path.read_text(encoding="utf-8")
            add_text = content
            if old_text and not old_text.endswith("\n"):
                old_text += "\n"
            if add_text and not add_text.endswith("\n"):
                add_text += "\n"
            new_text = old_text + add_text

        path.write_text(new_text, encoding="utf-8")

        return {
            "session_id": sid,
            "path": str(path),
            "mode": mode,
            "size": path.stat().st_size,
        }

    async def read(
        self,
        session_id: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        if limit <= 0:
            raise ValueError("limit must be > 0")

        sid = self._normalize_session_id(session_id)
        path = self._memory_path(sid)
        self._ensure_file(path, sid)

        text = path.read_text(encoding="utf-8")

        if not keyword:
            return {
                "session_id": sid,
                "path": str(path),
                "keyword": None,
                "content": text,
            }

        kw = keyword.lower().strip()
        matches: list[dict[str, Any]] = []

        for line_no, line in enumerate(text.splitlines(), start=1):
            if kw in line.lower():
                matches.append({"line": line_no, "text": line})

        return {
            "session_id": sid,
            "path": str(path),
            "keyword": keyword,
            "total": len(matches),
            "matches": matches[:limit],
        }
