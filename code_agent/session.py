"""
Session Management - Persist and restore agent conversations

Provides functionality to save, load, and manage agent sessions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from agent_core import (
    AgentMessage,
    AssistantMessage,
    Model,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

load_dotenv()
INFO_DIR = os.getenv("INFO_DIR", "./.personal")
SESSIONS_DIR = os.path.join(INFO_DIR, "sessions")


@dataclass
class SessionMetadata:
    """Metadata for a session"""

    session_id: str
    created_at: str
    updated_at: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = None
    message_count: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SessionManager:
    """
    Manage agent sessions with persistence

    Sessions are stored as JSON files in a sessions directory.
    Each session contains:
    - Metadata (ID, timestamps, title, etc.)
    - Agent configuration (system prompt, model, thinking level)
    - Conversation history (messages)
    """

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        """
        Initialize session manager

        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get path to session file"""
        return self.sessions_dir / f"{session_id}.json"

    def _serialize_message(self, message: AgentMessage) -> dict[str, Any]:
        """Serialize a message to JSON-compatible dict"""
        msg_dict = {
            "role": message.role,
            "timestamp": message.timestamp,
        }

        # Serialize content
        if hasattr(message, "content"):
            content_list = []
            for content in message.content:
                if isinstance(content, TextContent):
                    content_list.append(
                        {
                            "type": "text",
                            "text": content.text,
                        }
                    )
                elif isinstance(content, ToolCall):
                    # Serialize tool calls as part of content
                    content_list.append(
                        {
                            "type": "toolCall",
                            "id": content.id,
                            "name": content.name,
                            "arguments": content.arguments,
                        }
                    )
                # Add other content types as needed
            msg_dict["content"] = content_list

        # Serialize tool result
        if isinstance(message, ToolResultMessage):
            msg_dict["toolCallId"] = message.toolCallId
            msg_dict["toolName"] = message.toolName
            if message.isError:
                msg_dict["isError"] = True

        return msg_dict

    def _deserialize_message(self, msg_dict: dict[str, Any]) -> AgentMessage:
        """Deserialize a message from JSON dict"""
        role = msg_dict["role"]
        timestamp = msg_dict["timestamp"]

        # Deserialize content
        content = []
        if "content" in msg_dict:
            for content_item in msg_dict["content"]:
                if content_item["type"] == "text":
                    content.append(TextContent(text=content_item["text"]))
                elif content_item["type"] == "toolCall":
                    # Deserialize tool calls from content
                    content.append(
                        ToolCall(
                            id=content_item["id"],
                            name=content_item["name"],
                            arguments=content_item["arguments"],
                        )
                    )

        # Create appropriate message type
        if role == "user":
            return UserMessage(content=content, timestamp=timestamp)

        elif role == "assistant":
            return AssistantMessage(
                content=content,
                timestamp=timestamp,
            )

        elif role == "toolResult":
            return ToolResultMessage(
                toolCallId=msg_dict["toolCallId"],
                toolName=msg_dict["toolName"],
                content=content,
                timestamp=timestamp,
                isError=msg_dict.get("isError", False),
            )

        else:
            raise ValueError(f"Unknown message role: {role}")

    def save_session(
        self,
        session_id: str,
        system_prompt: str,
        model: Model,
        thinking_level: ThinkingLevel,
        messages: list[AgentMessage],
        metadata: Optional[SessionMetadata] = None,
    ) -> Path:
        """
        Save a session to disk

        Args:
            session_id: Session identifier
            system_prompt: System prompt
            model: Model configuration
            thinking_level: Thinking level
            messages: Conversation messages
            metadata: Optional session metadata

        Returns:
            Path to saved session file
        """
        # Create or update metadata
        now = datetime.now().isoformat()

        if metadata is None:
            metadata = SessionMetadata(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                message_count=len(messages),
            )
        else:
            metadata.updated_at = now
            metadata.message_count = len(messages)

        # Build session data
        session_data = {
            "metadata": asdict(metadata),
            "config": {
                "systemPrompt": system_prompt,
                "model": {
                    "api": model.api,
                    "provider": model.provider,
                    "id": model.id,
                    "baseUrl": model.baseUrl,
                    "name": model.name,
                },
                "thinkingLevel": thinking_level.value if isinstance(thinking_level, ThinkingLevel) else thinking_level,
            },
            "messages": [self._serialize_message(msg) for msg in messages],
        }

        # Save to file
        session_path = self._get_session_path(session_id)
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return session_path

    def load_session(self, session_id: str) -> dict[str, Any]:
        """
        Load a session from disk

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing:
            - metadata: SessionMetadata
            - system_prompt: str
            - model: Model
            - thinking_level: ThinkingLevel
            - messages: list[AgentMessage]

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        # Load from file
        with open(session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Parse metadata
        metadata = SessionMetadata(**session_data["metadata"])

        # Parse config
        config = session_data["config"]
        model = Model(
            api=config["model"]["api"],
            provider=config["model"]["provider"],
            id=config["model"]["id"],
            baseUrl=config["model"].get("baseUrl"),
            name=config["model"]["name"],
        )

        if model.provider == "openai" and not model.baseUrl:
            model.baseUrl = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")

        thinking_level = ThinkingLevel(config["thinkingLevel"])

        # Parse messages
        messages = [self._deserialize_message(msg_dict) for msg_dict in session_data["messages"]]

        return {
            "metadata": metadata,
            "system_prompt": config["systemPrompt"],
            "model": model,
            "thinking_level": thinking_level,
            "messages": messages,
        }

    def list_sessions(self) -> list[SessionMetadata]:
        """
        List all sessions

        Returns:
            List of session metadata, sorted by updated_at (newest first)
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    metadata = SessionMetadata(**session_data["metadata"])
                    sessions.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load session {session_file}: {e}")

        # Sort by updated_at (newest first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        session_path = self._get_session_path(session_id)

        if session_path.exists():
            session_path.unlink()
            return True

        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        return self._get_session_path(session_id).exists()

    def update_metadata(
        self,
        session_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        """
        Update session metadata

        Args:
            session_id: Session identifier
            title: New title (if provided)
            description: New description (if provided)
            tags: New tags (if provided)
        """
        session_data = self.load_session(session_id)
        metadata = session_data["metadata"]

        if title is not None:
            metadata.title = title
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags

        # Re-save session
        self.save_session(
            session_id=session_id,
            system_prompt=session_data["system_prompt"],
            model=session_data["model"],
            thinking_level=session_data["thinking_level"],
            messages=session_data["messages"],
            metadata=metadata,
        )


class CodingSession:
    """
    Convenient wrapper for CodingAgent with automatic session management

    This class combines CodingAgent with SessionManager to provide
    automatic session persistence.
    """

    def __init__(
        self,
        session_id: str,
        coding_agent: Any,  # CodingAgent instance
        session_manager: Optional[SessionManager] = None,
        auto_save: bool = True,
    ):
        """
        Initialize coding session

        Args:
            session_id: Session identifier
            coding_agent: CodingAgent instance
            session_manager: SessionManager (default: creates new one)
            auto_save: Automatically save after each prompt
        """
        self.session_id = session_id
        self.agent = coding_agent
        self.session_manager = session_manager or SessionManager()
        self.auto_save = auto_save

    async def prompt(self, text: str, images: Any = None):
        """
        Send a prompt and optionally auto-save

        Args:
            text: Prompt text
            images: Optional images
        """
        result = await self.agent.prompt(text, images)

        if self.auto_save:
            self.save()

        return result

    def save(self, title: Optional[str] = None, description: Optional[str] = None):
        """
        Save current session

        Args:
            title: Session title
            description: Session description
        """
        # Get current metadata if exists
        metadata = None
        if self.session_manager.session_exists(self.session_id):
            existing = self.session_manager.load_session(self.session_id)
            metadata = existing["metadata"]

            if title:
                metadata.title = title
            if description:
                metadata.description = description
        else:
            # Create new metadata
            metadata = SessionMetadata(
                session_id=self.session_id,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                title=title,
                description=description,
            )

        self.session_manager.save_session(
            session_id=self.session_id,
            system_prompt=self.agent.state.systemPrompt,
            model=self.agent.state.model,
            thinking_level=self.agent.state.thinkingLevel,
            messages=self.agent.state.messages,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls,
        session_id: str,
        coding_agent_class: Any,  # CodingAgent class
        session_manager: Optional[SessionManager] = None,
        auto_save: bool = True,
        **agent_kwargs,
    ) -> "CodingSession":
        """
        Load an existing session

        Args:
            session_id: Session identifier
            coding_agent_class: CodingAgent class
            session_manager: SessionManager
            auto_save: Enable auto-save
            **agent_kwargs: Additional args for CodingAgent

        Returns:
            CodingSession instance with restored state
        """
        session_manager = session_manager or SessionManager()

        # Load session data
        session_data = session_manager.load_session(session_id)

        # Create agent with restored state
        agent = coding_agent_class(
            model=session_data["model"],
            system_prompt=session_data["system_prompt"],
            thinking_level=session_data["thinking_level"],
            **agent_kwargs,
        )

        # Restore messages
        agent.agent.replace_messages(session_data["messages"])

        return cls(
            session_id=session_id,
            coding_agent=agent,
            session_manager=session_manager,
            auto_save=auto_save,
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        coding_agent_class: Any,
        session_manager: Optional[SessionManager] = None,
        auto_save: bool = True,
        **agent_kwargs,
    ) -> "CodingSession":
        """
        Create a new session

        Args:
            session_id: Session identifier
            coding_agent_class: CodingAgent class
            session_manager: SessionManager
            auto_save: Enable auto-save
            **agent_kwargs: Args for CodingAgent

        Returns:
            New CodingSession instance
        """
        session_manager = session_manager or SessionManager()

        # Create new agent
        agent = coding_agent_class(**agent_kwargs)

        return cls(
            session_id=session_id,
            coding_agent=agent,
            session_manager=session_manager,
            auto_save=auto_save,
        )
