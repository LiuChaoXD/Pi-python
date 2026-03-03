from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeAlias, Union


# Thinking/reasoning level for models that support it
class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # Only supported by specific OpenAI models


# Stop reasons
StopReason: TypeAlias = Literal["stop", "length", "toolUse", "aborted", "error"]


# Content types
@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""
    textSignature: str | None = None


@dataclass
class ImageContent:
    data: str  # base64 encoded
    mimeType: str
    type: Literal["image"] = "image"


@dataclass
class ThinkingContent:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinkingSignature: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"


Content: TypeAlias = Union[TextContent, ImageContent, ThinkingContent, ToolCall]


# Usage information
@dataclass
class Cost:
    input: float = 0.0
    output: float = 0.0
    cacheRead: float = 0.0
    cacheWrite: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cacheRead: int = 0
    cacheWrite: int = 0
    totalTokens: int = 0
    cost: Cost = field(default_factory=Cost)


# Message types
@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent] = field(default_factory=list)
    timestamp: int = 0


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[Content] = field(default_factory=list)
    stopReason: StopReason = "stop"
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    timestamp: int = 0
    errorMessage: str | None = None


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    toolCallId: str = ""
    toolName: str = ""
    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    isError: bool = False
    timestamp: int = 0


# Base message type (LLM-compatible messages)
Message: TypeAlias = Union[UserMessage, AssistantMessage, ToolResultMessage]


# AgentMessage: can be extended by applications
# In TypeScript this uses declaration merging, in Python we use a more direct union
AgentMessage: TypeAlias = Message


# Model information
@dataclass
class Model:
    api: str
    provider: str
    id: str
    baseUrl: str | None = None
    name: str | None = None
    contextWindow: int | None = None
    maxOutput: int | None = None
    pricing: dict[str, Any] = field(default_factory=dict)


# Tool result
@dataclass
class AgentToolResult:
    """Result from a tool execution."""

    content: list[TextContent | ImageContent]
    details: dict[str, Any]


# Tool update callback
AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], None]


# AgentTool protocol
class AgentTool(Protocol):
    """Protocol for agent tools."""

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,  # AbortSignal equivalent
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """Execute the tool with given parameters."""
        ...


# Context for agent operations
@dataclass
class AgentContext:
    systemPrompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


# Agent state
@dataclass
class AgentState:
    systemPrompt: str = ""
    model: Model | None = None
    thinkingLevel: ThinkingLevel = ThinkingLevel.OFF
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    isStreaming: bool = False
    streamMessage: AgentMessage | None = None
    pendingToolCalls: set[str] = field(default_factory=set)
    error: str | None = None


# Assistant message events
@dataclass
class StartEvent:
    type: Literal["start"] = "start"
    partial: AssistantMessage | None = None


@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"
    contentIndex: int = 0
    partial: AssistantMessage | None = None


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    contentIndex: int = 0
    delta: str = ""
    partial: AssistantMessage | None = None


@dataclass
class TextEndEvent:
    type: Literal["text_end"] = "text_end"
    contentIndex: int = 0
    content: str = ""
    partial: AssistantMessage | None = None


@dataclass
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"
    contentIndex: int = 0
    partial: AssistantMessage | None = None


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    contentIndex: int = 0
    delta: str = ""
    partial: AssistantMessage | None = None


@dataclass
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"
    contentIndex: int = 0
    content: str = ""
    partial: AssistantMessage | None = None


@dataclass
class ToolCallStartEvent:
    type: Literal["toolcall_start"] = "toolcall_start"
    contentIndex: int = 0
    partial: AssistantMessage | None = None


@dataclass
class ToolCallDeltaEvent:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    contentIndex: int = 0
    delta: str = ""
    partial: AssistantMessage | None = None


@dataclass
class ToolCallEndEvent:
    type: Literal["toolcall_end"] = "toolcall_end"
    contentIndex: int = 0
    toolCall: ToolCall | None = None
    partial: AssistantMessage | None = None


@dataclass
class DoneEvent:
    type: Literal["done"] = "done"
    reason: StopReason = "stop"
    message: AssistantMessage | None = None


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    reason: StopReason = "error"
    error: AssistantMessage | None = None


AssistantMessageEvent: TypeAlias = Union[
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]


# Agent events
@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    toolResults: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: Literal["message_start"] = "message_start"
    message: AgentMessage | None = None


@dataclass
class MessageUpdateEvent:
    type: Literal["message_update"] = "message_update"
    message: AgentMessage | None = None
    assistantMessageEvent: AssistantMessageEvent | None = None


@dataclass
class MessageEndEvent:
    type: Literal["message_end"] = "message_end"
    message: AgentMessage | None = None


@dataclass
class ToolExecutionStartEvent:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    toolCallId: str = ""
    toolName: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionUpdateEvent:
    type: Literal["tool_execution_update"] = "tool_execution_update"
    toolCallId: str = ""
    toolName: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    partialResult: AgentToolResult | None = None


@dataclass
class ToolExecutionEndEvent:
    type: Literal["tool_execution_end"] = "tool_execution_end"
    toolCallId: str = ""
    toolName: str = ""
    result: AgentToolResult | None = None
    isError: bool = False


AgentEvent: TypeAlias = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]


# Thinking budgets
@dataclass
class ThinkingBudgets:
    minimal: int = 128
    low: int = 512
    medium: int = 1024
    high: int = 2048
    xhigh: int = 8192


# Stream options
@dataclass
class SimpleStreamOptions:
    apiKey: str | None = None
    temperature: float | None = None
    maxTokens: int | None = None
    reasoning: ThinkingLevel | None = None
    sessionId: str | None = None
    thinkingBudgets: ThinkingBudgets | None = None
    maxRetryDelayMs: int | None = None


# Stream function type
StreamFn: TypeAlias = Callable[
    [Model, AgentContext, SimpleStreamOptions],
    Awaitable[Any],  # Should return an async generator
]


# Agent loop configuration
@dataclass
class AgentLoopConfig:
    model: Model
    convertToLlm: Callable[[list[AgentMessage]], Awaitable[list[Message]]]
    reasoning: ThinkingLevel | None = None
    sessionId: str | None = None
    thinkingBudgets: ThinkingBudgets | None = None
    maxRetryDelayMs: int | None = None
    apiKey: str | None = None
    temperature: float | None = None
    maxTokens: int | None = None
    transformContext: Callable[[list[AgentMessage], Any | None], Awaitable[list[AgentMessage]]] | None = None
    getApiKey: Callable[[str], Awaitable[str | None]] | None = None
    getSteeringMessages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    getFollowUpMessages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
