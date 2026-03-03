"""
Pi Agent Core - Python implementation
Stateful agent with tool execution and event streaming.

This is a Python port of the TypeScript @mariozechner/pi-agent-core package.
"""

"""
Pi Agent Core - Pure agent functionality without file system dependencies

This package provides the core agent loop, message handling, and LLM provider integration.
For file system operations and coding tools, see pi_coding_agent.
"""

from .agent import Agent, default_convert_to_llm
from .agent_loop import agent_loop, agent_loop_continue
from .proxy import ProxyStreamOptions, stream_proxy
from .types import (  # Core types; Message types; Content types; Event types; Model and config; Other types
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentState,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    AssistantMessage,
    AssistantMessageEvent,
    Content,
    Cost,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamFn,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
    UserMessage,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "Agent",
    "default_convert_to_llm",
    # Loop functions
    "agent_loop",
    "agent_loop_continue",
    # Proxy
    "stream_proxy",
    "ProxyStreamOptions",
    # Core types
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    # Message types
    "Message",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    # Content types
    "Content",
    "TextContent",
    "ImageContent",
    "ThinkingContent",
    "ToolCall",
    # Event types
    "AssistantMessageEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    # Model and config
    "Model",
    "ThinkingLevel",
    "ThinkingBudgets",
    "SimpleStreamOptions",
    "StreamFn",
    # Other types
    "Usage",
    "Cost",
    "StopReason",
]
