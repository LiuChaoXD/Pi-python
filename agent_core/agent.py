from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from .agent_loop import agent_loop, agent_loop_continue
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    Model,
    StreamFn,
    TextContent,
    ThinkingBudgets,
    ThinkingLevel,
    Usage,
    UserMessage,
)


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """
    Default convertToLlm: Keep only LLM-compatible messages.
    """
    return [msg for msg in messages if hasattr(msg, "role") and msg.role in ["user", "assistant", "toolResult"]]


class Agent:
    """
    Stateful agent with tool execution and event streaming.
    """

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] | None = None,
        transform_context: Callable[[list[AgentMessage], Any | None], Awaitable[list[AgentMessage]]] | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        stream_fn: StreamFn | None = None,
        session_id: str | None = None,
        get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
        max_retry_delay_ms: int | None = None,
    ):
        """
        Initialize the agent.

        Args:
            initial_state: Initial agent state (systemPrompt, model, etc.)
            convert_to_llm: Function to convert AgentMessage[] to LLM Message[]
            transform_context: Optional context transformation before LLM call
            steering_mode: "all" or "one-at-a-time" for steering messages
            follow_up_mode: "all" or "one-at-a-time" for follow-up messages
            stream_fn: Custom stream function for LLM calls
            session_id: Session ID for provider caching
            get_api_key: Function to resolve API keys dynamically
            thinking_budgets: Custom token budgets for thinking levels
            max_retry_delay_ms: Maximum retry delay in milliseconds
        """
        self._state = AgentState(
            systemPrompt="",
            model=None,
            thinkingLevel=ThinkingLevel.OFF,
            tools=[],
            messages=[],
            isStreaming=False,
            streamMessage=None,
            pendingToolCalls=set(),
            error=None,
        )

        # Apply initial state
        if initial_state:
            if "systemPrompt" in initial_state:
                self._state.systemPrompt = initial_state["systemPrompt"]
            if "model" in initial_state:
                self._state.model = initial_state["model"]
            if "thinkingLevel" in initial_state:
                self._state.thinkingLevel = initial_state["thinkingLevel"]
            if "tools" in initial_state:
                self._state.tools = initial_state["tools"]
            if "messages" in initial_state:
                self._state.messages = initial_state["messages"]

        self._listeners: set[Callable[[AgentEvent], None]] = set()
        self._abort_controller: Any | None = None
        self._convert_to_llm = convert_to_llm or default_convert_to_llm
        self._transform_context = transform_context
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._stream_fn = stream_fn
        self._session_id = session_id
        self._get_api_key = get_api_key
        self._running_prompt: asyncio.Task | None = None
        self._thinking_budgets = thinking_budgets
        self._max_retry_delay_ms = max_retry_delay_ms

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None):
        """Set the session ID for provider caching."""
        self._session_id = value

    @property
    def thinking_budgets(self) -> ThinkingBudgets | None:
        """Get the current thinking budgets."""
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: ThinkingBudgets | None):
        """Set custom thinking budgets."""
        self._thinking_budgets = value

    @property
    def max_retry_delay_ms(self) -> int | None:
        """Get the current max retry delay."""
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: int | None):
        """Set the maximum retry delay."""
        self._max_retry_delay_ms = value

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """
        Subscribe to agent events.

        Args:
            fn: Callback function to receive events

        Returns:
            Unsubscribe function
        """
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self._state.systemPrompt = prompt

    def set_model(self, model: Model):
        """Set the LLM model."""
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel):
        """Set the thinking/reasoning level."""
        self._state.thinkingLevel = level

    def set_steering_mode(self, mode: str):
        """Set steering mode: "all" or "one-at-a-time"."""
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        """Get current steering mode."""
        return self._steering_mode

    def set_follow_up_mode(self, mode: str):
        """Set follow-up mode: "all" or "one-at-a-time"."""
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> str:
        """Get current follow-up mode."""
        return self._follow_up_mode

    def set_tools(self, tools: list[AgentTool]):
        """Set the available tools."""
        self._state.tools = tools

    def replace_messages(self, messages: list[AgentMessage]):
        """Replace all messages."""
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage):
        """Append a message to the conversation."""
        self._state.messages.append(message)

    def steer(self, message: AgentMessage):
        """
        Queue a steering message to interrupt the agent mid-run.
        Delivered after current tool execution, skips remaining tools.
        """
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage):
        """
        Queue a follow-up message to be processed after the agent finishes.
        Delivered only when agent has no more tool calls or steering messages.
        """
        self._follow_up_queue.append(message)

    def clear_steering_queue(self):
        """Clear all steering messages."""
        self._steering_queue = []

    def clear_follow_up_queue(self):
        """Clear all follow-up messages."""
        self._follow_up_queue = []

    def clear_all_queues(self):
        """Clear both steering and follow-up queues."""
        self._steering_queue = []
        self._follow_up_queue = []

    def has_queued_messages(self) -> bool:
        """Check if there are any queued messages."""
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    def _dequeue_steering_messages(self) -> list[AgentMessage]:
        """Dequeue steering messages based on mode."""
        if self._steering_mode == "one-at-a-time":
            if len(self._steering_queue) > 0:
                first = self._steering_queue[0]
                self._steering_queue = self._steering_queue[1:]
                return [first]
            return []

        steering = list(self._steering_queue)
        self._steering_queue = []
        return steering

    def _dequeue_follow_up_messages(self) -> list[AgentMessage]:
        """Dequeue follow-up messages based on mode."""
        if self._follow_up_mode == "one-at-a-time":
            if len(self._follow_up_queue) > 0:
                first = self._follow_up_queue[0]
                self._follow_up_queue = self._follow_up_queue[1:]
                return [first]
            return []

        follow_up = list(self._follow_up_queue)
        self._follow_up_queue = []
        return follow_up

    def clear_messages(self):
        """Clear all messages."""
        self._state.messages = []

    def abort(self):
        """Abort the current operation."""
        if self._abort_controller:
            # Signal abort (implementation depends on abort mechanism)
            pass

    async def wait_for_idle(self):
        """Wait for current operation to complete."""
        if self._running_prompt:
            await self._running_prompt

    def reset(self):
        """Reset agent state."""
        self._state.messages = []
        self._state.isStreaming = False
        self._state.streamMessage = None
        self._state.pendingToolCalls = set()
        self._state.error = None
        self._steering_queue = []
        self._follow_up_queue = []

    async def prompt(
        self,
        input_msg: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ):
        """
        Send a prompt to the agent.

        Args:
            input_msg: Text prompt, single message, or list of messages
            images: Optional images to include with text prompt
        """
        if self._state.isStreaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages, or wait for completion."
            )

        if not self._state.model:
            raise ValueError("No model configured")

        # Convert input to messages
        if isinstance(input_msg, list):
            msgs = input_msg
        elif isinstance(input_msg, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input_msg)]
            if images:
                content.extend(images)
            msgs = [UserMessage(content=content, timestamp=int(time.time() * 1000))]
        else:
            msgs = [input_msg]

        await self._run_loop(msgs)

    async def continue_conversation(self):
        """
        Continue from current context (used for retries and resuming queued messages).
        """
        if self._state.isStreaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        if len(messages) == 0:
            raise ValueError("No messages to continue from")

        last_msg = messages[-1]
        if hasattr(last_msg, "role") and last_msg.role == "assistant":
            queued_steering = self._dequeue_steering_messages()
            if len(queued_steering) > 0:
                await self._run_loop(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_up = self._dequeue_follow_up_messages()
            if len(queued_follow_up) > 0:
                await self._run_loop(queued_follow_up)
                return

            raise ValueError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    async def _run_loop(
        self,
        messages: list[AgentMessage] | None,
        skip_initial_steering_poll: bool = False,
    ):
        """
        Run the agent loop.
        If messages are provided, starts a new conversation turn.
        Otherwise, continues from existing context.
        """
        if not self._state.model:
            raise ValueError("No model configured")

        self._state.isStreaming = True
        self._state.streamMessage = None
        self._state.error = None

        reasoning = None if self._state.thinkingLevel == ThinkingLevel.OFF else self._state.thinkingLevel

        context = AgentContext(
            systemPrompt=self._state.systemPrompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

        async def get_steering():
            if skip_initial_steering_poll:
                return []
            return self._dequeue_steering_messages()

        async def get_follow_up():
            return self._dequeue_follow_up_messages()

        # Convert convert_to_llm to async if needed
        async def convert_wrapper(msgs):
            result = self._convert_to_llm(msgs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        config = AgentLoopConfig(
            model=self._state.model,
            reasoning=reasoning,
            sessionId=self._session_id,
            thinkingBudgets=self._thinking_budgets,
            maxRetryDelayMs=self._max_retry_delay_ms,
            convertToLlm=convert_wrapper,
            transformContext=self._transform_context,
            getApiKey=self._get_api_key,
            getSteeringMessages=get_steering,
            getFollowUpMessages=get_follow_up,
        )

        partial: AgentMessage | None = None

        try:
            if messages:
                stream = agent_loop(messages, context, config, None, self._stream_fn)
            else:
                stream = agent_loop_continue(context, config, None, self._stream_fn)

            async for event in stream:
                # Update internal state based on events
                if event.type == "message_start":
                    partial = event.message
                    self._state.streamMessage = event.message
                elif event.type == "message_update":
                    partial = event.message
                    self._state.streamMessage = event.message
                elif event.type == "message_end":
                    partial = None
                    self._state.streamMessage = None
                    self.append_message(event.message)
                elif event.type == "tool_execution_start":
                    self._state.pendingToolCalls.add(event.toolCallId)
                elif event.type == "tool_execution_end":
                    self._state.pendingToolCalls.discard(event.toolCallId)
                elif event.type == "turn_end":
                    if hasattr(event.message, "errorMessage") and event.message.errorMessage:
                        self._state.error = event.message.errorMessage
                elif event.type == "agent_end":
                    self._state.isStreaming = False
                    self._state.streamMessage = None

                # Emit to listeners
                self._emit(event)

            # Handle any remaining partial message
            if partial and hasattr(partial, "role") and partial.role == "assistant":
                content = getattr(partial, "content", [])
                if len(content) > 0:
                    has_content = any(
                        (hasattr(c, "type") and c.type == "thinking" and hasattr(c, "thinking") and c.thinking.strip())
                        or (hasattr(c, "type") and c.type == "text" and hasattr(c, "text") and c.text.strip())
                        or (hasattr(c, "type") and c.type == "toolCall" and hasattr(c, "name") and c.name.strip())
                        for c in content
                    )
                    if has_content:
                        self.append_message(partial)

        except Exception as err:
            error_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(text="")],
                api=self._state.model.api,
                provider=self._state.model.provider,
                model=self._state.model.id,
                usage=Usage(
                    input=0,
                    output=0,
                    cacheRead=0,
                    cacheWrite=0,
                    totalTokens=0,
                    cost=Cost(),
                ),
                stopReason="error",
                errorMessage=str(err),
                timestamp=int(time.time() * 1000),
            )

            self.append_message(error_msg)
            self._state.error = str(err)
            self._emit({"type": "agent_end", "messages": [error_msg]})

        finally:
            self._state.isStreaming = False
            self._state.streamMessage = None
            self._state.pendingToolCalls = set()
            self._abort_controller = None
            self._running_prompt = None

    def _emit(self, event: AgentEvent | dict):
        """Emit an event to all listeners."""
        for listener in self._listeners:
            listener(event)
