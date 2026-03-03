from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from .logging import get_logger
from .providers.base import stream_simple as provider_stream_simple
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    AssistantMessageEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SimpleStreamOptions,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
)

logger = get_logger(__name__)


def _debug_print_context(prefix: str, context: AgentContext) -> None:
    """Emit context at debug level."""

    output = f"\n\nSYSTEM:\n {context.systemPrompt}" if context.systemPrompt else "<no system prompt>"
    for item in context.messages:
        if hasattr(item, "role") and item.role == "user":
            output += f"\n\nUSER:\n  {item}"
        elif hasattr(item, "role") and item.role == "assistant":
            output += f"\n\nASSISTANT:\n  {item}"
        elif hasattr(item, "role") and item.role == "toolResult":
            output += f"\n\nTOOLRESULT:\n  {item}"
        else:
            output += f"\nUNKNOWN MESSAGE TYPE: {item}"

    logger.debug(f"{prefix}: {output}")


async def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any | None = None,
    stream_fn: Any | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Start an agent loop with a new prompt message.
    The prompt is added to the context and events are emitted for it.
    """
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        systemPrompt=context.systemPrompt,
        messages=context.messages + prompts,
        tools=context.tools,
    )
    _debug_print_context("Starting agent loop with context messages", current_context)
    # print(f"Continuing agent loop with context messages: {current_context}")

    yield AgentStartEvent()
    yield TurnStartEvent()

    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    async for event in _run_loop(current_context, new_messages, config, signal, stream_fn):
        yield event


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any | None = None,
    stream_fn: Any | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    Continue an agent loop from the current context without adding a new message.
    Used for retries - context already has user message or tool results.

    Important: The last message in context must convert to a `user` or `toolResult` message
    via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last_message = context.messages[-1]
    if hasattr(last_message, "role") and last_message.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        systemPrompt=context.systemPrompt,
        messages=list(context.messages),
        tools=context.tools,
    )
    _debug_print_context("Continuing agent loop with context messages", current_context)

    yield AgentStartEvent()
    yield TurnStartEvent()

    async for event in _run_loop(current_context, new_messages, config, signal, stream_fn):
        yield event


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: Any | None,
    stream_fn: Any | None,
) -> AsyncGenerator[AgentEvent, None]:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    # Check for steering messages at start
    pending_messages: list[AgentMessage] = []
    if config.getSteeringMessages:
        pending_messages = await config.getSteeringMessages()

    # Outer loop: continues when queued follow-up messages arrive
    while True:
        has_more_tool_calls = True
        steering_after_tools: list[AgentMessage] | None = None

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                yield TurnStartEvent()
            else:
                first_turn = False

            # Process pending messages
            if len(pending_messages) > 0:
                for message in pending_messages:
                    yield MessageStartEvent(message=message)
                    yield MessageEndEvent(message=message)
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = None
            async for event_or_msg in _stream_assistant_response(current_context, config, signal, stream_fn):
                if isinstance(event_or_msg, AssistantMessage):
                    message = event_or_msg
                else:
                    yield event_or_msg

            if message is None:
                raise RuntimeError("No message received from assistant")

            new_messages.append(message)

            if message.stopReason in ["error", "aborted"]:
                yield TurnEndEvent(message=message, toolResults=[])
                yield AgentEndEvent(messages=new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if hasattr(c, "type") and c.type == "toolCall"]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                execution_result = None
                async for event_or_result in _execute_tool_calls(
                    current_context.tools,
                    message,
                    signal,
                    config.getSteeringMessages,
                ):
                    if isinstance(event_or_result, dict) and "toolResults" in event_or_result:
                        execution_result = event_or_result
                    else:
                        yield event_or_result

                if execution_result:
                    tool_results = execution_result["toolResults"]
                    steering_after_tools = execution_result.get("steeringMessages")

                    for result in tool_results:
                        current_context.messages.append(result)
                        new_messages.append(result)

            yield TurnEndEvent(message=message, toolResults=tool_results)

            # Get steering messages after turn completes
            if steering_after_tools and len(steering_after_tools) > 0:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif config.getSteeringMessages:
                pending_messages = await config.getSteeringMessages()
            else:
                pending_messages = []

        # Agent would stop here. Check for follow-up messages.
        follow_up_messages: list[AgentMessage] = []
        if config.getFollowUpMessages:
            follow_up_messages = await config.getFollowUpMessages()

        if len(follow_up_messages) > 0:
            pending_messages = follow_up_messages
            continue

        # No more messages, exit
        break

    yield AgentEndEvent(messages=new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any | None,
    stream_fn: Any | None,
) -> AsyncGenerator[AgentEvent | AssistantMessage, None]:
    """
    Stream an assistant response from the LLM.
    This is where AgentMessage[] gets transformed to Message[] for the LLM.
    Yields events and finally the complete AssistantMessage.
    """
    _debug_print_context("Streaming assistant response with context messages", context)

    # Apply context transform if configured
    messages = context.messages
    if config.transformContext:
        messages = await config.transformContext(messages, signal)

    # Convert to LLM-compatible messages
    llm_messages = await config.convertToLlm(messages)

    # Build LLM context
    llm_context = {
        "systemPrompt": context.systemPrompt,
        "messages": llm_messages,
        "tools": context.tools,
    }

    # Resolve API key
    resolved_api_key = config.apiKey
    if config.getApiKey:
        resolved_api_key = await config.getApiKey(config.model.provider) or config.apiKey

    # Build stream options
    stream_options = SimpleStreamOptions(
        apiKey=resolved_api_key,
        temperature=config.temperature,
        maxTokens=config.maxTokens,
        reasoning=config.reasoning,
        sessionId=config.sessionId,
        thinkingBudgets=config.thinkingBudgets,
        maxRetryDelayMs=config.maxRetryDelayMs,
    )
    stream_options.signal = signal  # Add signal dynamically

    # Use custom stream function if provided, otherwise use provider stream
    if stream_fn:
        response_stream = await stream_fn(config.model, llm_context, stream_options)
    else:
        response_stream = provider_stream_simple(config.model, llm_context, stream_options)

    partial_message: AssistantMessage | None = None
    added_partial = False

    # Iterate over LLM stream events
    async for event in response_stream:
        event_type = event.type if hasattr(event, "type") else None

        if event_type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            yield MessageStartEvent(message=partial_message)

        elif event_type in [
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ]:
            if partial_message:
                # Update partial in context
                context.messages[len(context.messages) - 1] = event.partial
                yield MessageUpdateEvent(
                    assistantMessageEvent=event,
                    message=event.partial,
                )

        elif event_type == "done":
            final_message = event.message
            if added_partial:
                context.messages[len(context.messages) - 1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                yield MessageStartEvent(message=final_message)
            yield MessageEndEvent(message=final_message)
            yield final_message  # Return the final message
            return

        elif event_type == "error":
            error_message = event.error
            if added_partial:
                context.messages[len(context.messages) - 1] = error_message
            else:
                context.messages.append(error_message)
            if not added_partial:
                yield MessageStartEvent(message=error_message)
            yield MessageEndEvent(message=error_message)
            yield error_message
            return

    # Fallback: return partial message if no done/error event
    if partial_message:
        yield MessageEndEvent(message=partial_message)
        yield partial_message


async def _execute_tool_calls(
    tools: list[AgentTool] | None,
    assistant_message: AssistantMessage,
    signal: Any | None,
    get_steering_messages: Any | None,
) -> AsyncGenerator[AgentEvent | dict, None]:
    """Execute tool calls from an assistant message."""
    tool_calls = [c for c in assistant_message.content if hasattr(c, "type") and c.type == "toolCall"]
    results: list[ToolResultMessage] = []
    steering_messages: list[AgentMessage] | None = None

    for index, tool_call in enumerate(tool_calls):
        # Find the tool
        tool = None
        if tools:
            tool = next((t for t in tools if t.name == tool_call.name), None)

        yield ToolExecutionStartEvent(
            toolCallId=tool_call.id,
            toolName=tool_call.name,
            args=tool_call.arguments,
        )

        result: AgentToolResult
        is_error = False

        try:
            if not tool:
                raise ValueError(f"Tool {tool_call.name} not found")

            # Validate and execute tool
            result = await tool.execute(
                tool_call.id,
                tool_call.arguments,
                signal,
                None,  # onUpdate callback
            )
        except Exception as e:
            result = AgentToolResult(
                content=[TextContent(text=str(e))],
                details={},
            )
            is_error = True

        yield ToolExecutionEndEvent(
            toolCallId=tool_call.id,
            toolName=tool_call.name,
            result=result,
            isError=is_error,
        )

        tool_result_message = ToolResultMessage(
            role="toolResult",
            toolCallId=tool_call.id,
            toolName=tool_call.name,
            content=result.content,
            details=result.details,
            isError=is_error,
            timestamp=int(time.time() * 1000),
        )

        results.append(tool_result_message)
        yield MessageStartEvent(message=tool_result_message)
        yield MessageEndEvent(message=tool_result_message)

        # Check for steering messages
        if get_steering_messages:
            steering = await get_steering_messages()
            if len(steering) > 0:
                steering_messages = steering
                # Skip remaining tool calls
                remaining_calls = tool_calls[index + 1 :]
                for skipped in remaining_calls:
                    skip_result = _skip_tool_call(skipped)
                    results.append(skip_result)
                    yield MessageStartEvent(message=skip_result)
                    yield MessageEndEvent(message=skip_result)
                break

    # Return final result as dict
    yield {
        "toolResults": results,
        "steeringMessages": steering_messages,
    }


def _skip_tool_call(tool_call: Any) -> ToolResultMessage:
    """Create a skipped tool result message."""
    return ToolResultMessage(
        role="toolResult",
        toolCallId=tool_call.id,
        toolName=tool_call.name,
        content=[TextContent(text="Skipped due to queued user message.")],
        details={},
        isError=True,
        timestamp=int(time.time() * 1000),
    )
