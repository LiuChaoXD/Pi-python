"""
OpenAI provider implementation.
Supports both OpenAI Completions API and Responses API.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

import aiohttp

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)


async def stream_openai(
    model: Model,
    context: dict[str, Any],
    options: SimpleStreamOptions,
    api_key: str | None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream from OpenAI API.

    Args:
        model: OpenAI model
        context: Context with messages and tools
        options: Stream options
        api_key: OpenAI API key

    Yields:
        AssistantMessageEvent instances
    """
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass apiKey in options.")

    # Build partial message
    partial = AssistantMessage(
        role="assistant",
        content=[],
        stopReason="stop",
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(
            input=0,
            output=0,
            cacheRead=0,
            cacheWrite=0,
            totalTokens=0,
            cost=Cost(),
        ),
        timestamp=int(time.time() * 1000),
    )

    # Build request payload
    messages = _convert_messages(context.get("messages", []))
    payload = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # Add system prompt
    system_prompt = context.get("systemPrompt", "")
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})

    # Add tools
    tools = context.get("tools")
    if tools:
        payload["tools"] = [_convert_tool(tool) for tool in tools]

    # Add options
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if options.maxTokens is not None:
        payload["max_tokens"] = options.maxTokens

    # Add reasoning (thinking)
    if options.reasoning:
        # For o1/o3/gpt-5 models
        if "o1" in model.id or "o3" in model.id or "gpt-5" in model.id:
            reasoning_map = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            }
            payload["reasoning_effort"] = reasoning_map.get(options.reasoning.value, "medium")

    # Determine base URL
    base_url = model.baseUrl if hasattr(model, "baseUrl") and model.baseUrl else "https://api.openai.com/v1"

    # Make streaming request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Add custom headers if specified
    if hasattr(model, "headers") and model.headers:
        headers.update(model.headers)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")

                yield StartEvent(partial=partial)

                buffer = ""

                async for line in response.content:
                    if options.signal and hasattr(options.signal, "is_set") and options.signal.is_set():
                        raise Exception("Request aborted by user")

                    buffer += line.decode("utf-8")
                    while "\n" in buffer:
                        line_str, buffer = buffer.split("\n", 1)
                        line_str = line_str.strip()

                        if not line_str or not line_str.startswith("data: "):
                            continue

                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Process chunk
                        async for event in _process_chunk(chunk, partial):
                            yield event

                # Done
                yield DoneEvent(reason="stop", message=partial)

    except Exception as e:
        error_msg = str(e)
        partial.stopReason = "error"
        partial.errorMessage = error_msg
        yield ErrorEvent(reason="error", error=partial)


async def _process_chunk(
    chunk: dict[str, Any],
    partial: AssistantMessage,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Process a single SSE chunk from OpenAI."""
    choices = chunk.get("choices", [])
    if not choices:
        # Check for usage info
        usage_data = chunk.get("usage")
        if usage_data:
            partial.usage.input = usage_data.get("prompt_tokens", 0)
            partial.usage.output = usage_data.get("completion_tokens", 0)
            partial.usage.totalTokens = partial.usage.input + partial.usage.output
            # Calculate costs (example rates, adjust as needed)
            partial.usage.cost.input = partial.usage.input * 0.000003
            partial.usage.cost.output = partial.usage.output * 0.000015
            partial.usage.cost.total = partial.usage.cost.input + partial.usage.cost.output
        return

    choice = choices[0]
    delta = choice.get("delta", {})

    # Keep a stable mapping from OpenAI tool call indexes to content indexes.
    # This avoids collisions when text and tool calls are mixed in one message.
    tool_index_map = getattr(partial, "_toolCallIndexMap", None)
    if tool_index_map is None:
        tool_index_map = {}
        setattr(partial, "_toolCallIndexMap", tool_index_map)

    # Handle text content
    if "content" in delta and delta["content"]:
        content = delta["content"]
        if partial.content and isinstance(partial.content[-1], TextContent):
            text_index = len(partial.content) - 1
        else:
            text_index = len(partial.content)
            partial.content.append(TextContent(text=""))
            yield TextStartEvent(contentIndex=text_index, partial=partial)

        text_content = partial.content[text_index]
        if isinstance(text_content, TextContent):
            text_content.text += content
            yield TextDeltaEvent(contentIndex=text_index, delta=content, partial=partial)

    # Handle tool calls
    tool_calls = delta.get("tool_calls", [])
    for tool_call in tool_calls:
        tc_index = tool_call.get("index", 0)

        mapped_index = tool_index_map.get(tc_index)
        if mapped_index is None:
            mapped_index = len(partial.content)
            partial.content.append(
                ToolCall(
                    id=tool_call.get("id", ""),
                    name=tool_call.get("function", {}).get("name", ""),
                    arguments={},
                )
            )
            tool_index_map[tc_index] = mapped_index
            yield ToolCallStartEvent(contentIndex=mapped_index, partial=partial)

        tc = partial.content[mapped_index] if mapped_index < len(partial.content) else None
        if not isinstance(tc, ToolCall):
            tc = ToolCall(
                id=tool_call.get("id", ""),
                name=tool_call.get("function", {}).get("name", ""),
                arguments={},
            )
            if mapped_index < len(partial.content):
                partial.content[mapped_index] = tc
            else:
                partial.content.append(tc)
                mapped_index = len(partial.content) - 1
                tool_index_map[tc_index] = mapped_index

        if tool_call.get("id"):
            tc.id = tool_call["id"]

        function_data = tool_call.get("function", {})
        if function_data.get("name"):
            tc.name = function_data["name"]

        # Handle function arguments
        if "arguments" in function_data:
            args_delta = function_data["arguments"]
            partial_json = getattr(tc, "_partialJson", "")
            partial_json += args_delta
            setattr(tc, "_partialJson", partial_json)

            try:
                tc.arguments = json.loads(partial_json)
            except json.JSONDecodeError:
                pass  # Keep partial

            yield ToolCallDeltaEvent(contentIndex=mapped_index, delta=args_delta, partial=partial)

    # Handle finish
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        if finish_reason == "tool_calls":
            partial.stopReason = "toolUse"
        elif finish_reason == "length":
            partial.stopReason = "length"
        else:
            partial.stopReason = "stop"


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert agent messages to OpenAI format."""
    result = []
    for msg in messages:
        if not hasattr(msg, "role"):
            continue

        if msg.role == "user":
            content_parts = []
            for c in msg.content:
                if hasattr(c, "type"):
                    if c.type == "text":
                        content_parts.append({"type": "text", "text": c.text})
                    elif c.type == "image":
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": f"data:{c.mimeType};base64,{c.data}"}}
                        )
            result.append(
                {
                    "role": "user",
                    "content": (
                        content_parts if len(content_parts) > 1 else content_parts[0]["text"] if content_parts else ""
                    ),
                }
            )

        elif msg.role == "assistant":
            content_parts = []
            tool_calls = []

            for c in msg.content:
                if hasattr(c, "type"):
                    if c.type == "text":
                        content_parts.append(c.text)
                    elif c.type == "toolCall":
                        tool_calls.append(
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments),
                                },
                            }
                        )

            msg_dict = {"role": "assistant"}
            if content_parts:
                msg_dict["content"] = " ".join(content_parts)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls

            result.append(msg_dict)

        elif msg.role == "toolResult":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.toolCallId,
                    "content": " ".join(c.text for c in msg.content if hasattr(c, "type") and c.type == "text"),
                }
            )

    return result


def _convert_tool(tool: Any) -> dict[str, Any]:
    """Convert agent tool to OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
