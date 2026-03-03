"""
Anthropic provider implementation (Claude models).
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    Usage,
)


async def stream_anthropic(
    model: Model,
    context: dict[str, Any],
    options: SimpleStreamOptions,
    api_key: str | None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream from Anthropic API.

    Args:
        model: Anthropic model (Claude)
        context: Context with messages and tools
        options: Stream options
        api_key: Anthropic API key

    Yields:
        AssistantMessageEvent instances
    """
    if not api_key:
        raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY or pass apiKey in options.")

    # Build partial message
    partial = AssistantMessage(
        role="assistant",
        content=[],
        stopReason="stop",
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(input=0, output=0, cacheRead=0, cacheWrite=0, totalTokens=0, cost=Cost()),
        timestamp=int(time.time() * 1000),
    )

    # Build request payload
    messages = _convert_messages(context.get("messages", []))
    payload = {
        "model": model.id,
        "messages": messages,
        "max_tokens": options.maxTokens or 4096,
        "stream": True,
    }

    # Add system prompt
    system_prompt = context.get("systemPrompt", "")
    if system_prompt:
        payload["system"] = system_prompt

    # Add tools
    tools = context.get("tools")
    if tools:
        payload["tools"] = [_convert_tool(tool) for tool in tools]

    # Add options
    if options.temperature is not None:
        payload["temperature"] = options.temperature

    # Determine base URL
    base_url = model.baseUrl if hasattr(model, "baseUrl") and model.baseUrl else "https://api.anthropic.com/v1"

    # Make streaming request
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")

                yield StartEvent(partial=partial)

                content_index = 0
                buffer = ""

                async for line in response.content:
                    buffer += line.decode("utf-8")
                    while "\n" in buffer:
                        line_str, buffer = buffer.split("\n", 1)
                        line_str = line_str.strip()

                        if not line_str or not line_str.startswith("data: "):
                            continue

                        data_str = line_str[6:]
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Process event
                        event_type = event_data.get("type")

                        if event_type == "content_block_start":
                            block = event_data.get("content_block", {})
                            if block.get("type") == "text":
                                partial.content.append(TextContent(text=""))
                                yield TextStartEvent(contentIndex=content_index, partial=partial)
                            elif block.get("type") == "tool_use":
                                partial.content.append(
                                    ToolCall(
                                        id=block.get("id", ""),
                                        name=block.get("name", ""),
                                        arguments={},
                                    )
                                )
                                yield ToolCallStartEvent(contentIndex=content_index, partial=partial)
                            content_index += 1

                        elif event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                idx = event_data.get("index", 0)
                                text = delta.get("text", "")
                                if idx < len(partial.content) and isinstance(partial.content[idx], TextContent):
                                    partial.content[idx].text += text
                                    yield TextDeltaEvent(contentIndex=idx, delta=text, partial=partial)

                            elif delta.get("type") == "input_json_delta":
                                idx = event_data.get("index", 0)
                                json_delta = delta.get("partial_json", "")
                                if idx < len(partial.content) and isinstance(partial.content[idx], ToolCall):
                                    tc = partial.content[idx]
                                    partial_json = getattr(tc, "_partialJson", "")
                                    partial_json += json_delta
                                    setattr(tc, "_partialJson", partial_json)
                                    try:
                                        tc.arguments = json.loads(partial_json)
                                    except json.JSONDecodeError:
                                        pass
                                    yield ToolCallDeltaEvent(contentIndex=idx, delta=json_delta, partial=partial)

                        elif event_type == "message_delta":
                            stop_reason = event_data.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                if stop_reason == "tool_use":
                                    partial.stopReason = "toolUse"
                                elif stop_reason == "max_tokens":
                                    partial.stopReason = "length"
                                else:
                                    partial.stopReason = "stop"

                            usage_delta = event_data.get("usage", {})
                            if usage_delta:
                                partial.usage.output = usage_delta.get("output_tokens", 0)

                        elif event_type == "message_stop":
                            break

                yield DoneEvent(reason=partial.stopReason, message=partial)

    except Exception as e:
        error_msg = str(e)
        partial.stopReason = "error"
        partial.errorMessage = error_msg
        yield ErrorEvent(reason="error", error=partial)


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert agent messages to Anthropic format."""
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
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": c.mimeType,
                                    "data": c.data,
                                },
                            }
                        )
            result.append({"role": "user", "content": content_parts})

        elif msg.role == "assistant":
            content_parts = []
            for c in msg.content:
                if hasattr(c, "type"):
                    if c.type == "text":
                        content_parts.append({"type": "text", "text": c.text})
                    elif c.type == "toolCall":
                        content_parts.append(
                            {
                                "type": "tool_use",
                                "id": c.id,
                                "name": c.name,
                                "input": c.arguments,
                            }
                        )
            result.append({"role": "assistant", "content": content_parts})

        elif msg.role == "toolResult":
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.toolCallId,
                            "content": " ".join(c.text for c in msg.content if hasattr(c, "type") and c.type == "text"),
                        }
                    ],
                }
            )

    return result


def _convert_tool(tool: Any) -> dict[str, Any]:
    """Convert agent tool to Anthropic format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }
