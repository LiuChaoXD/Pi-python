from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from .logging import get_logger
from .types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
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


logger = get_logger(__name__)

@dataclass
class ProxyStreamOptions(SimpleStreamOptions):
    """Options for proxy streaming."""

    authToken: str = ""
    proxyUrl: str = ""


async def stream_proxy(
    model: Model,
    context: dict[str, Any],
    options: ProxyStreamOptions,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream function that proxies through a server instead of calling LLM providers directly.
    The server strips the partial field from delta events to reduce bandwidth.
    We reconstruct the partial message client-side.

    Example:
        agent = Agent(
            stream_fn=lambda model, ctx, opts: stream_proxy(
                model, ctx, ProxyStreamOptions(**opts, authToken="...", proxyUrl="...")
            )
        )

    Args:
        model: LLM model to use
        context: Agent context with messages and tools
        options: Proxy stream options including auth token and URL

    Yields:
        AssistantMessageEvent events
    """
    # Initialize partial message
    partial = AssistantMessage(
        role="assistant",
        stopReason="stop",
        content=[],
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

    signal = options.signal if hasattr(options, "signal") else None

    try:
        # Make request to proxy
        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {options.authToken}",
                "Content-Type": "application/json",
            }

            body = {
                "model": {
                    "api": model.api,
                    "provider": model.provider,
                    "id": model.id,
                },
                "context": context,
                "options": {
                    "temperature": options.temperature,
                    "maxTokens": options.maxTokens,
                    "reasoning": options.reasoning.value if options.reasoning else None,
                },
            }

            async with session.post(
                f"{options.proxyUrl}/api/stream",
                headers=headers,
                json=body,
            ) as response:
                if response.status != 200:
                    error_message = f"Proxy error: {response.status} {response.reason}"
                    try:
                        error_data = await response.json()
                        if "error" in error_data:
                            error_message = f"Proxy error: {error_data['error']}"
                    except:
                        pass
                    raise Exception(error_message)

                # Read streaming response
                buffer = ""
                async for chunk in response.content.iter_any():
                    if signal and signal.is_set():
                        raise Exception("Request aborted by user")

                    buffer += chunk.decode("utf-8")
                    lines = buffer.split("\n")
                    buffer = lines[-1]

                    for line in lines[:-1]:
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data:
                                proxy_event = json.loads(data)
                                event = _process_proxy_event(proxy_event, partial)
                                if event:
                                    yield event

                if signal and signal.is_set():
                    raise Exception("Request aborted by user")

    except Exception as error:
        error_message = str(error)
        reason: StopReason = "aborted" if signal and signal.is_set() else "error"
        partial.stopReason = reason
        partial.errorMessage = error_message
        yield ErrorEvent(reason=reason, error=partial)


def _process_proxy_event(
    proxy_event: dict[str, Any],
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type")

    if event_type == "start":
        return StartEvent(partial=partial)

    elif event_type == "text_start":
        content_index = proxy_event["contentIndex"]
        partial.content.insert(content_index, TextContent(text=""))
        return TextStartEvent(contentIndex=content_index, partial=partial)

    elif event_type == "text_delta":
        content_index = proxy_event["contentIndex"]
        delta = proxy_event["delta"]
        content = partial.content[content_index]
        if isinstance(content, TextContent):
            content.text += delta
            return TextDeltaEvent(
                contentIndex=content_index,
                delta=delta,
                partial=partial,
            )
        raise ValueError("Received text_delta for non-text content")

    elif event_type == "text_end":
        content_index = proxy_event["contentIndex"]
        content = partial.content[content_index]
        if isinstance(content, TextContent):
            content.textSignature = proxy_event.get("contentSignature")
            return TextEndEvent(
                contentIndex=content_index,
                content=content.text,
                partial=partial,
            )
        raise ValueError("Received text_end for non-text content")

    elif event_type == "thinking_start":
        content_index = proxy_event["contentIndex"]
        partial.content.insert(content_index, ThinkingContent(thinking=""))
        return ThinkingStartEvent(contentIndex=content_index, partial=partial)

    elif event_type == "thinking_delta":
        content_index = proxy_event["contentIndex"]
        delta = proxy_event["delta"]
        content = partial.content[content_index]
        if isinstance(content, ThinkingContent):
            content.thinking += delta
            return ThinkingDeltaEvent(
                contentIndex=content_index,
                delta=delta,
                partial=partial,
            )
        raise ValueError("Received thinking_delta for non-thinking content")

    elif event_type == "thinking_end":
        content_index = proxy_event["contentIndex"]
        content = partial.content[content_index]
        if isinstance(content, ThinkingContent):
            content.thinkingSignature = proxy_event.get("contentSignature")
            return ThinkingEndEvent(
                contentIndex=content_index,
                content=content.thinking,
                partial=partial,
            )
        raise ValueError("Received thinking_end for non-thinking content")

    elif event_type == "toolcall_start":
        content_index = proxy_event["contentIndex"]
        partial.content.insert(
            content_index,
            ToolCall(
                id=proxy_event["id"],
                name=proxy_event["toolName"],
                arguments={},
            ),
        )
        return ToolCallStartEvent(contentIndex=content_index, partial=partial)

    elif event_type == "toolcall_delta":
        content_index = proxy_event["contentIndex"]
        delta = proxy_event["delta"]
        content = partial.content[content_index]
        if isinstance(content, ToolCall):
            # Parse streaming JSON (simplified version)
            partial_json = getattr(content, "_partialJson", "")
            partial_json += delta
            setattr(content, "_partialJson", partial_json)
            try:
                content.arguments = json.loads(partial_json)
            except json.JSONDecodeError:
                # Not yet valid JSON, keep partial
                pass
            return ToolCallDeltaEvent(
                contentIndex=content_index,
                delta=delta,
                partial=partial,
            )
        raise ValueError("Received toolcall_delta for non-toolCall content")

    elif event_type == "toolcall_end":
        content_index = proxy_event["contentIndex"]
        content = partial.content[content_index]
        if isinstance(content, ToolCall):
            if hasattr(content, "_partialJson"):
                delattr(content, "_partialJson")
            return ToolCallEndEvent(
                contentIndex=content_index,
                toolCall=content,
                partial=partial,
            )
        return None

    elif event_type == "done":
        partial.stopReason = proxy_event["reason"]
        usage_data = proxy_event["usage"]
        partial.usage = Usage(
            input=usage_data.get("input", 0),
            output=usage_data.get("output", 0),
            cacheRead=usage_data.get("cacheRead", 0),
            cacheWrite=usage_data.get("cacheWrite", 0),
            totalTokens=usage_data.get("totalTokens", 0),
            cost=Cost(
                input=usage_data.get("cost", {}).get("input", 0.0),
                output=usage_data.get("cost", {}).get("output", 0.0),
                cacheRead=usage_data.get("cost", {}).get("cacheRead", 0.0),
                cacheWrite=usage_data.get("cost", {}).get("cacheWrite", 0.0),
                total=usage_data.get("cost", {}).get("total", 0.0),
            ),
        )
        return DoneEvent(reason=proxy_event["reason"], message=partial)

    elif event_type == "error":
        partial.stopReason = proxy_event["reason"]
        partial.errorMessage = proxy_event.get("errorMessage")
        usage_data = proxy_event["usage"]
        partial.usage = Usage(
            input=usage_data.get("input", 0),
            output=usage_data.get("output", 0),
            cacheRead=usage_data.get("cacheRead", 0),
            cacheWrite=usage_data.get("cacheWrite", 0),
            totalTokens=usage_data.get("totalTokens", 0),
            cost=Cost(
                input=usage_data.get("cost", {}).get("input", 0.0),
                output=usage_data.get("cost", {}).get("output", 0.0),
                cacheRead=usage_data.get("cost", {}).get("cacheRead", 0.0),
                cacheWrite=usage_data.get("cost", {}).get("cacheWrite", 0.0),
                total=usage_data.get("cost", {}).get("total", 0.0),
            ),
        )
        return ErrorEvent(reason=proxy_event["reason"], error=partial)

    else:
        logger.warning(f"Unhandled proxy event type: {event_type}")
        return None
