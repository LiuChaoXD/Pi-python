"""
Google Gemini provider implementation (simplified).
"""

from __future__ import annotations

from typing import AsyncGenerator, Any

from ..types import (
    Model,
    SimpleStreamOptions,
    AssistantMessageEvent,
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    DoneEvent,
    ErrorEvent,
)


async def stream_google(
    model: Model,
    context: dict[str, Any],
    options: SimpleStreamOptions,
    api_key: str | None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream from Google Gemini API.

    Note: This is a simplified placeholder implementation.
    Full implementation requires Google's official SDK or direct REST API calls.

    Args:
        model: Google model (Gemini)
        context: Context with messages and tools
        options: Stream options
        api_key: Google API key

    Yields:
        AssistantMessageEvent instances
    """
    if not api_key:
        raise ValueError("Google API key is required. Set GEMINI_API_KEY or pass apiKey in options.")

    # TODO: Implement Google Gemini API streaming
    # For now, raise an error indicating this provider is not yet implemented
    raise NotImplementedError(
        "Google Gemini provider is not yet fully implemented. "
        "Use OpenAI or Anthropic providers for now."
    )
