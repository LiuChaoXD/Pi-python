"""
Base provider utilities and stream_simple implementation.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Any

from ..types import (
    Model,
    AgentContext,
    SimpleStreamOptions,
    AssistantMessageEvent,
)


def get_env_api_key(provider: str) -> str | None:
    """
    Get API key from environment variables for a given provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "google")

    Returns:
        API key if found, None otherwise
    """
    env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "google-vertex": ["GOOGLE_CLOUD_PROJECT"],
        "mistral": ["MISTRAL_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"],
        "xai": ["XAI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "github-copilot": ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"],
    }

    for var in env_vars.get(provider, []):
        value = os.getenv(var)
        if value:
            return value

    return None


async def stream_simple(
    model: Model,
    context: dict[str, Any] | AgentContext,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Unified streaming interface that routes to the appropriate provider.

    Args:
        model: LLM model to use
        context: Agent context with messages and tools
        options: Stream options including API key, temperature, etc.

    Yields:
        AssistantMessageEvent instances
    """
    if options is None:
        options = SimpleStreamOptions()

    # Convert AgentContext to dict if needed
    if hasattr(context, "systemPrompt"):
        context_dict = {
            "systemPrompt": context.systemPrompt,
            "messages": context.messages,
            "tools": context.tools,
        }
    else:
        context_dict = context

    # Resolve API key
    api_key = options.apiKey
    if not api_key:
        api_key = get_env_api_key(model.provider)

    # Route to appropriate provider
    if model.api == "openai-completions" or model.api == "openai-responses":
        from .openai_provider import stream_openai
        async for event in stream_openai(model, context_dict, options, api_key):
            yield event
    elif model.api == "anthropic-messages":
        from .anthropic_provider import stream_anthropic
        async for event in stream_anthropic(model, context_dict, options, api_key):
            yield event
    elif model.api == "google-generative-ai":
        from .google_provider import stream_google
        async for event in stream_google(model, context_dict, options, api_key):
            yield event
    else:
        raise ValueError(f"Unsupported API: {model.api}")
