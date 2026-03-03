# Pi Agent (Python)

A lightweight Python implementation of Pi Agent for learning and practicing **agent design patterns**.

This project is especially useful if you want to understand how to design:

- a **stateful agent**
- a **streaming event loop**
- **tool-calling workflows**
- layered architecture from **core agent** → **coding agent**

---

## Project Structure

The codebase is split into two layers:

### 1) `agent_core/` — Core Agent Engine

Pure agent runtime (no file-system-specific logic):

- `agent.py` — `Agent` class, state management, prompt/continue flow
- `agent_loop.py` — main agent loop, message turns, tool execution loop
- `types.py` — all core dataclasses/types (messages, events, tools, model)
- `providers/` — LLM provider adapters (OpenAI / Anthropic / Google)
- `proxy.py` — proxy streaming support

This layer teaches the **fundamental architecture** of an agent system.

### 2) `code_agent/` — Practical Coding Agent Layer

Builds on `agent_core` with coding-focused capabilities:

- `coding_agent.py` — high-level `CodingAgent` wrapper
- `tools/` — read/write/edit/bash/grep/find/ls tools
- `resources/` — skills, configs, memory tools
- `session.py` — session persistence and restore

This layer demonstrates how to compose a domain-specific agent from a generic core.

---

## Agent Design Patterns You Can Learn Here

1. **Stateful Agent Pattern**
   - Agent stores system prompt, model, tools, messages, streaming status.

2. **Event-Driven Streaming Pattern**
   - Emits events like:
     - `agent_start`, `turn_start`
     - `message_start/update/end`
     - `tool_execution_start/end`
     - `turn_end`, `agent_end`

3. **Tool Execution Loop Pattern**
   - Assistant emits tool calls → runtime executes tools → tool results are fed back into context.

4. **Queue/Interrupt Pattern**
   - `steer()` for interrupting in-progress flow
   - `follow_up()` for deferred user messages

5. **Layered Extension Pattern**
   - Keep core generic (`agent_core`), add domain modules separately (`code_agent`).

---

## Quick Start

### Option A: Use the high-level `CodingAgent`

```python
import asyncio
from code_agent import create_coding_agent


async def main():
    agent = create_coding_agent(
        model="gpt-4o-mini",   # or claude-*, gemini-*
        tools_mode="all",      # all | coding | readonly
    )

    await agent.prompt("Explain how this repository is structured.")


asyncio.run(main())
```

### Option B: Use low-level `Agent` from `agent_core`

```python
import asyncio
from agent_core import Agent, Model, ThinkingLevel


async def main():
    model = Model(api="openai-completions", provider="openai", id="gpt-4o-mini")

    agent = Agent(
        initial_state={
            "systemPrompt": "You are a helpful assistant.",
            "model": model,
            "thinkingLevel": ThinkingLevel.LOW,
            "tools": [],
            "messages": [],
        }
    )

    await agent.prompt("What are the key components of an agent loop?")


asyncio.run(main())
```

---

## Recommended Learning Path

If your goal is to learn agent architecture in Python:

1. Read `agent_core/types.py` first (data model and event contracts).
2. Read `agent_core/agent_loop.py` (runtime behavior and control flow).
3. Read `agent_core/agent.py` (stateful wrapper API).
4. Then read `code_agent/coding_agent.py` (how to package a practical agent).
5. Finally inspect `code_agent/session.py` and `code_agent/tools/` for real-world extensions.

---

## Notes

- This repo is ideal for studying **how to build an agent framework**, not only for end-user prompting.
- The design is intentionally modular, so you can reuse `agent_core` to build other specialized agents.

