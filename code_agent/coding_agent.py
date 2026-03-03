"""
CodingAgent - High-level coding agent with pre-configured tools

Provides an out-of-the-box coding agent with all file system tools,
resource management, and coding-specific features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from agent_core import (
    Agent,
    AgentMessage,
    ImageContent,
    Model,
    TextContent,
    ThinkingLevel,
    UserMessage,
)

from .resources import (
    AgentConfigManager,
    CreateAgentConfigTool,
    CreateSkillTool,
    ListResourcesTool,
    MemoryManager,
    MemoryReadTool,
    MemoryWriteTool,
    SkillManager,
)
from .tools import (
    BashTool,
    EditTool,
    ReadTool,
    WriteTool,
    get_all_tools,
    get_coding_tools,
    get_readonly_tools,
)

load_dotenv()
INFO_DIR = os.getenv("INFO_DIR", "./.personal")
SKILLS_DIR = os.path.join(INFO_DIR, "skills")
MEMORY_DIR = os.path.join(INFO_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = """
--- 
## General Purpose AI Agent

You are a general-purpose AI agent.

Your job is to:

- Understand the user's goals
- Think step-by-step
- Use tools effectively
- Produce clear, useful answers for a wide variety of tasks (planning, explanation, writing, coding, analysis, etc.)

You should be able to help with:

- Everyday questions and explanations
- Writing and editing text (emails, reports, documentation, blog posts, etc.)
- Brainstorming and planning
- Software engineering and debugging
- Personal productivity and study plans
- Light data analysis and reasoning tasks

---

## Communication style

- Default tone: concise, direct, and friendly; no unnecessary fluff.
- Use Markdown formatting (headings, lists, code blocks) when it improves clarity.
- Adapt depth to the user:
    - If they seem expert, be technical and precise.
    - If they seem non-expert, explain in simple language and give examples.
- Only use emojis if the user explicitly asks for them.
- Focus on facts, reasoning, and problem-solving; avoid over-the-top praise.


---

## Planning (no timelines)

When a task is more than one or two simple steps:

1. Briefly restate the goal in your own words.
2. Break the work into small, concrete steps.
3. Avoid time estimates like “this will take 2–3 days” or “later we can…”.
4. Focus on _what_ needs to be done and _how_, not _when_.

If assumptions are needed, state them explicitly.


---

## Professional objectivity

- Be honest and technically rigorous, even if it means gently disagreeing with the user.
- If you’re uncertain, say so, and either:
    - ask clarifying questions, or
    - explore with tools / reasoning and then update your answer.
- Avoid blindly confirming the user’s assumptions; check them where reasonable.
- Do not exaggerate; avoid phrases like “this is absolutely perfect” or “this is the best ever” unless truly warranted.


---

## Tool usage strategy

- Read tool descriptions and parameter schemas carefully; respect required fields and constraints.
- When tools are available that match the task (file tools, todo tool, memory tool, search tools, etc.), **prefer using them** over guessing.
- You may call multiple tools in parallel when they are independent.
- If a tool call depends on the result of a previous one, call them sequentially.
- Never fabricate tool outputs; only report what the tool actually returned.

"""


class CodingAgent:
    """
    High-level coding agent with pre-configured tools and resources.

    This class wraps the core Agent with coding-specific functionality:
    - Pre-loaded file system tools
    - Automatic Skills and Configs loading
    - Convenient helper methods
    - Coding-focused system prompt
    """

    def __init__(
        self,
        model: Model | str | None = None,
        tools_mode: str = "all",
        cwd: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_level: ThinkingLevel = ThinkingLevel.LOW,
        enable_resources: bool = True,
        auto_load_resources: bool = True,
        session_id: Optional[str] = None,
        **agent_kwargs,
    ):
        """
        Initialize CodingAgent

        Args:
            model: Model instance or model ID string (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
            tools_mode: "all" (all 7 tools), "coding" (read/write/edit/bash), "readonly" (read/grep/find/ls)
            cwd: Working directory for tools (default: current directory)
            system_prompt: Custom system prompt (default: coding-focused prompt)
            thinking_level: Thinking level (OFF, MINIMAL, LOW, MEDIUM, HIGH)
            enable_resources: Enable Skills and Configs management
            auto_load_resources: Automatically load existing Skills and Configs
            session_id: Session ID for provider caching
            **agent_kwargs: Additional arguments passed to Agent constructor
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.tools_mode = tools_mode
        self.enable_resources = enable_resources

        # Parse model
        if model is None:
            raise ValueError(
                "No model specified and no API keys found. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or provide a model."
            )
        elif isinstance(model, str):
            # Parse model string
            model = self._parse_model_string(model)

        # Get tools based on mode
        if tools_mode == "all":
            tools = get_all_tools(cwd=str(self.cwd))
        elif tools_mode == "coding":
            tools = get_coding_tools(cwd=str(self.cwd))
        elif tools_mode == "readonly":
            tools = get_readonly_tools(cwd=str(self.cwd))
        else:
            raise ValueError(f"Invalid tools_mode: {tools_mode}. Use 'all', 'coding', or 'readonly'.")

        # Add resource management tools
        if enable_resources:
            write_tool = WriteTool(cwd=str(self.cwd))
            self.skill_manager = SkillManager(skills_dir=SKILLS_DIR, write_tool=write_tool)
            self.config_manager = AgentConfigManager(write_tool=write_tool)
            self.memory_manager = MemoryManager(
                memory_dir=MEMORY_DIR,
                default_session_id=session_id or "default",
            )

            tools.extend(
                [
                    CreateSkillTool(self.skill_manager),
                    CreateAgentConfigTool(self.config_manager),
                    ListResourcesTool(self.skill_manager, self.config_manager),
                    MemoryWriteTool(self.memory_manager),
                    MemoryReadTool(self.memory_manager),
                ]
            )

            # Auto-load resources
            if auto_load_resources:
                self.skills = self.skill_manager.load_skills()
                self.configs = self.config_manager.load_configs()
        else:
            self.skill_manager = None
            self.config_manager = None
            self.skills = {}
            self.configs = {}
            self.memory_manager = None

        # Use custom or default system prompt
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # Create underlying agent
        self.agent = Agent(
            initial_state={
                "systemPrompt": system_prompt,
                "model": model,
                "thinkingLevel": thinking_level,
                "tools": tools,
                "messages": [],
            },
            session_id=session_id,
            **agent_kwargs,
        )

    def _parse_model_string(self, model_str: str) -> Model:
        """
        Parse model string to Model instance

        Supports:
        - "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
        - "claude-3-5-sonnet", "claude-3-opus"
        - "gemini-1.5-pro", "gemini-1.5-flash"
        """
        model_lower = model_str.lower()

        # OpenAI models
        if "minimax" in model_lower:
            return Model(
                api="openai-completions",
                provider="openai",
                id=model_str,
                baseUrl=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
                name=model_str,
            )
        # OpenAI models
        if "gpt" in model_lower:
            return Model(
                api="openai-completions",
                provider="openai",
                id=model_str,
                baseUrl=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
                name=model_str,
            )

        # Anthropic models
        if "claude" in model_lower:
            # Map short names to full IDs
            if model_str == "claude-3-5-sonnet":
                model_id = "claude-3-5-sonnet-20241022"
            elif model_str == "claude-3-opus":
                model_id = "claude-3-opus-20240229"
            elif model_str == "claude-3-sonnet":
                model_id = "claude-3-sonnet-20240229"
            else:
                model_id = model_str

            return Model(
                api="anthropic-messages",
                provider="anthropic",
                id=model_id,
                name=model_str,
            )

        # Google models
        if "gemini" in model_lower:
            return Model(
                api="google-generative-ai",
                provider="google",
                id=model_str,
                baseUrl=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
                name=model_str,
            )

        # Unknown model
        raise ValueError(f"Unknown model: {model_str}. " f"Supported: gpt-*, claude-*, gemini-*")

    # Convenience methods

    async def prompt(self, text: str, images: list[ImageContent] | None = None):
        """
        Send a prompt to the agent

        Args:
            text: Text prompt
            images: Optional list of images
        """
        return await self.agent.prompt(text, images)

    async def continue_conversation(self):
        """Continue the conversation from current context"""
        return await self.agent.continue_conversation()

    # Resource management

    async def create_skill(self, name: str, description: str, script_content: str):
        """
        Helper: Create a new Skill

        Args:
            name: Skill name
            description: Skill description
            script_content: Python script content
        """
        if not self.enable_resources:
            raise RuntimeError("Resources are not enabled. Set enable_resources=True.")

        return await self.prompt(
            f"Create a skill named '{name}' with description '{description}' "
            f"using this script:\n\n```python\n{script_content}\n```"
        )

    async def create_agent_config(self, name: str, description: str, system_prompt: str):
        """
        Helper: Create a new Agent config

        Args:
            name: Config name
            description: Config description
            system_prompt: System prompt
        """
        if not self.enable_resources:
            raise RuntimeError("Resources are not enabled. Set enable_resources=True.")

        return await self.prompt(
            f"Create an agent config named '{name}' with description '{description}' "
            f"and system prompt:\n\n{system_prompt}"
        )

    async def list_resources(self):
        """Helper: List all Skills and Configs"""
        if not self.enable_resources:
            raise RuntimeError("Resources are not enabled. Set enable_resources=True.")

        return await self.prompt("List all available skills and agent configs")

    # State management (delegate to underlying agent)

    @property
    def state(self):
        """Get agent state"""
        return self.agent.state

    @property
    def messages(self):
        """Get conversation messages"""
        return self.agent.state.messages

    def subscribe(self, callback: Callable[[Any], None]):
        """Subscribe to agent events"""
        return self.agent.subscribe(callback)

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.agent.set_system_prompt(prompt)

    def set_model(self, model: Model | str):
        """Update model"""
        if isinstance(model, str):
            model = self._parse_model_string(model)
        self.agent.set_model(model)

    def set_thinking_level(self, level: ThinkingLevel):
        """Update thinking level"""
        self.agent.set_thinking_level(level)

    def clear_messages(self):
        """Clear conversation history"""
        self.agent.clear_messages()

    def reset(self):
        """Reset agent to initial state"""
        self.agent.reset()

    def abort(self):
        """Abort current operation"""
        self.agent.abort()

    async def wait_for_idle(self):
        """Wait for agent to become idle"""
        await self.agent.wait_for_idle()


# Convenience function
def create_coding_agent(
    model: str = "gpt-4o-mini",
    tools_mode: str = "all",
    cwd: Optional[str] = None,
    **kwargs,
) -> CodingAgent:
    """
    Create a coding agent with sensible defaults

    Args:
        model: Model ID string (default: "gpt-4o-mini")
        tools_mode: "all", "coding", or "readonly"
        cwd: Working directory
        **kwargs: Additional arguments for CodingAgent

    Returns:
        CodingAgent instance
    """
    return CodingAgent(
        model=model,
        tools_mode=tools_mode,
        cwd=cwd,
        **kwargs,
    )
