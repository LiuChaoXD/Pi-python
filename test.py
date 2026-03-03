import asyncio
import os
import sys

from dotenv import load_dotenv

from agent_core import (
    Agent,
    AgentTool,
    AgentToolResult,
    Model,
    TextContent,
    ThinkingLevel,
)

load_dotenv()


class CalculatorTool:
    """Simple calculator tool."""

    name = "calculator"
    label = "Calculator"
    description = "Perform basic arithmetic: add, subtract, multiply, or divide two numbers"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The operation to perform",
            },
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["operation", "a", "b"],
    }

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        """Execute the calculation."""
        operation = params["operation"]
        a = params["a"]
        b = params["b"]

        await asyncio.sleep(0.2)  # Simulate some work

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return AgentToolResult(
            content=[TextContent(text=f"Result: {result}")],
            details={"operation": operation, "a": a, "b": b, "result": result},
        )


async def example_openai():
    """Example using OpenAI."""
    print("=" * 60)
    print("OpenAI Example (GPT-4o-mini)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI example")
        return

    agent = Agent(
        initial_state={
            "systemPrompt": "You are a helpful math assistant. Use the calculator tool for computations.",
            "model": Model(
                api="openai-completions",
                baseUrl="http://127.0.0.1:8080/v1",
                provider="openai",
                id="gpt-4o",
                name="gpt-4o",
            ),
            "tools": [CalculatorTool()],
        }
    )

    # Subscribe to events
    def on_event(event):
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

        if event_type == "message_update":
            msg_event = (
                event.get("assistantMessageEvent")
                if isinstance(event, dict)
                else getattr(event, "assistantMessageEvent", None)
            )
            if msg_event:
                evt_type = msg_event.get("type") if isinstance(msg_event, dict) else getattr(msg_event, "type", None)
                if evt_type == "text_delta":
                    delta = msg_event.get("delta") if isinstance(msg_event, dict) else getattr(msg_event, "delta", "")
                    print(delta, end="", flush=True)

        elif event_type == "tool_execution_start":
            tool_name = event.get("toolName") if isinstance(event, dict) else getattr(event, "toolName", "unknown")
            print(f"\n🔧 Executing tool: {tool_name}")

        elif event_type == "tool_execution_end":
            is_error = event.get("isError") if isinstance(event, dict) else getattr(event, "isError", False)
            print(f"   {'✅' if not is_error else '❌'} Tool execution {'succeeded' if not is_error else 'failed'}")

    agent.subscribe(on_event)

    # Test prompts
    print("\nUser: What is 156 multiplied by 23?\n")
    await agent.prompt("What is 156 multiplied by 23?")

    print("\n\n" + "=" * 60 + "\n")

    print("User: Now divide that result by 4\n")
    await agent.prompt("Now divide that result by 4")
    agent.state
    print("\n")


async def main():
    """Run examples."""
    print("\n🤖 Pi Agent Core - LLM Integration Examples\n")

    # Check which providers are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("❌ No API keys found!")
        print("\nPlease set at least one of these environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("\nExample:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)

    try:
        # Run available examples
        if has_openai:
            await example_openai()
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
