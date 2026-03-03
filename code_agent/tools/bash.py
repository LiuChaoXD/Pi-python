"""
Bash Tool - Execute shell commands

Executes bash commands with streaming output and timeout support.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from agent_core import AgentTool, AgentToolResult, TextContent


class BashTool:
    """
    Bash tool - Execute shell commands

    Features:
    - Execute arbitrary bash commands
    - Streaming output (stdout/stderr)
    - Timeout support
    - Output truncation (last 2000 lines or 2MB)
    """

    MAX_LINES = 2000
    MAX_BYTES = 2 * 1024 * 1024  # 2MB
    DEFAULT_TIMEOUT = 300  # 5 minutes

    def __init__(self, cwd: Optional[str] = None):
        """
        Initialize Bash tool

        Args:
            cwd: Working directory (default: current directory)
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

        self.name = "bash"
        self.label = "Execute Command"
        self.description = (
            "Execute a bash command. Returns stdout/stderr and exit code. "
            "Long output will be truncated to last 2000 lines."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 300)",
                    "minimum": 1,
                    "maximum": 3600,
                },
            },
            "required": ["command"],
        }

    def _truncate_output(self, output: str) -> tuple[str, bool]:
        """
        Truncate output to last N lines or bytes

        Args:
            output: Command output

        Returns:
            Tuple of (truncated_output, was_truncated)
        """
        lines = output.split("\n")

        # Check if truncation needed
        if len(lines) <= self.MAX_LINES and len(output) <= self.MAX_BYTES:
            return output, False

        # Truncate to last N lines
        if len(lines) > self.MAX_LINES:
            truncated_lines = lines[-self.MAX_LINES :]
            truncated = "\n".join(truncated_lines)
            removed_lines = len(lines) - self.MAX_LINES
            return f"... (truncated {removed_lines} lines) ...\n\n{truncated}", True

        # Truncate to last N bytes
        if len(output) > self.MAX_BYTES:
            truncated = output[-self.MAX_BYTES :]
            removed_bytes = len(output) - self.MAX_BYTES
            return f"... (truncated {removed_bytes} bytes) ...\n\n{truncated}", True

        return output, False

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Any | None = None,
    ) -> AgentToolResult:
        """
        Execute bash command

        Args:
            tool_call_id: Tool call identifier
            params: Parameters containing 'command' and optional 'timeout'
            signal: Abort signal (not used yet)
            on_update: Progress callback (not used yet)

        Returns:
            AgentToolResult with command output and exit code
        """
        command = params["command"]
        timeout = params.get("timeout", self.DEFAULT_TIMEOUT)

        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Combine stderr into stdout
                cwd=str(self.cwd),
                env=os.environ.copy(),
            )

            # Wait for completion with timeout
            try:
                stdout_data, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                # Kill process on timeout
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return AgentToolResult(
                    content=[TextContent(text=f"Command timed out after {timeout} seconds: {command}")],
                    details={
                        "error": "Timeout",
                        "command": command,
                        "timeout": timeout,
                    },
                )

            # Decode output
            output = stdout_data.decode("utf-8", errors="replace")

            # Get exit code
            exit_code = process.returncode

            # Truncate output
            truncated_output, was_truncated = self._truncate_output(output)

            # Build result message
            if exit_code == 0:
                status = "✓ Command succeeded"
            else:
                status = f"✗ Command failed with exit code {exit_code}"

            result_text = f"{status}\n\n"
            result_text += f"Command: {command}\n"
            result_text += f"Exit Code: {exit_code}\n\n"

            if was_truncated:
                result_text += "Output (truncated):\n"
            else:
                result_text += "Output:\n"

            result_text += truncated_output

            return AgentToolResult(
                content=[TextContent(text=result_text)],
                details={
                    "command": command,
                    "exit_code": exit_code,
                    "output": output,  # Full output in details
                    "truncated": was_truncated,
                    "cwd": str(self.cwd),
                },
            )

        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error executing command: {str(e)}\n\nCommand: {command}")],
                details={
                    "error": str(e),
                    "command": command,
                },
            )


def create_bash_tool(cwd: str) -> BashTool:
    """
    Factory function to create a bash tool for a specific directory

    Args:
        cwd: Working directory

    Returns:
        BashTool instance
    """
    return BashTool(cwd=cwd)


# Default instance using current directory
bash_tool = BashTool()
