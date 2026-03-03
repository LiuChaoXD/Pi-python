#!/usr/bin/env python3
"""
Final Interactive Terminal - 最终修复版本

修复所有问题：
1. 颜色代码正常显示
2. Skills 结构正确（metadata.parameters）
3. AgentConfig 结构正确（无 model 属性）
4. 空输入处理
5. 命令检测
"""

import asyncio
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from code_agent import CodingAgent, CodingSession, SessionManager

load_dotenv()
import json


class FinalTerminal:
    """最终修复版本的终端"""

    def __init__(self):
        self.session = None
        self.session_manager = SessionManager()
        self.current_session_id = None
        self.running = True
        self._stream_buf: list[str] = []
        self._stream_buf_chars = 0
        self._last_flush_ts = 0.0
        self._flush_interval_s = 0.04
        self._flush_chars = 2

    def print_header(self):
        """打印欢迎信息"""
        print("\n" + "=" * 70)
        print("🤖 Interactive Coding Terminal")
        print("=" * 70 + "\n")

        # 检查 API key
        if os.getenv("OPENAI_API_KEY"):
            print("✓ OPENAI_API_KEY found")
        elif os.getenv("ANTHROPIC_API_KEY"):
            print("✓ ANTHROPIC_API_KEY found")
        else:
            print("✗ No API key found")
            print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return False

        print()
        print("Commands:")
        print("  /help     - Show help")
        print("  /exit     - Exit")
        print("  /new      - New session")
        print("  /sessions - List sessions")
        print("  /info     - Session info")
        print("  /skills   - List skills")
        print("  /agents   - List agent configs")
        print()

        return True

    async def create_session(self, session_id=None):
        """创建会话"""
        if session_id is None:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            print("✗ No API key")
            return False

        try:
            # model = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "claude-3-5-sonnet"
            model = os.getenv("MODEL", "")

            self.session = CodingSession.create(
                session_id=session_id,
                coding_agent_class=CodingAgent,
                auto_save=True,
                model=model,
                tools_mode="all",
            )

            self.current_session_id = session_id
            self.session.agent.subscribe(self.on_event)

            print(f"✓ Session created: {session_id}")
            print(f"  Model: {self.session.agent.state.model.id}")
            print()
            return True

        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def list_skills(self):
        """列出所有 Skills"""
        if not self.session or not self.session.agent.enable_resources:
            print("⚠ Resources not enabled")
            return

        skill_names = self.session.agent.skills
        skill_manager = self.session.agent.skill_manager

        if not skill_names:
            print("ℹ No skills found")
            return

        print(f"\n🎯 Skills ({len(skill_names)}):\n")

        for name in skill_names:
            skill = skill_manager.skills.get(name)
            if skill:
                print(f"  • {name}")
                print(f"    {skill.description}")
                # 参数在 metadata 中
                if hasattr(skill, "metadata") and hasattr(skill.metadata, "parameters"):
                    if skill.metadata.parameters:
                        print(f"    Parameters: {list(skill.metadata.parameters.keys())}")
            else:
                print(f"  • {name}")
        print()

    def list_agent_configs(self):
        """列出所有 Agent Configs"""
        if not self.session or not self.session.agent.enable_resources:
            print("⚠ Resources not enabled")
            return

        config_names = self.session.agent.configs
        config_manager = self.session.agent.config_manager

        if not config_names:
            print("ℹ No agent configs found")
            return

        print(f"\n🤖 Agent Configs ({len(config_names)}):\n")

        for name in config_names:
            config = config_manager.configs.get(name)
            if config:
                print(f"  • {name}")
                print(f"    {config.description}")
                # AgentConfig 没有 model 属性，只有 tools, thinking_level 等
                if hasattr(config, "thinking_level") and config.thinking_level:
                    print(f"    Thinking: {config.thinking_level}")
                if hasattr(config, "tools") and config.tools:
                    print(f"    Tools: {', '.join(config.tools[:3])}{'...' if len(config.tools) > 3 else ''}")
            else:
                print(f"  • {name}")
        print()

    def _flush_stream_buffer(self, force: bool = False):
        if not self._stream_buf:
            return

        now = time.monotonic()
        if (
            not force
            and self._stream_buf_chars < self._flush_chars
            and (now - self._last_flush_ts) < self._flush_interval_s
        ):
            return

        chunk = "".join(self._stream_buf)
        self._stream_buf.clear()
        self._stream_buf_chars = 0
        self._last_flush_ts = now

        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_event(self, event):
        """处理事件"""
        event_type = event.type

        if event_type == "message_start":
            if hasattr(event, "message") and event.message and event.message.role == "assistant":
                self._flush_stream_buffer(force=True)
                sys.stdout.write("\n🤖 Assistant: ")
                sys.stdout.flush()

        elif event_type == "message_update":
            if hasattr(event, "assistantMessageEvent") and event.assistantMessageEvent:
                msg_event = event.assistantMessageEvent
                if getattr(msg_event, "type", "") == "text_delta":
                    delta = getattr(msg_event, "delta", "")
                    if delta:
                        self._stream_buf.append(delta)
                        self._stream_buf_chars += len(delta)
                        self._flush_stream_buffer(force=False)

        elif event_type == "message_end":
            if hasattr(event, "message") and event.message and event.message.role == "assistant":
                self._flush_stream_buffer(force=True)
                if getattr(event.message, "errorMessage", None):
                    sys.stdout.write(f"\n❌ LLM Error: {event.message.errorMessage}\n")
                else:
                    sys.stdout.write("\n")
                sys.stdout.flush()

        elif event_type == "tool_execution_start":
            if hasattr(event, "toolName"):
                self._flush_stream_buffer(force=True)
                args = getattr(event, "args", {}) or {}
                action = self._describe_tool_action(event.toolName, args)
                print(f"\n🔧 Tool: {event.toolName}")
                print(f"   ↳ {action}")

    async def handle_command(self, command):
        """处理命令"""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd in ["/exit", "/quit"]:
            return False

        elif cmd == "/help":
            self.print_header()

        elif cmd == "/new":
            await self.create_session(arg)

        elif cmd == "/sessions":
            sessions = self.session_manager.list_sessions()
            if sessions:
                print(f"\n📋 Sessions ({len(sessions)}):\n")
                for i, s in enumerate(sessions, 1):
                    marker = "→" if s.session_id == self.current_session_id else " "
                    print(f"{marker} {i}. {s.session_id}")
                    if s.title:
                        print(f"     {s.title}")
                    print(f"     Messages: {s.message_count}")
                print()
            else:
                print("ℹ No sessions found")

        elif cmd == "/info":
            if self.session:
                print(f"\n📊 Session: {self.current_session_id}")
                print(f"   Model: {self.session.agent.state.model.id}")
                print(f"   Messages: {len(self.session.agent.messages)}")
                print(f"   Tools: {len(self.session.agent.state.tools)}")
                print()
            else:
                print("ℹ No active session")

        elif cmd == "/skills":
            self.list_skills()

        elif cmd == "/agents":
            self.list_agent_configs()

        else:
            print(f"✗ Unknown command: {cmd}")
            print("  Type /help for available commands")

        return True

    async def run(self):
        """运行终端"""
        if not self.print_header():
            return

        # 创建或加载会话
        sessions = self.session_manager.list_sessions()
        if sessions:
            latest = sessions[0]
            print(f"Found recent session: {latest.session_id}")

            choice = input("Load it? (y/n): ").strip()

            if choice.startswith("/"):
                print("Creating new session first...")
                await self.create_session()
                self.running = await self.handle_command(choice)
            elif choice.lower() == "y":
                try:
                    self.session = CodingSession.load(
                        session_id=latest.session_id,
                        coding_agent_class=CodingAgent,
                    )
                    self.current_session_id = latest.session_id
                    self.session.agent.subscribe(self.on_event)
                    print(f"✓ Session loaded: {latest.session_id}\n")
                except Exception as e:
                    print(f"✗ Failed to load: {e}")
                    await self.create_session()
            else:
                await self.create_session()
        else:
            await self.create_session()

        if not self.session:
            print("✗ No session. Exiting.")
            return

        print("✨ Ready! Type your message or /help\n")

        # 主循环
        while self.running:
            try:
                # 获取输入（不使用颜色代码，避免显示问题）
                if self.session:
                    user_input = input("You> ").strip()
                else:
                    user_input = input("(No Session)> ").strip()

                # 处理空输入
                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    self.running = await self.handle_command(user_input)
                    continue

                # 检查会话
                if not self.session:
                    print("ℹ No session. Use /new")
                    continue

                # 发送消息
                try:
                    await self.session.prompt(user_input)
                    print()

                except Exception as e:
                    print(f"\n✗ Error: {e}\n")
                    import traceback

                    traceback.print_exc()

            except KeyboardInterrupt:
                print("\n")
                print("ℹ Use /exit to quit")
                print()
                continue

            except EOFError:
                print()
                break

        # 退出前保存
        if self.session and self.session.auto_save:
            self.session.save()
            print("✓ Session saved")

    def _describe_tool_action(self, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return f"执行命令: {args.get('command', '')}"
        if tool_name == "read":
            path = args.get("path", "")
            offset = args.get("offset")
            limit = args.get("limit")
            if offset or limit:
                return f"读取文件: {path} (offset={offset}, limit={limit})"
            return f"读取文件: {path}"
        if tool_name == "write":
            return f"写入文件: {args.get('path', '')}"
        if tool_name == "edit":
            return f"编辑文件: {args.get('path', '')}"
        if tool_name == "find":
            return f"查找文件: pattern={args.get('pattern', '')}, path={args.get('path', '.')}"
        if tool_name == "grep":
            return f"内容搜索: pattern={args.get('pattern', '')}, path={args.get('path', '.')}"
        if tool_name == "ls":
            return f"列目录: {args.get('path', '.')}"
        return json.dumps(args, ensure_ascii=False)


async def main():
    terminal = FinalTerminal()
    await terminal.run()


if __name__ == "__main__":
    print("🚀 Interactive Coding Terminal (Final)\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
