import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient
from session import SessionStore
from tools import ToolRegistry
from typing import Dict, List, Self, Tuple
from workspace import WorkspaceContext
from app_types import (
    HistoryEntry,
    Memory,
    MessageEntry,
    Session,
    ToolMessageEntry,
)
from utils import (
    MAX_HISTORY,
    now,
    clip,
)


class MiniAgent:
    def __init__(
        self: Self,
        model_client: LlamaCppModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session: Session | None = None,
        approval_policy: str = "ask",
        max_steps: int = 6,
        max_new_tokens: int = 512,
        depth: int = 0,
        max_depth: int = 1,
        read_only: bool = False,
        logger: AgentLogger | None = None,
    ) -> None:
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or Session(
            id=datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            created_at=now(),
            workspace_root=workspace.repo_root,
            history=[],
            memory=Memory(task="", files=[], notes=[]),
        )
        base_logger = logger or AgentLogger(None, enabled=False)
        self.logger = base_logger.child(session=self.session.id, depth=self.depth)
        self.logger.log(
            "session_start",
            workspace_root=self.session.workspace_root,
            approval_policy=self.approval_policy,
            read_only=self.read_only,
            resumed=session is not None,
            history_len=len(self.session.history),
        )
        self.tools = ToolRegistry(
            workspace=self.workspace,
            root=self.root,
            approval_policy=self.approval_policy,
            read_only=self.read_only,
            depth=self.depth,
            max_depth=self.max_depth,
            get_history=lambda: self.session.history,
            delegate_fn=self._make_delegate if self.depth < self.max_depth else None,
        )
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(
        cls: type[Self],
        model_client: LlamaCppModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session_id: str,
        **kwargs,
    ) -> Self:
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket: List[str], item: str, limit: int) -> None:
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def _make_delegate(self: Self, task: str, max_steps: int) -> str:
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            logger=self.logger,
        )
        child.session.memory.task = task
        child.session.memory.notes = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)

    def build_prefix(self: Self) -> str:
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool.schema.items())
            risk = "approval required" if tool.risky else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool.description}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        rules = "\n".join(
            [
                "- Use tools instead of guessing about the workspace.",
                "- Return exactly one <tool>...</tool> or one <final>...</final>.",
                "- Tool calls must look like:",
                '  <tool>{"name":"tool_name","args":{...}}</tool>',
                "- For write_file and patch_file with multi-line text, prefer XML style:",
                '  <tool name="write_file" path="file.py"><content>...</content></tool>',
                "- Final answers must look like:",
                "  <final>your answer</final>",
                "- Never invent tool results.",
                "- Keep answers concise and concrete.",
                "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
                "- Before writing tests for existing code, read the implementation first.",
                "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
                "- New files should be complete and runnable, including obvious imports.",
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
                "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
            ]
        )
        return "\n\n".join(
            [
                "You are Mini-Coding-Agent, a small local coding agent running through llama-server.",
                "Rules:\n" + rules,
                "Tools:\n" + tool_text,
                "Valid response examples:\n" + examples,
                self.workspace.text(),
            ]
        )

    def memory_text(self: Self) -> str:
        memory: Memory = self.session.memory
        notes = "\n".join(f"- {note}" for note in memory.notes) or "- none"
        return "\n".join(
            [
                "Memory:",
                f"- task: {memory.task or '-'}",
                f"- files: {', '.join(memory.files) or '-'}",
                "- notes:",
                notes,
            ]
        )

    def history_text(self: Self) -> str:
        history: List[HistoryEntry] = self.session.history
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if isinstance(item, ToolMessageEntry) and item.name in (
                "write_file",
                "patch_file",
            ):
                path = str(item.args.get("path", ""))
                seen_reads.discard(path)
            if (
                isinstance(item, ToolMessageEntry)
                and item.name == "read_file"
                and not recent
            ):
                path = str(item.args.get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if isinstance(item, ToolMessageEntry):
                limit = 900 if recent else 180
                lines.append(
                    f"[tool:{item.name}] {json.dumps(item.args, sort_keys=True)}"
                )
                lines.append(clip(item.content, limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item.role}] {clip(item.content, limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def prompt(self: Self, user_message: str) -> Tuple[str, str]:
        return self.prefix, "\n\n".join(
            [
                self.memory_text(),
                "Transcript:\n" + self.history_text(),
                "Current user request:\n" + user_message,
            ]
        )

    def memory_snapshot(self: Self) -> Dict[str, object]:
        memory = self.session.memory
        return {
            "task": memory.task,
            "files": list(memory.files),
            "notes": list(memory.notes),
        }

    def log_memory(self: Self, reason: str) -> None:
        self.logger.log("memory_update", reason=reason, memory=self.memory_snapshot())

    def record(self: Self, item: HistoryEntry) -> None:
        self.session.history.append(item)
        if isinstance(item, ToolMessageEntry):
            self.logger.log(
                "history_append",
                entry="tool",
                index=len(self.session.history) - 1,
                name=item.name,
                args=item.args,
                content=clip(item.content, 2000),
            )
        else:
            self.logger.log(
                "history_append",
                entry="message",
                index=len(self.session.history) - 1,
                role=item.role,
                content=clip(item.content, 2000),
            )
        self.session_path = self.session_store.save(self.session)

    def note_tool(self: Self, name: str, args: Dict[str, str], result: str) -> None:
        memory = self.session.memory
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory.files, str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory.notes, note, 5)
        self.log_memory(f"note_tool:{name}")

    def ask(self: Self, user_message: str) -> str:
        memory = self.session.memory
        if not memory.task:
            memory.task = clip(user_message.strip(), 300)
        self.logger.log(
            "request_start",
            max_steps=self.max_steps,
            max_new_tokens=self.max_new_tokens,
            user_message=clip(user_message, 2000),
        )
        self.log_memory("request_start")
        self.record(MessageEntry(role="user", content=user_message, created_at=now()))

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            system_prompt, user_prompt = self.prompt(user_message)
            self.logger.log(
                "prompt_built",
                attempt=attempts,
                tool_step=tool_steps,
                system_len=len(system_prompt),
                user_len=len(user_prompt),
                memory_text=self.memory_text(),
                history_text=self.history_text(),
            )
            raw = self.model_client.complete(
                system_prompt, user_prompt, self.max_new_tokens
            )
            kind, payload = self.parse(raw)
            self.logger.log(
                "model_output",
                attempt=attempts,
                tool_step=tool_steps,
                parse_kind=kind,
                raw=clip(raw, 2000),
            )

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                self.logger.log("tool_call", name=name, args=args, step=tool_steps)
                result = self.tools.run(name, args)
                self.logger.log(
                    "tool_result",
                    name=name,
                    step=tool_steps,
                    result=clip(result, 2000),
                )
                self.record(
                    ToolMessageEntry(
                        role="tool",
                        name=name,
                        args=args,
                        content=result,
                        created_at=now(),
                    )
                )
                self.note_tool(name, args, result)
                continue

            if kind == "retry":
                self.logger.log("retry", attempt=attempts, notice=clip(str(payload), 500))
                self.record(
                    MessageEntry(role="assistant", content=payload, created_at=now())
                )
                continue

            final = (payload or raw).strip()
            self.record(MessageEntry(role="assistant", content=final, created_at=now()))
            self.remember(memory.notes, clip(final, 220), 5)
            self.log_memory("final")
            self.logger.log(
                "final",
                reason="answer",
                tool_steps=tool_steps,
                attempts=attempts,
                final=clip(final, 2000),
            )
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            reason = "max_attempts"
        else:
            final = "Stopped after reaching the step limit without a final answer."
            reason = "max_steps"
        self.record(MessageEntry(role="assistant", content=final, created_at=now()))
        self.logger.log(
            "final",
            reason=reason,
            tool_steps=tool_steps,
            attempts=attempts,
            final=clip(final, 2000),
        )
        return final

    @staticmethod
    def parse(raw: str) -> Tuple[str, str | Dict[str, str]]:
        raw = str(raw)
        if "<tool>" in raw and (
            "<final>" not in raw or raw.find("<tool>") < raw.find("<final>")
        ):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", MiniAgent.retry_notice(
                    "model returned malformed tool JSON"
                )
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice(
                    "tool payload must be a JSON object"
                )
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice(
                    "tool payload is missing a tool name"
                )
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and (
            "<final>" not in raw or raw.find("<tool") < raw.find("<final>")
        ):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice(
                "model returned an empty <final> answer"
            )
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem: str | None = None) -> str:
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw: str) -> Dict[str, str | Dict[str, str]] | None:
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in (
            "content",
            "old_text",
            "new_text",
            "command",
            "task",
            "pattern",
            "path",
        ):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text: str) -> Dict[str, str]:
        attrs = {}
        for match in re.finditer(
            r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text
        ):
            attrs[match.group(1)] = (
                match.group(2) if match.group(2) is not None else match.group(3)
            )
        return attrs

    @staticmethod
    def extract(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self: Self) -> None:
        self.session.history = []
        self.session.memory = Memory(task="", files=[], notes=[])
        self.session_store.save(self.session)
        self.logger.log("reset")

    @property
    def log_path(self: Self) -> str:
        return str(self.logger.path) if self.logger.path else "(logging disabled)"
