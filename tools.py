import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

from workspace import WorkspaceContext
from utils import (
    IGNORED_PATH_NAMES,
    HistoryEntry,
    ToolMessageEntry,
    Tools,
    ToolDescriptionEntry,
    clip,
)


class ToolRegistry:
    def __init__(
        self,
        workspace: WorkspaceContext,
        root: Path,
        approval_policy: str,
        read_only: bool,
        depth: int,
        max_depth: int,
        get_history: Callable[[], List[HistoryEntry]],
        delegate_fn: Callable[[str, int], str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.root = root
        self.approval_policy = approval_policy
        self.read_only = read_only
        self.depth = depth
        self.max_depth = max_depth
        self.get_history = get_history
        self.delegate_fn = delegate_fn
        self._registry: Tools = self._build()

    def items(self):
        return self._registry.items()

    def run(self, name: str, args: Dict) -> str:
        tool = self._registry.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self._validate(name, args)
        except Exception as exc:
            example = self._example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return message
        if self._repeated_call(name, args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool.risky and not self._approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool.run(args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def _build(self) -> Tools:
        tools = {
            "list_files": ToolDescriptionEntry(
                schema={"path": "str='.'"},
                risky=False,
                description="List files in the workspace.",
                run=self._tool_list_files,
            ),
            "read_file": ToolDescriptionEntry(
                schema={"path": "str", "start": "int=1", "end": "int=200"},
                risky=False,
                description="Read a UTF-8 file by line range.",
                run=self._tool_read_file,
            ),
            "search": ToolDescriptionEntry(
                schema={"pattern": "str", "path": "str='.'"},
                risky=False,
                description="Search the workspace with rg or a simple fallback.",
                run=self._tool_search,
            ),
            "run_shell": ToolDescriptionEntry(
                schema={"command": "str", "timeout": "int=20"},
                risky=True,
                description="Run a shell command in the repo root.",
                run=self._tool_run_shell,
            ),
            "write_file": ToolDescriptionEntry(
                schema={"path": "str", "content": "str"},
                risky=True,
                description="Write a text file.",
                run=self._tool_write_file,
            ),
            "patch_file": ToolDescriptionEntry(
                schema={"path": "str", "old_text": "str", "new_text": "str"},
                risky=True,
                description="Replace one exact text block in a file.",
                run=self._tool_patch_file,
            ),
        }
        if self.delegate_fn is not None:
            tools["delegate"] = ToolDescriptionEntry(
                schema={"task": "str", "max_steps": "int=3"},
                risky=False,
                description="Ask a bounded read-only child agent to investigate.",
                run=self._tool_delegate,
            )
        return tools

    def _path_is_within_root(self, resolved: Path) -> bool:
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def _path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self._path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def _validate(self, name: str, args: Dict) -> None:
        args = args or {}

        if name == "list_files":
            path = self._path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self._path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self._path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self._path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self._path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "delegate":
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

    def _approve(self, name: str, args: Dict) -> bool:
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(
                f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] "
            )
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def _example(self, name: str) -> str:
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def _repeated_call(self, name: str, args: Dict) -> bool:
        tool_events = [
            item for item in self.get_history() if isinstance(item, ToolMessageEntry)
        ]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item.name == name and item.args == args for item in recent)

    def _tool_list_files(self, args: Dict) -> str:
        path = self._path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item
            for item in sorted(
                path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())
            )
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def _tool_read_file(self, args: Dict) -> str:
        path = self._path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{number:>4}: {line}"
            for number, line in enumerate(lines[start - 1 : end], start=start)
        )
        return f"# {path.relative_to(self.root)}\n{body}"

    def _tool_search(self, args: Dict) -> str:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self._path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = (
            [path]
            if path.is_file()
            else [
                item
                for item in path.rglob("*")
                if item.is_file()
                and not any(
                    part in IGNORED_PATH_NAMES
                    for part in item.relative_to(self.root).parts
                )
            ]
        )
        for file_path in files:
            for number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if pattern.lower() in line.lower():
                    matches.append(
                        f"{file_path.relative_to(self.root)}:{number}:{line}"
                    )
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def _tool_run_shell(self, args: Dict) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    def _tool_write_file(self, args: Dict) -> str:
        path = self._path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def _tool_patch_file(self, args: Dict) -> str:
        path = self._path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(
            text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8"
        )
        return f"patched {path.relative_to(self.root)}"

    def _tool_delegate(self, args: Dict) -> str:
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        max_steps = int(args.get("max_steps", 3))
        return self.delegate_fn(task, max_steps)
