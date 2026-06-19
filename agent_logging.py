import json
import threading
from pathlib import Path
from typing import Any, Dict, Self

from utils import now


class AgentLogger:
    """Structured JSONL logger for the agent.

    Writes one JSON object per line to a single per-run log file. It is used to
    record everything the agent stores and exchanges:

    - ``llm_request`` / ``llm_response`` / ``llm_continuation`` — the raw
      communications with the llama-server backend (prompts and completions).
    - ``memory_update`` — snapshots of the distilled working memory.
    - ``history_append`` — every entry appended to the session transcript.
    - ``tool_call`` / ``tool_result`` — tool invocations and their output.
    - ``model_output`` / ``parse`` — raw model text and how it was interpreted.
    - ``request_start`` / ``final`` — the lifecycle of a single ``ask()``.

    A disabled logger (``enabled=False`` or ``path=None``) is a cheap no-op, so
    the rest of the code can always call :meth:`log` unconditionally.
    """

    def __init__(
        self: Self,
        path: Path | None,
        enabled: bool = True,
        defaults: Dict[str, Any] | None = None,
        lock: "threading.Lock | None" = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = bool(enabled) and self.path is not None
        self._defaults: Dict[str, Any] = dict(defaults or {})
        self._lock = lock or threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def bind(self: Self, **fields: Any) -> None:
        """Attach default fields included on every subsequent record."""
        for key, value in fields.items():
            if value is not None:
                self._defaults[key] = value

    def child(self: Self, **fields: Any) -> "AgentLogger":
        """Return a logger writing to the same file with extra default fields.

        Used by nested (delegate) agents so their records share the run file
        but carry their own ``depth``/``session`` without clobbering the parent.
        """
        merged = {**self._defaults, **{k: v for k, v in fields.items() if v is not None}}
        return AgentLogger(
            path=self.path,
            enabled=self.enabled,
            defaults=merged,
            lock=self._lock,
        )

    def log(self: Self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {"ts": now(), "event": event, **self._defaults, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
