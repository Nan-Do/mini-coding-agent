from collections.abc import Callable
from datetime import datetime, timezone
from typing import Dict, List, TypeAlias, TypedDict


DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/reset   Clear the current session history and memory.",
        "/exit    Exit the agent.",
    ]
)
MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
IGNORED_PATH_NAMES = {
    ".git",
    ".mini-coding-agent",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

# Custom types:
Tools: TypeAlias = Dict[str, Dict[str, str | bool | Callable[..., None]]]


class MemoryDict(TypedDict):
    task: str
    files: List[str]
    notes: List[str]


class HistoryDict(TypedDict):
    role: str
    name: str
    content: str
    created_at: str


class SessionDict(TypedDict):
    id: str
    created_at: datetime
    workspace_root: str
    history: List[HistoryDict]
    memory: MemoryDict


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text: str, limit: int) -> str:
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]
