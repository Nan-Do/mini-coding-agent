import dataclasses
import json
from pathlib import Path
from utils import Memory, Session, history_entry_from_dict


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        path = self.path(session.id)
        serializable = {
            **dataclasses.asdict(session),
            "history": [
                dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item
                for item in session.history
            ],
            "memory": dataclasses.asdict(session.memory),
        }
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> Session:
        data = json.loads(self.path(session_id).read_text(encoding="utf-8"))
        data["memory"] = Memory(**data["memory"])
        data["history"] = [history_entry_from_dict(item) for item in data["history"]]
        return Session(**data)

    def latest(self) -> str | None:
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None
