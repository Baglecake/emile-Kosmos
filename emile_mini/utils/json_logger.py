
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

class JSONLWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Optional[TextIO] = open(self.path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        if self._f is None:
            return
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
