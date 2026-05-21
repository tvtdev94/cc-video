"""Per-stage checkpoint persistence.

Each pipeline stage writes its output to {work_dir}/state/{stage}.json (or
.jsonl for incremental stages). Subsequent runs load instead of recomputing.

Layout:
  state/shots.json       — list of Shot dicts (one-shot)
  state/keyframes.json   — list of Keyframe dicts (one-shot)
  state/transcript.json  — list of TranscriptSegment dicts (one-shot)
  state/ocr.jsonl        — one OCR result per line, append-only (incremental)
  state/vlm.jsonl        — one ShotCaption per line, append-only (incremental)

JSONL stages survive mid-stage crashes — completed items are kept, missing
items get retried on resume.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


class Store:
    """Per-stage checkpoint store rooted at {work_dir}/state."""

    def __init__(self, work_dir: Path):
        self.dir = Path(work_dir) / "state"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    # --- one-shot JSON (shots, keyframes, transcript) -------------------

    def save_json(self, name: str, items: Iterable[Any]) -> None:
        data = [_to_dict(it) for it in items]
        tmp = self.path(name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path(name))

    def load_json(self, name: str, factory: Callable[[dict], T]) -> list[T]:
        raw = json.loads(self.path(name).read_text(encoding="utf-8"))
        return [factory(d) for d in raw]

    # --- incremental JSONL (ocr, vlm) -----------------------------------

    def jsonl_done_keys(self, name: str, key_field: str) -> set:
        """Return set of `key_field` values already recorded in JSONL."""
        p = self.path(name)
        if not p.exists():
            return set()
        keys: set = set()
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line).get(key_field))
                except json.JSONDecodeError:
                    continue  # tolerate corrupt trailing line
        keys.discard(None)
        return keys

    def jsonl_load_all(self, name: str, factory: Callable[[dict], T]) -> list[T]:
        p = self.path(name)
        if not p.exists():
            return []
        out: list[T] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(factory(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return out

    def jsonl_append(self, name: str, obj: Any) -> None:
        with self.path(name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_dict(obj), ensure_ascii=False) + "\n")


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj
