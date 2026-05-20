"""Cross-modal merger — align transcript, OCR, captions to the shot timeline.

Also groups consecutive similar shots into "chapters" using a heuristic over
caption similarity + dialogue continuity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .keyframes import Keyframe
from .ocr import OCRResult
from .shots import Shot
from .transcript import TranscriptSegment
from .vlm import ShotCaption


@dataclass
class ShotRecord:
    shot: Shot
    keyframes: list[Keyframe]
    transcript: list[TranscriptSegment]
    ocr_text: str
    caption: ShotCaption | None


@dataclass
class Chapter:
    index: int
    start_seconds: float
    end_seconds: float
    title: str
    shot_indices: list[int] = field(default_factory=list)


def slice_transcript_per_shot(
    shots: list[Shot], segments: list[TranscriptSegment],
) -> dict[int, list[TranscriptSegment]]:
    """Split transcript by shot boundaries. Segments that straddle a boundary are
    assigned to the shot containing their midpoint."""
    out: dict[int, list[TranscriptSegment]] = {s.index: [] for s in shots}
    for seg in segments:
        mid = (seg.start_seconds + seg.end_seconds) / 2
        owner = _find_shot(shots, mid)
        if owner is not None:
            out[owner].append(seg)
    return out


def group_keyframes_by_shot(keyframes: list[Keyframe]) -> dict[int, list[Keyframe]]:
    out: dict[int, list[Keyframe]] = {}
    for kf in keyframes:
        out.setdefault(kf.shot_index, []).append(kf)
    return out


def group_ocr_by_shot(
    shots: list[Shot],
    keyframes: list[Keyframe],
    ocr_by_path: dict[str, OCRResult],
) -> dict[int, str]:
    """Merge OCR text across keyframes within the same shot, deduped."""
    per_shot: dict[int, list[str]] = {s.index: [] for s in shots}
    for kf in keyframes:
        r = ocr_by_path.get(kf.path)
        if not r:
            continue
        per_shot.setdefault(kf.shot_index, []).extend(line.text for line in r.lines)
    out: dict[int, str] = {}
    for idx, texts in per_shot.items():
        seen: set[str] = set()
        keep: list[str] = []
        for t in texts:
            key = t.lower().strip()
            if key and key not in seen:
                seen.add(key)
                keep.append(t)
        out[idx] = "\n".join(keep)
    return out


def build_shot_records(
    shots: list[Shot],
    keyframes: list[Keyframe],
    segments: list[TranscriptSegment],
    ocr_by_path: dict[str, OCRResult],
    captions: list[ShotCaption],
) -> list[ShotRecord]:
    kfs = group_keyframes_by_shot(keyframes)
    trans = slice_transcript_per_shot(shots, segments)
    ocr = group_ocr_by_shot(shots, keyframes, ocr_by_path)
    cap_by_idx = {c.shot_index: c for c in captions}
    return [
        ShotRecord(
            shot=s,
            keyframes=kfs.get(s.index, []),
            transcript=trans.get(s.index, []),
            ocr_text=ocr.get(s.index, ""),
            caption=cap_by_idx.get(s.index),
        )
        for s in shots
    ]


def group_into_chapters(
    records: list[ShotRecord], min_chapter_seconds: float = 20.0,
) -> list[Chapter]:
    """Heuristic: start a new chapter when setting/action keywords change strongly
    OR transcript topic shifts (proxy: long silence ≥3s). Otherwise extend."""
    if not records:
        return []

    chapters: list[Chapter] = []
    current_indices: list[int] = [records[0].shot.index]
    chapter_start = records[0].shot.start_seconds
    last_setting_tokens = _tokens(_setting_of(records[0]))
    last_end = records[0].shot.end_seconds

    for rec in records[1:]:
        setting_tokens = _tokens(_setting_of(rec))
        silence_gap = rec.shot.start_seconds - last_end
        sim = _jaccard(last_setting_tokens, setting_tokens)
        chapter_dur = rec.shot.end_seconds - chapter_start
        new_chapter = (
            chapter_dur > min_chapter_seconds and (sim < 0.25 or silence_gap > 3.0)
        )
        if new_chapter:
            chapters.append(_make_chapter(len(chapters), records, current_indices))
            current_indices = []
            chapter_start = rec.shot.start_seconds
        current_indices.append(rec.shot.index)
        last_setting_tokens = setting_tokens or last_setting_tokens
        last_end = rec.shot.end_seconds

    if current_indices:
        chapters.append(_make_chapter(len(chapters), records, current_indices))
    return chapters


def _make_chapter(idx: int, records: list[ShotRecord], shot_indices: list[int]) -> Chapter:
    by_idx = {r.shot.index: r for r in records}
    first = by_idx[shot_indices[0]]
    last = by_idx[shot_indices[-1]]
    title = _setting_of(first) or _action_of(first) or f"Chapter {idx+1}"
    return Chapter(
        index=idx,
        start_seconds=first.shot.start_seconds,
        end_seconds=last.shot.end_seconds,
        title=_short(title, 80),
        shot_indices=shot_indices,
    )


def _setting_of(r: ShotRecord) -> str:
    return r.caption.setting if r.caption else ""


def _action_of(r: ShotRecord) -> str:
    return r.caption.action if r.caption else ""


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _short(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _find_shot(shots: list[Shot], t: float) -> int | None:
    for s in shots:
        if s.start_seconds <= t < s.end_seconds:
            return s.index
    return shots[-1].index if shots and t >= shots[-1].start_seconds else None
