"""Render the final timeline as Markdown (Claude-readable) + JSON (queryable)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .merger import Chapter, ShotRecord
from .probe import VideoMeta


def write_outputs(
    work_dir: Path,
    source: str,
    meta: VideoMeta,
    records: list[ShotRecord],
    chapters: list[Chapter],
    info: dict,
) -> dict[str, Path]:
    md_path = work_dir / "video.understanding.md"
    json_path = work_dir / "video.understanding.json"
    md_path.write_text(_render_markdown(source, meta, records, chapters, info), encoding="utf-8")
    json_path.write_text(_render_json(source, meta, records, chapters, info), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def _render_markdown(
    source: str, meta: VideoMeta, records: list[ShotRecord],
    chapters: list[Chapter], info: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Video Understanding")
    lines.append("")
    lines.append(f"- **Source:** `{source}`")
    if info.get("title"):
        lines.append(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        lines.append(f"- **Uploader:** {info['uploader']}")
    dur = meta["duration_seconds"]
    lines.append(f"- **Duration:** {_fmt(dur)} ({dur:.1f}s)")
    if meta.get("width") and meta.get("height"):
        lines.append(f"- **Resolution:** {meta['width']}×{meta['height']}")
    lines.append(f"- **Shots detected:** {len(records)}")
    lines.append(f"- **Chapters:** {len(chapters)}")
    lines.append("")

    if chapters:
        lines.append("## Chapters (high-level)")
        lines.append("")
        for ch in chapters:
            lines.append(
                f"- **Ch {ch.index+1}** [{_fmt(ch.start_seconds)} → {_fmt(ch.end_seconds)}] "
                f"({len(ch.shot_indices)} shots) — {ch.title}"
            )
        lines.append("")

    lines.append("## Shot-by-shot timeline")
    lines.append("")
    lines.append(
        "Each shot is one camera take. Keyframes are the representative images; OCR is "
        "text read directly off the frame; caption is a VLM summary of the visual content."
    )
    lines.append("")
    for r in records:
        lines.extend(_render_shot(r))
        lines.append("")

    lines.append("## Full transcript")
    lines.append("")
    lines.append("```")
    for r in records:
        for seg in r.transcript:
            tag = f"({seg.speaker}) " if seg.speaker else ""
            lines.append(f"[{_fmt(seg.start_seconds)}] {tag}{seg.text}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def _render_shot(r: ShotRecord) -> list[str]:
    out: list[str] = []
    shot = r.shot
    header = (
        f"### Shot {shot.index} · {_fmt(shot.start_seconds)} → {_fmt(shot.end_seconds)} "
        f"({shot.duration:.1f}s)"
    )
    out.append(header)
    out.append("")

    if r.keyframes:
        kf_paths = [Path(k.path).as_posix() for k in r.keyframes]
        out.append("**Keyframes:** " + " ".join(f"`{p}`" for p in kf_paths))
        out.append("")

    if r.caption:
        c = r.caption
        if c.change_from_prev:
            out.append(f"**Change:** {c.change_from_prev}")
        if c.action:
            out.append(f"**Action:** {c.action}")
        if c.setting:
            out.append(f"**Setting:** {c.setting}")
        if c.on_screen_text_summary:
            out.append(f"**On-screen text (summary):** {c.on_screen_text_summary}")
        if c.notable_objects:
            out.append("**Notable objects:** " + ", ".join(c.notable_objects))
        out.append("")

    if r.ocr_text:
        out.append("**OCR (raw text on screen):**")
        out.append("```")
        out.append(r.ocr_text)
        out.append("```")

    if r.transcript:
        out.append("**Spoken:**")
        out.append("```")
        for seg in r.transcript:
            tag = f"({seg.speaker}) " if seg.speaker else ""
            out.append(f"[{_fmt(seg.start_seconds)}] {tag}{seg.text}")
        out.append("```")
    return out


def _render_json(
    source: str, meta: VideoMeta, records: list[ShotRecord],
    chapters: list[Chapter], info: dict,
) -> str:
    payload = {
        "source": source,
        "info": info,
        "meta": dict(meta),
        "chapters": [asdict(c) for c in chapters],
        "shots": [
            {
                "index": r.shot.index,
                "start_seconds": r.shot.start_seconds,
                "end_seconds": r.shot.end_seconds,
                "duration_seconds": r.shot.duration,
                "keyframes": [asdict(k) for k in r.keyframes],
                "ocr_text": r.ocr_text,
                "caption": asdict(r.caption) if r.caption else None,
                "transcript": [
                    {
                        "start": s.start_seconds,
                        "end": s.end_seconds,
                        "speaker": s.speaker,
                        "text": s.text,
                        "words": [asdict(w) for w in s.words],
                    }
                    for s in r.transcript
                ],
            }
            for r in records
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _fmt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
