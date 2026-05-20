"""VLM shot captioning — Gemini 2.5 reads keyframes + local audio context per shot.

Why pre-caption: the alternative is dumping raw frames into Claude's context,
which costs ~1k image tokens per frame. Captioning first gives Claude a compact
text timeline.

Input per shot: 1-5 keyframe JPEGs + transcript text in shot's time range +
OCR text. Output: structured caption {action, setting, on_screen_text_summary,
change_from_prev, notable_objects}.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .keyframes import Keyframe
from .shots import Shot
from .transcript import TranscriptSegment


@dataclass
class ShotCaption:
    shot_index: int
    action: str
    setting: str
    on_screen_text_summary: str
    notable_objects: list[str]
    change_from_prev: str
    raw: dict


SHOT_PROMPT = """You describe one shot from a video for an LLM that cannot see it.
Be specific and concrete. No "appears to" hedging. No prose padding.

Return strict JSON only, no markdown fences:
{
  "action": "<what is happening — verbs, motion, interaction>",
  "setting": "<environment, screen vs real-world, app/UI shown>",
  "on_screen_text_summary": "<short summary of meaningful text visible; not character-by-character>",
  "notable_objects": ["<object1>", "<object2>"],
  "change_from_prev": "<one sentence: what changed since the previous shot, or 'opening shot'>"
}

Shot index: %(idx)d  (%(start)s → %(end)s, %(dur).1fs)
Audio transcript during this shot:
%(transcript)s

OCR text already extracted from this shot:
%(ocr)s
"""


def caption_shots(
    shots: list[Shot],
    keyframes_by_shot: dict[int, list[Keyframe]],
    transcript_by_shot: dict[int, list[TranscriptSegment]],
    ocr_by_shot: dict[int, str],
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> list[ShotCaption]:
    """Caption every shot. Returns one ShotCaption per shot. Skips silently on missing SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return []
    client = genai.Client(api_key=key)

    from . import log_util as logu
    captions: list[ShotCaption] = []
    n = len(shots)
    t0 = time.time()
    for s_idx, shot in enumerate(shots, 1):
        kfs = keyframes_by_shot.get(shot.index, [])
        if not kfs:
            continue
        trans_text = "\n".join(
            f"[{_fmt(s.start_seconds)}] {('('+s.speaker+') ') if s.speaker else ''}{s.text}"
            for s in transcript_by_shot.get(shot.index, [])
        ) or "(no speech)"
        ocr_text = ocr_by_shot.get(shot.index, "") or "(no text on screen)"

        prompt = SHOT_PROMPT % dict(
            idx=shot.index,
            start=_fmt(shot.start_seconds),
            end=_fmt(shot.end_seconds),
            dur=shot.duration,
            transcript=trans_text,
            ocr=ocr_text,
        )
        parts: list = [prompt]
        for kf in kfs[:5]:
            parts.append(types.Part.from_bytes(
                data=Path(kf.path).read_bytes(),
                mime_type="image/jpeg",
            ))

        text = ""
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(model=model, contents=parts)
                text = resp.text or ""
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "503" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg:
                    delay = _parse_retry_delay(msg) or min(60, 12 * (2 ** attempt))
                    code = "429" if "429" in msg else "503"
                    logu.warn(f"shot {shot.index} attempt {attempt+1}: {code}, sleeping {delay:.0f}s")
                    time.sleep(delay)
                    continue
                break  # non-retryable
        if last_err is not None:
            captions.append(_fallback(shot.index, f"vlm-error: {last_err}"))
            elapsed = time.time() - t0
            rate = s_idx / elapsed if elapsed > 0 else 0
            eta = (n - s_idx) / rate if rate > 0 else 0
            logu.progress(s_idx, n, "shots (err)", rate=rate, eta_seconds=eta)
            continue

        data = _extract_json(text)
        if not data:
            captions.append(_fallback(shot.index, text[:200]))
            continue

        captions.append(ShotCaption(
            shot_index=shot.index,
            action=str(data.get("action", "")).strip(),
            setting=str(data.get("setting", "")).strip(),
            on_screen_text_summary=str(data.get("on_screen_text_summary", "")).strip(),
            notable_objects=list(data.get("notable_objects", []))[:10],
            change_from_prev=str(data.get("change_from_prev", "")).strip(),
            raw=data,
        ))
        if s_idx == 1 or s_idx % 10 == 0 or s_idx == n:
            elapsed = time.time() - t0
            rate = s_idx / elapsed if elapsed > 0 else 0
            eta = (n - s_idx) / rate if rate > 0 else 0
            logu.progress(s_idx, n, "shots", rate=rate, eta_seconds=eta)
    return captions


def _parse_retry_delay(msg: str) -> float | None:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\s*s", msg)
    if m:
        return float(m.group(1)) + 1.0  # small buffer
    return None


def _fallback(idx: int, note: str) -> ShotCaption:
    return ShotCaption(
        shot_index=idx, action="", setting="", on_screen_text_summary="",
        notable_objects=[], change_from_prev=note, raw={},
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


def _fmt(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"
