"""ffprobe wrapper — duration, resolution, audio presence."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict


class VideoMeta(TypedDict):
    duration_seconds: float
    width: int | None
    height: int | None
    codec: str | None
    fps: float | None
    has_audio: bool
    size_bytes: int


def get_metadata(video_path: str | Path) -> VideoMeta:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe not on PATH. Install ffmpeg (apt/brew/winget install ffmpeg).")

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams",
            str(Path(video_path).resolve()),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = None
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate")
    if rate and "/" in rate:
        num, den = rate.split("/")
        try:
            fps = float(num) / float(den) if float(den) else None
        except ValueError:
            fps = None

    return VideoMeta(
        duration_seconds=float(fmt.get("duration") or v.get("duration") or 0),
        width=v.get("width"),
        height=v.get("height"),
        codec=v.get("codec_name"),
        fps=fps,
        has_audio=a is not None,
        size_bytes=int(fmt.get("size") or 0),
    )


def format_time(seconds: float) -> str:
    """Seconds → HH:MM:SS.mmm string."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_time(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise ValueError(f"Cannot parse time: {value!r}")
