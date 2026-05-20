"""yt-dlp wrapper — download URL, prefer mp4, also fetch native captions."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


URL_PREFIXES = ("http://", "https://", "ftp://")


def is_url(source: str) -> bool:
    return source.startswith(URL_PREFIXES)


def download(source: str, out_dir: Path, prefer_lang: str = "en",
             fetch_subs: bool = True) -> dict:
    """Download video + optionally subtitle track. Returns dict with paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"local video not found: {path}")
        return {"video_path": str(path), "subtitle_path": None, "info": {}}

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp not on PATH. Install: pipx install yt-dlp")

    tpl = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "--write-info-json",
        "--no-playlist",
        "-o", tpl,
        source,
    ]
    if fetch_subs:
        cmd[5:5] = [
            "--write-auto-sub", "--write-sub",
            "--sub-lang", f"{prefer_lang},en,en-orig",
            "--convert-subs", "vtt",
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"yt-dlp failed: {result.stderr.strip()[:500]}")

    video_files = sorted(
        p for p in out_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    )
    if not video_files:
        raise SystemExit(f"yt-dlp produced no video file in {out_dir}")
    video_path = video_files[0]

    vtt = next(out_dir.glob(f"{video_path.stem}*.vtt"), None)

    info: dict = {}
    info_json = next(out_dir.glob(f"{video_path.stem}*.info.json"), None)
    if info_json:
        try:
            info = json.loads(info_json.read_text(encoding="utf-8"))
        except Exception:
            info = {}

    return {
        "video_path": str(video_path),
        "subtitle_path": str(vtt) if vtt else None,
        "info": {
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "webpage_url": info.get("webpage_url"),
            "duration": info.get("duration"),
        },
    }
