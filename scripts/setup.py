#!/usr/bin/env python3
"""Preflight: check ffmpeg/yt-dlp + report which extras are present.

Exit codes:
  0 — ready
  2 — missing binaries (ffmpeg, ffprobe, yt-dlp)
  3 — missing optional Python extras (warning only)
"""
from __future__ import annotations

import json
import shutil
import sys


REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "yt-dlp")


def check() -> dict:
    missing_bin = [b for b in REQUIRED_BINARIES if shutil.which(b) is None]
    extras = {
        "scenedetect": _have("scenedetect"),
        "faster_whisper": _have("faster_whisper"),
        "pyannote": _have("pyannote.audio"),
        "paddleocr": _have("paddleocr"),
        "google_genai": _have("google.genai"),
        "open_clip": _have("open_clip"),
        "sklearn": _have("sklearn"),
        "torch": _have("torch"),
    }
    keys = {
        "gemini": _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        "groq": _env("GROQ_API_KEY"),
        "openai": _env("OPENAI_API_KEY"),
        "hf": _env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN"),
    }
    return {"missing_binaries": missing_bin, "extras": extras, "api_keys_set": keys}


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _env(name: str) -> bool:
    import os
    return bool(os.environ.get(name))


def main() -> int:
    report = check()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        print("# cc-video preflight\n")
        bins_ok = not report["missing_binaries"]
        print(f"Binaries: {'OK' if bins_ok else 'MISSING ' + ','.join(report['missing_binaries'])}")
        print("Extras (optional):")
        for k, v in report["extras"].items():
            print(f"  - {k}: {'installed' if v else 'not installed'}")
        print("API keys:")
        for k, v in report["api_keys_set"].items():
            print(f"  - {k}: {'set' if v else 'not set'}")
        if report["missing_binaries"]:
            print("\nInstall hints:")
            print("  macOS:   brew install ffmpeg yt-dlp")
            print("  Linux:   sudo apt install ffmpeg && pipx install yt-dlp")
            print("  Windows: winget install ffmpeg && pip install yt-dlp")
        if not report["extras"]["scenedetect"]:
            print("\nCore install:  pip install -e .  (or  pip install scenedetect[opencv])")
        if not any(report["api_keys_set"].values()) and not report["extras"]["faster_whisper"]:
            print("\nFor transcripts: set GROQ_API_KEY (cheap+fast) OR install [asr] extra for local.")

    if report["missing_binaries"]:
        return 2
    if not report["extras"]["scenedetect"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
