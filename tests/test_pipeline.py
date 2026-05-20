"""Smoke tests — module imports + dataclass round-trips. No video processing
since CI typically lacks ffmpeg.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_imports():
    from cc_video import cli, config, download, keyframes, merger, ocr  # noqa: F401
    from cc_video import output, pipeline, probe, shots, transcript, vlm  # noqa: F401


def test_config_from_env(monkeypatch):
    from cc_video.config import Config
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY", "k2")
    cfg = Config.from_env()
    assert cfg.gemini_key == "k1"
    assert cfg.groq_key == "k2"


def test_probe_format_time():
    from cc_video.probe import format_time, parse_time
    assert format_time(0) == "00:00:00.000"
    assert format_time(65.5) == "00:01:05.500"
    assert parse_time("1:23") == 83.0
    assert parse_time("01:02:03") == 3723.0


def test_shots_fallback(tmp_path, monkeypatch):
    """Single-shot fallback when scenedetect can't open file."""
    from cc_video.shots import _single_shot_fallback
    # _single_shot_fallback calls ffprobe — only sensible to test the shape
    from cc_video.shots import Shot
    s = Shot(0, 0.0, 10.0, 0, 250)
    assert s.duration == 10.0
    assert s.middle_seconds == 5.0


def test_merger_chapters_single_shot():
    from cc_video.merger import build_shot_records, group_into_chapters
    from cc_video.shots import Shot
    shots = [Shot(0, 0.0, 5.0, 0, 125)]
    records = build_shot_records(shots, [], [], {}, [])
    chapters = group_into_chapters(records)
    assert len(chapters) == 1
    assert chapters[0].start_seconds == 0.0


def test_output_renders(tmp_path):
    from cc_video.output import write_outputs
    from cc_video.probe import VideoMeta
    from cc_video.shots import Shot
    from cc_video.merger import build_shot_records, group_into_chapters

    shots = [Shot(0, 0.0, 5.0, 0, 125)]
    records = build_shot_records(shots, [], [], {}, [])
    chapters = group_into_chapters(records)
    meta = VideoMeta(
        duration_seconds=5.0, width=1920, height=1080, codec="h264",
        fps=25.0, has_audio=False, size_bytes=0,
    )
    out = write_outputs(tmp_path, "fake.mp4", meta, records, chapters, {})
    assert out["markdown"].exists()
    assert out["json"].exists()
    md = out["markdown"].read_text(encoding="utf-8")
    assert "Video Understanding" in md
    assert "Shot 0" in md
    js = json.loads(out["json"].read_text(encoding="utf-8"))
    assert js["meta"]["duration_seconds"] == 5.0
    assert len(js["shots"]) == 1
