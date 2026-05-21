"""CLI entry — `cc-video analyze <source> [opts]`."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Config
from .pipeline import run


def _source_from_work_dir(work: Path) -> str | None:
    """Best-effort: recover source URL/path from a previous run's artifacts."""
    info_jsons = list((work / "download").glob("*.info.json"))
    for ij in info_jsons:
        try:
            raw = json.loads(ij.read_text(encoding="utf-8"))
            if raw.get("webpage_url"):
                return raw["webpage_url"]
        except Exception:
            continue
    for video in (work / "download").glob("*.mp4"):
        return str(video)
    return None


def _make_slug(source: str) -> str:
    """Derive a folder-safe slug from URL (prefer video ID) or local path stem."""
    if source.startswith(("http://", "https://", "ftp://")):
        parsed = urlparse(source)
        host = parsed.netloc.lower()
        if "youtube.com" in host or "youtu.be" in host:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
            m = re.search(r"/(?:shorts/|embed/)?([A-Za-z0-9_-]{6,})/?$", parsed.path)
            if m:
                return m.group(1)
        last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if last:
            return re.sub(r"[^A-Za-z0-9_-]+", "-", last)[:40]
        return hashlib.sha1(source.encode()).hexdigest()[:10]
    return re.sub(r"[^A-Za-z0-9_-]+", "-", Path(source).stem)[:60] or "video"


def main(argv: list[str] | None = None) -> int:
    # If first positional is "query", route there; otherwise treat as analyze.
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] == "query":
        return _cmd_query(_parse_query(raw[1:]))
    # Strip optional "analyze" subcommand prefix for ergonomics.
    if raw and raw[0] == "analyze":
        raw = raw[1:]
    return _cmd_analyze(_parse_analyze(raw))


def _parse_analyze(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cc-video analyze",
        description="Analyze a video — shot-aware, cross-modal.",
    )
    p.add_argument("source", nargs="?",
                   help="Video URL or local file path (omit when --resume)")
    p.add_argument("--out-dir", type=str, default="./out", help="Output dir (default: ./out)")
    p.add_argument("--resume", metavar="WORK_DIR",
                   help="Resume an existing work dir instead of creating a new one")
    p.add_argument("--no-deep-keyframes", action="store_true",
                   help="Disable CLIP-cluster keyframe selection (on by default)")
    p.add_argument("--no-vlm", action="store_true", help="Skip Gemini per-shot captioning")
    p.add_argument("--no-ocr", action="store_true", help="Skip PaddleOCR text extraction")
    p.add_argument("--no-asr", action="store_true", help="Skip transcription entirely")
    p.add_argument("--whisper-model", default=None,
                   help="faster-whisper model size: tiny|base|small|medium|large-v3")
    p.add_argument("--vlm-model", default=None, help="Gemini model name (default: gemini-2.5-flash)")
    p.add_argument("--resolution", type=int, default=None, help="Keyframe width px (default 720)")
    return p.parse_args(argv)


def _parse_query(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="cc-video query")
    p.add_argument("work_dir", help="Directory produced by `analyze`")
    p.add_argument("question", nargs="+", help="Question to answer")
    return p.parse_args(argv)


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.vlm_model:
        cfg.vlm_model = args.vlm_model
    if args.resolution:
        cfg.keyframe_resolution = args.resolution

    if args.resume:
        work = Path(args.resume).expanduser().resolve()
        if not work.exists():
            print(f"error: --resume dir does not exist: {work}", file=sys.stderr)
            return 2
        source = args.source or _source_from_work_dir(work)
        if not source:
            print(f"error: cannot determine source from {work}. "
                  f"Pass a URL/path explicitly.", file=sys.stderr)
            return 2
    else:
        if not args.source:
            print("error: missing video source. Use `cc-video analyze <source>` "
                  "or `cc-video analyze --resume <dir>`.", file=sys.stderr)
            return 2
        source = args.source
        parent = Path(args.out_dir).expanduser().resolve()
        slug = _make_slug(source)
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        work = parent / f"{slug}-{timestamp}"
        work.mkdir(parents=True, exist_ok=True)

    print(f"[cc-video] work_dir: {work}", file=sys.stderr)

    out = run(
        source=source,
        work_dir=work,
        config=cfg,
        no_vlm=args.no_vlm,
        no_ocr=args.no_ocr,
        no_asr=args.no_asr,
        deep_keyframes=not args.no_deep_keyframes,
    )

    print(f"\n# cc-video output\n")
    print(f"- Markdown timeline: `{out['markdown']}`")
    print(f"- JSON structure:    `{out['json']}`")
    print(f"\n**For Claude Code:** Read `{out['markdown']}` to get the full timeline. "
          f"Read individual keyframe JPEGs from `{work / 'keyframes'}` only if you need "
          f"finer detail than the per-shot captions provide.")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    import json
    work = Path(args.work_dir).expanduser().resolve()
    js = work / "video.understanding.json"
    if not js.exists():
        print(f"error: no analysis found at {js}", file=sys.stderr)
        return 2
    data = json.loads(js.read_text(encoding="utf-8"))
    q = " ".join(args.question).lower()
    matches: list[str] = []
    for shot in data["shots"]:
        hay = " ".join([
            shot.get("ocr_text", "") or "",
            (shot.get("caption") or {}).get("action", "") or "",
            (shot.get("caption") or {}).get("setting", "") or "",
            (shot.get("caption") or {}).get("on_screen_text_summary", "") or "",
            " ".join(s.get("text", "") for s in shot.get("transcript", [])),
        ]).lower()
        if all(tok in hay for tok in q.split() if len(tok) > 2):
            matches.append(
                f"- Shot {shot['index']} [{shot['start_seconds']:.1f}s→{shot['end_seconds']:.1f}s]: "
                f"{(shot.get('caption') or {}).get('action', '')[:120]}"
            )
    if not matches:
        print(f"No shots matched: {q}")
        return 1
    print(f"# Matches for: {q}\n")
    print("\n".join(matches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
