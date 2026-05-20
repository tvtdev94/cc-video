"""Resume VLM stage on an existing cc-video output dir.

Re-runs shot detection + reuses cached keyframes/audio/transcript + redoes VLM
captions + rewrites output. Saves ~6 min of download/keyframe/transcribe time
when the previous run's VLM stage hit quota errors.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from cc_video.config import Config
from cc_video.keyframes import Keyframe
from cc_video.merger import (
    build_shot_records, group_into_chapters,
    group_keyframes_by_shot, group_ocr_by_shot, slice_transcript_per_shot,
)
from cc_video.ocr import ocr_keyframes
from cc_video.output import write_outputs
from cc_video.probe import get_metadata
from cc_video.shots import detect_shots, merge_micro_shots
from cc_video.transcript import TranscriptSegment, transcribe_api
from cc_video.vlm import caption_shots


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("work_dir", help="Existing cc-video output directory")
    args = p.parse_args()

    work = Path(args.work_dir).expanduser().resolve()
    video_files = list((work / "download").glob("*.mp4")) + list((work / "download").glob("*.webm"))
    if not video_files:
        print(f"no video in {work / 'download'}", file=sys.stderr)
        return 1
    video = video_files[0]
    print(f"[resume] video: {video}", file=sys.stderr)

    meta = get_metadata(str(video))
    print(f"[resume] re-detecting shots…", file=sys.stderr)
    shots = merge_micro_shots(detect_shots(str(video)), min_duration=0.6)
    print(f"[resume] shots: {len(shots)}", file=sys.stderr)

    # Reuse keyframes from disk
    kf_paths = sorted((work / "keyframes").glob("shot*.jpg"))
    keyframes: list[Keyframe] = []
    for p in kf_paths:
        m = re.match(r"shot(\d+)_", p.name)
        if m:
            keyframes.append(Keyframe(
                shot_index=int(m.group(1)), timestamp_seconds=0.0,
                path=str(p), role="cluster",
            ))
    print(f"[resume] keyframes: {len(keyframes)}", file=sys.stderr)

    cfg = Config.from_env()

    # Reuse transcript from previous JSON if exists, else re-transcribe from audio
    segments: list[TranscriptSegment] = []
    prev_json = work / "video.understanding.json"
    source = ""
    info: dict = {}
    if prev_json.exists():
        prev = json.loads(prev_json.read_text(encoding="utf-8"))
        seen_segs: set[tuple[float, float, str]] = set()
        for shot in prev["shots"]:
            for s in shot.get("transcript", []):
                start = s.get("start", s.get("start_seconds"))
                end = s.get("end", s.get("end_seconds"))
                key = (start, end, s["text"][:50])
                if key in seen_segs:
                    continue
                seen_segs.add(key)
                segments.append(TranscriptSegment(
                    start, end, s["text"], s.get("speaker"), [],
                ))
        source = prev.get("source", "")
        info = {"title": prev.get("title"), "uploader": prev.get("uploader")}
        print(f"[resume] reused transcript from JSON: {len(segments)} segments", file=sys.stderr)
    else:
        audio = work / "audio.wav"
        if not audio.exists():
            print(f"no transcript JSON and no audio.wav at {audio}", file=sys.stderr)
            return 1
        if not cfg.groq_key:
            print("no GROQ_API_KEY in env — cannot re-transcribe", file=sys.stderr)
            return 1
        print(f"[resume] re-transcribing via Groq Whisper large-v3…", file=sys.stderr)
        segments = transcribe_api(audio, backend="groq")
        print(f"[resume] transcript: {len(segments)} segments", file=sys.stderr)
        # Load info.json if available for source/title/uploader
        info_jsons = list((work / "download").glob("*.info.json"))
        if info_jsons:
            try:
                raw = json.loads(info_jsons[0].read_text(encoding="utf-8"))
                source = raw.get("webpage_url", "")
                info = {"title": raw.get("title"), "uploader": raw.get("uploader")}
            except Exception:
                pass

    kfs_by_shot = group_keyframes_by_shot(keyframes)
    trans_by_shot = slice_transcript_per_shot(shots, segments)

    # OCR on keyframes (auto-detect lang from transcript)
    from cc_video.pipeline import _guess_ocr_lang
    ocr_lang = _guess_ocr_lang(segments) if segments else "en"
    print(f"[resume] OCR on keyframes (lang={ocr_lang})…", file=sys.stderr)
    try:
        ocr_by_path = ocr_keyframes([k.path for k in keyframes], lang=ocr_lang)
    except Exception as e:
        print(f"[resume] OCR failed, skipping: {e}", file=sys.stderr)
        ocr_by_path = {}
    ocr_by_shot = group_ocr_by_shot(shots, keyframes, ocr_by_path)
    ocr_lines = sum(1 for v in ocr_by_shot.values() if v)
    print(f"[resume] OCR: text from {ocr_lines}/{len(shots)} shots", file=sys.stderr)

    print(f"[resume] running VLM (Gemini {cfg.vlm_model}) on {len(shots)} shots…", file=sys.stderr)
    captions = caption_shots(
        shots, kfs_by_shot, trans_by_shot, ocr_by_shot,
        model=cfg.vlm_model, api_key=cfg.gemini_key,
    )
    print(f"[resume] captions returned: {len(captions)}", file=sys.stderr)

    records = build_shot_records(shots, keyframes, segments, ocr_by_path, captions)
    chapters = group_into_chapters(records)
    print(f"[resume] chapters: {len(chapters)}", file=sys.stderr)

    out = write_outputs(work, source, meta, records, chapters, info)
    print(f"[resume] wrote → {out['markdown']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
