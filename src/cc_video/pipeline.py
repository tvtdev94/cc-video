"""Top-level orchestrator — probe → shots → keyframes → ASR → OCR → VLM → output.

Every stage checkpoints to {work_dir}/state/. Re-running on the same work_dir
loads completed stages from disk and only recomputes what's missing — crash
recovery and fast iteration come for free.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .config import Config
from .download import download
from .keyframes import Keyframe, extract_keyframes
from . import log_util as logu
from .merger import build_shot_records, group_into_chapters
from .ocr import OCRLine, OCRResult, ocr_keyframes
from .output import write_outputs
from .probe import get_metadata
from .shots import Shot, detect_shots, merge_micro_shots
from .state import Store
from .transcript import (
    TranscriptSegment, Word, attach_speakers, extract_audio, parse_vtt,
    transcribe_api, transcribe_local,
)
from .vlm import ShotCaption, caption_shots


def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


_VI_DIACRITICS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def _guess_ocr_lang(segments: list[TranscriptSegment]) -> str:
    sample = " ".join(s.text for s in segments[:30])
    if any(c in _VI_DIACRITICS for c in sample):
        return "vi"
    return "en"


def _smart_ocr_paths(keyframes: list[Keyframe], shots: list[Shot]) -> list[str]:
    """Pick which keyframes to OCR: 1 per short shot, 2 (first+last) per long shot.

    Within a single shot, on-screen text is nearly always identical across
    keyframes — running OCR on every cluster is wasted compute. For shots
    longer than ~8s we sample two endpoints in case text content shifts.
    """
    by_shot: dict[int, list[Keyframe]] = {}
    for kf in keyframes:
        by_shot.setdefault(kf.shot_index, []).append(kf)
    shot_dur = {s.index: s.duration for s in shots}

    paths: list[str] = []
    for shot_idx, kfs in by_shot.items():
        kfs_sorted = sorted(kfs, key=lambda k: k.path)
        dur = shot_dur.get(shot_idx, 0.0)
        if dur > 8.0 and len(kfs_sorted) >= 2:
            paths.append(kfs_sorted[0].path)
            paths.append(kfs_sorted[-1].path)
        else:
            paths.append(kfs_sorted[len(kfs_sorted) // 2].path)
    return paths


# --- factories for loading state files back into dataclasses ----------------

def _shot_from_dict(d: dict) -> Shot:
    return Shot(**d)


def _keyframe_from_dict(d: dict) -> Keyframe:
    return Keyframe(**d)


def _segment_from_dict(d: dict) -> TranscriptSegment:
    words = [Word(**w) for w in d.get("words", [])]
    return TranscriptSegment(
        start_seconds=d["start_seconds"],
        end_seconds=d["end_seconds"],
        text=d["text"],
        speaker=d.get("speaker"),
        words=words,
    )


def _ocr_from_dict(d: dict) -> OCRResult:
    lines = [
        OCRLine(text=l["text"], confidence=l["confidence"], bbox=tuple(l["bbox"]))
        for l in d.get("lines", [])
    ]
    return OCRResult(keyframe_path=d["keyframe_path"], lines=lines)


def _caption_from_dict(d: dict) -> ShotCaption:
    return ShotCaption(
        shot_index=d["shot_index"],
        action=d.get("action", ""),
        setting=d.get("setting", ""),
        on_screen_text_summary=d.get("on_screen_text_summary", ""),
        notable_objects=d.get("notable_objects", []),
        change_from_prev=d.get("change_from_prev", ""),
        raw=d.get("raw", {}),
    )


# --- pipeline ---------------------------------------------------------------


def run(
    source: str,
    work_dir: Path,
    config: Config | None = None,
    no_vlm: bool = False,
    no_ocr: bool = False,
    no_asr: bool = False,
    deep_keyframes: bool = False,
) -> dict[str, Path]:
    """Run the full pipeline with stage-level checkpointing."""
    cfg = config or Config.from_env()
    work_dir.mkdir(parents=True, exist_ok=True)
    store = Store(work_dir)
    t0 = time.time()

    # 1. Download / locate (yt-dlp is idempotent if video exists)
    dl = _download_or_reuse(source, work_dir, store, cfg)
    video_path = dl["video_path"]
    info = dl["info"]

    # 2. Probe — always cheap
    meta = get_metadata(video_path)
    logu.step(f"video: {meta['duration_seconds']:.1f}s @ {meta.get('fps') or '?'} fps, "
              f"{meta['width']}x{meta['height']}, audio={'yes' if meta['has_audio'] else 'no'}")

    # 3. Shot detection
    if store.has("shots.json"):
        shots = store.load_json("shots.json", _shot_from_dict)
        logu.stage(f"shots: {len(shots)} (cached)")
    else:
        logu.stage("detecting shots")
        shots = merge_micro_shots(
            detect_shots(video_path, adaptive_threshold=cfg.adaptive_threshold,
                         min_scene_len_seconds=cfg.min_shot_seconds),
            min_duration=cfg.min_shot_seconds,
        )
        store.save_json("shots.json", shots)
        logu.step(f"shots: {len(shots)}")

    # 4. Keyframes
    if store.has("keyframes.json") and (work_dir / "keyframes").exists():
        keyframes = store.load_json("keyframes.json", _keyframe_from_dict)
        keyframes = [k for k in keyframes if Path(k.path).exists()]
        logu.stage(f"keyframes: {len(keyframes)} (cached)")
    else:
        logu.stage(f"extracting keyframes (deep={deep_keyframes}, {cfg.keyframe_resolution}p)")
        keyframes = extract_keyframes(
            video_path, shots, work_dir / "keyframes",
            resolution=cfg.keyframe_resolution, deep=deep_keyframes,
        )
        store.save_json("keyframes.json", keyframes)
        logu.step(f"keyframes: {len(keyframes)}")

    # 5. Transcript + diarization
    segments, transcript_source = _transcribe_or_reuse(
        video_path, work_dir, store, dl, meta, cfg, no_asr,
    )

    # 6. OCR — incremental, smart sampling
    ocr_by_path = _ocr_or_reuse(keyframes, shots, segments, store, cfg, no_ocr)

    # 7. VLM — incremental, parallel
    captions = _vlm_or_reuse(shots, keyframes, segments, ocr_by_path, store, cfg, no_vlm)

    # 8. Merge + write final output
    logu.stage("merging and writing output")
    records = build_shot_records(shots, keyframes, segments, ocr_by_path, captions)
    chapters = group_into_chapters(records)
    logu.step(f"chapters: {len(chapters)}")

    out = write_outputs(work_dir, source, meta, records, chapters, info)
    elapsed = time.time() - t0
    logu.stage(f"done in {elapsed:.1f}s → {out['markdown']}")
    return out


def _download_or_reuse(source: str, work_dir: Path, store: Store, cfg: Config) -> dict:
    dl_dir = work_dir / "download"
    existing = list(dl_dir.glob("*.mp4")) + list(dl_dir.glob("*.webm")) + list(dl_dir.glob("*.mkv"))
    if existing:
        info_json = next(dl_dir.glob("*.info.json"), None)
        info: dict = {}
        if info_json:
            try:
                raw = json.loads(info_json.read_text(encoding="utf-8"))
                info = {
                    "title": raw.get("title"),
                    "uploader": raw.get("uploader"),
                    "webpage_url": raw.get("webpage_url"),
                    "duration": raw.get("duration"),
                }
            except Exception:
                pass
        logu.stage(f"source: {source} (cached)")
        sub = next(dl_dir.glob("*.vtt"), None)
        return {"video_path": str(existing[0]), "subtitle_path": str(sub) if sub else None, "info": info}

    logu.stage(f"resolving source: {source}")
    return download(source, dl_dir, fetch_subs=not cfg.prefer_whisper_over_captions)


def _transcribe_or_reuse(
    video_path: str, work_dir: Path, store: Store, dl: dict, meta: dict,
    cfg: Config, no_asr: bool,
) -> tuple[list[TranscriptSegment], str]:
    if store.has("transcript.json"):
        segments = store.load_json("transcript.json", _segment_from_dict)
        logu.stage(f"transcript: {len(segments)} segments (cached)")
        return segments, "cached"

    segments: list[TranscriptSegment] = []
    transcript_source = "none"
    if no_asr or not meta["has_audio"]:
        return segments, transcript_source

    logu.stage("transcribing audio")
    if not cfg.prefer_whisper_over_captions and dl.get("subtitle_path"):
        try:
            segments = parse_vtt(dl["subtitle_path"])
            transcript_source = "captions"
        except Exception as e:
            logu.warn(f"caption parse failed: {e}")

    audio_path: Path | None = None
    if not segments:
        audio_path = work_dir / "audio.wav"
        if not audio_path.exists():
            logu.step("extracting audio (mono 16kHz wav)")
            audio_path = extract_audio(video_path, audio_path)
        try:
            if cfg.groq_key:
                logu.step("Groq Whisper large-v3 API call")
                segments = transcribe_api(audio_path, backend="groq")
                transcript_source = "whisper-api-groq"
            elif _have_faster_whisper():
                logu.step(f"faster-whisper local ({cfg.whisper_model})")
                segments = transcribe_local(
                    audio_path, model_size=cfg.whisper_model,
                    diarize=False, hf_token=None,
                )
                transcript_source = "faster-whisper-local"
            elif cfg.openai_key:
                logu.step("OpenAI Whisper API call")
                segments = transcribe_api(audio_path, backend="openai")
                transcript_source = "whisper-api-openai"
        except SystemExit as e:
            logu.warn(f"ASR skipped: {e}")

    if segments and audio_path and cfg.enable_diarization and cfg.hf_token:
        logu.stage("diarizing speakers (pyannote)")
        try:
            attach_speakers(segments, audio_path, cfg.hf_token)
            transcript_source += "+diarization"
            logu.step("speakers attached")
        except Exception as e:
            logu.warn(f"diarization skipped: {e}")

    if segments:
        store.save_json("transcript.json", segments)
    logu.step(f"transcript: {len(segments)} segments via {transcript_source}")
    return segments, transcript_source


def _ocr_or_reuse(
    keyframes: list[Keyframe], shots: list[Shot], segments: list[TranscriptSegment],
    store: Store, cfg: Config, no_ocr: bool,
) -> dict[str, OCRResult]:
    if no_ocr or not keyframes:
        return {}

    # Load any per-frame results already on disk
    prev = {r.keyframe_path: r for r in store.jsonl_load_all("ocr.jsonl", _ocr_from_dict)}
    paths_to_ocr = _smart_ocr_paths(keyframes, shots)
    pending = [p for p in paths_to_ocr if p not in prev]

    if not pending:
        if prev:
            logu.stage(f"OCR: {len(prev)} frames (cached)")
        return prev

    ocr_lang = _guess_ocr_lang(segments) if segments else "en"
    logu.stage(f"OCR on {len(pending)} keyframes (lang={ocr_lang}, "
               f"{len(prev)} cached, {len(keyframes) - len(paths_to_ocr)} skipped via shot dedup)")

    try:
        new_results = ocr_keyframes(
            pending,
            lang=ocr_lang,
            on_result=lambda r: store.jsonl_append("ocr.jsonl", r),
        )
    except Exception as e:
        logu.warn(f"OCR failed, skipping: {e}")
        return prev

    merged = {**prev, **new_results}
    if merged and any(r.lines for r in merged.values()):
        total = sum(len(r.lines) for r in merged.values())
        logu.step(f"OCR: {total} lines across {len(merged)} frames")
    else:
        logu.step("OCR: no text extracted")
    return merged


def _vlm_or_reuse(
    shots: list[Shot], keyframes: list[Keyframe], segments: list[TranscriptSegment],
    ocr_by_path: dict[str, OCRResult], store: Store, cfg: Config, no_vlm: bool,
) -> list[ShotCaption]:
    if no_vlm or not cfg.has_vlm() or not keyframes:
        return []

    from .merger import group_keyframes_by_shot, group_ocr_by_shot, slice_transcript_per_shot

    prev = store.jsonl_load_all("vlm.jsonl", _caption_from_dict)
    done_indices = {c.shot_index for c in prev}
    pending = [s for s in shots if s.index not in done_indices]

    if not pending:
        if prev:
            logu.stage(f"VLM: {len(prev)} captions (cached)")
        return prev

    logu.stage(f"VLM captioning {len(pending)} shots (model={cfg.vlm_model}, "
               f"{len(prev)} cached)")
    kfs_by_shot = group_keyframes_by_shot(keyframes)
    trans_by_shot = slice_transcript_per_shot(shots, segments)
    ocr_by_shot = group_ocr_by_shot(shots, keyframes, ocr_by_path)

    new_caps = caption_shots(
        pending, kfs_by_shot, trans_by_shot, ocr_by_shot,
        model=cfg.vlm_model, api_key=cfg.gemini_key,
        on_caption=lambda c: store.jsonl_append("vlm.jsonl", c),
    )
    merged = prev + new_caps
    merged.sort(key=lambda c: c.shot_index)
    logu.step(f"VLM: {len(merged)} captions total ({len(new_caps)} new)")
    return merged
