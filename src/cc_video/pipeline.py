"""Top-level orchestrator — ties probe → shots → keyframes → ASR → OCR → VLM → output."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .config import Config
from .download import download
from .keyframes import extract_keyframes
from . import log_util as logu
from .merger import build_shot_records, group_into_chapters
from .ocr import ocr_keyframes
from .output import write_outputs
from .probe import get_metadata
from .shots import detect_shots, merge_micro_shots
from .transcript import (
    TranscriptSegment, attach_speakers, extract_audio, parse_vtt,
    transcribe_api, transcribe_local,
)
from .vlm import caption_shots


def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


_VI_DIACRITICS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def _guess_ocr_lang(segments: list[TranscriptSegment]) -> str:
    """Pick PaddleOCR lang code from a few transcript samples. Defaults to 'en'."""
    sample = " ".join(s.text for s in segments[:30])
    if any(c in _VI_DIACRITICS for c in sample):
        return "vi"
    return "en"


def run(
    source: str,
    work_dir: Path,
    config: Config | None = None,
    no_vlm: bool = False,
    no_ocr: bool = False,
    no_asr: bool = False,
    deep_keyframes: bool = False,
) -> dict[str, Path]:
    """Run the full pipeline. Returns {markdown, json} output paths."""
    cfg = config or Config.from_env()
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Download / locate
    logu.stage(f"resolving source: {source}")
    dl = download(source, work_dir / "download",
                  fetch_subs=not cfg.prefer_whisper_over_captions)
    video_path = dl["video_path"]
    info = dl["info"]

    # 2. Probe
    meta = get_metadata(video_path)
    logu.step(f"video: {meta['duration_seconds']:.1f}s @ {meta.get('fps') or '?'} fps, "
              f"{meta['width']}x{meta['height']}, audio={'yes' if meta['has_audio'] else 'no'}")

    # 3. Shot detection
    logu.stage("detecting shots")
    shots = merge_micro_shots(
        detect_shots(video_path, adaptive_threshold=cfg.adaptive_threshold,
                     min_scene_len_seconds=cfg.min_shot_seconds),
        min_duration=cfg.min_shot_seconds,
    )
    logu.step(f"shots: {len(shots)}")

    # 4. Keyframes
    logu.stage(f"extracting keyframes (deep={deep_keyframes}, {cfg.keyframe_resolution}p)")
    keyframes = extract_keyframes(
        video_path, shots, work_dir / "keyframes",
        resolution=cfg.keyframe_resolution, deep=deep_keyframes,
    )
    logu.step(f"keyframes: {len(keyframes)}")

    # 5. Transcript + diarization (auto-detect best path)
    segments: list[TranscriptSegment] = []
    transcript_source = "none"
    if not no_asr and meta["has_audio"]:
        logu.stage("transcribing audio")
        # Native captions only if explicitly preferred (off by default — Whisper is better)
        if not cfg.prefer_whisper_over_captions and dl.get("subtitle_path"):
            try:
                segments = parse_vtt(dl["subtitle_path"])
                transcript_source = "captions"
            except Exception as e:
                logu.warn(f"caption parse failed: {e}")

        audio: Path | None = None
        if not segments:
            logu.step("extracting audio (mono 16kHz wav)")
            audio = extract_audio(video_path, work_dir / "audio.wav")
            try:
                if cfg.groq_key:
                    logu.step("Groq Whisper large-v3 API call")
                    segments = transcribe_api(audio, backend="groq")
                    transcript_source = "whisper-api-groq"
                elif _have_faster_whisper():
                    logu.step(f"faster-whisper local ({cfg.whisper_model})")
                    segments = transcribe_local(
                        audio, model_size=cfg.whisper_model,
                        diarize=False, hf_token=None,  # diarization done separately below
                    )
                    transcript_source = "faster-whisper-local"
                elif cfg.openai_key:
                    logu.step("OpenAI Whisper API call")
                    segments = transcribe_api(audio, backend="openai")
                    transcript_source = "whisper-api-openai"
            except SystemExit as e:
                logu.warn(f"ASR skipped: {e}")

        # Attach speaker labels via pyannote (independent of transcript backend)
        if segments and audio and cfg.enable_diarization and cfg.hf_token:
            logu.stage("diarizing speakers (pyannote)")
            try:
                attach_speakers(segments, audio, cfg.hf_token)
                transcript_source += "+diarization"
                logu.step("speakers attached")
            except Exception as e:
                logu.warn(f"diarization skipped: {e}")
    logu.step(f"transcript: {len(segments)} segments via {transcript_source}")

    # 6. OCR
    ocr_by_path = {}
    if not no_ocr and keyframes:
        ocr_lang = _guess_ocr_lang(segments) if segments else "en"
        logu.stage(f"OCR on keyframes (lang={ocr_lang})")
        try:
            ocr_by_path = ocr_keyframes([k.path for k in keyframes], lang=ocr_lang)
        except Exception as e:
            logu.warn(f"OCR failed, skipping: {e}")
            ocr_by_path = {}
        if ocr_by_path and any(r.lines for r in ocr_by_path.values()):
            total = sum(len(r.lines) for r in ocr_by_path.values())
            logu.step(f"OCR: {total} lines across {len(ocr_by_path)} frames")
        else:
            logu.step("OCR: no text extracted")

    # 7. VLM shot captions
    captions = []
    if not no_vlm and cfg.has_vlm() and keyframes:
        logu.stage(f"VLM captioning per shot (model={cfg.vlm_model})")
        # Need shot-grouped inputs
        from .merger import group_keyframes_by_shot, group_ocr_by_shot, slice_transcript_per_shot
        kfs_by_shot = group_keyframes_by_shot(keyframes)
        trans_by_shot = slice_transcript_per_shot(shots, segments)
        ocr_by_shot = group_ocr_by_shot(shots, keyframes, ocr_by_path)
        captions = caption_shots(
            shots, kfs_by_shot, trans_by_shot, ocr_by_shot,
            model=cfg.vlm_model, api_key=cfg.gemini_key,
        )
        logu.step(f"VLM: {len(captions)} captions")

    # 8. Merge + write
    logu.stage("merging and writing output")
    records = build_shot_records(shots, keyframes, segments, ocr_by_path, captions)
    chapters = group_into_chapters(records)
    logu.step(f"chapters: {len(chapters)}")

    out = write_outputs(work_dir, source, meta, records, chapters, info)
    elapsed = time.time() - t0
    logu.stage(f"done in {elapsed:.1f}s → {out['markdown']}")
    return out
