"""Speech-to-text with three fallbacks.

Priority:
  1. Native captions (yt-dlp .vtt) — free, instant, often accurate
  2. faster-whisper local + pyannote diarization — best quality, no API key
  3. Whisper API (Groq/OpenAI) — fast, costs a few cents

Output is normalized to a list of TranscriptSegment with speaker tags when available.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)


def extract_audio(video_path: str | Path, out_path: Path) -> Path:
    """Mono 16kHz WAV — what whisper and pyannote both want."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"audio extraction failed: {result.stderr.strip()}")
    return out_path


def parse_vtt(path: str | Path) -> list[TranscriptSegment]:
    """Naive VTT parser. yt-dlp auto-captions have heavy overlap → we dedupe."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text)
    cue_re = re.compile(r"(\d+:\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+)")
    segs: list[TranscriptSegment] = []
    seen: set[str] = set()
    for blk in blocks:
        m = cue_re.search(blk)
        if not m:
            continue
        start, end = _vtt_time(m.group(1)), _vtt_time(m.group(2))
        body = "\n".join(
            line for line in blk.splitlines() if not cue_re.search(line)
        )
        body = re.sub(r"<[^>]+>", "", body).strip()
        if not body or body in seen:
            continue
        seen.add(body)
        segs.append(TranscriptSegment(start, end, body))
    return segs


def _vtt_time(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def transcribe_local(
    audio_path: Path,
    model_size: str = "small",
    language: str | None = None,
    diarize: bool = True,
    hf_token: str | None = None,
) -> list[TranscriptSegment]:
    """faster-whisper transcription, optionally aligned to pyannote diarization."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit("faster-whisper missing. Install: pip install 'cc-video[asr]'") from e

    device = "cuda" if _has_cuda() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments_iter, _info = model.transcribe(
        str(audio_path), language=language, word_timestamps=True, vad_filter=True,
    )

    segs: list[TranscriptSegment] = []
    for s in segments_iter:
        words = [Word(w.start, w.end, w.word) for w in (s.words or [])]
        segs.append(TranscriptSegment(s.start, s.end, s.text.strip(), None, words))

    if diarize and hf_token:
        try:
            attach_speakers(segs, audio_path, hf_token)
        except Exception as e:
            print(f"[cc-video] diarization skipped: {e}")
    return segs


def attach_speakers(segs: list[TranscriptSegment], audio_path: Path, hf_token: str) -> None:
    from pyannote.audio import Pipeline
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except TypeError:
        # Older pyannote API
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diar = pipe(str(audio_path))

    spans: list[tuple[float, float, str]] = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        spans.append((turn.start, turn.end, spk))

    for seg in segs:
        mid = (seg.start_seconds + seg.end_seconds) / 2
        best = next(
            (spk for (s, e, spk) in spans if s <= mid <= e),
            None,
        )
        seg.speaker = best


def transcribe_api(
    audio_path: Path,
    backend: str = "groq",
    api_key: str | None = None,
) -> list[TranscriptSegment]:
    """Whisper API path (Groq preferred, OpenAI fallback). Word timestamps included."""
    import json
    import urllib.request

    if backend == "groq":
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        model = "whisper-large-v3"
        key = api_key or os.environ.get("GROQ_API_KEY")
        size_limit = 24 * 1024 * 1024  # Groq free tier hard-caps at 25MB
    else:
        url = "https://api.openai.com/v1/audio/transcriptions"
        model = "whisper-1"
        key = api_key or os.environ.get("OPENAI_API_KEY")
        size_limit = 24 * 1024 * 1024  # OpenAI: 25MB
    if not key:
        raise SystemExit(f"no API key for {backend}")

    upload_path = _shrink_for_upload(audio_path, size_limit)

    boundary = "----ccvideoform"
    body = _multipart(upload_path, model, boundary)
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "cc-video/0.1 (+https://github.com/anthropics/cc-video)",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())

    segs: list[TranscriptSegment] = []
    raw_segments = data.get("segments") or []
    raw_words = data.get("words") or []

    if raw_segments:
        # Build a quick lookup of words by their start time for segment assignment
        word_pool = [Word(w["start"], w["end"], w["word"]) for w in raw_words]
        for s in raw_segments:
            seg_words = s.get("words")
            if seg_words:
                words = [Word(w["start"], w["end"], w["word"]) for w in seg_words]
            else:
                words = [w for w in word_pool if s["start"] <= w.start < s["end"]]
            segs.append(TranscriptSegment(
                s["start"], s["end"], s["text"].strip(), None, words,
            ))
    elif raw_words:
        # No segments returned (Groq sometimes does this) — chunk words into ~10s segments
        chunk: list[Word] = []
        for w in raw_words:
            word_obj = Word(w["start"], w["end"], w["word"])
            if chunk and (word_obj.end - chunk[0].start) > 10.0:
                segs.append(TranscriptSegment(
                    chunk[0].start, chunk[-1].end,
                    "".join(x.text for x in chunk).strip(), None, chunk,
                ))
                chunk = []
            chunk.append(word_obj)
        if chunk:
            segs.append(TranscriptSegment(
                chunk[0].start, chunk[-1].end,
                "".join(x.text for x in chunk).strip(), None, chunk,
            ))
    elif data.get("text"):
        segs.append(TranscriptSegment(0.0, float(data.get("duration", 0.0)),
                                      data["text"].strip(), None, []))
    return segs


def _shrink_for_upload(audio_path: Path, size_limit: int) -> Path:
    """If audio exceeds API size limit, transcode to compact m4a (64kbps mono 16kHz)."""
    if audio_path.stat().st_size <= size_limit:
        return audio_path
    out = audio_path.with_name(audio_path.stem + ".upload.m4a")
    if not out.exists() or out.stat().st_size > size_limit:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio_path),
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000",
            str(out),
        ]
        subprocess.run(cmd, check=True)
    return out


_MIME_BY_EXT = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac"}


def _multipart(audio_path: Path, model: str, boundary: str) -> bytes:
    mime = _MIME_BY_EXT.get(audio_path.suffix.lower(), "application/octet-stream")
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nsegment\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nword\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio_path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode())
    parts.append(audio_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
