---
name: cc-video
description: Deep video understanding for Claude Code. Detects shots (not uniform fps), picks representative keyframes per shot, transcribes with speaker tags, OCRs on-screen text, and pre-captions each shot with a VLM — producing a structured Markdown timeline Claude can read instead of being flooded with raw frames.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read
homepage: https://github.com/anthropics/cc-video
repository: https://github.com/anthropics/cc-video
license: MIT
user-invocable: true
---

# /cc-video — Claude sees a video deeply

Unlike a uniform-fps frame dump, this skill produces a structured timeline:
shot-by-shot summaries + word-level transcript + speaker tags + OCR of on-screen
text. Claude reads compact text instead of dozens of raw images.

## When to use

- User pastes a video URL/path and asks anything that depends on *what's actually
  shown*: code on screen, tools used, UI state, who said what.
- User has a screen recording of a bug — cc-video extracts terminal/UI text via
  OCR per shot, plus the verbal explanation, so Claude can diagnose without
  watching pixel-by-pixel.
- The video is longer than `/watch` handles well (>10 min) — shot-based coverage
  scales without the "sparse scan" warning.

## Invocation

```bash
cc-video analyze "<source>" --out-dir /tmp/myvideo
```

Then `Read /tmp/myvideo/video.understanding.md` for the full timeline.

Optional flags:

| Flag | Purpose |
|---|---|
| `--out-dir DIR` | Where to write the analysis (default: tmp) |
| `--deep-keyframes` | CLIP-cluster keyframes inside each shot (more diversity, needs `[clip]`) |
| `--no-vlm` | Skip Gemini per-shot caption (use when no `GEMINI_API_KEY`) |
| `--no-ocr` | Skip PaddleOCR |
| `--no-asr` | Skip transcription entirely |
| `--whisper-model SIZE` | tiny/base/small/medium/large-v3 (local) |
| `--vlm-model NAME` | Gemini model (default `gemini-2.5-flash`) |

## How to use the output in Claude

1. **Always read the `.md` first** — it has shot timeline + chapters + transcript.
2. **Only read raw keyframe JPEGs when** the user asks about a detail not in the
   shot caption (e.g., "what color is X at 2:30?").
3. **For follow-up questions**, run `cc-video query <work-dir> "<question>"` to
   grep the JSON locally without re-doing the pipeline.

## Comparison with `/watch` (bradautomates/claude-video)

| | `/watch` | `cc-video` |
|---|---|---|
| Sampling | Uniform fps, capped at 100 frames | Shot-aware: 1-5 keyframes per shot, no count cap |
| Transcript | Plain segments | Word-level + speaker diarization (WhisperX) |
| OCR | None | PaddleOCR per keyframe |
| Visual description | Raw frames dumped to Claude | Pre-captioned per shot by Gemini |
| Long video | "Sparse scan" warning >10min | Full coverage regardless of length |
| Cross-modal | None | Speaker timeline ↔ shot timeline aligned |
| Token cost | Dominated by image tokens | Mostly text — far cheaper |

## Failure modes

- **ffmpeg/yt-dlp missing** → run `python3 scripts/setup.py` for install commands.
- **No Gemini key** → captions skipped, but shots + transcript + OCR still
  produced. Pass `--no-vlm` to suppress the warning.
- **No ASR available** → no API key AND `faster-whisper` not installed → timeline
  has shots + visual captions only. Install with `pip install 'cc-video[asr]'`.
- **PySceneDetect missing** → falls back to a single shot (whole video). Install
  with `pip install scenedetect[opencv]`.

## Security

- All processing is local (ffmpeg/PySceneDetect/PaddleOCR/faster-whisper).
- Network calls only when: (a) downloading via yt-dlp, (b) Gemini VLM, (c) Whisper
  API fallback. All optional.
- API keys read from env vars (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`,
  `HF_TOKEN`). Never written to output.
