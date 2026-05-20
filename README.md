# cc-video

**Deep video understanding for Claude Code.** Shot-aware, cross-modal — not
uniform frame sampling.

```
video → shots → per-shot keyframes → ASR (+ speakers) → OCR → per-shot VLM caption
      → cross-modal merger → video.understanding.md  (Claude reads this)
```

## What this gives Claude that uniform sampling can't

The conventional approach (e.g. [bradautomates/claude-video `/watch`](https://github.com/bradautomates/claude-video))
samples frames at a fixed fps, caps at 100 frames, and hands raw JPEGs +
transcript to Claude. That works for short videos, but:

- A 30-second shot of a static IDE wastes 30+ frames; a 1-second cut between
  shots gets the same coverage as the boring static one.
- The transcript has no speakers, no word timestamps.
- On-screen text (code, terminal, slides) is buried in the image — Claude has to
  re-OCR every frame visually, burning tokens.
- Visual descriptions are computed at query time, redundantly for follow-ups.
- Videos >10 min get a "sparse scan" warning because the 100-frame cap is brutal.

`cc-video` fixes each of these:

| | `/watch` | `cc-video` |
|---|---|---|
| Sampling | Uniform fps, cap 100 | Shot-aware, 1-5 keyframes per shot, no cap |
| Transcript | Plain segments | Word-level + speaker diarization (WhisperX) |
| OCR | — | PaddleOCR per keyframe → per-shot deduped block |
| Visual description | Raw frames dumped to Claude context | Pre-captioned per shot by Gemini 2.5 |
| Long video | "Sparse scan" >10min | Full coverage regardless of length |
| Cross-modal | — | Speaker timeline aligned to shot timeline |
| Re-query | Re-runs everything | `cc-video query` greps cached JSON |

## Quick start

```bash
# install (Python 3.10+)
pip install -e .                  # core only — needs ffmpeg + yt-dlp + scenedetect
pip install -e '.[asr]'           # + faster-whisper + pyannote (local transcripts)
pip install -e '.[ocr]'           # + PaddleOCR
pip install -e '.[vlm]'           # + google-genai for shot captions
pip install -e '.[all]'           # everything

# binaries (one-time)
brew install ffmpeg yt-dlp        # macOS
sudo apt install ffmpeg && pipx install yt-dlp   # Linux
winget install ffmpeg && pip install yt-dlp      # Windows

# preflight
python3 scripts/setup.py

# run
cc-video analyze sample.mp4 --out-dir ./out
cc-video analyze https://youtu.be/abc --out-dir ./out

# query the cached analysis (no re-run)
cc-video query ./out "what tool was used to deploy"
```

The output:

```
out/
├── video.understanding.md      # Claude reads this (timeline + transcript)
├── video.understanding.json    # programmatic access
├── keyframes/                  # only the selected frames
│   ├── shot0000_middle.jpg
│   ├── shot0001_start.jpg
│   └── …
└── audio.wav                   # if ASR ran
```

## Pipeline detail

1. **Probe** (`probe.py`) — ffprobe for duration/resolution/audio.
2. **Download** (`download.py`) — yt-dlp for URLs (also fetches native captions);
   local files pass through.
3. **Shot detection** (`shots.py`) — PySceneDetect `AdaptiveDetector` with a
   rolling-average threshold robust to camera motion. Micro-shots <0.6s are
   merged into neighbors.
4. **Keyframes** (`keyframes.py`) — middle frame per shot by default; shots >8s
   get start+middle+end. `--deep-keyframes` runs CLIP embedding + k-means inside
   each shot to pick diverse representatives.
5. **Transcript** (`transcript.py`) — three-way fallback:
   1. yt-dlp native captions if present
   2. faster-whisper local (with pyannote diarization if `HF_TOKEN` set)
   3. Whisper API (Groq → OpenAI)
6. **OCR** (`ocr.py`) — PaddleOCR per keyframe, deduped per shot.
7. **VLM caption** (`vlm.py`) — Gemini 2.5 reads (keyframes + transcript-in-shot
   + OCR-in-shot) and returns structured JSON: action, setting,
   on_screen_text_summary, notable_objects, change_from_prev.
8. **Merger** (`merger.py`) — slice transcript by shot boundaries; group
   consecutive similar shots into chapters via setting-token Jaccard similarity.
9. **Output** (`output.py`) — render Markdown timeline + JSON.

## Configuration

Env vars (all optional):

| Var | Purpose |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Per-shot VLM captioning |
| `GROQ_API_KEY` | Whisper API fallback (preferred — fast, cheap) |
| `OPENAI_API_KEY` | Whisper API fallback (alt) |
| `HF_TOKEN` | pyannote.audio speaker diarization (needs HF model access) |

Without any keys, cc-video still produces shot detection + keyframes + (if
`[asr]` extra installed) local faster-whisper transcript + (if `[ocr]` extra
installed) OCR. The output degrades gracefully — no key just means no VLM
captions; the timeline still has everything else.

## Claude Code integration

```bash
# Install as a skill
git clone https://github.com/anthropics/cc-video.git ~/.claude/skills/cc-video
pip install -e ~/.claude/skills/cc-video
```

In Claude Code:

```
/cc-video https://youtu.be/abc what stack did they use?
/cc-video screen-recording.mp4 where does the error appear?
```

The skill (`SKILL.md`) instructs Claude to run `cc-video analyze`, then `Read`
the resulting `video.understanding.md`. Only individual keyframe JPEGs are
opened if the question needs detail beyond the captions.

## License

MIT.
