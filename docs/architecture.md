# Architecture

```
┌────────────┐      ┌─────────────┐      ┌─────────────┐
│  Source    │      │   probe     │      │   shots     │
│ URL / path │─────▶│  (ffprobe)  │─────▶│ (PySceneDet)│
└────────────┘      │  duration   │      │  Adaptive   │
       │            │  res, fps   │      │  thresholding│
       ▼            └─────────────┘      └──────┬──────┘
┌────────────┐                                  │
│  download  │                                  ▼
│  (yt-dlp)  │                          ┌─────────────┐
│  → video   │                          │  keyframes  │
│  → vtt sub │                          │ mid-frame   │
└─────┬──────┘                          │ or CLIP-KM  │
      │                                 └──────┬──────┘
      │                                        │
      ▼                                        ▼
┌─────────────┐  parallel  ┌─────────┐  ┌──────────┐  ┌──────────┐
│  transcript │            │   OCR   │  │   VLM    │  │          │
│  WhisperX   │◀──audio────│Paddle   │  │ Gemini   │  │  merger  │
│  or API     │            │  OCR    │  │  2.5     │  │          │
│  or VTT     │            └────┬────┘  └────┬─────┘  └────┬─────┘
└──────┬──────┘                 │            │              │
       │     ┌──────────────────┴────────────┘              │
       └────▶│  cross-modal alignment, chapter grouping     │
             └────────────────────┬──────────────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ video.understanding │
                       │      .md + .json    │
                       └─────────────────────┘
                                  ▼
                             Claude Code
```

## Why shot-aware beats uniform fps

| Scenario | Uniform fps @ 2fps cap=100 | Shot-aware |
|---|---|---|
| 5-min tutorial with 20 long shots | 100 frames, ~5s apart | 20-60 keyframes, one per scene change |
| 30-second highlight reel with 60 cuts | 60 frames evenly spaced (≈1 per cut by luck) | 60 keyframes, each at the *actual* cut |
| 2-hour podcast (1 camera, no cuts) | 100 sparse frames, 72s apart, useless | 1 shot, 3 keyframes, transcript carries the load |
| Screen recording with intermittent IDE → terminal → browser switches | Misses any switch shorter than ~5s | Detects every switch via content delta |

## Module dependencies

```
cli.py
 └─▶ pipeline.py
      ├─▶ probe.py        (stdlib + ffprobe)
      ├─▶ download.py     (stdlib + yt-dlp binary)
      ├─▶ shots.py        (scenedetect[opencv])
      ├─▶ keyframes.py    (ffmpeg; deep mode: open-clip + sklearn + torch)
      ├─▶ transcript.py   (faster-whisper + pyannote OR Whisper API)
      ├─▶ ocr.py          (paddleocr)
      ├─▶ vlm.py          (google-genai)
      ├─▶ merger.py       (pure stdlib)
      └─▶ output.py       (pure stdlib)
```

Optional deps fail gracefully — pipeline detects missing extras at runtime and
skips that enrichment stage with a clear log line.

## Token math (rough)

For a 5-minute coding tutorial:

| Approach | Image tokens | Text tokens | Total |
|---|---|---|---|
| `/watch` (60 frames @ 512px) | ~30,000 | ~3,000 | ~33,000 |
| `cc-video` (15 shots × 1 keyframe sent to VLM offline + Claude reads md) | 0 (VLM done offline) | ~5,000 (md timeline) | ~5,000 |

Image tokens dominate `/watch`. `cc-video` pre-captions offline so Claude reads
text. Claude can still `Read` specific keyframes on demand for visual detail.
