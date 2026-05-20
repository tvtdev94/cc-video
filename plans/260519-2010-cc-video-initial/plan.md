# cc-video — Deep Video Understanding for Claude Code

**Date:** 2026-05-19
**Goal:** Repo cho phép Claude Code "hiểu" video chi tiết, không phải chỉ snapshot đều theo thời gian.

## Vấn đề với cách hiện tại (bradautomates/claude-video `/watch`)

- Lấy frame *đều* theo fps (max 2 fps, 100 frames). Bỏ sót shot ngắn, lãng phí frame ở shot dài tĩnh.
- Transcript thô — không có speaker, không word-level timestamp.
- Không có OCR — Claude phải "đọc" text trên màn hình từ ảnh thô, tốn token.
- Không synthesize — đẩy raw frames vào context Claude, frame token chiếm 90%+ budget.
- Video > 10 phút "sparse scan warning" — không cover được hết.

## Cách của cc-video

Pipeline shot-aware + cross-modal alignment:

```
video → probe → shot-detect → per-shot keyframe → enrich (ASR/OCR/VLM) → merge → timeline.md+json
```

Chi tiết:

1. **Shot detection (PySceneDetect AdaptiveDetector)** — chia video theo nội dung, không theo thời gian. 1 shot = 1 đơn vị ngữ nghĩa.
2. **Keyframe per shot** — mặc định lấy middle frame (đại diện shot). Mode `--deep`: dùng CLIP embedding + clustering trong shot để chọn 1–N frame đa dạng (AdaRD-Key/MaxInfo).
3. **Transcript (WhisperX = faster-whisper + pyannote)** — word-level + speaker diarization. Fallback: native captions (yt-dlp), Whisper API (Groq/OpenAI).
4. **OCR per keyframe (PaddleOCR)** — tách text on-screen (terminal, IDE, slide, subtitle, lower-third). Quan trọng cho video lập trình/giảng dạy.
5. **VLM caption per shot (Gemini 2.5)** — mỗi shot, đẩy keyframes + transcript-in-shot vào Gemini, lấy ra: action, setting, on-screen text summary, UI elements, change-from-previous-shot.
6. **Cross-modal merger** — align speaker với shot, gom shot tương tự thành "chapters", dò inconsistency (người nói X nhưng màn hình Y).
7. **Output:**
   - `video.understanding.md` — timeline Claude đọc.
   - `video.understanding.json` — query-able structure.
   - `keyframes/` — chỉ frame được chọn (5–10× ít hơn `/watch`).

## So sánh

| Aspect | `/watch` | `cc-video` |
|---|---|---|
| Frame sampling | Uniform, max 100 | Shot-aware, adaptive per-shot |
| Transcript | Plain segments | Word-level + speaker |
| OCR | — | PaddleOCR per keyframe |
| Visual description | Raw images vào Claude | Pre-captioned per shot |
| Long video | "Sparse warning" | Full coverage, không cap thời gian |
| Cross-modal | — | Speaker ↔ visual alignment |
| Re-query | Phải chạy lại | Cache JSON, hỏi lại không tốn |

## File layout

```
cc-video/
├── README.md
├── pyproject.toml
├── SKILL.md
├── .claude-plugin/plugin.json
├── commands/cc-video.md
├── src/cc_video/
│   ├── cli.py              # argparse entry
│   ├── pipeline.py         # orchestrator
│   ├── probe.py            # ffprobe
│   ├── download.py         # yt-dlp
│   ├── shots.py            # PySceneDetect
│   ├── keyframes.py        # middle-frame + optional CLIP cluster
│   ├── transcript.py       # WhisperX | captions | whisper-api
│   ├── ocr.py              # PaddleOCR
│   ├── vlm.py              # Gemini 2.5 shot caption
│   ├── merger.py           # cross-modal timeline
│   ├── output.py           # md + json writer
│   └── config.py           # env, model paths
├── scripts/setup.py        # preflight & install
├── docs/architecture.md
├── docs/usage.md
└── examples/sample.md      # output ví dụ
```

## Phases

### Phase 1 — Skeleton + core pipeline (lightweight path)
- pyproject.toml + cli.py
- probe.py + download.py (port từ ref)
- shots.py với PySceneDetect AdaptiveDetector
- keyframes.py (middle frame mỗi shot, ffmpeg seek)
- transcript.py (3 paths: captions → whisper-api → faster-whisper)
- output.py (md + json)
- pipeline.py wire all together
- Smoke test với sample.mp4

### Phase 2 — Enrichment (optional deep mode)
- ocr.py (PaddleOCR, optional)
- vlm.py (Gemini 2.5 per-shot caption, optional)
- keyframes deep mode (CLIP cluster)

### Phase 3 — Claude Code integration
- SKILL.md + commands/cc-video.md
- .claude-plugin/plugin.json
- Hook để Claude đọc file output

### Phase 4 — Docs + examples
- README.md
- docs/architecture.md, docs/usage.md
- examples/sample.md (real run output)

## Todo

- [x] Plan
- [ ] pyproject.toml
- [ ] src/cc_video/probe.py
- [ ] src/cc_video/download.py
- [ ] src/cc_video/shots.py
- [ ] src/cc_video/keyframes.py
- [ ] src/cc_video/transcript.py
- [ ] src/cc_video/ocr.py
- [ ] src/cc_video/vlm.py
- [ ] src/cc_video/merger.py
- [ ] src/cc_video/output.py
- [ ] src/cc_video/pipeline.py
- [ ] src/cc_video/cli.py
- [ ] src/cc_video/config.py
- [ ] SKILL.md
- [ ] commands/cc-video.md
- [ ] .claude-plugin/plugin.json
- [ ] README.md
- [ ] docs/architecture.md, docs/usage.md
- [ ] examples/sample.md
- [ ] tests/test_pipeline.py

## Success criteria

1. Chạy `cc-video analyze sample.mp4` produce `video.understanding.md` đầy đủ.
2. So với `/watch`: cùng video < 5 phút, cc-video dùng < 1/3 frame, output có speaker + OCR + per-shot summary.
3. Claude Code đọc file md trả lời được câu hỏi về nội dung video chi tiết hơn `/watch`.

## Open questions

- Gemini API key có sẵn không? Nếu không, fallback ra sao? → Mode `--no-vlm` cho lightweight.
- Local VLM thay thế Gemini? → Qwen2.5-VL/LLaVA-Video, để sau Phase 4.
- Real-time streaming? → ngoài scope, batch-only.
