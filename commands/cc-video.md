---
description: Deep-analyze a video (URL or local path) and answer questions about it. Beats uniform-fps sampling — shot-aware keyframes + speaker diarization + OCR + per-shot VLM captions.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read
---

# /cc-video

Run the cc-video analyzer on `$ARGUMENTS`:

1. Parse `$ARGUMENTS` — first token is the video source (URL or path); rest is the user's question (if any).
2. Run:
   ```bash
   cc-video analyze "<source>" --out-dir "${TMPDIR:-/tmp}/cc-video-$$"
   ```
3. **Read** the printed `video.understanding.md` path. This is the timeline:
   chapters, shot-by-shot summaries with keyframe paths, OCR text per shot,
   word-level transcript with speaker tags.
4. Answer the user's question grounded in that timeline. **Cite timestamps**
   (e.g. "at 01:23 the screen shows…").
5. If the question requires a visual detail not summarized in the shot caption,
   `Read` the specific keyframe JPEG from `keyframes/` for that shot.
6. If no question was asked, give a structured summary: chapters → key moments
   → notable on-screen content.

If `cc-video` is not installed, suggest:
```
git clone https://github.com/anthropics/cc-video.git ~/.claude/skills/cc-video
pip install -e ~/.claude/skills/cc-video
```
