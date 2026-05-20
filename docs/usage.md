# Usage

## Install

```bash
git clone <repo> cc-video && cd cc-video
pip install -e '.[all]'             # full features
# or: pip install -e .              # core only (shots + keyframes)

# Binaries (one time):
brew install ffmpeg yt-dlp          # macOS
sudo apt install ffmpeg && pipx install yt-dlp   # Debian/Ubuntu
winget install ffmpeg && pip install yt-dlp      # Windows
```

Preflight to see what's wired up:
```bash
python3 scripts/setup.py            # human-readable
python3 scripts/setup.py --json     # for scripting
```

## CLI

```bash
cc-video analyze <source> [options]
cc-video query <work-dir> "<question>"
```

### `analyze` options

| Flag | Default | Effect |
|---|---|---|
| `--out-dir DIR` | tmp | Where to write `video.understanding.md`, `keyframes/`, JSON |
| `--deep-keyframes` | off | Inside each shot, pick frames by CLIP clustering (more diverse) |
| `--no-vlm` | off | Skip Gemini shot captions |
| `--no-ocr` | off | Skip PaddleOCR |
| `--no-asr` | off | Skip transcription |
| `--whisper-model SIZE` | `small` | Local whisper size: tiny/base/small/medium/large-v3 |
| `--vlm-model NAME` | `gemini-2.5-flash` | Gemini variant |
| `--resolution N` | 720 | Keyframe width px |

### `query` (no re-run)

After an `analyze` has produced a work dir, you can grep the JSON:

```bash
cc-video query ./out "terminal error"
cc-video query ./out "deploy to vercel"
```

Returns matching shots with timestamps. Fast — no model calls.

## Output format

`video.understanding.md` structure:

```markdown
# Video Understanding

- Source / Title / Duration / Resolution
- Shots detected: N
- Chapters: M

## Chapters (high-level)
- Ch 1 [00:00 → 02:15] (12 shots) — Setting: editor + terminal

## Shot-by-shot timeline

### Shot 0 · 00:00 → 00:05 (5.0s)

**Keyframes:** `keyframes/shot0000_middle.jpg`
**Change:** opening shot
**Action:** developer opens a terminal in VS Code
**Setting:** VS Code editor with integrated terminal, dark theme
**On-screen text (summary):** filenames in sidebar, prompt $ cursor
**Notable objects:** VS Code, terminal, sidebar

**OCR (raw text on screen):**
```
src/app.tsx
package.json
$ npm run dev
```

**Spoken:**
```
[00:01] (SPEAKER_00) Let's start by opening the project.
```

### Shot 1 · ...
```

`video.understanding.json` has the same data structured for programmatic
access — every shot has `start_seconds`, `end_seconds`, `keyframes[]`, `caption`,
`ocr_text`, `transcript[]` with `words[]` (start/end per word).

## Using in Claude Code

Once installed as a skill (`~/.claude/skills/cc-video/`):

```
/cc-video <url-or-path> [question]
```

The skill instructs Claude to:
1. Run `cc-video analyze` on the source
2. Read `video.understanding.md`
3. Answer the question citing timestamps
4. Open specific keyframes only when needed
