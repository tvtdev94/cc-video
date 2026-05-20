# Example output

For a 3-minute screencast of someone deploying a Next.js app to Vercel, the
generated `video.understanding.md` would look roughly like:

```markdown
# Video Understanding

- **Source:** `/tmp/deploy-nextjs.mp4`
- **Title:** Deploy Next.js to Vercel in 3 minutes
- **Duration:** 03:12 (192.3s)
- **Resolution:** 1920×1080
- **Shots detected:** 18
- **Chapters:** 4

## Chapters (high-level)

- **Ch 1** [00:00 → 00:32] (3 shots) — Talking-head intro
- **Ch 2** [00:32 → 01:45] (8 shots) — VS Code editor, building the app
- **Ch 3** [01:45 → 02:40] (5 shots) — Terminal: vercel CLI commands
- **Ch 4** [02:40 → 03:12] (2 shots) — Browser, showing deployed site

## Shot-by-shot timeline

### Shot 0 · 00:00 → 00:11 (11.2s)

**Keyframes:** `keyframes/shot0000_start.jpg` `keyframes/shot0000_middle.jpg` `keyframes/shot0000_end.jpg`
**Change:** opening shot
**Action:** presenter introduces the tutorial directly to camera
**Setting:** home office, neutral background, presenter centered
**On-screen text (summary):** lower-third with presenter name and "Next.js → Vercel"
**Notable objects:** presenter, microphone, lower-third title

**OCR (raw text on screen):**
```
Alex Chen
Next.js → Vercel in 3 minutes
```

**Spoken:**
```
[00:01] (SPEAKER_00) Hey everyone, in the next three minutes I'll show you how
to take a Next.js app from local dev to a live Vercel URL.
```

### Shot 4 · 00:42 → 00:58 (16.1s)

**Keyframes:** `keyframes/shot0004_start.jpg` `keyframes/shot0004_middle.jpg` `keyframes/shot0004_end.jpg`
**Change:** cut from talking-head to VS Code editor
**Action:** developer edits the home page, adds an "Hello, world" heading
**Setting:** VS Code, dark theme, file explorer on left, editor on right
**On-screen text (summary):** TSX code for the home page, sidebar shows project files
**Notable objects:** VS Code, terminal pane, TSX file

**OCR (raw text on screen):**
```
app/page.tsx
package.json
next.config.mjs
export default function Home() {
  return <h1>Hello, world</h1>;
}
```

**Spoken:**
```
[00:43] (SPEAKER_00) I'll start by editing app/page.tsx and just dropping in
a basic Hello, world heading so we have something visible to deploy.
```

### Shot 11 · 01:52 → 02:08 (16.0s)

**Keyframes:** `keyframes/shot0011_start.jpg` `keyframes/shot0011_middle.jpg`
**Change:** switch from editor to terminal
**Action:** developer runs vercel CLI, types login + project link
**Setting:** terminal, dark theme, npm install just completed
**On-screen text (summary):** vercel login URL, project link prompts, environment variable warnings
**Notable objects:** terminal, vercel command, browser URL prompt

**OCR (raw text on screen):**
```
$ npx vercel
Vercel CLI 32.4.1
? Set up and deploy "~/projects/demo"? [Y/n] y
? Which scope do you want to deploy to? alex-chen
? Link to existing project? [y/N] n
? What's your project's name? demo
? In which directory is your code located? ./
```

**Spoken:**
```
[01:53] (SPEAKER_00) Now we run npx vercel. It asks for scope, name, and the
location of the code — for a brand-new project I just press enter through these.
```

…

## Full transcript

```
[00:01] (SPEAKER_00) Hey everyone, in the next three minutes I'll show you…
[00:43] (SPEAKER_00) I'll start by editing app/page.tsx…
[01:53] (SPEAKER_00) Now we run npx vercel…
[02:41] (SPEAKER_00) And there it is — live on the vercel.app URL.
```
```

## What Claude can do with this

When the user asks _"What command was used to deploy?"_, Claude reads the
markdown above, sees Shot 11 with OCR containing `npx vercel`, and answers:

> The deploy command shown at 01:53 is `npx vercel` (CLI v32.4.1), which then
> prompts for scope, project name, and code directory. See the OCR block in
> Shot 11.

No images needed in Claude's context — the text was enough.

If the user asks something visual ("what color is the lower-third?"), Claude
can `Read keyframes/shot0000_start.jpg` to see it directly.
