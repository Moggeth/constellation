# Constellation

Constellation is the umbrella name for this growing local system of notes, voice interfaces, tools, and linked data sources.

Some file names remain legacy for compatibility, but this is now the canonical Constellation repo.

Current entrypoints in this repo:

- `constellation.py` as the top-level launcher
- `faq_notes_ingest.py` for importing and transcribing recordings
- `faq_notes_chat.py` for chatting with the archive and turning notes into actionable outputs
- `voice_notes_realtime.py` for optional live speech-to-speech use

Preferred entrypoints now use `voice_notes_*` aliases:

- `voice_notes_ingest.py`
- `voice_notes_chat.py`
- `voice_notes_realtime.py`
- `constellation.py`
- `AGENTS.md` for future Codex-thread continuity inside this repo

Top-level launcher:

```bash
python constellation.py chat
python constellation.py ingest --yes
python constellation.py ideas mine
python constellation.py realtime --list-voices
python constellation.py realtime --tray
python constellation.py codex status
python constellation.py toolbelt list
```

Fresh-thread guidance for future Codex work in this repo lives in [AGENTS.md](AGENTS.md).

## What it does

- Acts as the current core of Constellation.
- Detects connected devices on Windows, Ubuntu/Linux mount points, and macOS `/Volumes`.
- Scans media files from the selected device/path.
- Uses a fast metadata index to skip re-hashing known files, then hashes only unknown candidates.
- Copies new files to monthly folders under this tool folder: `faq_notes/YYYY-MM/audio/`.
- Transcribes with OpenAI Whisper (`whisper-1`) using ffmpeg normalization/chunking.
- Writes monthly transcripts:
  - Combined (time-ordered across all devices): `faq_notes/YYYY-MM/FAQ_YYYY-MM.txt`
  - Per-device file(s): `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_1.txt`, `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_2.txt`

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- `OPENAI_API_KEY` environment variable
- Python package: `openai`
- Optional for realtime voice mode: `sounddevice`
- Optional for PostgreSQL backend: `psycopg[binary]`
- Optional for tray mode on Windows, macOS, and Linux: `pystray`, `Pillow`
- Optional for voice-driven coding tasks: a working `codex` CLI command on `PATH` or configured via `CONSTELLATION_CODEX_COMMAND`

Codex CLI install used on this machine:

```bash
"C:\Program Files\nodejs\npm.cmd" install -g @openai/codex
```

Install package:

```bash
pip install openai
```

or install from requirements:

```bash
pip install -r requirements.txt
```

For a fresh Windows machine, the quickest setup path is:

```powershell
pwsh -File .\bootstrap.ps1
```

To also mount an external read-only library root on that machine:

```powershell
pwsh -File .\bootstrap.ps1 -MountLibrary "C:\path\to\older\repos"
```

For tray mode:

```bash
pip install pystray pillow
```

For realtime voice mode:

```bash
pip install sounddevice
```

## Usage

Run from this folder:

```bash
python faq_notes_ingest.py
```

Open the chat interface:

```bash
python faq_notes_chat.py
```

Preferred alias:

```bash
python voice_notes_chat.py
```

Import first, then open chat:

```bash
python faq_notes_chat.py --import-first
```

Ask one question without entering the REPL:

```bash
python faq_notes_chat.py --prompt "Find my latest coding ideas and turn them into a build brief."
```

Inspect the archive without calling GPT:

```bash
python faq_notes_chat.py --stats
python faq_notes_chat.py --list-recent 5
```

Start realtime voice chat:

```bash
python voice_notes_realtime.py
```

This is the most direct current entrypoint for the live Constellation experience.

List supported realtime voices:

```bash
python voice_notes_realtime.py --list-voices
```

Start with a specific voice:

```bash
python voice_notes_realtime.py --voice verse
```

Run the realtime voice controller from the system tray:

```bash
python voice_notes_realtime.py --tray
```

Realtime session logs now go to `toolbelt/realtime_sessions/<timestamp>/` by default and include:

- native user/assistant transcript lines
- tool calls and tool results
- session start/stop/error events
- enough context to reconstruct voice-switch continuity and tool failures
- enough information for a fresh Codex thread to diagnose recent runtime behavior

Disable that logging if needed:

```bash
python voice_notes_realtime.py --no-session-logging
```

When you want to review the most recent voice session, inspect:

- `toolbelt/realtime_sessions/<timestamp>/transcript.md` for the user/assistant exchange
- `toolbelt/realtime_sessions/<timestamp>/events.jsonl` for tool calls, timings, errors, and reconnects
- `toolbelt/realtime_voice.log` for tray-level session startup/shutdown context

In Constellation terms, "review the logs", "introspect on the last session", or "self-reflect on the last session" should mean:

- look at the latest session folder first
- check the transcript tail to see what the user was trying to do
- check timestamps on tool starts/completions
- identify failed tool calls, abandoned actions, runtime errors, and slow operations
- summarize what went wrong and implement fixes when the user asks for improvements

Notes on realtime voices:

- Supported realtime voices are currently `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, and `cedar`.
- If you ask the assistant to switch voices mid-session, it now reconnects briefly and continues in the new voice.
- Voice changes now carry forward recent conversation context instead of starting from a blank slate.
- The live runtime now uses UTF-8-safe console output on Windows so transcribed characters do not crash the session.
- Tray mode remembers the default voice, can relaunch the live session without a terminal window, and uses a Constellation-themed starry tray icon.
- Speech speed is clamped to the current realtime-supported range before session updates are sent.

## Workspace Boundaries

Constellation now treats this repo as the canonical app home:

- Default writable workspace root: `workspace/`
- Optional mounted library roots: zero or more external folders exposed as read-only local context
- Local direct-write tools should stay in `workspace/`
- Mounted libraries stay readable and openable locally
- If you want to modify a mounted repo, target it explicitly through Codex

Machine-specific runtime paths live in `toolbelt/runtime_paths.json`, which is intentionally not tracked by Git. The checked-in example lives at `toolbelt/runtime_paths.example.json`.

Show the active runtime paths:

```bash
python constellation.py paths show
```

Mount an external read-only library root:

```bash
python constellation.py paths mount-library "C:\path\to\older\repos"
```

Reset to the default repo-local workspace with no mounted libraries:

```bash
python constellation.py paths reset
```

Check Codex bridge status:

```bash
python constellation.py codex status
```

Inspect recent live-session logs when debugging the runtime:

```bash
toolbelt/realtime_sessions/<timestamp>/
```

List candidate repos for Codex work:

```bash
python constellation.py codex repos
```

Queue a Codex task against a repo:

```bash
python constellation.py codex run --repo auto_roster "Review the repo and implement the requested change."
```

If your current shell has not picked up the npm install path yet, point Constellation at the installed binary explicitly:

```bash
set CONSTELLATION_CODEX_COMMAND=C:\Users\mog\AppData\Roaming\npm\codex.cmd
python constellation.py codex status
```

Inspect recent Codex runs:

```bash
python constellation.py codex runs
```

Mine notes for likely build ideas:

```bash
python constellation.py ideas mine --limit 5
```

Focus the mining pass on a topic:

```bash
python constellation.py ideas mine --query "robotics arm" --limit 3
```

Draft a Codex prompt from one chosen idea:

```bash
python constellation.py ideas prompt --title "Left arm control stack" --summary "Build the first left arm subsystem..." --source-note-id 20260301_101530_ab12cd34 --repo vibe_coded_wheelchair
```

Run from project root:

```bash
python constellation/faq_notes_ingest.py
```

Use explicit source path:

```bash
python faq_notes_ingest.py --source "E:\\"
```

Auto-confirm and run without prompt:

```bash
python faq_notes_ingest.py --yes
```

Watch for newly connected devices:

```bash
python faq_notes_ingest.py --watch
```

Run in the system tray (first time each device is seen, approval is required):

```bash
python faq_notes_ingest.py --tray
```

Run quietly in the background and write logs to `faq_notes/faq_notes.log`:

```bash
python faq_notes_ingest.py --tray --quiet
```

Install tray mode to launch automatically when you sign in:

```bash
python faq_notes_ingest.py --install-startup
```

Remove the startup launcher:

```bash
python faq_notes_ingest.py --remove-startup
```

Set parallel workers (default is parallel already):

```bash
python faq_notes_ingest.py --workers 4
```

Use strict duplicate detection (slower, full-file SHA-256):

```bash
python faq_notes_ingest.py --dedupe-mode strict
```

Custom archive root:

```bash
python faq_notes_ingest.py --archive-root "./faq_notes"
```

Use PostgreSQL for ingest state (devices, dedupe index, processed files):

```bash
python faq_notes_ingest.py --storage-backend postgres --account-id "alice" --database-url "postgresql://user:pass@host:5432/faq_notes"
```

Use env vars instead of CLI args:

```bash
set FAQ_NOTES_DATABASE_URL=postgresql://user:pass@host:5432/faq_notes
set FAQ_NOTES_ACCOUNT_ID=alice
python faq_notes_ingest.py --storage-backend postgres
```

One-time migration from local `manifest.json` into PostgreSQL:

```bash
python faq_notes_ingest.py --storage-backend postgres --account-id "alice" --database-url "postgresql://user:pass@host:5432/faq_notes" --migrate-manifest --migrate-only
```

Notes on archive root:
- Default is `faq_notes` resolved relative to the script location, not the current working directory.
- Manifest `copied_path` and `transcript_path` entries are stored archive-relative for portability.

## Watch/tray approval behavior

- In watch/tray mode, newly detected device IDs require manual approval once.
- After approval, that exact device ID is auto-processed on future reconnects.
- If the tray app starts while an already-approved device is still attached, it is processed immediately on startup.
- If a new device is not approved, it is ignored until approved on a later connection.
- Tray mode now uses GUI approval prompts on Windows, macOS, and Linux when available, so first-time approval still works after launch-at-login.

## Tray mode polish

- Tray mode now animates the tray icon while an import is actively running.
- Tray icon states are visually distinct for monitoring, paused, restart-pending, restarting, and active import states.
- Tray mode uses desktop banner notifications (`icon.notify`) for import start and import completion when supported by the OS tray backend.
- Tray mode watches the local project code and automatically restarts itself when `.py` files or startup-relevant config files change.
- If code changes while an import or scan is in progress, restart is deferred until the current work finishes.
- The top-level tray menu stays minimal: status, pause/resume, scan now, restart service, advanced options, and quit.
- `Advanced` contains the import start/end notification toggles, last event/last scan status, and shortcuts to the archive folder and log file.
- Tray notification preferences are stored per machine in `faq_notes/tray_settings.json`.
- `--install-startup` now installs a per-user launcher on each supported OS:
  - Windows: Startup folder `.vbs`
  - Ubuntu/Linux: `~/.config/autostart/*.desktop`
  - macOS: `~/Library/LaunchAgents/*.plist`
- Tray mode logs to `faq_notes/faq_notes.log` by default, or to a custom path with `--log-file`.
- On Ubuntu/Linux, `zenity` is recommended so first-time approval prompts appear cleanly from background tray mode.

## Duplicate detection modes

- `fast` (default): sampled content fingerprint (size + sampled bytes from start/middle/end) for much faster duplicate checks.
- `strict`: full-file SHA-256 (slower, highest confidence).

## Storage backends

- `json` (default): writes `faq_notes/manifest.json` as the ingest index.
- `postgres`: stores ingest index in PostgreSQL tables (`faq_devices`, `faq_files`, `faq_quick_index`).
- Postgres mode requires an account ID (`--account-id` or `FAQ_NOTES_ACCOUNT_ID`) so each user's state is isolated.
- Monthly human-readable outputs are unchanged in both modes:
  - `faq_notes/YYYY-MM/FAQ_YYYY-MM.txt`
  - `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_*.txt`

## Interrupt/disconnect safety

- Source reads now fail gracefully if a USB device disconnects mid-run.
- Audio copies are written to a temporary `.part` file and atomically renamed only when complete.
- Transcript, manifest, and monthly aggregate files are written atomically to avoid partial/corrupted outputs.
- If disconnect happens mid-ingest, already completed items remain intact and the rest can be resumed on reconnect.

## Output structure

```text
constellation/
  constellation.py
  faq_notes_ingest.py
  faq_notes_chat.py
  voice_notes_ingest.py
  voice_notes_chat.py
  voice_notes_realtime.py
  voice_notes_toolbelt.py
  ROADMAP.md
  README.md
  toolbelt/
    registry.json
    plugins/
    skills/
  faq_notes/
    manifest.json
    YYYY-MM/
      audio/
        YYYYMMDD_HHMMSS_<hash8>.<ext>
      transcripts/
        YYYYMMDD_HHMMSS_<hash8>.txt
      FAQ_YYYY-MM.txt
      FAQ_YYYY-MM_device_1.txt
      FAQ_YYYY-MM_device_2.txt
```

## Chat commands

- `/help`
- `/stats`
- `/months`
- `/latest`
- `/find <query>`
- `/read <note_id>`
- `/import [path]`
- `/refresh`
- `/reset`
- `/exit`

## Toolbelt Direction

The repo now includes a small `toolbelt/` seed plus `voice_notes_toolbelt.py` as the start of the Constellation toolbelt layer. The current split is:

- notes/archive behavior stays here
- reusable runtime controls and voice metadata begin moving into the toolbelt
- note mining now turns captured ideas into build candidates and Codex-ready prompts
- Codex CLI queueing and run state now live in the toolbelt boundary
- broader agent/tool growth can be extracted there later without re-splitting the notes archive itself
