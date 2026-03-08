# Constellation Agent Notes

This repo is the canonical local home of `Constellation`, a voice-first system for notes, realtime assistance, local tools, and Codex-driven coding workflows.

## Start Here

- Read [README.md](README.md) for current entrypoints and operator-facing usage.
- Read [ROADMAP.md](ROADMAP.md) for the intended product direction.
- Treat [constellation.py](constellation.py) as the top-level launcher.
- Treat [voice_notes_realtime.py](voice_notes_realtime.py) as the live runtime.

## Core Entry Points

- `python constellation.py realtime --tray`
- `python constellation.py ideas mine --limit 5`
- `python constellation.py codex status`
- `python constellation.py codex repos`
- `python constellation.py codex run --repo <repo> "<task>"`
- `python constellation.py paths show`

## Preferred Behavior

- Prefer direct local workspace tools for small file operations.
  - Examples: create a simple file, replace text, read a file, open a workspace path.
- Prefer Codex CLI for larger coding tasks, repo-wide edits, verification, or execution chains.
- Keep the experience voice-first for the primary user, but do not force voice/tray assumptions into every workflow.
- Preserve continuity across reconnects and restarts where possible.

## Logs And State

- Realtime session logs live in `toolbelt/realtime_sessions/<timestamp>/`.
  - `transcript.md` contains user/assistant transcript lines.
  - `events.jsonl` contains tool calls, tool results, and session lifecycle events.
- Codex run records live in `toolbelt/codex_runs/<task_id>/`.
- Tray session output also lands in `toolbelt/realtime_voice.log`.
- Machine-specific runtime paths live in `toolbelt/runtime_paths.json`.
- These runtime artifacts are intentionally local and ignored by Git.

If the user says "review the logs", "introspect on the last session", or "self-reflect on the last session", treat that as a concrete workflow:

1. Inspect the newest folder under `toolbelt/realtime_sessions/`.
2. Read `transcript.md` to understand the user request flow.
3. Read `events.jsonl` to inspect tool calls, timings, failures, reconnects, and errors.
4. Check `toolbelt/realtime_voice.log` for tray/session startup context when relevant.
5. Summarize what went wrong, what was slow, and what should be fixed.
6. If the user asked for improvements, implement the runtime or prompt changes instead of stopping at analysis.

## New Thread Checklist

When starting a fresh Codex thread in this repo:

1. Check `git status` before editing anything.
2. Read `README.md`, `ROADMAP.md`, and this file.
3. If the task involves recent runtime behavior, inspect the latest folder under `toolbelt/realtime_sessions/`.
4. If the task involves Codex execution, inspect `python constellation.py codex status` and recent runs under `toolbelt/codex_runs/`.
5. Do not overwrite unrelated local changes.

## Safety Rules

- Do not revert user changes unless explicitly asked.
- Do not assume the realtime runtime can open arbitrary external apps unless a tool exists for it.
- Keep changes local-first and practical.
- Treat `workspace/` as the default writable scratch area for this repo.
- Treat any mounted library roots as read-only local context unless the user explicitly asks to modify one through Codex.
- Avoid broad refactors unless they clearly support the Constellation runtime, toolbelt, or note-to-action workflow.

## Naming Notes

- `Constellation` is the umbrella name.
- Some `faq_notes_*` filenames remain for compatibility.
- The current repo/folder is canonical even where legacy names still appear internally.
- Personal archive roots such as `things_are_different_now` are optional mounted libraries, not required runtime dependencies.
