# Constellation Roadmap

Near-term direction:

1. Keep Constellation as the local hub for notes, voice interfaces, and reusable tools.
2. Use the Codex CLI bridge so a spoken request can trigger coding work in a chosen repo.
3. Let the assistant mine the notes archive for candidate ideas, confirm one by voice, and then hand execution off to Codex CLI.
4. Keep the realtime experience available as a tray-based personal default without forcing that mode on other users.

Target workflow:

1. User asks by voice for ideas from notes.
2. Constellation searches and summarizes likely build candidates.
3. User picks one.
4. Constellation invokes Codex CLI against the target repo with GPT-5.4-oriented prompting where possible.
5. Constellation reports progress and results back through the voice interface.

Current implementation:

- `constellation.py codex ...` exposes repo discovery, bridge status, task queueing, task status, and cancellation.
- `constellation.py ideas ...` exposes note mining and Codex prompt drafting from captured ideas.
- `voice_notes_realtime.py` can call the Codex bridge from live voice sessions.
- `voice_notes_realtime.py` can now mine notes for ideas and queue Codex work directly from a chosen idea.
- `voice_notes_realtime.py --tray` starts a starry tray controller that can launch or stop the realtime voice session and switch default voices.

Constraints to preserve:

- voice-first for the user, but optional for others
- local-first data access
- reusable toolbelt and skill boundaries instead of one monolithic script
- compatibility wrappers while the naming migration is in progress
