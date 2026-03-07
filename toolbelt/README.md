# Constellation Toolbelt

This folder marks the future boundary for reusable local tools that Constellation interfaces can call.

Current direction:

- workspace inspection tools
- notes archive search/read/import tools
- runtime controls such as voice, speech speed, and verbosity
- Codex CLI bridging for queued coding work against local repos
- runtime logs for realtime sessions and Codex task history

Short-term rule:

- keep domain-specific tools close to `voice_notes`
- extract them here once they are reused across multiple entrypoints
- keep bridge state and logs out of Git so Constellation can run locally without polluting the repo
- use these local logs as the first debugging surface in a fresh Codex thread
