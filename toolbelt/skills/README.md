# Constellation Skills

This folder is for higher-level reusable skill definitions that describe how Constellation should approach recurring tasks.

Examples:

- search notes, summarize ideas, and turn them into build briefs
- inspect a workspace and prepare a coding plan
- invoke Codex CLI to implement a selected idea
- switch between voice-first exploration and queued coding execution without losing context

Near-term goal:

- keep the skill contract explicit so voice-driven orchestration can choose the right workflow without hardcoding every path in one giant script
- keep enough local structure and logging that a fresh Codex thread can resume work safely
