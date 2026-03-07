# Constellation Plugins

This folder is for reusable plugin-style tools that Constellation can call across multiple interfaces.

Expected use:

- wrappers around local CLIs
- integrations with external services such as Slack
- adapters for coding agents and automation tools
- queueable bridges like the Codex CLI runner

Extraction rule:

- keep a tool in the main repo until it is stable and used in more than one entrypoint
- move it here once it becomes a reusable plugin rather than a one-off feature
