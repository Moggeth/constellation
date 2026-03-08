#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

from constellation_runtime import (
    RUNTIME_PATHS_PATH,
    ensure_runtime_layout,
    load_runtime_paths,
    mount_library,
    reset_runtime_paths,
    unmount_library,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLBELT_REGISTRY = SCRIPT_DIR / "toolbelt" / "registry.json"


def run_script(filename: str, forwarded_args: list[str]) -> int:
    script_path = SCRIPT_DIR / filename
    sys.argv = [str(script_path), *forwarded_args]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


def load_registry() -> dict[str, Any]:
    return json.loads(TOOLBELT_REGISTRY.read_text(encoding="utf-8"))


def cmd_toolbelt_list() -> int:
    payload = load_registry()
    print(f"Constellation toolbelt: {payload.get('name', 'toolbelt')}")
    for tool in payload.get("tools", []):
        name = tool.get("name", "unknown")
        kind = tool.get("kind", "unspecified")
        status = tool.get("status", "unknown")
        print(f"- {name} [{kind}] ({status})")
        print(f"  {tool.get('description', '').strip()}")
    return 0


def cmd_toolbelt_show(tool_name: str) -> int:
    payload = load_registry()
    for tool in payload.get("tools", []):
        if tool.get("name") == tool_name:
            print(json.dumps(tool, indent=2))
            return 0
    print(f"Tool not found: {tool_name}", file=sys.stderr)
    return 1


def cmd_paths_show() -> int:
    paths = ensure_runtime_layout(load_runtime_paths())
    print(f"Runtime config: {RUNTIME_PATHS_PATH}")
    print(f"Workspace root: {paths.workspace_path}")
    print(f"Workspace exists: {paths.workspace_path.exists()}")
    if paths.library_paths:
        print("Mounted libraries:")
        for root in paths.library_paths:
            print(f"- {root} (exists={root.exists() and root.is_dir()})")
    else:
        print("Mounted libraries: none")
    return 0


def cmd_paths_set_workspace(path_value: str) -> int:
    paths = ensure_runtime_layout(load_runtime_paths())
    paths.workspace_root = path_value
    ensure_runtime_layout(paths)
    print(f"Workspace root set to: {paths.workspace_path}")
    return 0


def cmd_paths_mount_library(path_value: str) -> int:
    paths = ensure_runtime_layout(load_runtime_paths())
    mount_library(paths, path_value)
    print(f"Mounted library: {path_value}")
    return 0


def cmd_paths_unmount_library(path_value: str) -> int:
    paths = ensure_runtime_layout(load_runtime_paths())
    unmount_library(paths, path_value)
    print(f"Unmounted library: {path_value}")
    return 0


def cmd_paths_reset() -> int:
    paths = reset_runtime_paths()
    print(f"Runtime paths reset. Workspace root: {paths.workspace_path}")
    if RUNTIME_PATHS_PATH.exists():
        print(f"Config written to: {RUNTIME_PATHS_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constellation launcher for notes, chat, realtime voice, and toolbelt workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat", help="Open the Constellation notes chat.")
    subparsers.add_parser("ingest", help="Import and transcribe new recordings.")
    subparsers.add_parser("ideas", help="Mine notes for build ideas and Codex prompts.")
    subparsers.add_parser("realtime", help="Start the live speech-to-speech interface.")
    subparsers.add_parser("codex", help="Inspect or queue Codex CLI tasks through Constellation.")
    paths_parser = subparsers.add_parser(
        "paths",
        help="Show or update the writable workspace and optional mounted library roots.",
    )
    paths_subparsers = paths_parser.add_subparsers(dest="paths_command", required=True)
    paths_subparsers.add_parser("show", help="Print the current runtime paths.")
    set_workspace_parser = paths_subparsers.add_parser("set-workspace", help="Set the writable workspace root.")
    set_workspace_parser.add_argument("path", help="Directory path for the writable Constellation workspace.")
    mount_parser = paths_subparsers.add_parser("mount-library", help="Mount a read-only library root.")
    mount_parser.add_argument("path", help="Directory path to expose as a mounted library.")
    unmount_parser = paths_subparsers.add_parser("unmount-library", help="Remove a mounted library root.")
    unmount_parser.add_argument("path", help="Mounted library path to remove.")
    paths_subparsers.add_parser("reset", help="Reset to the default repo-local workspace with no mounted libraries.")

    toolbelt_parser = subparsers.add_parser("toolbelt", help="Inspect the Constellation toolbelt registry.")
    toolbelt_subparsers = toolbelt_parser.add_subparsers(dest="toolbelt_command", required=True)
    toolbelt_subparsers.add_parser("list", help="List known toolbelt tools.")
    show_parser = toolbelt_subparsers.add_parser("show", help="Show one toolbelt entry.")
    show_parser.add_argument("name", help="Tool name from toolbelt/registry.json")

    return parser


def main() -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    if args.command == "chat":
        return run_script("voice_notes_chat.py", unknown)
    if args.command == "ingest":
        return run_script("voice_notes_ingest.py", unknown)
    if args.command == "ideas":
        return run_script("constellation_ideas.py", unknown)
    if args.command == "realtime":
        return run_script("voice_notes_realtime.py", unknown)
    if args.command == "codex":
        return run_script("constellation_codex.py", unknown)
    if args.command == "paths":
        if args.paths_command == "show":
            return cmd_paths_show()
        if args.paths_command == "set-workspace":
            return cmd_paths_set_workspace(args.path)
        if args.paths_command == "mount-library":
            return cmd_paths_mount_library(args.path)
        if args.paths_command == "unmount-library":
            return cmd_paths_unmount_library(args.path)
        if args.paths_command == "reset":
            return cmd_paths_reset()
    if args.command == "toolbelt":
        if args.toolbelt_command == "list":
            return cmd_toolbelt_list()
        if args.toolbelt_command == "show":
            return cmd_toolbelt_show(args.name)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
