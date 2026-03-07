#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constellation launcher for notes, chat, realtime voice, and toolbelt workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat", help="Open the Constellation notes chat.")
    subparsers.add_parser("ingest", help="Import and transcribe new recordings.")
    subparsers.add_parser("realtime", help="Start the live speech-to-speech interface.")
    subparsers.add_parser("codex", help="Inspect or queue Codex CLI tasks through Constellation.")

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
    if args.command == "realtime":
        return run_script("voice_notes_realtime.py", unknown)
    if args.command == "codex":
        return run_script("constellation_codex.py", unknown)
    if args.command == "toolbelt":
        if args.toolbelt_command == "list":
            return cmd_toolbelt_list()
        if args.toolbelt_command == "show":
            return cmd_toolbelt_show(args.name)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
