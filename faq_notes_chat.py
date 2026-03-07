#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_ROOT = (SCRIPT_DIR / "faq_notes").resolve()
DEFAULT_INGEST_SCRIPT = (SCRIPT_DIR / "faq_notes_ingest.py").resolve()
DEFAULT_MODEL = "gpt-5.4"

SYSTEM_PROMPT = """You are Voice Notes Copilot.

You help the user talk to their imported microphone notes and turn rough spoken ideas
into concrete outputs, especially software plans, code prompts, specs, task breakdowns,
data analysis, workflow drafts, and code.

Rules:
- Use the local notes tools before claiming what is or is not in the archive.
- Prefer search_notes first, then read_note for the most relevant entries.
- If the user asks to import newly recorded notes, call import_notes.
- Reference note IDs and dates when summarizing source material.
- If the user wants code-related help, transform the notes into something actionable:
  a coding brief, implementation plan, prompt for a code-generation tool, pseudo-code,
  or actual code.
- Be concise and practical.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_archive_overview",
        "description": "Return archive stats, month coverage, and recent note counts.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "list_recent_notes",
        "description": "List the most recent imported notes with IDs and short previews.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "How many recent notes to return.",
                },
                "month": {
                    "type": "string",
                    "description": "Optional month filter in YYYY-MM format.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_notes",
        "description": "Search note transcripts for ideas, topics, dates, or phrases.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words or phrase to search for."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "How many matches to return.",
                },
                "month": {
                    "type": "string",
                    "description": "Optional month filter in YYYY-MM format.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_note",
        "description": "Read one note in more detail by note_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "The note_id returned from search_notes or list_recent_notes.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 400,
                    "maximum": 20000,
                    "description": "Maximum transcript characters to return.",
                },
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "refresh_notes_index",
        "description": "Reload the local notes archive after changes or imports.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "import_notes",
        "description": "Run the notes ingest tool to copy and transcribe new recordings.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Optional explicit device or folder path.",
                },
                "auto_confirm": {
                    "type": "boolean",
                    "description": "If true, passes --yes to the ingest script.",
                },
            },
            "additionalProperties": False,
        },
    },
]


@dataclass(slots=True)
class NoteRecord:
    note_id: str
    content_id: str
    month: str
    captured_at: datetime
    source_name: str
    transcript_path: Path
    audio_path: Path | None
    text: str


def compact_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text.lower())


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def resolve_archive_path(archive_root: Path, stored_path: str | None) -> Path | None:
    if not stored_path:
        return None
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    return archive_root / candidate


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class NotesIndex:
    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root
        self.manifest_path = archive_root / "manifest.json"
        self.notes: list[NoteRecord] = []
        self.notes_by_id: dict[str, NoteRecord] = {}
        self.refresh()

    def refresh(self) -> dict[str, Any]:
        self.notes = []
        self.notes_by_id = {}
        if not self.archive_root.exists():
            return self.stats()
        if not self.manifest_path.exists():
            self._load_loose_transcripts()
            return self.stats()

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        files = payload.get("files", {})
        for content_id, entry in files.items():
            transcript_path = resolve_archive_path(self.archive_root, entry.get("transcript_path"))
            if transcript_path is None or not transcript_path.exists():
                continue
            captured_at = parse_iso_datetime(entry.get("captured_at_local"))
            if captured_at is None:
                try:
                    captured_at = datetime.fromtimestamp(transcript_path.stat().st_mtime)
                except OSError:
                    continue
            try:
                text = transcript_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            note_id = transcript_path.stem
            if note_id in self.notes_by_id:
                note_id = f"{note_id}_{content_id[:8]}"
            self.notes_by_id[note_id] = NoteRecord(
                note_id=note_id,
                content_id=content_id,
                month=entry.get("month_key") or captured_at.strftime("%Y-%m"),
                captured_at=captured_at,
                source_name=entry.get("source_name") or entry.get("source_id") or "unknown",
                transcript_path=transcript_path,
                audio_path=resolve_archive_path(self.archive_root, entry.get("copied_path")),
                text=text,
            )

        self.notes = sorted(self.notes_by_id.values(), key=lambda item: item.captured_at)
        return self.stats()

    def _load_loose_transcripts(self) -> None:
        for transcript_path in sorted(self.archive_root.glob("*-*/transcripts/*.txt")):
            try:
                text = transcript_path.read_text(encoding="utf-8").strip()
                captured_at = datetime.fromtimestamp(transcript_path.stat().st_mtime)
            except OSError:
                continue
            record = NoteRecord(
                note_id=transcript_path.stem,
                content_id=transcript_path.stem,
                month=transcript_path.parent.parent.name,
                captured_at=captured_at,
                source_name="unknown",
                transcript_path=transcript_path,
                audio_path=None,
                text=text,
            )
            self.notes.append(record)
            self.notes_by_id[record.note_id] = record

    def stats(self) -> dict[str, Any]:
        counts_by_month: dict[str, int] = {}
        for note in self.notes:
            counts_by_month[note.month] = counts_by_month.get(note.month, 0) + 1
        return {
            "archive_root": str(self.archive_root),
            "manifest_path": str(self.manifest_path),
            "note_count": len(self.notes),
            "months": counts_by_month,
            "first_note": self.notes[0].captured_at.isoformat(timespec="seconds") if self.notes else None,
            "last_note": self.notes[-1].captured_at.isoformat(timespec="seconds") if self.notes else None,
        }

    def list_recent(self, limit: int = 8, month: str | None = None) -> list[dict[str, Any]]:
        items = [note for note in self.notes if month is None or note.month == month]
        return [self._note_summary(note) for note in reversed(items[-limit:])]

    def search(self, query: str, limit: int = 5, month: str | None = None) -> list[dict[str, Any]]:
        phrase = query.strip().lower()
        if not phrase:
            return self.list_recent(limit=limit, month=month)
        words = list(dict.fromkeys(tokenize(query)))
        matches: list[tuple[float, NoteRecord]] = []
        for note in self.notes:
            if month is not None and note.month != month:
                continue
            haystack = note.text.lower()
            score = 0.0
            if phrase in haystack:
                score += max(4.0, len(words) * 2.0)
            for word in words:
                hits = haystack.count(word)
                if hits:
                    score += min(hits, 5)
            if score > 0:
                score += note.captured_at.timestamp() / 1_000_000_000_000
                matches.append((score, note))
        matches.sort(key=lambda item: item[0], reverse=True)
        return [self._note_summary(note, query=query) for _, note in matches[:limit]]

    def read(self, note_id: str, max_chars: int = 8000) -> dict[str, Any]:
        record = self.notes_by_id.get(note_id)
        if record is None:
            candidates = [note for note in self.notes if note_id.lower() in note.note_id.lower()]
            if len(candidates) != 1:
                return {"found": False, "note_id": note_id, "matches": [note.note_id for note in candidates[:10]]}
            record = candidates[0]
        text = record.text
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
            truncated = True
        return {
            "found": True,
            "note_id": record.note_id,
            "captured_at": record.captured_at.isoformat(timespec="seconds"),
            "month": record.month,
            "source_name": record.source_name,
            "audio_path": str(record.audio_path) if record.audio_path else None,
            "transcript_path": str(record.transcript_path),
            "truncated": truncated,
            "text": text,
        }

    def _note_summary(self, note: NoteRecord, query: str | None = None) -> dict[str, Any]:
        return {
            "note_id": note.note_id,
            "captured_at": note.captured_at.isoformat(timespec="seconds"),
            "month": note.month,
            "source_name": note.source_name,
            "transcript_path": str(note.transcript_path),
            "preview": self._excerpt(note.text, query),
        }

    @staticmethod
    def _excerpt(text: str, query: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not query:
            return compact_text(cleaned, 260)
        lowered = cleaned.lower()
        start = lowered.find(query.lower())
        if start < 0:
            for word in tokenize(query):
                start = lowered.find(word)
                if start >= 0:
                    break
        if start < 0:
            return compact_text(cleaned, 260)
        snippet_start = max(0, start - 100)
        snippet_end = min(len(cleaned), start + 220)
        snippet = cleaned[snippet_start:snippet_end].strip()
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(cleaned):
            snippet = snippet + "..."
        return snippet


class NotesChatApp:
    def __init__(self, archive_root: Path, ingest_script: Path, model: str, temperature: float | None, max_output_tokens: int) -> None:
        self.archive_root = archive_root
        self.ingest_script = ingest_script
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.index = NotesIndex(archive_root)
        self._client: OpenAI | None = None
        self.previous_response_id: str | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def ensure_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set.")

    def print_banner(self) -> None:
        stats = self.index.stats()
        print("=== Voice Notes Chat ===")
        print(f"Archive: {stats['archive_root']}")
        print(f"Notes loaded: {stats['note_count']}")
        if stats["last_note"]:
            print(f"Latest note: {stats['last_note']}")
        print("Commands: /help  /stats  /months  /latest  /find <query>  /read <note_id>  /import [path]  /refresh  /reset  /exit")
        print()

    def run_repl(self) -> None:
        self.print_banner()
        while True:
            try:
                user_input = input("notes> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                if self.handle_command(user_input):
                    break
                continue
            self.ensure_api_key()
            try:
                reply = self.ask(user_input)
            except Exception as exc:
                print(f"[error] {exc}")
                continue
            print()
            print(reply)
            print()

    def handle_command(self, raw_command: str) -> bool:
        parts = shlex.split(raw_command)
        command = parts[0].lower()
        if command in {"/exit", "/quit"}:
            return True
        if command == "/help":
            self.print_banner()
            print("Natural language works too. Ask things like:")
            print('- "Import my new notes and tell me what coding ideas I captured."')
            print('- "Search for notes about radial menu integration and turn them into a code prompt."')
            print('- "Find my latest robotics idea and write an implementation brief."')
            print()
            return False
        if command == "/stats":
            print(json.dumps(self.index.stats(), indent=2))
            print()
            return False
        if command == "/months":
            print(json.dumps(self.index.stats()["months"], indent=2))
            print()
            return False
        if command == "/latest":
            limit = max(1, min(20, int(parts[1]))) if len(parts) > 1 else 8
            print(json.dumps(self.index.list_recent(limit=limit), indent=2))
            print()
            return False
        if command == "/find":
            query = raw_command[len(parts[0]) :].strip()
            if not query:
                print("Usage: /find <query>\n")
                return False
            print(json.dumps(self.index.search(query=query, limit=8), indent=2))
            print()
            return False
        if command == "/read":
            if len(parts) < 2:
                print("Usage: /read <note_id>\n")
                return False
            print(json.dumps(self.index.read(parts[1]), indent=2))
            print()
            return False
        if command == "/refresh":
            print(json.dumps(self.index.refresh(), indent=2))
            print()
            return False
        if command == "/reset":
            self.previous_response_id = None
            print("Conversation reset.\n")
            return False
        if command == "/import":
            source_path = parts[1] if len(parts) > 1 else None
            print(json.dumps(self.import_notes(source_path=source_path, auto_confirm=True), indent=2))
            print()
            return False
        print(f"Unknown command: {parts[0]}\n")
        return False

    def ask(self, user_text: str) -> str:
        pending_input: Any = user_text
        previous_response_id = self.previous_response_id
        while True:
            response = self.client.responses.create(
                **self._build_response_request(previous_response_id=previous_response_id, input_payload=pending_input)
            )
            previous_response_id = response.id
            tool_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not tool_calls:
                self.previous_response_id = previous_response_id
                return response.output_text.strip()
            tool_outputs = []
            for tool_call in tool_calls:
                args = json.loads(tool_call.arguments or "{}")
                print(f"[tool] {tool_call.name}({json.dumps(args)})")
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(self.execute_tool(tool_call.name, args), ensure_ascii=False),
                    }
                )
            pending_input = tool_outputs

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_archive_overview":
            return self.index.stats()
        if name == "list_recent_notes":
            return {"items": self.index.list_recent(limit=int(args.get("limit", 8)), month=normalize_optional_string(args.get("month")))}
        if name == "search_notes":
            return {"items": self.index.search(query=str(args.get("query", "")), limit=int(args.get("limit", 5)), month=normalize_optional_string(args.get("month")))}
        if name == "read_note":
            return self.index.read(note_id=str(args.get("note_id", "")), max_chars=int(args.get("max_chars", 8000)))
        if name == "refresh_notes_index":
            return self.index.refresh()
        if name == "import_notes":
            return self.import_notes(source_path=normalize_optional_string(args.get("source_path")), auto_confirm=bool(args.get("auto_confirm", True)))
        return {"error": f"Unknown tool: {name}"}

    def import_notes(self, source_path: str | None, auto_confirm: bool) -> dict[str, Any]:
        if not self.ingest_script.exists():
            return {"ok": False, "error": f"Ingest script not found: {self.ingest_script}"}
        command = [sys.executable, str(self.ingest_script)]
        if auto_confirm:
            command.append("--yes")
        if source_path:
            command.extend(["--source", source_path])
        try:
            completed = subprocess.run(command, cwd=str(self.ingest_script.parent), capture_output=True, text=True, check=False)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": command,
            "stdout_tail": compact_text(completed.stdout, 1600),
            "stderr_tail": compact_text(completed.stderr, 1600),
            "archive_stats": self.index.refresh(),
        }

    def _build_response_request(self, *, previous_response_id: str | None, input_payload: Any) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": input_payload,
            "tools": TOOLS,
            "max_output_tokens": self.max_output_tokens,
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with imported microphone notes using GPT.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Path to notes archive root.")
    parser.add_argument("--ingest-script", default=str(DEFAULT_INGEST_SCRIPT), help="Path to faq_notes_ingest.py.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature for chat responses.")
    parser.add_argument("--max-output-tokens", type=int, default=4000, help="Maximum output tokens per response.")
    parser.add_argument("--import-first", action="store_true", help="Run the ingest tool before opening chat.")
    parser.add_argument("--source", help="Optional explicit source path for --import-first or /import flows.")
    parser.add_argument("--prompt", help="Send one prompt non-interactively and exit.")
    parser.add_argument("--stats", action="store_true", help="Print archive stats and exit.")
    parser.add_argument("--list-recent", type=int, metavar="N", help="Print N recent notes and exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = NotesChatApp(
        archive_root=Path(args.archive_root).resolve(),
        ingest_script=Path(args.ingest_script).resolve(),
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    if args.import_first:
        print(json.dumps(app.import_notes(source_path=args.source, auto_confirm=True), indent=2))
        print()
    if args.stats:
        print(json.dumps(app.index.stats(), indent=2))
        return 0
    if args.list_recent:
        print(json.dumps(app.index.list_recent(limit=max(1, min(20, args.list_recent))), indent=2))
        return 0
    if args.prompt:
        app.ensure_api_key()
        print(app.ask(args.prompt))
        return 0
    app.run_repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
