#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import sounddevice as sd
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from faq_notes_chat import (
    DEFAULT_ARCHIVE_ROOT as DEFAULT_NOTES_ARCHIVE_ROOT,
    DEFAULT_INGEST_SCRIPT as DEFAULT_NOTES_INGEST_SCRIPT,
    NotesIndex,
    compact_text,
)
from voice_notes_toolbelt import (
    AVAILABLE_REALTIME_VOICES,
    DEFAULT_SPEECH_SPEED,
    VOICE_CHANGE_REQUIRES_RECONNECT,
    RuntimePreferences,
    build_voice_catalog,
    normalize_voice_name,
)

DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "alloy"
SAMPLE_RATE = 24_000
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
WORKSPACE_ROOT = SCRIPT_DIR.parent.resolve()
NOTES_IMPORT_ERROR: str | None = None

TEXT_FILE_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SESSION_INSTRUCTIONS = (
    "You are the user's live voice assistant running on their computer. "
    "Be practical, helpful, and very concise. "
    "For spoken replies, default to 1-3 short sentences unless the user asks for more detail. "
    "Use the minimum words needed. "
    "Speak English unless the user explicitly asks for another language. "
    "You can inspect the user's FAQ note archive and current coding workspace with tools. "
    "For note-storage or archive-location questions, prefer the archive overview/location tools instead of transcript search. "
    "Use tools before claiming specifics about notes or project files. "
    "If the user asks you to talk faster, slower, shorter, more detailed, or switch voices, call the runtime preference tools. "
    "If the user asks what voices are available, call the voice catalog tool. "
    "Do not mention internal implementation details unless asked."
)


class ThinkingIndicator:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._active = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if not self.enabled or self._active:
            return
        self._active = True
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._active = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while self._active:
            await asyncio.to_thread(self._beep_once)
            try:
                await asyncio.sleep(0.8)
            except asyncio.CancelledError:
                return

    @staticmethod
    def _beep_once() -> None:
        if os.name == "nt":
            import winsound

            with contextlib.suppress(RuntimeError):
                winsound.Beep(880, 60)
            return
        print("\a", end="", flush=True)


def normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_max_output_tokens(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "inf":
        return "inf"
    return int(text)


def list_realtime_models_sync() -> list[str]:
    from openai import OpenAI

    client = OpenAI()
    models = sorted({model.id for model in client.models.list().data})
    return [model for model in models if "realtime" in model.lower() or "audio" in model.lower()]


def ensure_within_root(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes workspace root: {resolved}")
    return resolved


class WorkspaceTools:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def overview(self) -> dict[str, Any]:
        directories: list[str] = []
        files: list[str] = []
        for entry in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if entry.name.startswith(".") and entry.name not in {".gitignore"}:
                continue
            if entry.is_dir():
                directories.append(entry.name)
            else:
                files.append(entry.name)
        return {
            "workspace_root": str(self.root),
            "top_level_directories": directories[:40],
            "top_level_files": files[:40],
        }

    def list_directory(self, relative_path: str = "", max_entries: int = 40) -> dict[str, Any]:
        relative = normalize_optional_string(relative_path) or "."
        target = ensure_within_root(self.root, self.root / relative)
        if not target.exists():
            return {"ok": False, "error": f"Path not found: {target}"}
        if not target.is_dir():
            return {"ok": False, "error": f"Not a directory: {target}"}
        entries = []
        for entry in sorted(target.iterdir(), key=lambda item: item.name.lower())[:max_entries]:
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "relative_path": str(entry.relative_to(self.root)),
                }
            )
        return {"ok": True, "directory": str(target.relative_to(self.root)), "entries": entries}

    def read_file(self, relative_path: str, max_chars: int = 6000) -> dict[str, Any]:
        target = ensure_within_root(self.root, self.root / relative_path)
        if not target.exists():
            return {"ok": False, "error": f"File not found: {relative_path}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {relative_path}"}
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": f"File is not UTF-8 text: {relative_path}"}
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
            truncated = True
        return {
            "ok": True,
            "relative_path": str(target.relative_to(self.root)),
            "truncated": truncated,
            "text": text,
        }

    def search(self, query: str, max_results: int = 8, subpath: str | None = None) -> dict[str, Any]:
        base = self.root if subpath is None else ensure_within_root(self.root, self.root / subpath)
        if not base.exists():
            return {"ok": False, "error": f"Search path not found: {base}"}
        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for file_path in base.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                    continue
                candidates.append(file_path)

        query_lower = query.lower().strip()
        results = []
        for file_path in candidates:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lines = text.splitlines()
            for line_number, line in enumerate(lines, start=1):
                if query_lower in line.lower():
                    results.append(
                        {
                            "relative_path": str(file_path.relative_to(self.root)),
                            "line_number": line_number,
                            "line": compact_text(line.strip(), 220),
                        }
                    )
                    if len(results) >= max_results:
                        return {"ok": True, "results": results}
        return {"ok": True, "results": results}


class RealtimeAssistantContext:
    def __init__(self, *, workspace_root: Path, archive_root: Path, ingest_script: Path) -> None:
        self.workspace = WorkspaceTools(workspace_root)
        self.archive_root = archive_root.resolve()
        self.ingest_script = ingest_script.resolve()
        self.notes_index = NotesIndex(self.archive_root) if NotesIndex is not None else None

    def tool_definitions(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "name": "get_workspace_overview",
                "description": "Return a high-level overview of the current coding workspace root.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "type": "function",
                "name": "get_runtime_preferences",
                "description": "Return current local assistant runtime preferences such as speech speed and concise mode.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "type": "function",
                "name": "set_runtime_preferences",
                "description": "Adjust local runtime preferences. Use this when the user asks you to talk faster, slower, shorter, more concise, more detailed, or to switch voices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speech_speed": {"type": "number", "description": "Voice speed from 0.75 to 2.0."},
                        "concise_mode": {"type": "boolean", "description": "Whether to keep replies very concise by default."},
                        "thinking_sound_enabled": {"type": "boolean", "description": "Whether to play a small thinking sound while waiting on a response."},
                        "voice": {
                            "type": "string",
                            "description": f"Realtime voice. Available: {', '.join(AVAILABLE_REALTIME_VOICES)}.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_available_voices",
                "description": "Return the supported realtime voices and the current voice.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "type": "function",
                "name": "list_directory",
                "description": "List files or folders inside the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "search_workspace",
                "description": "Search text files in the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "subpath": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read one UTF-8 text file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
        ]

        if self.notes_index is not None:
            tools.extend(
                [
                    {
                        "type": "function",
                        "name": "get_archive_overview",
                        "description": "Return stats about the notes archive, including the archive root path.",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    },
                    {
                        "type": "function",
                        "name": "get_notes_storage_location",
                        "description": "Return where imported audio, transcripts, and combined monthly note files are stored on disk.",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    },
                    {
                        "type": "function",
                        "name": "list_recent_notes",
                        "description": "List the most recent imported notes.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                                "month": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "search_notes",
                        "description": "Search the imported note transcripts.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                                "month": {"type": "string"},
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "read_note",
                        "description": "Read one note transcript in more detail.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "note_id": {"type": "string"},
                                "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
                            },
                            "required": ["note_id"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "refresh_notes_index",
                        "description": "Reload the notes archive after changes.",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                    },
                    {
                        "type": "function",
                        "name": "import_notes",
                        "description": "Run the existing notes ingest tool to import and transcribe new recordings.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source_path": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                ]
            )
        return tools

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_workspace_overview":
            return self.workspace.overview()
        if name == "get_runtime_preferences":
            return {"ok": False, "error": "Runtime preferences must be handled by the live session."}
        if name == "set_runtime_preferences":
            return {"ok": False, "error": "Runtime preferences must be handled by the live session."}
        if name == "get_available_voices":
            return build_voice_catalog()
        if name == "list_directory":
            return self.workspace.list_directory(
                relative_path=normalize_optional_string(args.get("relative_path")) or "",
                max_entries=int(args.get("max_entries", 40)),
            )
        if name == "search_workspace":
            return self.workspace.search(
                query=str(args.get("query", "")),
                subpath=normalize_optional_string(args.get("subpath")),
                max_results=int(args.get("max_results", 8)),
            )
        if name == "read_file":
            return self.workspace.read_file(
                relative_path=str(args.get("relative_path", "")),
                max_chars=int(args.get("max_chars", 6000)),
            )

        if self.notes_index is None:
            return {"ok": False, "error": f"Notes tools unavailable: {NOTES_IMPORT_ERROR or 'unknown import error'}"}

        if name == "get_archive_overview":
            return self.notes_index.stats()
        if name == "get_notes_storage_location":
            return self.notes_storage_location()
        if name == "list_recent_notes":
            return {
                "items": self.notes_index.list_recent(
                    limit=int(args.get("limit", 8)),
                    month=normalize_optional_string(args.get("month")),
                )
            }
        if name == "search_notes":
            return {
                "items": self.notes_index.search(
                    query=str(args.get("query", "")),
                    limit=int(args.get("limit", 5)),
                    month=normalize_optional_string(args.get("month")),
                )
            }
        if name == "read_note":
            return self.notes_index.read(
                note_id=str(args.get("note_id", "")),
                max_chars=int(args.get("max_chars", 8000)),
            )
        if name == "refresh_notes_index":
            return self.notes_index.refresh()
        if name == "import_notes":
            return self.import_notes(source_path=normalize_optional_string(args.get("source_path")))
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def import_notes(self, source_path: str | None) -> dict[str, Any]:
        if not self.ingest_script.exists():
            return {"ok": False, "error": f"Ingest script not found: {self.ingest_script}"}
        command = [sys.executable, str(self.ingest_script), "--yes"]
        if source_path:
            command.extend(["--source", source_path])
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.ingest_script.parent),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        stats = self.notes_index.refresh() if self.notes_index is not None else None
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_tail": compact_text(completed.stdout, 1600),
            "stderr_tail": compact_text(completed.stderr, 1600),
            "archive_stats": stats,
        }

    def notes_storage_location(self) -> dict[str, Any]:
        return {
            "archive_root": str(self.archive_root),
            "manifest_path": str(self.archive_root / "manifest.json"),
            "transcript_glob": str(self.archive_root / "*-*" / "transcripts" / "*.txt"),
            "monthly_combined_glob": str(self.archive_root / "*-*" / "FAQ_*.txt"),
            "audio_glob": str(self.archive_root / "*-*" / "audio" / "*"),
            "example_month_folder": str(self.archive_root / "2026-03"),
        }


class AudioPlayer:
    def __init__(self, *, output_device: str | int | None, enabled: bool) -> None:
        self.output_device = output_device
        self.enabled = enabled
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stream: sd.RawOutputStream | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=CHANNELS,
            dtype="int16",
            device=self.output_device,
        )
        self._stream.start()

    async def stop(self) -> None:
        await self.queue.put(None)
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None

    async def play_worker(self) -> None:
        if not self.enabled:
            return
        if self._stream is None:
            raise RuntimeError("Audio output stream has not been started.")
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                break
            await asyncio.to_thread(self._stream.write, chunk)

    async def enqueue(self, pcm_bytes: bytes) -> None:
        if not self.enabled:
            return
        await self.queue.put(pcm_bytes)

    def clear(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class MicCapture:
    def __init__(
        self,
        *,
        input_device: str | int | None,
        enabled: bool,
        loop: asyncio.AbstractEventLoop,
        target_queue: asyncio.Queue[bytes],
    ) -> None:
        self.input_device = input_device
        self.enabled = enabled
        self.loop = loop
        self.target_queue = target_queue
        self.stream: sd.RawInputStream | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=CHANNELS,
            dtype="int16",
            device=self.input_device,
            callback=self._callback,
        )
        self.stream.start()

    async def stop(self) -> None:
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.stop()
            with contextlib.suppress(Exception):
                self.stream.close()
            self.stream = None

    def _callback(self, indata: Any, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"[audio warning] input status: {status}")
        chunk = bytes(indata)
        self.loop.call_soon_threadsafe(self._push_chunk, chunk)

    def _push_chunk(self, chunk: bytes) -> None:
        try:
            self.target_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass


async def forward_microphone_audio(conn: Any, mic_queue: asyncio.Queue[bytes]) -> None:
    while True:
        chunk = await mic_queue.get()
        await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))


def build_runtime_instructions(preferences: RuntimePreferences) -> str:
    concise_clause = (
        "Be extremely concise. Default to short direct answers."
        if preferences.concise_mode
        else "The user currently allows more detail when it helps."
    )
    return f"{SESSION_INSTRUCTIONS} {concise_clause}"


def build_session_payload(
    args: argparse.Namespace,
    assistant_context: RealtimeAssistantContext,
    preferences: RuntimePreferences,
) -> dict[str, Any]:
    preferences.clamp()
    session: dict[str, Any] = {
        "type": "realtime",
        "instructions": build_runtime_instructions(preferences),
        "output_modalities": ["audio"],
        "tool_choice": "auto",
        "tools": assistant_context.tool_definitions(),
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": SAMPLE_RATE,
                },
                "transcription": {
                    "model": args.transcription_model,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                    "prefix_padding_ms": 250,
                    "silence_duration_ms": 500,
                },
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": SAMPLE_RATE,
                },
                "voice": preferences.voice,
                "speed": preferences.speech_speed,
            },
        },
        "max_output_tokens": parse_max_output_tokens(args.max_output_tokens),
    }
    if args.temperature is not None:
        session["temperature"] = args.temperature
    return session


async def trigger_initial_greeting(conn: Any) -> None:
    await conn.response.create(
        response={
            "output_modalities": ["audio"],
            "instructions": "In English, greet the user very briefly, say hello, and mention that you're ready to talk.",
        }
    )


async def run_single_voice_session(
    args: argparse.Namespace,
    assistant_context: RealtimeAssistantContext,
    runtime_preferences: RuntimePreferences,
    *,
    prompt: str | None,
    greeting_enabled: bool,
) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = AsyncOpenAI()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()
    player = AudioPlayer(output_device=args.output_device, enabled=not args.no_audio_output)
    mic = MicCapture(
        input_device=args.input_device,
        enabled=not args.no_mic,
        loop=loop,
        target_queue=mic_queue,
    )

    assistant_partial: dict[str, str] = {}
    seen_user_transcripts: set[str] = set()
    assistant_output_received = asyncio.Event()
    function_call_names: dict[str, str] = {}
    thinking_indicator = ThinkingIndicator(enabled=runtime_preferences.thinking_sound_enabled)
    reconnect_prompt: str | None = None

    print(f"Connecting to realtime model: {args.model}")
    print("Press Ctrl+C to quit.")

    async with client.realtime.connect(model=args.model) as conn:
        await conn.session.update(session=build_session_payload(args, assistant_context, runtime_preferences))
        await player.start()
        await mic.start()

        tasks = []
        if not args.no_mic:
            tasks.append(asyncio.create_task(forward_microphone_audio(conn, mic_queue)))
        if not args.no_audio_output:
            tasks.append(asyncio.create_task(player.play_worker()))

        if greeting_enabled:
            await trigger_initial_greeting(conn)
        if prompt:
            await conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
            await conn.response.create(response={"output_modalities": ["audio"]})

        try:
            async for event in conn:
                event_type = getattr(event, "type", "")

                if event_type == "session.created":
                    print("[session] created")
                    continue
                if event_type == "session.updated":
                    print("[session] updated and ready")
                    continue
                if event_type == "input_audio_buffer.speech_started":
                    player.clear()
                    thinking_indicator.stop()
                    print("[you] listening...")
                    continue
                if event_type == "input_audio_buffer.speech_stopped":
                    thinking_indicator.start()
                    continue
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.transcript.strip()
                    if transcript and event.item_id not in seen_user_transcripts:
                        seen_user_transcripts.add(event.item_id)
                        print(f"[you] {transcript}")
                    continue
                if event_type == "response.created":
                    thinking_indicator.start()
                    continue
                if event_type == "response.output_audio.delta":
                    thinking_indicator.stop()
                    if not args.no_audio_output:
                        await player.enqueue(base64.b64decode(event.delta))
                    continue
                if event_type == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", "") == "function_call":
                        function_call_names[item.id] = item.name
                    continue
                if event_type == "response.output_audio_transcript.delta":
                    current = assistant_partial.get(event.item_id, "")
                    assistant_partial[event.item_id] = current + event.delta
                    continue
                if event_type == "response.output_audio_transcript.done":
                    thinking_indicator.stop()
                    transcript = event.transcript.strip()
                    if transcript:
                        print(f"[assistant] {transcript}")
                        assistant_output_received.set()
                    assistant_partial.pop(event.item_id, None)
                    continue
                if event_type == "response.output_text.done":
                    thinking_indicator.stop()
                    if args.no_audio_output:
                        text = event.text.strip()
                        if text:
                            print(f"[assistant] {text}")
                            assistant_output_received.set()
                    continue
                if event_type == "response.function_call_arguments.done":
                    tool_name = function_call_names.get(event.item_id, "unknown_tool")
                    try:
                        tool_args = json.loads(event.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        tool_result = {"ok": False, "error": f"Invalid tool arguments: {exc}"}
                    else:
                        print(f"[tool] {tool_name}({json.dumps(tool_args)})")
                        if tool_name == "get_runtime_preferences":
                            tool_result = {
                                "speech_speed": runtime_preferences.speech_speed,
                                "concise_mode": runtime_preferences.concise_mode,
                                "thinking_sound_enabled": runtime_preferences.thinking_sound_enabled,
                                "voice": runtime_preferences.voice,
                                "supported_runtime_controls": [
                                    "speech_speed",
                                    "concise_mode",
                                    "thinking_sound_enabled",
                                    "voice",
                                ],
                            }
                        elif tool_name == "set_runtime_preferences":
                            if "speech_speed" in tool_args:
                                runtime_preferences.speech_speed = float(tool_args["speech_speed"])
                            if "concise_mode" in tool_args:
                                runtime_preferences.concise_mode = bool(tool_args["concise_mode"])
                            if "thinking_sound_enabled" in tool_args:
                                runtime_preferences.thinking_sound_enabled = bool(tool_args["thinking_sound_enabled"])
                            requested_voice: str | None = None
                            if "voice" in tool_args:
                                try:
                                    requested_voice = normalize_voice_name(tool_args["voice"])
                                except ValueError as exc:
                                    tool_result = {
                                        "ok": False,
                                        "error": str(exc),
                                        "available_realtime_voices": list(AVAILABLE_REALTIME_VOICES),
                                    }
                                    await conn.conversation.item.create(
                                        item={
                                            "type": "function_call_output",
                                            "call_id": event.call_id,
                                            "output": json.dumps(tool_result, ensure_ascii=False),
                                        }
                                    )
                                    await conn.response.create(response={"output_modalities": ["audio"]})
                                    continue
                                if requested_voice != runtime_preferences.voice:
                                    runtime_preferences.voice = requested_voice
                                    reconnect_prompt = (
                                        f"Briefly confirm that you switched to the {requested_voice} voice and ask how you can help next."
                                    )
                            runtime_preferences.clamp()
                            thinking_indicator.enabled = runtime_preferences.thinking_sound_enabled
                            if not runtime_preferences.thinking_sound_enabled:
                                thinking_indicator.stop()
                            tool_result = {
                                "ok": True,
                                "speech_speed": runtime_preferences.speech_speed,
                                "concise_mode": runtime_preferences.concise_mode,
                                "thinking_sound_enabled": runtime_preferences.thinking_sound_enabled,
                                "voice": runtime_preferences.voice,
                            }
                            if reconnect_prompt is None:
                                await conn.session.update(
                                    session=build_session_payload(args, assistant_context, runtime_preferences)
                                )
                            else:
                                tool_result["reconnect_required"] = True
                                tool_result["note"] = VOICE_CHANGE_REQUIRES_RECONNECT
                        elif tool_name == "get_available_voices":
                            tool_result = build_voice_catalog(runtime_preferences.voice)
                        else:
                            tool_result = assistant_context.execute_tool(tool_name, tool_args)
                    await conn.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": event.call_id,
                            "output": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    if reconnect_prompt is not None:
                        break
                    await conn.response.create(response={"output_modalities": ["audio"]})
                    continue
                if event_type == "response.done":
                    thinking_indicator.stop()
                    if args.greeting_only and assistant_output_received.is_set():
                        break
                    if prompt and assistant_output_received.is_set() and args.no_mic:
                        break
                    continue
                if event_type == "error":
                    thinking_indicator.stop()
                    print(f"[error] {event.error.message}")
                    if args.greeting_only or args.prompt:
                        break
        finally:
            thinking_indicator.stop()
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await mic.stop()
            await player.stop()
    return reconnect_prompt


async def run_voice_chat(args: argparse.Namespace) -> None:
    assistant_context = RealtimeAssistantContext(
        workspace_root=Path(args.workspace_root).resolve(),
        archive_root=Path(args.archive_root).resolve(),
        ingest_script=Path(args.ingest_script).resolve(),
    )
    runtime_preferences = RuntimePreferences(
        speech_speed=args.speech_speed,
        concise_mode=not args.verbose_responses,
        thinking_sound_enabled=not args.no_thinking_sound,
        voice=args.voice,
    )
    prompt = args.prompt
    greeting_enabled = not args.no_greeting

    while True:
        reconnect_prompt = await run_single_voice_session(
            args,
            assistant_context,
            runtime_preferences,
            prompt=prompt,
            greeting_enabled=greeting_enabled,
        )
        if reconnect_prompt is None:
            return
        prompt = reconnect_prompt
        greeting_enabled = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime speech-to-speech chat MVP using OpenAI Realtime.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Realtime model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE}).")
    parser.add_argument("--list-voices", action="store_true", help="List supported realtime voices and exit.")
    parser.add_argument("--speech-speed", type=float, default=DEFAULT_SPEECH_SPEED, help=f"Initial speech speed (default: {DEFAULT_SPEECH_SPEED}).")
    parser.add_argument("--verbose-responses", action="store_true", help="Allow more detailed default replies instead of concise mode.")
    parser.add_argument("--transcription-model", default="gpt-4o-mini-transcribe", help="Input transcription model.")
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT), help="Workspace root to expose to file tools.")
    parser.add_argument("--archive-root", default=str(DEFAULT_NOTES_ARCHIVE_ROOT), help="Voice notes archive root.")
    parser.add_argument("--ingest-script", default=str(DEFAULT_NOTES_INGEST_SCRIPT), help="Voice notes ingest script path.")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature.")
    parser.add_argument("--max-output-tokens", default="inf", help="Max response output tokens, or inf.")
    parser.add_argument("--input-device", help="Optional sounddevice input device name or index.")
    parser.add_argument("--output-device", help="Optional sounddevice output device name or index.")
    parser.add_argument("--list-models", action="store_true", help="List realtime/audio-capable models visible to this account and exit.")
    parser.add_argument("--list-devices", action="store_true", help="List local audio devices and exit.")
    parser.add_argument("--greeting-only", action="store_true", help="Connect, request the initial greeting, print it, and exit.")
    parser.add_argument("--prompt", help="Inject one text prompt after connect, useful for testing tools without a mic.")
    parser.add_argument("--no-greeting", action="store_true", help="Do not make the assistant greet on startup.")
    parser.add_argument("--no-mic", action="store_true", help="Do not capture microphone input.")
    parser.add_argument("--no-audio-output", action="store_true", help="Do not play assistant audio; print text only.")
    parser.add_argument("--no-thinking-sound", action="store_true", help="Disable the local thinking indicator sound.")
    return parser


def parse_device(value: str | None) -> str | int | None:
    value = normalize_optional_string(value)
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def print_audio_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(
            f"{index}: {device['name']} "
            f"(in={device['max_input_channels']}, out={device['max_output_channels']}, "
            f"default_sr={device['default_samplerate']})"
        )


def print_available_voices(current_voice: str) -> None:
    catalog = build_voice_catalog(current_voice)
    print("Realtime voices:")
    for voice in catalog["available_realtime_voices"]:
        marker = " (current)" if voice == current_voice else ""
        print(f"- {voice}{marker}")
    print(catalog["note"])


async def async_main(args: argparse.Namespace) -> int:
    if args.list_models:
        for model in list_realtime_models_sync():
            print(model)
        return 0
    if args.list_voices:
        print_available_voices(args.voice)
        return 0
    if args.list_devices:
        print_audio_devices()
        return 0

    args.voice = normalize_voice_name(args.voice) or DEFAULT_VOICE
    args.input_device = parse_device(args.input_device)
    args.output_device = parse_device(args.output_device)
    await run_voice_chat(args)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
