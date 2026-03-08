#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import asyncio
import base64
import contextlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from collections import defaultdict
from datetime import datetime
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
from constellation_ideas import ConstellationIdeaMiner
from constellation_codex import (
    ALLOWED_APPROVAL_MODES,
    ALLOWED_PROVIDERS,
    ALLOWED_SANDBOXES,
    DEFAULT_APPROVAL_MODE,
    DEFAULT_CODEX_COMMAND,
    DEFAULT_CODEX_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_SANDBOX,
    DEFAULT_WSL_DISTRO,
    CodexBridgeManager,
)
from constellation_runtime import ensure_runtime_layout, load_runtime_paths
from constellation_realtime_tray import run_realtime_tray
from voice_notes_toolbelt import (
    AVAILABLE_REALTIME_VOICES,
    DEFAULT_SPEECH_SPEED,
    MAX_SPEECH_SPEED,
    MIN_SPEECH_SPEED,
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
RUNTIME_PATHS = ensure_runtime_layout(load_runtime_paths())
WORKSPACE_ROOT = RUNTIME_PATHS.workspace_path
DEFAULT_SESSION_LOG_ROOT = SCRIPT_DIR / "toolbelt" / "realtime_sessions"
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


class SingleInstanceGuard:
    def __init__(self, name: str) -> None:
        self.path = Path(tempfile.gettempdir()) / f"{name}.lock"
        self.handle: Any | None = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        handle.write(b" ")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False

        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("utf-8"))
        handle.flush()
        self.handle = handle
        atexit.register(self.release)
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        with contextlib.suppress(OSError):
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            self.handle.close()
        self.handle = None

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
    "If the user explicitly asks you to build, edit, review, or wire up code, use the Codex bridge tools instead of pretending the work is already done. "
    "For small direct workspace changes like creating or editing a simple file, use the local workspace tools directly before escalating to Codex. "
    "Treat the dedicated workspace as writable scratch space for new files and experiments. "
    "Treat any mounted library roots as read-only unless the user explicitly asks to modify one of those older repos, in which case use Codex against the chosen repo. "
    "Never blank, delete, or overwrite an existing file with empty content unless the user explicitly asks for that destructive action. "
    "If no mounted libraries are configured, say so briefly and keep working inside the Constellation repo workspace. "
    "If the user asks to open a file or folder from the workspace, use the local open-path tool. "
    "If the user asks you to run or verify a small script and a direct local path is not available, use the Codex bridge rather than claiming you cannot do it. "
    "Before launching a Codex task that could change files or install software, briefly restate the target repo and intended action. "
    "Use the safest sandbox that still fits the job, and only use danger-full-access if the user clearly wants machine-level changes. "
    "If the user asks you to review the logs, introspect on the last session, or self-reflect on recent behavior, inspect the latest realtime session logs and transcript, look for errors, failed or abandoned tool calls, slow actions, timing gaps, and what the user last asked for. "
    "When that review reveals a concrete runtime bug and the user wants it fixed, implement the change rather than only describing it. "
    "Do not mention internal implementation details unless asked."
)


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def console_print(*parts: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    text = sep.join(str(part) for part in parts) + end
    stream = sys.stdout
    try:
        stream.write(text)
        if flush:
            stream.flush()
        return
    except UnicodeEncodeError:
        pass

    buffer = getattr(stream, "buffer", None)
    encoding = getattr(stream, "encoding", None) or "utf-8"
    encoded = text.encode(encoding, errors="replace")
    if buffer is not None:
        buffer.write(encoded)
        if flush:
            buffer.flush()
        return
    stream.write(encoded.decode(encoding, errors="replace"))
    if flush:
        stream.flush()


class ThinkingIndicator:
    BACKGROUND_START_DELAY_SECONDS = 0.65
    BACKGROUND_REPEAT_SECONDS = 1.6
    HANDOFF_COOLDOWN_SECONDS = 0.25

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._background_active = False
        self._background_task: asyncio.Task[None] | None = None
        self._last_handoff_at = 0.0

    def play_handoff(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_handoff_at < self.HANDOFF_COOLDOWN_SECONDS:
            return
        self._last_handoff_at = now
        asyncio.create_task(asyncio.to_thread(self._play_handoff_once))

    def start_background(self) -> None:
        if not self.enabled or self._background_active:
            return
        self._background_active = True
        self._background_task = asyncio.create_task(self._background_loop())

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.stop()

    def stop(self) -> None:
        self._background_active = False
        if self._background_task is not None:
            self._background_task.cancel()
            self._background_task = None

    async def _background_loop(self) -> None:
        try:
            await asyncio.sleep(self.BACKGROUND_START_DELAY_SECONDS)
            while self._background_active:
                await asyncio.to_thread(self._play_background_once)
                await asyncio.sleep(self.BACKGROUND_REPEAT_SECONDS)
        except asyncio.CancelledError:
            return

    @classmethod
    def _play_handoff_once(cls) -> None:
        if os.name == "nt":
            import winsound

            with contextlib.suppress(RuntimeError):
                winsound.Beep(880, 60)
            return
        console_print("\a", end="", flush=True)

    @classmethod
    def _play_background_once(cls) -> None:
        if os.name == "nt":
            cls._play_windows_wave(
                [
                    (720.0, 0.028, 0.14),
                    (510.0, 0.04, 0.1),
                ]
            )
            return
        console_print("\a", end="", flush=True)

    @classmethod
    def _play_windows_wave(cls, segments: list[tuple[float, float, float]]) -> None:
        import winsound

        sample_rate = 24_000
        frames = bytearray()
        for frequency, duration, amplitude in segments:
            sample_count = max(1, int(sample_rate * duration))
            for index in range(sample_count):
                progress = index / sample_count
                envelope = math.sin(math.pi * progress)
                sample = math.sin(2 * math.pi * frequency * progress * duration)
                value = int(32767 * amplitude * envelope * sample)
                frames.extend(struct.pack("<h", value))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(frames))
        with contextlib.suppress(RuntimeError):
            winsound.PlaySound(buffer.getvalue(), winsound.SND_MEMORY)


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

    def write_file(
        self,
        relative_path: str,
        content: str,
        *,
        overwrite: bool = True,
        create_parents: bool = True,
    ) -> dict[str, Any]:
        target = ensure_within_root(self.root, self.root / relative_path)
        if target.exists() and target.is_dir():
            return {"ok": False, "error": f"Path is a directory: {relative_path}"}
        if target.exists() and not overwrite:
            return {"ok": False, "error": f"File already exists: {relative_path}"}
        if target.exists() and not content:
            return {
                "ok": False,
                "error": (
                    "Refusing to overwrite an existing file with empty content. "
                    "Use a non-empty update or ask for an explicit deletion workflow."
                ),
            }
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "relative_path": str(target.relative_to(self.root)),
            "bytes_written": len(content.encode("utf-8")),
            "restart_recommended": target.suffix == ".py",
        }

    def replace_text(
        self,
        relative_path: str,
        old_text: str,
        new_text: str,
        *,
        count: int = 1,
    ) -> dict[str, Any]:
        target = ensure_within_root(self.root, self.root / relative_path)
        if not target.exists():
            return {"ok": False, "error": f"File not found: {relative_path}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {relative_path}"}
        text = target.read_text(encoding="utf-8")
        replacements = text.count(old_text)
        if replacements == 0:
            return {"ok": False, "error": f"Text not found in {relative_path}"}
        target.write_text(text.replace(old_text, new_text, count), encoding="utf-8")
        return {
            "ok": True,
            "relative_path": str(target.relative_to(self.root)),
            "replacements_available": replacements,
            "replacements_applied": min(replacements, count),
            "restart_recommended": target.suffix == ".py",
        }

    def open_path(self, relative_path: str) -> dict[str, Any]:
        target = ensure_within_root(self.root, self.root / relative_path)
        if not target.exists():
            return {"ok": False, "error": f"Path not found: {relative_path}"}
        try:
            if os.name == "nt":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "relative_path": str(target.relative_to(self.root)),
            "opened_in_default_app": True,
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


class SessionLogger:
    def __init__(self, *, root: Path, enabled: bool) -> None:
        self.enabled = enabled
        self.root = root.resolve()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.root / timestamp
        self.events_path = self.session_dir / "events.jsonl"
        self.transcript_path = self.session_dir / "transcript.md"
        if self.enabled:
            self.session_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_transcript(self, speaker: str, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {speaker}: {text.strip()}\n"
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def latest_session_dir(root: Path) -> Path | None:
    candidates = [entry for entry in root.iterdir() if entry.is_dir()] if root.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def build_history_reconnect_prompt(
    conversation_history: list[dict[str, str]],
    *,
    follow_up: str,
    max_turns: int = 12,
    max_chars: int = 6000,
) -> str:
    recent_turns = conversation_history[-max_turns:]
    chunks: list[str] = []
    for item in recent_turns:
        role = item.get("role", "assistant").capitalize()
        text = item.get("text", "").strip()
        if text:
            chunks.append(f"{role}: {text}")
    history_text = "\n".join(chunks)
    if len(history_text) > max_chars:
        history_text = history_text[-max_chars:]
    return (
        "Conversation continuity context. Do not repeat this context back unless useful.\n"
        f"{history_text}\n\n"
        f"Continue seamlessly from there. {follow_up}"
    ).strip()


class RealtimeAssistantContext:
    def __init__(
        self,
        *,
        runtime_paths: Any,
        session_log_root: Path,
        archive_root: Path,
        ingest_script: Path,
        codex_bridge: CodexBridgeManager,
        idea_miner: ConstellationIdeaMiner,
    ) -> None:
        self.runtime_paths = runtime_paths
        self.workspace = WorkspaceTools(runtime_paths.workspace_path)
        self.mounted_libraries = [WorkspaceTools(path) for path in runtime_paths.existing_library_paths]
        self.session_log_root = session_log_root.resolve()
        self.archive_root = archive_root.resolve()
        self.ingest_script = ingest_script.resolve()
        self.codex_bridge = codex_bridge
        self.idea_miner = idea_miner
        self.notes_index = NotesIndex(self.archive_root) if NotesIndex is not None else None

    def workspace_overview(self) -> dict[str, Any]:
        workspace_overview = self.workspace.overview()
        mounted_library_overviews = [library.overview() for library in self.mounted_libraries]
        return {
            "workspace_root": workspace_overview["workspace_root"],
            "workspace_top_level_directories": workspace_overview["top_level_directories"],
            "workspace_top_level_files": workspace_overview["top_level_files"],
            "mounted_library_roots": [overview["workspace_root"] for overview in mounted_library_overviews],
            "mounted_library_summaries": [
                {
                    "root": overview["workspace_root"],
                    "top_level_directories": overview["top_level_directories"][:20],
                    "top_level_files": overview["top_level_files"][:20],
                }
                for overview in mounted_library_overviews
            ],
            "mounted_libraries_are_read_only_via_local_tools": True,
            "codex_repository_roots": [str(root) for root in self.runtime_paths.repo_roots],
        }

    def allowed_open_roots(self) -> list[Path]:
        return [self.workspace.root, *(library.root for library in self.mounted_libraries), self.archive_root, self.session_log_root]

    def resolve_mounted_library(self, library_root: str | None = None) -> WorkspaceTools | None:
        if not self.mounted_libraries:
            return None
        requested = normalize_optional_string(library_root)
        if requested is None:
            return self.mounted_libraries[0]
        try:
            requested_path = Path(requested).expanduser().resolve()
        except OSError:
            requested_path = None
        requested_lower = requested.lower()
        for library in self.mounted_libraries:
            if requested_path is not None and library.root == requested_path:
                return library
            if library.root.name.lower() == requested_lower:
                return library
            if str(library.root).lower() == requested_lower:
                return library
        return None

    def missing_library_error(self, requested_root: str | None = None) -> dict[str, Any]:
        if not self.mounted_libraries:
            return {
                "ok": False,
                "error": "No mounted library roots are configured.",
                "hint": "Use `python constellation.py paths mount-library <path>` to add one on this machine.",
            }
        if requested_root is None:
            return {"ok": False, "error": "Mounted library root not found."}
        return {
            "ok": False,
            "error": f"Mounted library root not found: {requested_root}",
            "available_library_roots": [str(library.root) for library in self.mounted_libraries],
        }

    def open_allowed_path(self, path_value: str) -> dict[str, Any]:
        raw_value = normalize_optional_string(path_value)
        if raw_value is None:
            return {"ok": False, "error": "Path is required."}
        candidate = Path(raw_value).expanduser()
        allowed_roots = self.allowed_open_roots()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not any(resolved.is_relative_to(root) for root in allowed_roots):
                return {"ok": False, "error": f"Path is outside the allowed open roots: {resolved}"}
            if not resolved.exists():
                return {"ok": False, "error": f"Path not found: {resolved}"}
            try:
                open_path(resolved)
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "path": str(resolved), "opened_in_default_app": True}

        for root in allowed_roots:
            resolved = (root / raw_value).resolve()
            if not resolved.is_relative_to(root):
                continue
            if not resolved.exists():
                continue
            try:
                open_path(resolved)
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "path": str(resolved), "opened_in_default_app": True}
        return {"ok": False, "error": f"Path not found inside allowed roots: {raw_value}"}

    def review_recent_runtime_session(
        self,
        *,
        slow_tool_seconds: float = 8.0,
        transcript_tail_lines: int = 12,
    ) -> dict[str, Any]:
        session_dir = latest_session_dir(self.session_log_root)
        if session_dir is None:
            return {"ok": False, "error": f"No session folders found in {self.session_log_root}"}
        events_path = session_dir / "events.jsonl"
        transcript_path = session_dir / "transcript.md"
        events = load_jsonl_records(events_path)
        transcript_lines = transcript_path.read_text(encoding="utf-8").splitlines() if transcript_path.exists() else []

        pending_tools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        slow_tools: list[dict[str, Any]] = []
        failed_tools: list[dict[str, Any]] = []
        session_errors: list[dict[str, Any]] = []

        for event in events:
            event_type = str(event.get("type", ""))
            timestamp_raw = str(event.get("timestamp", ""))
            payload = event.get("payload", {})
            try:
                event_timestamp = datetime.fromisoformat(timestamp_raw)
            except ValueError:
                event_timestamp = None
            if event_type == "tool_call_started" and isinstance(payload, dict):
                tool_name = str(payload.get("tool_name", "unknown_tool"))
                pending_tools[tool_name].append(
                    {
                        "started_at": timestamp_raw,
                        "timestamp": event_timestamp,
                        "arguments": payload.get("arguments", {}),
                    }
                )
                continue
            if event_type == "tool_call_completed" and isinstance(payload, dict):
                tool_name = str(payload.get("tool_name", "unknown_tool"))
                result = payload.get("result", {})
                started_entry = pending_tools[tool_name].pop(0) if pending_tools.get(tool_name) else None
                duration_seconds: float | None = None
                if started_entry is not None and started_entry.get("timestamp") is not None and event_timestamp is not None:
                    duration_seconds = round((event_timestamp - started_entry["timestamp"]).total_seconds(), 3)
                    if duration_seconds >= slow_tool_seconds:
                        slow_tools.append(
                            {
                                "tool_name": tool_name,
                                "started_at": started_entry["started_at"],
                                "completed_at": timestamp_raw,
                                "duration_seconds": duration_seconds,
                                "arguments": started_entry.get("arguments", {}),
                            }
                        )
                has_error = False
                error_text: str | None = None
                if isinstance(result, dict):
                    has_error = result.get("ok") is False or "error" in result
                    error_text = str(result.get("error")) if "error" in result else None
                if has_error:
                    failed_tools.append(
                        {
                            "tool_name": tool_name,
                            "started_at": started_entry["started_at"] if started_entry is not None else None,
                            "completed_at": timestamp_raw,
                            "duration_seconds": duration_seconds,
                            "error": error_text,
                            "result": result,
                        }
                    )
                continue
            if event_type == "session_error" and isinstance(payload, dict):
                session_errors.append(
                    {
                        "timestamp": timestamp_raw,
                        "message": str(payload.get("message", "Unknown runtime error")),
                    }
                )

        abandoned_tools: list[dict[str, Any]] = []
        for tool_name, entries in pending_tools.items():
            for entry in entries:
                abandoned_tools.append(
                    {
                        "tool_name": tool_name,
                        "started_at": entry.get("started_at"),
                        "arguments": entry.get("arguments", {}),
                    }
                )

        return {
            "ok": True,
            "latest_session_dir": str(session_dir),
            "events_path": str(events_path),
            "transcript_path": str(transcript_path),
            "event_count": len(events),
            "session_errors": session_errors,
            "failed_tools": failed_tools,
            "slow_tools": slow_tools,
            "abandoned_tools": abandoned_tools,
            "transcript_tail": transcript_lines[-max(1, transcript_tail_lines):],
            "last_user_lines": [line for line in transcript_lines if "] user:" in line][-5:],
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "name": "get_workspace_overview",
                "description": "Return the writable workspace plus any separate read-only mounted library roots.",
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
                        "speech_speed": {
                            "type": "number",
                            "description": f"Voice speed from {MIN_SPEECH_SPEED} to {MAX_SPEECH_SPEED}.",
                        },
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
                "description": "List files or folders inside the writable Constellation workspace.",
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
                "description": "Search text files inside the writable Constellation workspace.",
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
                "description": "Read one UTF-8 text file from the writable Constellation workspace.",
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
            {
                "type": "function",
                "name": "write_file",
                "description": "Create or overwrite one UTF-8 text file inside the writable Constellation workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "content": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                        "create_parents": {"type": "boolean"},
                    },
                    "required": ["relative_path", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "replace_text_in_file",
                "description": "Replace a text snippet inside one UTF-8 text file in the writable Constellation workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["relative_path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "open_path_in_default_app",
                "description": "Open a file or folder in the default app. Relative paths are resolved inside the workspace, mounted library roots, notes archive, or session log roots. Absolute paths are allowed if they stay inside those roots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string"},
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_legacy_directory",
                "description": "List files or folders inside a mounted library root. Read-only via local tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library_root": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "search_legacy_library",
                "description": "Search text files in a mounted library root. Read-only via local tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library_root": {"type": "string"},
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
                "name": "read_legacy_file",
                "description": "Read one UTF-8 text file from a mounted library root. Read-only via local tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library_root": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000},
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "open_legacy_path",
                "description": "Open a file or folder from a mounted library root in the default app. Read-only via local tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library_root": {"type": "string"},
                        "relative_path": {"type": "string"},
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "review_recent_runtime_session",
                "description": "Inspect the latest realtime session logs and transcript to self-review what happened, including failures, slow tool calls, abandoned actions, and recent user requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slow_tool_seconds": {"type": "number", "minimum": 0.5, "maximum": 60},
                        "transcript_tail_lines": {"type": "integer", "minimum": 3, "maximum": 50},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_codex_bridge_status",
                "description": "Check whether Constellation can launch Codex CLI from this machine right now.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "type": "function",
                "name": "list_codex_repositories",
                "description": "List likely repo targets for Codex CLI work across the writable workspace and any mounted library roots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "start_codex_task",
                "description": "Queue a Codex CLI task against a chosen repo or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "repo_name_or_path": {"type": "string"},
                        "title": {"type": "string"},
                        "model": {"type": "string"},
                        "sandbox": {
                            "type": "string",
                            "enum": list(ALLOWED_SANDBOXES),
                            "description": "Codex sandbox mode.",
                        },
                        "approval_mode": {
                            "type": "string",
                            "enum": list(ALLOWED_APPROVAL_MODES),
                            "description": "Codex approval mode.",
                        },
                        "add_dirs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional extra directories to expose to Codex.",
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_codex_task_status",
                "description": "Read the latest status of one queued Codex CLI task.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_codex_tasks",
                "description": "List recent Codex CLI tasks queued through Constellation.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 30}},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "cancel_codex_task",
                "description": "Cancel a running Codex CLI task.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "restart_runtime_session",
                "description": "Restart the live runtime session after the assistant warns the user. Use when code or configuration changes need a reconnect.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "mine_notes_for_ideas",
                "description": "Extract likely build ideas from recent or matching notes using GPT-5.4.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "month": {"type": "string"},
                        "max_notes": {"type": "integer", "minimum": 1, "maximum": 20},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "build_codex_prompt_from_idea",
                "description": "Turn a chosen idea into a concrete Codex task prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_note_ids": {"type": "array", "items": {"type": "string"}},
                        "repo_name_or_path": {"type": "string"},
                        "extra_instruction": {"type": "string"},
                    },
                    "required": ["title", "summary", "source_note_ids"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "start_codex_task_from_idea",
                "description": "Build a Codex prompt from a chosen note-derived idea and queue it immediately.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_note_ids": {"type": "array", "items": {"type": "string"}},
                        "repo_name_or_path": {"type": "string"},
                        "extra_instruction": {"type": "string"},
                        "sandbox": {"type": "string", "enum": list(ALLOWED_SANDBOXES)},
                        "approval_mode": {"type": "string", "enum": list(ALLOWED_APPROVAL_MODES)},
                    },
                    "required": ["title", "summary", "source_note_ids"],
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
            return self.workspace_overview()
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
        if name == "write_file":
            return self.workspace.write_file(
                relative_path=str(args.get("relative_path", "")),
                content=str(args.get("content", "")),
                overwrite=bool(args.get("overwrite", True)),
                create_parents=bool(args.get("create_parents", True)),
            )
        if name == "replace_text_in_file":
            return self.workspace.replace_text(
                relative_path=str(args.get("relative_path", "")),
                old_text=str(args.get("old_text", "")),
                new_text=str(args.get("new_text", "")),
                count=int(args.get("count", 1)),
            )
        if name == "open_path_in_default_app":
            return self.open_allowed_path(path_value=str(args.get("relative_path", "")))
        if name == "list_legacy_directory":
            library = self.resolve_mounted_library(normalize_optional_string(args.get("library_root")))
            if library is None:
                return self.missing_library_error(normalize_optional_string(args.get("library_root")))
            return library.list_directory(
                relative_path=normalize_optional_string(args.get("relative_path")) or "",
                max_entries=int(args.get("max_entries", 40)),
            )
        if name == "search_legacy_library":
            library = self.resolve_mounted_library(normalize_optional_string(args.get("library_root")))
            if library is None:
                return self.missing_library_error(normalize_optional_string(args.get("library_root")))
            return library.search(
                query=str(args.get("query", "")),
                subpath=normalize_optional_string(args.get("subpath")),
                max_results=int(args.get("max_results", 8)),
            )
        if name == "read_legacy_file":
            library = self.resolve_mounted_library(normalize_optional_string(args.get("library_root")))
            if library is None:
                return self.missing_library_error(normalize_optional_string(args.get("library_root")))
            return library.read_file(
                relative_path=str(args.get("relative_path", "")),
                max_chars=int(args.get("max_chars", 6000)),
            )
        if name == "open_legacy_path":
            library = self.resolve_mounted_library(normalize_optional_string(args.get("library_root")))
            if library is None:
                return self.missing_library_error(normalize_optional_string(args.get("library_root")))
            return library.open_path(relative_path=str(args.get("relative_path", "")))
        if name == "review_recent_runtime_session":
            return self.review_recent_runtime_session(
                slow_tool_seconds=float(args.get("slow_tool_seconds", 8.0)),
                transcript_tail_lines=int(args.get("transcript_tail_lines", 12)),
            )
        if name == "get_codex_bridge_status":
            return self.codex_bridge.status()
        if name == "list_codex_repositories":
            return {"items": self.codex_bridge.discover_repositories(max_repos=int(args.get("limit", 20)))}
        if name == "start_codex_task":
            try:
                task = self.codex_bridge.submit_task(
                    prompt=str(args.get("prompt", "")),
                    repo_name_or_path=normalize_optional_string(args.get("repo_name_or_path")),
                    title=normalize_optional_string(args.get("title")),
                    model=normalize_optional_string(args.get("model")),
                    sandbox=str(args.get("sandbox", "workspace-write")),
                    approval_mode=str(args.get("approval_mode", DEFAULT_APPROVAL_MODE)),
                    add_dirs=[
                        str(item)
                        for item in args.get("add_dirs", [])
                        if normalize_optional_string(str(item)) is not None
                    ],
                )
            except (OSError, ValueError, FileNotFoundError, NotADirectoryError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "task": task}
        if name == "get_codex_task_status":
            try:
                return self.codex_bridge.get_task_status(str(args.get("task_id", "")))
            except FileNotFoundError as exc:
                return {"ok": False, "error": str(exc)}
        if name == "list_codex_tasks":
            return self.codex_bridge.list_tasks(limit=int(args.get("limit", 10)))
        if name == "cancel_codex_task":
            try:
                return self.codex_bridge.cancel_task(str(args.get("task_id", "")))
            except FileNotFoundError as exc:
                return {"ok": False, "error": str(exc)}
        if name == "mine_notes_for_ideas":
            try:
                return self.idea_miner.mine_ideas(
                    query=normalize_optional_string(args.get("query")),
                    month=normalize_optional_string(args.get("month")),
                    max_notes=int(args.get("max_notes", 8)),
                    limit=int(args.get("limit", 5)),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return {"ok": False, "error": str(exc)}
        if name == "build_codex_prompt_from_idea":
            try:
                return self.idea_miner.build_codex_prompt(
                    title=str(args.get("title", "")),
                    summary=str(args.get("summary", "")),
                    source_note_ids=[str(item) for item in args.get("source_note_ids", [])],
                    repo_name_or_path=normalize_optional_string(args.get("repo_name_or_path")),
                    extra_instruction=normalize_optional_string(args.get("extra_instruction")),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return {"ok": False, "error": str(exc)}
        if name == "start_codex_task_from_idea":
            try:
                prompt_payload = self.idea_miner.build_codex_prompt(
                    title=str(args.get("title", "")),
                    summary=str(args.get("summary", "")),
                    source_note_ids=[str(item) for item in args.get("source_note_ids", [])],
                    repo_name_or_path=normalize_optional_string(args.get("repo_name_or_path")),
                    extra_instruction=normalize_optional_string(args.get("extra_instruction")),
                )
                task = self.codex_bridge.submit_task(
                    prompt=prompt_payload["prompt"],
                    repo_name_or_path=normalize_optional_string(args.get("repo_name_or_path")),
                    title=str(args.get("title", "")),
                    sandbox=str(args.get("sandbox", DEFAULT_SANDBOX)),
                    approval_mode=str(args.get("approval_mode", DEFAULT_APPROVAL_MODE)),
                )
            except (OSError, ValueError, FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "prompt": prompt_payload, "task": task}

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
            console_print(f"[audio warning] input status: {status}")
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
    session_logger: SessionLogger,
    conversation_history: list[dict[str, str]],
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
    pending_restart_announcement: str | None = None
    restart_after_response = False

    console_print(f"Connecting to realtime model: {args.model}")
    console_print("Press Ctrl+C to quit.")
    session_logger.log_event(
        "session_start",
        {
            "model": args.model,
            "voice": runtime_preferences.voice,
            "greeting_enabled": greeting_enabled,
            "prompt_injected": bool(prompt),
        },
    )

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
                    session_logger.log_event("session_created", {"event_type": event_type})
                    console_print("[session] created")
                    continue
                if event_type == "session.updated":
                    session_logger.log_event("session_updated", {"voice": runtime_preferences.voice})
                    console_print("[session] updated and ready")
                    continue
                if event_type == "input_audio_buffer.speech_started":
                    player.clear()
                    thinking_indicator.stop()
                    console_print("[you] listening...")
                    continue
                if event_type == "input_audio_buffer.speech_stopped":
                    thinking_indicator.play_handoff()
                    continue
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.transcript.strip()
                    if transcript and event.item_id not in seen_user_transcripts:
                        seen_user_transcripts.add(event.item_id)
                        conversation_history.append({"role": "user", "text": transcript})
                        session_logger.log_transcript("user", transcript)
                        console_print(f"[you] {transcript}")
                    continue
                if event_type == "response.created":
                    thinking_indicator.start_background()
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
                        conversation_history.append({"role": "assistant", "text": transcript})
                        session_logger.log_transcript("assistant", transcript)
                        console_print(f"[assistant] {transcript}")
                        assistant_output_received.set()
                    assistant_partial.pop(event.item_id, None)
                    continue
                if event_type == "response.output_text.done":
                    thinking_indicator.stop()
                    if args.no_audio_output:
                        text = event.text.strip()
                        if text:
                            conversation_history.append({"role": "assistant", "text": text})
                            session_logger.log_transcript("assistant", text)
                            console_print(f"[assistant] {text}")
                            assistant_output_received.set()
                    continue
                if event_type == "response.function_call_arguments.done":
                    tool_name = function_call_names.get(event.item_id, "unknown_tool")
                    try:
                        tool_args = json.loads(event.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        tool_result = {"ok": False, "error": f"Invalid tool arguments: {exc}"}
                    else:
                        console_print(f"[tool] {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                        session_logger.log_event(
                            "tool_call_started",
                            {"tool_name": tool_name, "arguments": tool_args},
                        )
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
                            requested_speed: float | None = None
                            if "speech_speed" in tool_args:
                                requested_speed = float(tool_args["speech_speed"])
                                runtime_preferences.speech_speed = requested_speed
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
                                    reconnect_prompt = build_history_reconnect_prompt(
                                        conversation_history,
                                        follow_up=(
                                            f"Briefly confirm that you switched to the {requested_voice} voice and then continue helping seamlessly."
                                        ),
                                    )
                            runtime_preferences.clamp()
                            thinking_indicator.set_enabled(runtime_preferences.thinking_sound_enabled)
                            tool_result = {
                                "ok": True,
                                "speech_speed": runtime_preferences.speech_speed,
                                "concise_mode": runtime_preferences.concise_mode,
                                "thinking_sound_enabled": runtime_preferences.thinking_sound_enabled,
                                "voice": runtime_preferences.voice,
                            }
                            if requested_speed is not None and requested_speed != runtime_preferences.speech_speed:
                                tool_result["note"] = (
                                    f"Speech speed was clamped to {runtime_preferences.speech_speed:.2f}. "
                                    f"Realtime currently supports {MIN_SPEECH_SPEED} to {MAX_SPEECH_SPEED}."
                                )
                            if reconnect_prompt is None:
                                await conn.session.update(
                                    session=build_session_payload(args, assistant_context, runtime_preferences)
                                )
                            else:
                                tool_result["reconnect_required"] = True
                                tool_result["note"] = VOICE_CHANGE_REQUIRES_RECONNECT
                        elif tool_name == "restart_runtime_session":
                            reason = normalize_optional_string(tool_args.get("reason")) or "A runtime refresh was requested."
                            reconnect_prompt = build_history_reconnect_prompt(
                                conversation_history,
                                follow_up=(
                                    f"Briefly tell the user you restarted successfully and continue seamlessly. Context for restart: {reason}"
                                ),
                            )
                            pending_restart_announcement = (
                                "Briefly tell the user you are restarting now, that it should only take a couple of seconds, and that you will be right back."
                            )
                            tool_result = {
                                "ok": True,
                                "reconnect_required": True,
                                "note": reason,
                            }
                        elif tool_name == "get_available_voices":
                            tool_result = build_voice_catalog(runtime_preferences.voice)
                        else:
                            tool_result = assistant_context.execute_tool(tool_name, tool_args)
                    session_logger.log_event(
                        "tool_call_completed",
                        {"tool_name": tool_name, "result": tool_result},
                    )
                    await conn.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": event.call_id,
                            "output": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    if reconnect_prompt is not None:
                        if pending_restart_announcement is not None:
                            restart_after_response = True
                            await conn.response.create(
                                response={
                                    "output_modalities": ["audio"],
                                    "instructions": pending_restart_announcement,
                                }
                            )
                            pending_restart_announcement = None
                            continue
                        break
                    await conn.response.create(response={"output_modalities": ["audio"]})
                    continue
                if event_type == "response.done":
                    thinking_indicator.stop()
                    if restart_after_response:
                        break
                    if args.greeting_only and assistant_output_received.is_set():
                        break
                    if prompt and assistant_output_received.is_set() and args.no_mic:
                        break
                    continue
                if event_type == "error":
                    thinking_indicator.stop()
                    session_logger.log_event(
                        "session_error",
                        {"message": getattr(event.error, "message", "Unknown realtime error")},
                    )
                    console_print(f"[error] {event.error.message}")
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
            session_logger.log_event(
                "session_stop",
                {"voice": runtime_preferences.voice, "reconnect_requested": bool(reconnect_prompt)},
            )
    return reconnect_prompt


async def run_voice_chat(args: argparse.Namespace) -> None:
    session_logger = SessionLogger(
        root=Path(args.session_log_root).resolve(),
        enabled=not args.no_session_logging,
    )
    conversation_history: list[dict[str, str]] = []
    runtime_paths = ensure_runtime_layout(load_runtime_paths())
    requested_workspace_root = Path(args.workspace_root).resolve()
    if requested_workspace_root != runtime_paths.workspace_path:
        runtime_paths.workspace_root = str(requested_workspace_root)
        runtime_paths = ensure_runtime_layout(runtime_paths)
    assistant_context = RealtimeAssistantContext(
        runtime_paths=runtime_paths,
        session_log_root=Path(args.session_log_root).resolve(),
        archive_root=Path(args.archive_root).resolve(),
        ingest_script=Path(args.ingest_script).resolve(),
        codex_bridge=CodexBridgeManager(
            workspace_root=runtime_paths.workspace_path,
            repo_roots=runtime_paths.repo_roots,
            codex_command=args.codex_command,
            default_model=args.codex_model,
            provider=args.codex_provider,
            wsl_distro=args.codex_wsl_distro,
        ),
        idea_miner=ConstellationIdeaMiner(
            archive_root=Path(args.archive_root).resolve(),
            model=args.idea_model,
        ),
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
            session_logger,
            conversation_history,
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
    parser.add_argument(
        "--speech-speed",
        type=float,
        default=DEFAULT_SPEECH_SPEED,
        help=f"Initial speech speed from {MIN_SPEECH_SPEED} to {MAX_SPEECH_SPEED} (default: {DEFAULT_SPEECH_SPEED}).",
    )
    parser.add_argument("--verbose-responses", action="store_true", help="Allow more detailed default replies instead of concise mode.")
    parser.add_argument("--transcription-model", default="gpt-4o-mini-transcribe", help="Input transcription model.")
    parser.add_argument(
        "--workspace-root",
        default=str(WORKSPACE_ROOT),
        help="Writable workspace root for direct file tools and the default Codex scratch area.",
    )
    parser.add_argument("--archive-root", default=str(DEFAULT_NOTES_ARCHIVE_ROOT), help="Voice notes archive root.")
    parser.add_argument("--ingest-script", default=str(DEFAULT_NOTES_INGEST_SCRIPT), help="Voice notes ingest script path.")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature.")
    parser.add_argument("--max-output-tokens", default="inf", help="Max response output tokens, or inf.")
    parser.add_argument("--input-device", help="Optional sounddevice input device name or index.")
    parser.add_argument("--output-device", help="Optional sounddevice output device name or index.")
    parser.add_argument("--list-models", action="store_true", help="List realtime/audio-capable models visible to this account and exit.")
    parser.add_argument("--list-devices", action="store_true", help="List local audio devices and exit.")
    parser.add_argument("--tray", action="store_true", help="Run the Constellation realtime voice controller in the system tray.")
    parser.add_argument("--tray-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--greeting-only", action="store_true", help="Connect, request the initial greeting, print it, and exit.")
    parser.add_argument("--prompt", help="Inject one text prompt after connect, useful for testing tools without a mic.")
    parser.add_argument("--no-greeting", action="store_true", help="Do not make the assistant greet on startup.")
    parser.add_argument("--no-mic", action="store_true", help="Do not capture microphone input.")
    parser.add_argument("--no-audio-output", action="store_true", help="Do not play assistant audio; print text only.")
    parser.add_argument(
        "--no-thinking-sound",
        action="store_true",
        help="Disable the local handoff and background work indicator sounds.",
    )
    parser.add_argument(
        "--session-log-root",
        default=str(DEFAULT_SESSION_LOG_ROOT),
        help="Directory for transcript and tool-call logs.",
    )
    parser.add_argument("--no-session-logging", action="store_true", help="Disable local transcript and tool-call logging.")
    parser.add_argument(
        "--codex-command",
        default=DEFAULT_CODEX_COMMAND,
        help=f"Codex executable or alias for bridge tasks (default: {DEFAULT_CODEX_COMMAND}).",
    )
    parser.add_argument(
        "--codex-provider",
        default=DEFAULT_PROVIDER,
        choices=ALLOWED_PROVIDERS,
        help=f"How to launch Codex bridge tasks (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--codex-wsl-distro",
        default=DEFAULT_WSL_DISTRO,
        help=f"WSL distro to use when Codex provider is wsl (default: {DEFAULT_WSL_DISTRO}).",
    )
    parser.add_argument(
        "--codex-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Default Codex model for bridge tasks (default: {DEFAULT_CODEX_MODEL}).",
    )
    parser.add_argument(
        "--idea-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Model for mining note ideas and drafting Codex prompts (default: {DEFAULT_CODEX_MODEL}).",
    )
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
        console_print(
            f"{index}: {device['name']} "
            f"(in={device['max_input_channels']}, out={device['max_output_channels']}, "
            f"default_sr={device['default_samplerate']})"
        )


def print_available_voices(current_voice: str) -> None:
    catalog = build_voice_catalog(current_voice)
    console_print("Realtime voices:")
    for voice in catalog["available_realtime_voices"]:
        marker = " (current)" if voice == current_voice else ""
        console_print(f"- {voice}{marker}")
    console_print(catalog["note"])


async def async_main(args: argparse.Namespace) -> int:
    if args.list_models:
        for model in list_realtime_models_sync():
            console_print(model)
        return 0
    if args.list_voices:
        print_available_voices(args.voice)
        return 0
    if args.list_devices:
        print_audio_devices()
        return 0
    if args.tray and not args.tray_child:
        tray_guard = SingleInstanceGuard("constellation_realtime_tray")
        if not tray_guard.try_acquire():
            console_print("Constellation tray is already running.")
            return 0
        run_realtime_tray(args)
        return 0

    args.voice = normalize_voice_name(args.voice) or DEFAULT_VOICE
    args.input_device = parse_device(args.input_device)
    args.output_device = parse_device(args.output_device)
    await run_voice_chat(args)
    return 0


def main() -> int:
    configure_console_output()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        console_print("\nStopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
