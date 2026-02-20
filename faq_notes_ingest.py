#!/usr/bin/env python3
"""
FAQ note ingestor for USB recorders.

Workflow:
1) Detect connected devices (Windows + Ubuntu/Linux mount points)
2) Scan for media files
3) Skip known files using a fast metadata index, hash only unknown candidates
4) Copy new files into archive_root/YYYY-MM/audio/
5) Transcribe with OpenAI Whisper (whisper-1), with ffmpeg normalization/chunking
6) Write three monthly transcript files:
   - Combined: FAQ_YYYY-MM.txt (time-ordered across all devices)
   - Per device: FAQ_YYYY-MM_device_1.txt, FAQ_YYYY-MM_device_2.txt (and so on)
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

MODEL = "whisper-1"
LIMIT_BYTES = 25 * 1024 * 1024
AUDIO_OPTS = ["-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "192k"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_SUBDIR = "faq_notes"
FAST_FINGERPRINT_PREFIX = "fpv1"
FAST_FINGERPRINT_SAMPLE_BYTES = 1024 * 1024
ACCOUNT_SCOPE_PREFIX = "acct::"
DEFAULT_ACCOUNT_ID = "default"
SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".amr",
    ".avi",
    ".flac",
    ".m2ts",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".oga",
    ".ogg",
    ".opus",
    ".ts",
    ".wav",
    ".webm",
    ".wma",
    ".3gp",
}
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, TimeoutError)


@dataclass
class Device:
    device_id: str
    name: str
    path: Path


class SourceUnavailableError(RuntimeError):
    """Raised when the currently selected source device disappears mid-run."""


class ManifestStore(Protocol):
    def load(self) -> dict[str, Any]:
        ...

    def save(self, payload: dict[str, Any]) -> None:
        ...


class JsonManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 2,
                "files": {},
                "quick_index": {},
                "devices": {},
                "fast_index": {},
            }

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest root must be an object")
            payload.setdefault("schema_version", 2)
            payload.setdefault("files", {})
            payload.setdefault("quick_index", {})
            payload.setdefault("devices", {})
            payload.setdefault("fast_index", {})
            if payload["schema_version"] < 2:
                payload["schema_version"] = 2
            return payload
        except Exception as exc:
            raise RuntimeError(f"Failed to read manifest at {self.path}: {exc}") from exc

    def save(self, payload: dict[str, Any]) -> None:
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class PostgresManifestStore:
    def __init__(self, database_url: str, account_id: str) -> None:
        normalized_account = account_id.strip()
        if not normalized_account:
            raise SystemExit("PostgreSQL backend requires a non-empty account ID.")

        self.database_url = database_url
        self.account_id = normalized_account
        self._scope_prefix = f"{ACCOUNT_SCOPE_PREFIX}{self.account_id}::"
        self._psycopg = None
        self._schema_ready = False
        self._last_saved: dict[str, Any] | None = None

    def _scope_token(self, raw: str) -> str:
        return f"{self._scope_prefix}{raw}"

    def _unscope_token(self, stored: str) -> str | None:
        if stored.startswith(ACCOUNT_SCOPE_PREFIX):
            if stored.startswith(self._scope_prefix):
                return stored[len(self._scope_prefix) :]
            return None
        if self.account_id == DEFAULT_ACCOUNT_ID:
            return stored
        return None

    def _driver(self) -> Any:
        if self._psycopg is not None:
            return self._psycopg
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "PostgreSQL backend requires 'psycopg'. Install with: pip install psycopg[binary]"
            ) from exc
        self._psycopg = psycopg
        return psycopg

    def _connect(self) -> Any:
        psycopg = self._driver()
        return psycopg.connect(self.database_url)

    def _ensure_schema(self, conn: Any) -> None:
        if self._schema_ready:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_devices (
                    device_id TEXT PRIMARY KEY,
                    alias TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    auto_approved BOOLEAN NOT NULL DEFAULT FALSE,
                    approved_at_local TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_files (
                    content_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    device_alias TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    copied_path TEXT NOT NULL,
                    transcript_path TEXT NOT NULL,
                    captured_at_local TEXT NOT NULL,
                    processed_at_local TEXT NOT NULL,
                    month_key TEXT NOT NULL,
                    dedupe_mode TEXT NOT NULL,
                    fast_fingerprint TEXT UNIQUE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_quick_index (
                    source_id TEXT NOT NULL,
                    quick_key TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    PRIMARY KEY (source_id, quick_key)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_faq_files_month_key ON faq_files (month_key)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_faq_quick_index_source_id ON faq_quick_index (source_id)"
            )
        conn.commit()
        self._schema_ready = True

    def load(self) -> dict[str, Any]:
        devices: dict[str, Any] = {}
        files: dict[str, Any] = {}
        quick_index: dict[str, dict[str, str]] = defaultdict(dict)
        fast_index: dict[str, str] = {}

        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT device_id, alias, display_name, auto_approved, approved_at_local
                    FROM faq_devices
                    """
                )
                for row in cur.fetchall():
                    device_id, alias, display_name, auto_approved, approved_at_local = row
                    unscoped_device_id = self._unscope_token(str(device_id))
                    unscoped_alias = self._unscope_token(str(alias))
                    if unscoped_device_id is None or unscoped_alias is None:
                        continue
                    devices[unscoped_device_id] = {
                        "alias": unscoped_alias,
                        "display_name": str(display_name),
                        "auto_approved": bool(auto_approved),
                        "approved_at_local": str(approved_at_local) if approved_at_local else None,
                    }

                cur.execute(
                    """
                    SELECT
                        content_id,
                        source_id,
                        source_name,
                        device_alias,
                        source_path,
                        copied_path,
                        transcript_path,
                        captured_at_local,
                        processed_at_local,
                        month_key,
                        dedupe_mode,
                        fast_fingerprint
                    FROM faq_files
                    """
                )
                for row in cur.fetchall():
                    (
                        content_id,
                        source_id,
                        source_name,
                        device_alias,
                        source_path,
                        copied_path,
                        transcript_path,
                        captured_at_local,
                        processed_at_local,
                        month_key,
                        dedupe_mode,
                        fast_fingerprint,
                    ) = row
                    unscoped_content_id = self._unscope_token(str(content_id))
                    unscoped_source_id = self._unscope_token(str(source_id))
                    unscoped_device_alias = self._unscope_token(str(device_alias))
                    if (
                        unscoped_content_id is None
                        or unscoped_source_id is None
                        or unscoped_device_alias is None
                    ):
                        continue

                    fp = ""
                    if fast_fingerprint:
                        unscoped_fingerprint = self._unscope_token(str(fast_fingerprint).strip())
                        if unscoped_fingerprint is None:
                            continue
                        fp = unscoped_fingerprint

                    files[unscoped_content_id] = {
                        "source_id": unscoped_source_id,
                        "source_name": str(source_name),
                        "device_alias": unscoped_device_alias,
                        "source_path": str(source_path),
                        "copied_path": str(copied_path),
                        "transcript_path": str(transcript_path),
                        "captured_at_local": str(captured_at_local),
                        "processed_at_local": str(processed_at_local),
                        "month_key": str(month_key),
                        "content_id": unscoped_content_id,
                        "dedupe_mode": str(dedupe_mode),
                        "fast_fingerprint": fp,
                    }
                    if fp and fp not in fast_index:
                        fast_index[fp] = unscoped_content_id

                cur.execute("SELECT source_id, quick_key, content_id FROM faq_quick_index")
                for row in cur.fetchall():
                    source_id, quick_key, content_id = row
                    unscoped_source_id = self._unscope_token(str(source_id))
                    unscoped_content_id = self._unscope_token(str(content_id))
                    if unscoped_source_id is None or unscoped_content_id is None:
                        continue
                    quick_index[unscoped_source_id][str(quick_key)] = unscoped_content_id

        payload = {
            "schema_version": 2,
            "files": files,
            "quick_index": dict(quick_index),
            "devices": devices,
            "fast_index": fast_index,
        }
        self._last_saved = copy.deepcopy(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        current_devices: dict[str, Any] = payload.get("devices", {})
        current_files: dict[str, Any] = payload.get("files", {})
        current_quick_index: dict[str, Any] = payload.get("quick_index", {})

        previous = self._last_saved or {
            "devices": {},
            "files": {},
            "quick_index": {},
        }
        previous_devices: dict[str, Any] = previous.get("devices", {})
        previous_files: dict[str, Any] = previous.get("files", {})
        previous_quick_index: dict[str, Any] = previous.get("quick_index", {})

        device_rows: list[tuple[Any, ...]] = []
        for device_id, meta in current_devices.items():
            if not isinstance(meta, dict):
                continue
            if previous_devices.get(device_id) == meta:
                continue
            alias = str(meta.get("alias", "")).strip()
            if not alias:
                continue
            approved_at_local = meta.get("approved_at_local")
            device_rows.append(
                (
                    self._scope_token(str(device_id)),
                    self._scope_token(alias),
                    str(meta.get("display_name", "")),
                    bool(meta.get("auto_approved", False)),
                    str(approved_at_local) if approved_at_local else None,
                )
            )

        file_rows: list[tuple[Any, ...]] = []
        for content_id, meta in current_files.items():
            if not isinstance(meta, dict):
                continue
            if previous_files.get(content_id) == meta:
                continue
            fingerprint = str(meta.get("fast_fingerprint", "")).strip()
            file_rows.append(
                (
                    self._scope_token(str(content_id)),
                    self._scope_token(str(meta.get("source_id", ""))),
                    str(meta.get("source_name", "")),
                    self._scope_token(str(meta.get("device_alias", ""))),
                    str(meta.get("source_path", "")),
                    str(meta.get("copied_path", "")),
                    str(meta.get("transcript_path", "")),
                    str(meta.get("captured_at_local", "")),
                    str(meta.get("processed_at_local", "")),
                    str(meta.get("month_key", "")),
                    str(meta.get("dedupe_mode", "")),
                    self._scope_token(fingerprint) if fingerprint else None,
                )
            )

        quick_rows: list[tuple[str, str, str]] = []
        for source_id, source_map in current_quick_index.items():
            if not isinstance(source_map, dict):
                continue
            prev_source_map = previous_quick_index.get(source_id, {})
            for quick_key, content_id in source_map.items():
                if prev_source_map.get(quick_key) == content_id:
                    continue
                quick_rows.append(
                    (
                        self._scope_token(str(source_id)),
                        str(quick_key),
                        self._scope_token(str(content_id)),
                    )
                )

        if not device_rows and not file_rows and not quick_rows:
            return

        with self._connect() as conn:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                if device_rows:
                    cur.executemany(
                        """
                        INSERT INTO faq_devices (
                            device_id, alias, display_name, auto_approved, approved_at_local
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (device_id) DO UPDATE
                        SET
                            alias = EXCLUDED.alias,
                            display_name = EXCLUDED.display_name,
                            auto_approved = EXCLUDED.auto_approved,
                            approved_at_local = EXCLUDED.approved_at_local
                        """,
                        device_rows,
                    )
                if file_rows:
                    cur.executemany(
                        """
                        INSERT INTO faq_files (
                            content_id,
                            source_id,
                            source_name,
                            device_alias,
                            source_path,
                            copied_path,
                            transcript_path,
                            captured_at_local,
                            processed_at_local,
                            month_key,
                            dedupe_mode,
                            fast_fingerprint
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_id) DO UPDATE
                        SET
                            source_id = EXCLUDED.source_id,
                            source_name = EXCLUDED.source_name,
                            device_alias = EXCLUDED.device_alias,
                            source_path = EXCLUDED.source_path,
                            copied_path = EXCLUDED.copied_path,
                            transcript_path = EXCLUDED.transcript_path,
                            captured_at_local = EXCLUDED.captured_at_local,
                            processed_at_local = EXCLUDED.processed_at_local,
                            month_key = EXCLUDED.month_key,
                            dedupe_mode = EXCLUDED.dedupe_mode,
                            fast_fingerprint = EXCLUDED.fast_fingerprint
                        """,
                        file_rows,
                    )
                if quick_rows:
                    cur.executemany(
                        """
                        INSERT INTO faq_quick_index (source_id, quick_key, content_id)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (source_id, quick_key) DO UPDATE
                        SET content_id = EXCLUDED.content_id
                        """,
                        quick_rows,
                    )
            conn.commit()

        self._last_saved = copy.deepcopy(payload)


def default_worker_count() -> int:
    return max(2, min(4, os.cpu_count() or 2))


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"Missing required tool on PATH: {tool}")


def run_checked(cmd: list[str], capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding=encoding)
    tmp_path.replace(path)


def is_disconnect_error(exc: OSError) -> bool:
    disconnect_errnos = {
        errno.ENODEV,
        errno.EIO,
        errno.ENXIO,
        getattr(errno, "ESTALE", errno.EIO),
    }
    disconnect_winerrors = {
        21,    # ERROR_NOT_READY
        64,    # ERROR_NETNAME_DELETED
        1117,  # ERROR_IO_DEVICE
        1167,  # ERROR_DEVICE_NOT_CONNECTED
    }
    if getattr(exc, "errno", None) in disconnect_errnos:
        return True
    if getattr(exc, "winerror", None) in disconnect_winerrors:
        return True
    return False


def ensure_source_available(source: Device) -> None:
    if not source.path.exists():
        raise SourceUnavailableError(f"Source is not available: {source.path}")


def convert_source_io_error(source: Device, file_path: Path, exc: OSError) -> SourceUnavailableError:
    if is_disconnect_error(exc) or not source.path.exists():
        return SourceUnavailableError(f"Source disconnected while reading {file_path}")
    return SourceUnavailableError(f"Source I/O error while reading {file_path}: {exc}")


def copy_file_atomic(src: Path, dst: Path, expected_size: int | None = None, chunk_size: int = 4 * 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    bytes_written = 0
    try:
        with src.open("rb") as in_file, tmp_path.open("wb") as out_file:
            while True:
                chunk = in_file.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_written += len(chunk)
            out_file.flush()
            os.fsync(out_file.fileno())

        if expected_size is not None and bytes_written != expected_size:
            raise RuntimeError(
                f"Incomplete copy for {src}: wrote {bytes_written} bytes, expected {expected_size}"
            )

        try:
            shutil.copystat(src, tmp_path)
        except OSError:
            pass
        tmp_path.replace(dst)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def probe_audio_params(path: Path) -> tuple[int | None, int | None]:
    try:
        result = run_checked(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels,sample_rate",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(result.stdout or "{}")
        stream = (payload.get("streams") or [None])[0] or {}
        channels = int(stream["channels"]) if stream.get("channels") else None
        sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
        return channels, sample_rate
    except Exception:
        return None, None


def is_mp3_16k_mono(path: Path) -> bool:
    if path.suffix.lower() != ".mp3":
        return False
    channels, sample_rate = probe_audio_params(path)
    return channels == 1 and sample_rate == 16000


def media_duration_seconds(path: Path) -> float:
    result = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def normalize_audio(src: Path, out_dir: Path) -> Path:
    if is_mp3_16k_mono(src):
        return src

    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = out_dir / f"{src.stem}.whisper.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-sn",
        "-map",
        "0:a:0",
        *AUDIO_OPTS,
        str(normalized),
    ]
    try:
        run_checked(cmd)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "matches no streams" in stderr or "Stream specifier" in stderr:
            raise RuntimeError(f"No audio stream found in {src}") from exc
        raise
    return normalized


def chunk_audio(path: Path, out_dir: Path) -> list[Path]:
    if path.stat().st_size <= LIMIT_BYTES:
        return [path]

    parts = math.ceil(path.stat().st_size / LIMIT_BYTES)
    part_seconds = math.ceil(media_duration_seconds(path) / parts)
    chunk_dir = out_dir / f"{path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunk_dir / "part_%03d.mp3"

    run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(part_seconds),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    chunk_paths = sorted(chunk_dir.glob("part_*.mp3"))
    if not chunk_paths:
        raise RuntimeError(f"Failed to create chunks for {path}")
    return chunk_paths


def transcribe_part(client: OpenAI, part_path: Path, max_attempts: int = 3) -> str:
    delay_seconds = 2
    for attempt in range(1, max_attempts + 1):
        try:
            with part_path.open("rb") as f:
                response = client.audio.transcriptions.create(
                    model=MODEL,
                    file=f,
                    timeout=300,
                )
            return response.text.strip()
        except RETRYABLE_ERRORS as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"Transcription failed for {part_path}: {exc}") from exc
            time.sleep(delay_seconds)
            delay_seconds *= 2
    raise RuntimeError(f"Transcription failed for {part_path}")


def transcribe_media_file(client: OpenAI, media_path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="faq_transcribe_") as tmp:
        tmp_dir = Path(tmp)
        normalized = normalize_audio(media_path, tmp_dir)
        chunks = chunk_audio(normalized, tmp_dir)
        texts: list[str] = []
        for idx, chunk_path in enumerate(chunks, start=1):
            print(f"  - Transcribing chunk {idx}/{len(chunks)}: {media_path.name}")
            texts.append(transcribe_part(client, chunk_path))
        return "\n\n".join(t for t in texts if t).strip()


def transcribe_media_file_with_api_key(api_key: str, media_path: Path) -> str:
    client = OpenAI(api_key=api_key)
    return transcribe_media_file(client, media_path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sampled_file_fingerprint(
    path: Path,
    file_size: int | None = None,
    sample_bytes: int = FAST_FINGERPRINT_SAMPLE_BYTES,
) -> str:
    size = file_size if file_size is not None else path.stat().st_size
    digest = hashlib.blake2b(digest_size=20)
    digest.update(size.to_bytes(8, "little", signed=False))

    with path.open("rb") as f:
        if size <= sample_bytes * 3:
            digest.update(f.read())
        else:
            digest.update(f.read(sample_bytes))

            middle_offset = max((size // 2) - (sample_bytes // 2), 0)
            f.seek(middle_offset)
            digest.update(f.read(sample_bytes))

            tail_offset = max(size - sample_bytes, 0)
            f.seek(tail_offset)
            digest.update(f.read(sample_bytes))

    return f"{FAST_FINGERPRINT_PREFIX}:{size}:{digest.hexdigest()}"


def load_manifest(store: ManifestStore) -> dict[str, Any]:
    payload = store.load()
    payload.setdefault("schema_version", 2)
    payload.setdefault("files", {})
    payload.setdefault("quick_index", {})
    payload.setdefault("devices", {})
    payload.setdefault("fast_index", {})
    if payload["schema_version"] < 2:
        payload["schema_version"] = 2
    return payload


def save_manifest(store: ManifestStore, payload: dict[str, Any]) -> None:
    store.save(payload)


def resolve_postgres_account_id(args: argparse.Namespace) -> str:
    account_id = (args.account_id or os.getenv("FAQ_NOTES_ACCOUNT_ID") or "").strip()
    if not account_id:
        raise SystemExit(
            "PostgreSQL backend selected but no account ID was provided. "
            "Set --account-id or FAQ_NOTES_ACCOUNT_ID."
        )
    return account_id


def create_manifest_store(args: argparse.Namespace, manifest_path: Path) -> ManifestStore:
    if args.storage_backend == "json":
        return JsonManifestStore(manifest_path)

    db_url = (
        args.database_url
        or os.getenv("FAQ_NOTES_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not db_url:
        raise SystemExit(
            "PostgreSQL backend selected but no database URL was provided. "
            "Set --database-url or FAQ_NOTES_DATABASE_URL."
        )
    account_id = resolve_postgres_account_id(args)
    return PostgresManifestStore(db_url, account_id)


def merge_manifest_data(target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    target_devices: dict[str, Any] = target.setdefault("devices", {})
    for source_id, meta in incoming.get("devices", {}).items():
        if isinstance(meta, dict):
            target_devices[source_id] = meta

    target_files: dict[str, Any] = target.setdefault("files", {})
    for content_id, meta in incoming.get("files", {}).items():
        if isinstance(meta, dict):
            target_files[content_id] = meta

    target_quick: dict[str, Any] = target.setdefault("quick_index", {})
    for source_id, entries in incoming.get("quick_index", {}).items():
        if not isinstance(entries, dict):
            continue
        bucket: dict[str, Any] = target_quick.setdefault(source_id, {})
        for quick_key, content_id in entries.items():
            bucket[quick_key] = content_id

    fast_index: dict[str, str] = {}
    for content_id, meta in target_files.items():
        if not isinstance(meta, dict):
            continue
        fingerprint = str(meta.get("fast_fingerprint", "")).strip()
        if fingerprint and fingerprint not in fast_index:
            fast_index[fingerprint] = str(content_id)
    target["fast_index"] = fast_index
    target["schema_version"] = 2
    return target


def migrate_manifest_to_postgres(args: argparse.Namespace, manifest_path: Path) -> None:
    if args.storage_backend != "postgres":
        raise SystemExit("--migrate-manifest requires --storage-backend postgres.")
    if not manifest_path.exists():
        print(f"No local manifest found at {manifest_path}; skipping migration.")
        return

    json_store = JsonManifestStore(manifest_path)
    source_manifest = load_manifest(json_store)
    target_store = create_manifest_store(args, manifest_path)
    target_manifest = load_manifest(target_store)
    merged = merge_manifest_data(target_manifest, source_manifest)
    save_manifest(target_store, merged)
    print(
        f"Migrated manifest to PostgreSQL: {len(source_manifest.get('files', {}))} file record(s), "
        f"{len(source_manifest.get('devices', {}))} device(s)."
    )


def get_or_create_device_alias(manifest: dict[str, Any], source_id: str, display_name: str) -> str:
    devices: dict[str, Any] = manifest.setdefault("devices", {})
    current = devices.get(source_id)
    aliases = {str(v.get("alias", "")) for v in devices.values()}
    if current:
        current["display_name"] = display_name
        alias = str(current.get("alias", "")).strip()
        if not alias:
            idx = 1
            while f"device_{idx}" in aliases:
                idx += 1
            alias = f"device_{idx}"
            current["alias"] = alias
        current.setdefault("auto_approved", False)
        return alias

    idx = 1
    while f"device_{idx}" in aliases:
        idx += 1
    alias = f"device_{idx}"
    devices[source_id] = {
        "alias": alias,
        "display_name": display_name,
        "auto_approved": False,
    }
    return alias


def make_quick_key(source_root: Path, file_path: Path, size: int, mtime_ns: int) -> str:
    try:
        relative = file_path.relative_to(source_root).as_posix().lower()
    except ValueError:
        relative = file_path.name.lower()
    return f"{relative}|{size}|{mtime_ns}"


def write_entries_file(path: Path, month_key: str, entries: list[dict[str, Any]]) -> None:
    lines: list[str] = [f"# FAQ Notes {month_key}\n"]
    last_day = ""
    for entry in entries:
        captured_at: datetime = entry["captured_at"]
        day_key = captured_at.strftime("%Y-%m-%d")
        if day_key != last_day:
            lines.append(f"=== {day_key} ===\n")
            last_day = day_key
        lines.append(f"[{captured_at.strftime('%H:%M:%S')}]\n")
        lines.append(f"{entry['text'].strip()}\n\n")
    atomic_write_text(path, "".join(lines), encoding="utf-8")


def resolve_archive_root(raw_path: str) -> Path:
    archive_path = Path(raw_path).expanduser()
    if archive_path.is_absolute():
        return archive_path.resolve()
    return (SCRIPT_DIR / archive_path).resolve()


def resolve_manifest_file_path(archive_root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        return (archive_root / candidate).resolve()
    if candidate.exists():
        return candidate.resolve()

    parts_lower = [part.lower() for part in candidate.parts]
    if "faq_notes" in parts_lower:
        marker = parts_lower.index("faq_notes")
        tail = Path(*candidate.parts[marker + 1 :])
        remapped = (archive_root / tail).resolve()
        if remapped.exists():
            return remapped
    return candidate


def path_for_manifest(archive_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(archive_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def normalize_manifest_paths(archive_root: Path, manifest: dict[str, Any]) -> bool:
    files: dict[str, Any] = manifest.get("files", {})
    changed = False
    for meta in files.values():
        if not isinstance(meta, dict):
            continue
        for key in ("copied_path", "transcript_path"):
            raw_value = meta.get(key)
            if not raw_value:
                continue
            resolved = resolve_manifest_file_path(archive_root, str(raw_value))
            normalized = path_for_manifest(archive_root, resolved)
            if normalized != raw_value:
                meta[key] = normalized
                changed = True
    return changed


def refresh_fast_index(archive_root: Path, manifest: dict[str, Any]) -> bool:
    files: dict[str, Any] = manifest.get("files", {})
    rebuilt: dict[str, str] = {}
    changed = False

    for content_id, meta in files.items():
        if not isinstance(meta, dict):
            continue

        fingerprint = str(meta.get("fast_fingerprint", "")).strip()
        if not fingerprint:
            copied_path_raw = meta.get("copied_path")
            if copied_path_raw:
                copied_path = resolve_manifest_file_path(archive_root, str(copied_path_raw))
                if copied_path.exists():
                    try:
                        fingerprint = sampled_file_fingerprint(copied_path)
                    except Exception:
                        fingerprint = ""
                    if fingerprint:
                        meta["fast_fingerprint"] = fingerprint
                        changed = True

        if fingerprint and fingerprint not in rebuilt:
            rebuilt[fingerprint] = content_id

    if manifest.get("fast_index") != rebuilt:
        manifest["fast_index"] = rebuilt
        changed = True
    return changed


def rebuild_month_outputs(archive_root: Path, manifest: dict[str, Any], month_key: str) -> None:
    files: dict[str, Any] = manifest.get("files", {})
    month_entries: list[dict[str, Any]] = []

    for digest, meta in files.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("month_key") != month_key:
            continue
        transcript_path_raw = meta.get("transcript_path")
        if not transcript_path_raw:
            continue
        transcript_path = resolve_manifest_file_path(archive_root, str(transcript_path_raw))
        if not transcript_path.exists():
            continue
        captured_raw = meta.get("captured_at_local")
        if not captured_raw:
            continue
        try:
            captured_at = datetime.fromisoformat(str(captured_raw))
        except ValueError:
            continue
        text = transcript_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        month_entries.append(
            {
                "digest": digest,
                "captured_at": captured_at,
                "device_alias": str(meta.get("device_alias", "device_unknown")),
                "text": text,
            }
        )

    if not month_entries:
        return

    month_entries.sort(
        key=lambda e: (
            e["captured_at"].isoformat(timespec="seconds"),
            e["device_alias"],
            e["digest"],
        )
    )

    month_dir = archive_root / month_key
    combined_path = month_dir / f"FAQ_{month_key}.txt"
    write_entries_file(combined_path, month_key, month_entries)

    by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in month_entries:
        by_device[entry["device_alias"]].append(entry)

    for device_alias, entries in by_device.items():
        device_path = month_dir / f"FAQ_{month_key}_{device_alias}.txt"
        write_entries_file(device_path, month_key, entries)


def get_windows_drive_type(root: str) -> int:
    return ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))


def get_windows_volume_info(root: str) -> tuple[str | None, int | None]:
    kernel32 = ctypes.windll.kernel32
    volume_name = ctypes.create_unicode_buffer(261)
    fs_name = ctypes.create_unicode_buffer(261)
    serial = ctypes.c_uint(0)
    max_component_length = ctypes.c_uint(0)
    fs_flags = ctypes.c_uint(0)

    ok = kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(root),
        volume_name,
        len(volume_name),
        ctypes.byref(serial),
        ctypes.byref(max_component_length),
        ctypes.byref(fs_flags),
        fs_name,
        len(fs_name),
    )
    if not ok:
        return None, None
    label = volume_name.value.strip() if volume_name.value else None
    return label, int(serial.value)


def discover_windows_devices() -> list[Device]:
    devices: list[Device] = []
    system_root = Path(os.environ.get("SystemDrive", "C:") + "\\").resolve()
    for code in range(ord("A"), ord("Z") + 1):
        drive = f"{chr(code)}:\\"
        drive_path = Path(drive)
        if not drive_path.exists():
            continue
        if drive_path.resolve() == system_root:
            continue

        drive_type = get_windows_drive_type(drive)
        if drive_type not in {2, 3}:
            continue

        label, serial = get_windows_volume_info(drive)
        stable_id = f"winvol:{serial:08X}" if serial is not None else f"win:{drive}"
        display_label = label if label else "NO_LABEL"
        drive_kind = "Removable" if drive_type == 2 else "Fixed"
        devices.append(
            Device(
                device_id=stable_id,
                name=f"{drive} [{display_label}] ({drive_kind})",
                path=drive_path,
            )
        )
    return devices


def linux_mount_uuid(path: Path) -> str | None:
    if shutil.which("findmnt") is None:
        return None
    try:
        result = run_checked(["findmnt", "-no", "UUID", "-T", str(path)])
        uuid = result.stdout.strip()
        return uuid or None
    except Exception:
        return None


def discover_linux_devices() -> list[Device]:
    devices: list[Device] = []
    candidates = ("/media", "/run/media", "/mnt")
    for base in candidates:
        base_path = Path(base)
        if not base_path.exists():
            continue

        probe_paths: set[Path] = {base_path}
        try:
            for child in base_path.iterdir():
                if not child.is_dir():
                    continue
                probe_paths.add(child)
                try:
                    for grandchild in child.iterdir():
                        if grandchild.is_dir():
                            probe_paths.add(grandchild)
                except PermissionError:
                    continue
        except PermissionError:
            continue

        for path in probe_paths:
            try:
                if not path.is_mount():
                    continue
            except PermissionError:
                continue
            uuid = linux_mount_uuid(path)
            device_id = f"linuxuuid:{uuid}" if uuid else f"linux:{path}"
            display = f"{path} [{uuid}]" if uuid else str(path)
            devices.append(Device(device_id=device_id, name=display, path=path))

    unique = {d.device_id: d for d in devices}
    return sorted(unique.values(), key=lambda d: d.path.as_posix().lower())


def discover_devices() -> list[Device]:
    if os.name == "nt":
        return discover_windows_devices()
    if platform.system().lower() == "linux":
        return discover_linux_devices()
    return []


def find_media_files(root: Path) -> list[Path]:
    scored_paths: list[tuple[float, str, Path]] = []

    def on_walk_error(exc: OSError) -> None:
        if is_disconnect_error(exc) or not root.exists():
            raise SourceUnavailableError(f"Source disconnected while scanning {root}") from exc
        raise RuntimeError(f"Failed to scan source {root}: {exc}") from exc

    for dir_path, _, file_names in os.walk(root, onerror=on_walk_error):
        for file_name in file_names:
            path = Path(dir_path) / file_name
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            scored_paths.append((stat.st_mtime, path.name.lower(), path))

    scored_paths.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored_paths]


def choose_device_interactive(devices: list[Device]) -> Device | None:
    if not devices:
        return None
    print("\nDetected devices:")
    for idx, device in enumerate(devices, start=1):
        print(f"  {idx}. {device.name}")
    print("  M. Manual path")
    choice = input("Select a device [number/M]: ").strip().lower()
    if choice == "m":
        return None
    if not choice.isdigit():
        raise SystemExit("Invalid selection.")
    index = int(choice)
    if index < 1 or index > len(devices):
        raise SystemExit("Selection out of range.")
    return devices[index - 1]


def resolve_source(args: argparse.Namespace) -> Device:
    if args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.exists():
            raise SystemExit(f"Source path does not exist: {source}")
        return Device(
            device_id=f"manual:{source}",
            name=f"Manual {source}",
            path=source,
        )

    devices = discover_devices()
    selected = choose_device_interactive(devices)
    if selected is not None:
        return selected

    manual = input("Enter source path: ").strip()
    manual_path = Path(manual).expanduser().resolve()
    if not manual_path.exists():
        raise SystemExit(f"Source path does not exist: {manual_path}")
    return Device(
        device_id=f"manual:{manual_path}",
        name=f"Manual {manual_path}",
        path=manual_path,
    )


def make_archive_paths(archive_root: Path) -> tuple[Path, Path]:
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_root / "manifest.json"
    return archive_root, manifest_path


def copied_filename(captured_at: datetime, file_hash: str, suffix: str) -> str:
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S")
    token = hashlib.blake2s(file_hash.encode("utf-8"), digest_size=4).hexdigest()
    return f"{timestamp}_{token}{suffix.lower()}"


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{message} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def prompt_yes_no_windows(message: str, title: str, default_yes: bool = False) -> bool:
    if os.name != "nt":
        return False
    user32 = ctypes.windll.user32
    flags = 0x00000004 | 0x00000020  # MB_YESNO | MB_ICONQUESTION
    if not default_yes:
        flags |= 0x00000100  # MB_DEFBUTTON2
    result = user32.MessageBoxW(None, message, title, flags)
    return result == 6  # IDYES


def prompt_device_auto_approval(args: argparse.Namespace, source: Device, device_alias: str) -> bool:
    message = (
        "New device detected.\n\n"
        f"Alias: {device_alias}\n"
        f"Device ID: {source.device_id}\n"
        f"Name: {source.name}\n\n"
        "Approve this device for automatic processing on future reconnects?\n"
        "If approved, this connection will process now."
    )

    if args.tray and os.name == "nt":
        return prompt_yes_no_windows(message, "FAQ Notes Tool", default_yes=False)

    print(f"\nDetected new device (approval required): {source.name}")
    print(f"Device ID: {source.device_id} ({device_alias})")
    if sys.stdin.isatty():
        return prompt_yes_no(
            "Approve this device for automatic processing in watch mode?",
            default_yes=False,
        )

    print("Skipping unapproved device: no interactive prompt is available.")
    return False


def should_process_in_watch_mode(args: argparse.Namespace, source: Device) -> bool:
    archive_root = resolve_archive_root(args.archive_root)
    archive_root, manifest_path = make_archive_paths(archive_root)
    store = create_manifest_store(args, manifest_path)
    manifest = load_manifest(store)
    normalize_manifest_paths(archive_root, manifest)

    device_alias = get_or_create_device_alias(manifest, source.device_id, source.name)
    devices: dict[str, Any] = manifest.setdefault("devices", {})
    device_meta: dict[str, Any] = devices.setdefault(source.device_id, {})
    auto_approved = bool(device_meta.get("auto_approved"))

    if auto_approved:
        # Persist display name/alias schema updates for previously approved devices.
        save_manifest(store, manifest)
        return True

    approved = prompt_device_auto_approval(args, source, device_alias)
    device_meta["auto_approved"] = approved
    if approved:
        device_meta["approved_at_local"] = datetime.now().isoformat(timespec="seconds")
        print(f"Approved {source.name}; starting ingestion.")
    else:
        print(f"Ignored {source.name}; not approved for auto-processing.")
    save_manifest(store, manifest)
    return approved


def notify_complete() -> None:
    if os.name == "nt":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)


def process_source(args: argparse.Namespace, source: Device) -> int:
    archive_root = resolve_archive_root(args.archive_root)
    archive_root, manifest_path = make_archive_paths(archive_root)
    store = create_manifest_store(args, manifest_path)
    manifest = load_manifest(store)
    if normalize_manifest_paths(archive_root, manifest):
        save_manifest(store, manifest)
    if args.dedupe_mode == "fast" and refresh_fast_index(archive_root, manifest):
        save_manifest(store, manifest)
    known_hashes: dict[str, Any] = manifest.get("files", {})
    quick_index_root: dict[str, Any] = manifest.setdefault("quick_index", {})
    fast_index: dict[str, str] = manifest.setdefault("fast_index", {})
    source_quick: dict[str, str] = quick_index_root.setdefault(source.device_id, {})
    device_alias = get_or_create_device_alias(manifest, source.device_id, source.name)

    print(f"\nSource: {source.path}")
    print(f"Device ID: {source.device_id} ({device_alias})")
    print(f"Duplicate mode: {args.dedupe_mode}")
    try:
        ensure_source_available(source)
        media_files = find_media_files(source.path)
    except SourceUnavailableError as exc:
        print(str(exc))
        save_manifest(store, manifest)
        return 0
    print(f"Media files found: {len(media_files)}")
    if not media_files:
        save_manifest(store, manifest)
        return 0

    new_items: list[dict[str, Any]] = []
    quick_index_changed = False
    source_disconnected = False
    disconnect_reason = ""

    for idx, media_file in enumerate(media_files, start=1):
        try:
            ensure_source_available(source)
            stat = media_file.stat()
        except SourceUnavailableError as exc:
            source_disconnected = True
            disconnect_reason = str(exc)
            break
        except OSError as exc:
            source_disconnected = True
            disconnect_reason = str(convert_source_io_error(source, media_file, exc))
            break

        quick_key = make_quick_key(source.path, media_file, stat.st_size, stat.st_mtime_ns)
        known_content_id = source_quick.get(quick_key)
        if known_content_id and known_content_id in known_hashes:
            continue

        fast_fingerprint = ""
        try:
            if args.dedupe_mode == "fast":
                print(f"Fingerprinting {idx}/{len(media_files)}: {media_file.name}")
                fast_fingerprint = sampled_file_fingerprint(media_file, stat.st_size)
                known_by_fingerprint = fast_index.get(fast_fingerprint)
                if known_by_fingerprint and known_by_fingerprint in known_hashes:
                    source_quick[quick_key] = known_by_fingerprint
                    quick_index_changed = True
                    continue
                content_id = fast_fingerprint
            else:
                print(f"Hashing {idx}/{len(media_files)}: {media_file.name}")
                content_id = sha256_file(media_file)
        except OSError as exc:
            source_disconnected = True
            disconnect_reason = str(convert_source_io_error(source, media_file, exc))
            break

        source_quick[quick_key] = content_id
        quick_index_changed = True

        if content_id in known_hashes:
            continue

        captured_at = datetime.fromtimestamp(stat.st_mtime)
        month_key = captured_at.strftime("%Y-%m")
        month_dir = archive_root / month_key
        audio_dir = month_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        target_name = copied_filename(captured_at, content_id, media_file.suffix)
        target_path = audio_dir / target_name
        try:
            needs_copy = True
            if target_path.exists():
                existing_size = target_path.stat().st_size
                if existing_size == stat.st_size:
                    needs_copy = False
                else:
                    print(f"Existing copy size mismatch, recopying: {target_path.name}")
                    target_path.unlink()
            if needs_copy:
                copy_file_atomic(media_file, target_path, expected_size=stat.st_size)
        except OSError as exc:
            source_disconnected = True
            disconnect_reason = str(convert_source_io_error(source, media_file, exc))
            break
        except Exception as exc:
            if "Incomplete copy" in str(exc) or not source.path.exists():
                source_disconnected = True
                disconnect_reason = f"Source disconnected while copying {media_file}"
                break
            print(f"Failed to copy {media_file.name}: {exc}")
            continue

        new_items.append(
            {
                "source_file": media_file,
                "content_id": content_id,
                "captured_at": captured_at,
                "month_key": month_key,
                "month_dir": month_dir,
                "target_path": target_path,
                "fast_fingerprint": fast_fingerprint,
            }
        )

    if quick_index_changed:
        save_manifest(store, manifest)

    if source_disconnected:
        print(disconnect_reason)
        print("Source disconnected mid-run. Already copied/transcribed files are safe; resume by reconnecting.")

    print(f"New files to process: {len(new_items)}")
    if not new_items:
        return 0

    if not args.yes and not args.watch and not prompt_yes_no("Copy and transcribe new files?"):
        print("Cancelled by user.")
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    requested_workers = max(1, args.workers)
    worker_count = min(requested_workers, len(new_items))
    print(f"Queued {len(new_items)} file(s) for transcription with {worker_count} worker(s).")

    processed = 0
    failed = 0
    touched_months: set[str] = set()

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(transcribe_media_file_with_api_key, api_key, item["target_path"]): item
            for item in new_items
        }

        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            target_path: Path = item["target_path"]
            print(f"\n[{index}/{len(new_items)}] Completed task for {target_path.name}")
            try:
                transcript = future.result()
            except Exception as exc:
                failed += 1
                print(f"Failed {target_path.name}: {exc}")
                continue

            captured_at: datetime = item["captured_at"]
            month_key: str = item["month_key"]
            month_dir: Path = item["month_dir"]
            source_file: Path = item["source_file"]
            content_id: str = item["content_id"]
            fast_fingerprint: str = item["fast_fingerprint"]

            transcript_dir = month_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            token = hashlib.blake2s(content_id.encode("utf-8"), digest_size=4).hexdigest()
            transcript_path = transcript_dir / f"{captured_at.strftime('%Y%m%d_%H%M%S')}_{token}.txt"
            try:
                atomic_write_text(transcript_path, transcript.strip(), encoding="utf-8")
            except OSError as exc:
                failed += 1
                print(f"Failed to write transcript {transcript_path.name}: {exc}")
                continue

            known_hashes[content_id] = {
                "source_id": source.device_id,
                "source_name": source.name,
                "device_alias": device_alias,
                "source_path": str(source_file),
                "copied_path": path_for_manifest(archive_root, target_path),
                "transcript_path": path_for_manifest(archive_root, transcript_path),
                "captured_at_local": captured_at.isoformat(timespec="seconds"),
                "processed_at_local": datetime.now().isoformat(timespec="seconds"),
                "month_key": month_key,
                "content_id": content_id,
                "dedupe_mode": args.dedupe_mode,
                "fast_fingerprint": fast_fingerprint,
            }
            if fast_fingerprint:
                fast_index[fast_fingerprint] = content_id
            manifest["files"] = known_hashes
            manifest["fast_index"] = fast_index
            save_manifest(store, manifest)
            touched_months.add(month_key)
            processed += 1

    for month_key in sorted(touched_months):
        rebuild_month_outputs(archive_root, manifest, month_key)

    print(f"\nCompleted. Processed {processed} new file(s). Failed {failed} file(s).")
    notify_complete()
    return processed


def poll_watch_cycle(args: argparse.Namespace, seen_ids: set[str]) -> set[str]:
    current = discover_devices()
    current_ids = {d.device_id for d in current}
    new_devices = [d for d in current if d.device_id not in seen_ids]
    for device in new_devices:
        print(f"\nDetected new device: {device.name}")
        try:
            if should_process_in_watch_mode(args, device):
                process_source(args, device)
        except Exception as exc:
            print(f"Error processing {device.path}: {exc}")
    return current_ids


def run_watch_mode(args: argparse.Namespace, stop_event: threading.Event | None = None) -> None:
    print("Watch mode enabled.")
    print("Waiting for newly connected devices. Press Ctrl+C to stop.")
    seen_ids = {d.device_id for d in discover_devices()}
    interval = max(2, args.interval)

    while True:
        if stop_event is None:
            time.sleep(interval)
        elif stop_event.wait(interval):
            break
        seen_ids = poll_watch_cycle(args, seen_ids)


def build_tray_image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (64, 64), "#0f3a2e")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((6, 6, 58, 58), radius=12, fill="#1a5e49")
    draw.rectangle((18, 14, 46, 50), fill="#d9f3e8")
    draw.rectangle((22, 20, 42, 26), fill="#1a5e49")
    draw.rectangle((22, 30, 42, 36), fill="#1a5e49")
    draw.rectangle((22, 40, 34, 46), fill="#1a5e49")
    return image


def run_tray_mode(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise SystemExit("--tray is currently supported on Windows only.")
    try:
        import pystray
    except ImportError as exc:
        raise SystemExit("Tray mode requires 'pystray' and 'Pillow'. Install with: pip install pystray pillow") from exc

    print("Tray mode enabled.")
    print("The app is running in the system tray.")
    stop_event = threading.Event()
    watch_thread = threading.Thread(target=run_watch_mode, args=(args, stop_event), daemon=True)
    watch_thread.start()

    def on_quit(icon: Any, item: Any) -> None:
        stop_event.set()
        icon.stop()

    icon = pystray.Icon(
        "faq_notes_ingest",
        build_tray_image(),
        "FAQ Notes Tool",
        pystray.Menu(pystray.MenuItem("Quit", on_quit)),
    )
    try:
        icon.run()
    finally:
        stop_event.set()
        watch_thread.join(timeout=max(2, args.interval) + 2)


def parse_args() -> argparse.Namespace:
    recommended_workers = default_worker_count()
    parser = argparse.ArgumentParser(
        description="Copy + transcribe FAQ recorder files into monthly archives."
    )
    parser.add_argument(
        "--archive-root",
        default=DEFAULT_ARCHIVE_SUBDIR,
        help="Local archive folder. Relative paths are resolved from script dir (default: faq_notes).",
    )
    parser.add_argument(
        "--source",
        help="Source folder/device path. If omitted, interactive device selection is used.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for one-off (non-watch) processing.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for newly connected devices and auto-process.",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Windows system tray mode (watch mode with first-time device approval prompts).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Watch poll interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=recommended_workers,
        help=f"Parallel transcription workers (default: {recommended_workers}). Use 1 for sequential.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=("fast", "strict"),
        default="fast",
        help="Duplicate detection mode: 'fast' uses sampled content fingerprints, 'strict' uses full SHA-256.",
    )
    parser.add_argument(
        "--storage-backend",
        choices=("json", "postgres"),
        default="json",
        help="Manifest storage backend: 'json' writes manifest.json, 'postgres' stores state in PostgreSQL.",
    )
    parser.add_argument(
        "--database-url",
        help=(
            "PostgreSQL connection URL for --storage-backend postgres. "
            "Falls back to FAQ_NOTES_DATABASE_URL or DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--account-id",
        help=(
            "Logical user/account ID for Postgres record isolation. "
            "Required for --storage-backend postgres (or set FAQ_NOTES_ACCOUNT_ID)."
        ),
    )
    parser.add_argument(
        "--migrate-manifest",
        action="store_true",
        help=(
            "One-time import of archive_root/manifest.json into PostgreSQL before running. "
            "Requires --storage-backend postgres."
        ),
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Exit after --migrate-manifest completes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.migrate_only and not args.migrate_manifest:
        raise SystemExit("--migrate-only requires --migrate-manifest.")
    if args.migrate_manifest:
        archive_root = resolve_archive_root(args.archive_root)
        _, manifest_path = make_archive_paths(archive_root)
        migrate_manifest_to_postgres(args, manifest_path)
        if args.migrate_only:
            return

    require_tool("ffmpeg")
    require_tool("ffprobe")

    if args.tray:
        args.watch = True
        run_tray_mode(args)
        return

    if args.watch:
        run_watch_mode(args)
        return

    source = resolve_source(args)
    process_source(args, source)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
