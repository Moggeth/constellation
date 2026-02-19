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
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

MODEL = "whisper-1"
LIMIT_BYTES = 25 * 1024 * 1024
AUDIO_OPTS = ["-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "192k"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_SUBDIR = "faq_notes"
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


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 2,
            "files": {},
            "quick_index": {},
            "devices": {},
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
        payload.setdefault("schema_version", 2)
        payload.setdefault("files", {})
        payload.setdefault("quick_index", {})
        payload.setdefault("devices", {})
        if payload["schema_version"] < 2:
            payload["schema_version"] = 2
        return payload
    except Exception as exc:
        raise RuntimeError(f"Failed to read manifest at {path}: {exc}") from exc


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def get_or_create_device_alias(manifest: dict[str, Any], source_id: str, display_name: str) -> str:
    devices: dict[str, Any] = manifest.setdefault("devices", {})
    current = devices.get(source_id)
    if current:
        current["display_name"] = display_name
        return str(current.get("alias", "device_1"))

    aliases = {str(v.get("alias", "")) for v in devices.values()}
    idx = 1
    while f"device_{idx}" in aliases:
        idx += 1
    alias = f"device_{idx}"
    devices[source_id] = {"alias": alias, "display_name": display_name}
    return alias


def make_quick_key(source_root: Path, file_path: Path, size: int, mtime_ns: int) -> str:
    try:
        relative = file_path.relative_to(source_root).as_posix().lower()
    except ValueError:
        relative = file_path.name.lower()
    return f"{relative}|{size}|{mtime_ns}"


def write_entries_file(path: Path, month_key: str, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.write_text("".join(lines), encoding="utf-8")


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
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    files.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
    return files


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
    return f"{timestamp}_{file_hash[:8]}{suffix.lower()}"


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{message} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


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
    manifest = load_manifest(manifest_path)
    if normalize_manifest_paths(archive_root, manifest):
        save_manifest(manifest_path, manifest)
    known_hashes: dict[str, Any] = manifest.get("files", {})
    quick_index_root: dict[str, Any] = manifest.setdefault("quick_index", {})
    source_quick: dict[str, str] = quick_index_root.setdefault(source.device_id, {})
    device_alias = get_or_create_device_alias(manifest, source.device_id, source.name)

    print(f"\nSource: {source.path}")
    print(f"Device ID: {source.device_id} ({device_alias})")
    media_files = find_media_files(source.path)
    print(f"Media files found: {len(media_files)}")
    if not media_files:
        save_manifest(manifest_path, manifest)
        return 0

    new_items: list[dict[str, Any]] = []
    quick_index_changed = False

    for idx, media_file in enumerate(media_files, start=1):
        stat = media_file.stat()
        quick_key = make_quick_key(source.path, media_file, stat.st_size, stat.st_mtime_ns)
        known_digest = source_quick.get(quick_key)
        if known_digest and known_digest in known_hashes:
            continue

        print(f"Hashing {idx}/{len(media_files)}: {media_file.name}")
        digest = sha256_file(media_file)
        source_quick[quick_key] = digest
        quick_index_changed = True

        if digest in known_hashes:
            continue

        captured_at = datetime.fromtimestamp(stat.st_mtime)
        month_key = captured_at.strftime("%Y-%m")
        month_dir = archive_root / month_key
        audio_dir = month_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        target_name = copied_filename(captured_at, digest, media_file.suffix)
        target_path = audio_dir / target_name
        if not target_path.exists():
            shutil.copy2(media_file, target_path)

        new_items.append(
            {
                "source_file": media_file,
                "digest": digest,
                "captured_at": captured_at,
                "month_key": month_key,
                "month_dir": month_dir,
                "target_path": target_path,
            }
        )

    if quick_index_changed:
        save_manifest(manifest_path, manifest)

    print(f"New files to process: {len(new_items)}")
    if not new_items:
        return 0

    if not args.yes and not prompt_yes_no("Copy and transcribe new files?"):
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
            digest: str = item["digest"]

            transcript_dir = month_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{captured_at.strftime('%Y%m%d_%H%M%S')}_{digest[:8]}.txt"
            transcript_path.write_text(transcript.strip(), encoding="utf-8")

            known_hashes[digest] = {
                "source_id": source.device_id,
                "source_name": source.name,
                "device_alias": device_alias,
                "source_path": str(source_file),
                "copied_path": path_for_manifest(archive_root, target_path),
                "transcript_path": path_for_manifest(archive_root, transcript_path),
                "captured_at_local": captured_at.isoformat(timespec="seconds"),
                "processed_at_local": datetime.now().isoformat(timespec="seconds"),
                "month_key": month_key,
            }
            manifest["files"] = known_hashes
            save_manifest(manifest_path, manifest)
            touched_months.add(month_key)
            processed += 1

    for month_key in sorted(touched_months):
        rebuild_month_outputs(archive_root, manifest, month_key)

    print(f"\nCompleted. Processed {processed} new file(s). Failed {failed} file(s).")
    notify_complete()
    return processed


def run_watch_mode(args: argparse.Namespace) -> None:
    print("Watch mode enabled.")
    print("Waiting for newly connected devices. Press Ctrl+C to stop.")
    seen_ids = {d.device_id for d in discover_devices()}
    interval = max(2, args.interval)

    while True:
        time.sleep(interval)
        current = discover_devices()
        current_ids = {d.device_id for d in current}
        new_devices = [d for d in current if d.device_id not in seen_ids]
        for device in new_devices:
            print(f"\nDetected new device: {device.name}")
            try:
                process_source(args, device)
            except Exception as exc:
                print(f"Error processing {device.path}: {exc}")
        seen_ids = current_ids


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
        help="Skip confirmation prompt and process immediately.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for newly connected devices and auto-process.",
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
    return parser.parse_args()


def main() -> None:
    require_tool("ffmpeg")
    require_tool("ffprobe")

    args = parse_args()
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
