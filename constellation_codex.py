from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = SCRIPT_DIR / "toolbelt" / "codex_runs"
APPDATA_DIR = Path(os.getenv("APPDATA", ""))
DEFAULT_NATIVE_CODEX_PATH = APPDATA_DIR / "npm" / "codex.cmd"
DEFAULT_NODE_DIR = Path(r"C:\Program Files\nodejs")
DEFAULT_CODEX_COMMAND = os.getenv(
    "CONSTELLATION_CODEX_COMMAND",
    str(DEFAULT_NATIVE_CODEX_PATH if DEFAULT_NATIVE_CODEX_PATH.exists() else "codex"),
)
DEFAULT_CODEX_MODEL = os.getenv("CONSTELLATION_CODEX_MODEL", "gpt-5.4")
DEFAULT_SANDBOX = os.getenv("CONSTELLATION_CODEX_SANDBOX", "workspace-write")
DEFAULT_APPROVAL_MODE = os.getenv("CONSTELLATION_CODEX_APPROVAL_MODE", "never")
DEFAULT_PROVIDER = os.getenv("CONSTELLATION_CODEX_PROVIDER", "native")
DEFAULT_WSL_DISTRO = os.getenv("CONSTELLATION_CODEX_WSL_DISTRO", "Ubuntu-22.04")
ALLOWED_SANDBOXES = ("read-only", "workspace-write", "danger-full-access")
ALLOWED_APPROVAL_MODES = ("untrusted", "on-request", "never")
ALLOWED_PROVIDERS = ("native", "wsl")
ACTIVE_STATUSES = {"starting", "running"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tail_text(path: Path, max_chars: int = 4000) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > max_chars:
        return "..." + text[-max_chars:]
    return text


def validate_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported {label} '{value}'. Allowed values: {', '.join(allowed)}")
    return normalized


def looks_like_completed_run(last_message: str | None, json_tail: str | None) -> bool:
    if not last_message:
        return False
    if not json_tail:
        return False
    return '"turn.completed"' in json_tail or '"item.completed"' in json_tail


@dataclass
class CodexTaskRecord:
    task_id: str
    title: str
    prompt: str
    repo_label: str
    repo_path: str
    command: list[str]
    command_path: str | None
    model: str
    sandbox: str
    approval_mode: str
    skip_git_repo_check: bool
    json_output_path: str
    stderr_path: str
    last_message_path: str
    created_at: str
    status: str = "starting"
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    last_message: str | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class CodexBridgeManager:
    def __init__(
        self,
        workspace_root: Path,
        *,
        codex_command: str = DEFAULT_CODEX_COMMAND,
        default_model: str = DEFAULT_CODEX_MODEL,
        runs_root: Path = RUNS_ROOT,
        provider: str = DEFAULT_PROVIDER,
        wsl_distro: str = DEFAULT_WSL_DISTRO,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.codex_command = codex_command
        self.default_model = default_model
        self.runs_root = runs_root.resolve()
        self.provider = validate_choice(provider, ALLOWED_PROVIDERS, "provider")
        self.wsl_distro = wsl_distro
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active_processes: dict[str, subprocess.Popen[str]] = {}

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        extras: list[str] = []
        native_parent = Path(self.codex_command).expanduser().resolve().parent if any(sep in self.codex_command for sep in ("\\", "/")) else None
        if native_parent is not None and native_parent.exists():
            extras.append(str(native_parent))
        if DEFAULT_NODE_DIR.exists():
            extras.append(str(DEFAULT_NODE_DIR))
        if DEFAULT_NATIVE_CODEX_PATH.parent.exists():
            extras.append(str(DEFAULT_NATIVE_CODEX_PATH.parent))
        existing = env.get("PATH", "")
        env["PATH"] = ";".join([*extras, existing])
        return env

    def _resolve_native_command(self) -> str | None:
        configured = self.codex_command.strip()
        if any(sep in configured for sep in ("\\", "/")):
            candidate = Path(configured).expanduser()
            if candidate.exists():
                return str(candidate)
        resolved = shutil.which(configured, path=self._build_env().get("PATH"))
        if resolved:
            return resolved
        if DEFAULT_NATIVE_CODEX_PATH.exists():
            return str(DEFAULT_NATIVE_CODEX_PATH)
        return None

    def _windows_path_to_wsl(self, value: Path) -> str:
        completed = subprocess.run(
            ["wsl.exe", "-d", self.wsl_distro, "wslpath", "-a", str(value)],
            cwd=str(self.workspace_root),
            capture_output=True,
            text=True,
            timeout=20,
            env=self._build_env(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip() or "wslpath failed")
        return completed.stdout.strip()

    def _build_command(
        self,
        *,
        prompt: str,
        model: str,
        sandbox: str,
        approval_mode: str,
        repo_path: Path,
        last_message_path: Path,
        add_dirs: list[str],
        skip_git_repo_check: bool,
    ) -> tuple[list[str], str | None]:
        common_args = [
            "-a",
            approval_mode,
            "exec",
            "--json",
            "--output-last-message",
            str(last_message_path),
            "--model",
            model,
            "--sandbox",
            sandbox,
        ]
        for extra_dir in add_dirs:
            common_args.extend(["--add-dir", extra_dir])
        if skip_git_repo_check:
            common_args.append("--skip-git-repo-check")
        common_args.append(prompt)

        if self.provider == "native":
            resolved_command = self._resolve_native_command()
            if resolved_command is None:
                raise FileNotFoundError(
                    f"Could not resolve the Codex command '{self.codex_command}'. "
                    "Set CONSTELLATION_CODEX_COMMAND if Codex lives somewhere else."
                )
            return [resolved_command, *common_args], resolved_command

        repo_wsl = self._windows_path_to_wsl(repo_path)
        last_message_wsl = self._windows_path_to_wsl(last_message_path)
        wsl_args = [
            "-a",
            approval_mode,
            "exec",
            "--json",
            "--output-last-message",
            last_message_wsl,
            "--model",
            model,
            "--sandbox",
            sandbox,
        ]
        for extra_dir in add_dirs:
            wsl_args.extend(["--add-dir", self._windows_path_to_wsl(Path(extra_dir).resolve())])
        if skip_git_repo_check:
            wsl_args.append("--skip-git-repo-check")
        wsl_args.append(prompt)
        command_name = self.codex_command.strip() or "codex"
        shell_command = f"cd {shlex.quote(repo_wsl)} && {shlex.join([command_name, *wsl_args])}"
        return ["wsl.exe", "-d", self.wsl_distro, "bash", "-lc", shell_command], f"wsl:{command_name}"

    def discover_repositories(self, max_repos: int = 20) -> list[dict[str, Any]]:
        repositories: list[dict[str, Any]] = []
        for entry in sorted(self.workspace_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            git_marker = entry / ".git"
            if not git_marker.exists():
                continue
            repositories.append(
                {
                    "name": entry.name,
                    "path": str(entry.resolve()),
                    "git_repository": True,
                }
            )
            if len(repositories) >= max_repos:
                break
        return repositories

    def resolve_repo(self, repo_name_or_path: str | None) -> tuple[Path, bool, str]:
        raw_value = (repo_name_or_path or "").strip()
        if not raw_value:
            candidate = self.workspace_root
        else:
            direct = Path(raw_value).expanduser()
            if direct.exists():
                candidate = direct
            else:
                candidate = self.workspace_root / raw_value
                if not candidate.exists():
                    match = next(
                        (
                            Path(repo["path"])
                            for repo in self.discover_repositories(max_repos=200)
                            if repo["name"].lower() == raw_value.lower()
                        ),
                        None,
                    )
                    if match is None:
                        raise FileNotFoundError(f"Repository or directory not found: {repo_name_or_path}")
                    candidate = match
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Repository path not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {resolved}")
        skip_git_repo_check = not (resolved / ".git").exists()
        return resolved, skip_git_repo_check, resolved.name

    def status(self) -> dict[str, Any]:
        resolved_command = self._resolve_native_command() if self.provider == "native" else self.codex_command
        payload: dict[str, Any] = {
            "workspace_root": str(self.workspace_root),
            "codex_command": self.codex_command,
            "command_path": resolved_command,
            "default_model": self.default_model,
            "provider": self.provider,
            "wsl_distro": self.wsl_distro if self.provider == "wsl" else None,
            "available": False,
            "active_task_ids": sorted(self._active_processes.keys()),
            "known_repositories": self.discover_repositories(max_repos=20),
        }
        if self.provider == "native" and resolved_command is None:
            payload["error"] = (
                f"Could not resolve the Codex command '{self.codex_command}'. "
                "Set CONSTELLATION_CODEX_COMMAND if Codex lives somewhere else."
            )
            return payload
        try:
            if self.provider == "native":
                completed = subprocess.run(
                    [str(resolved_command), "--version"],
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=12,
                    env=self._build_env(),
                    check=False,
                )
            else:
                completed = subprocess.run(
                    [
                        "wsl.exe",
                        "-d",
                        self.wsl_distro,
                        "bash",
                        "-lc",
                        f"{shlex.quote(self.codex_command or 'codex')} --version",
                    ],
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=20,
                    env=self._build_env(),
                    check=False,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            payload["error"] = (
                f"Codex was found at '{resolved_command}' but could not be executed from this environment: {exc}"
            )
            return payload
        payload["available"] = completed.returncode == 0
        payload["version_output"] = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            payload["error"] = (completed.stderr or completed.stdout).strip() or "Codex returned a non-zero exit code."
        return payload

    def _record_path(self, task_id: str) -> Path:
        return self.runs_root / task_id / "task.json"

    def _load_record(self, task_id: str) -> CodexTaskRecord:
        record_path = self._record_path(task_id)
        if not record_path.exists():
            raise FileNotFoundError(f"Codex task not found: {task_id}")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        return CodexTaskRecord(**payload)

    def _save_record(self, record: CodexTaskRecord) -> None:
        task_dir = self.runs_root / record.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        self._record_path(record.task_id).write_text(
            json.dumps(record.to_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _monitor_process(
        self,
        task_id: str,
        process: subprocess.Popen[str],
        stdout_handle: Any,
        stderr_handle: Any,
    ) -> None:
        exit_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        with self._lock:
            self._active_processes.pop(task_id, None)
        record = self._load_record(task_id)
        record.exit_code = exit_code
        record.finished_at = utc_now()
        record.last_message = tail_text(Path(record.last_message_path), max_chars=4000)
        json_tail = tail_text(Path(record.json_output_path), max_chars=2500)
        stderr_tail = tail_text(Path(record.stderr_path), max_chars=2500)
        if exit_code == 0 or looks_like_completed_run(record.last_message, json_tail):
            record.status = "completed"
        else:
            record.status = "failed"
            record.error = stderr_tail or "Codex exited with a non-zero code."
        self._save_record(record)

    def submit_task(
        self,
        *,
        prompt: str,
        repo_name_or_path: str | None = None,
        title: str | None = None,
        model: str | None = None,
        sandbox: str = DEFAULT_SANDBOX,
        approval_mode: str = DEFAULT_APPROVAL_MODE,
        add_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Codex prompt cannot be empty.")
        sandbox = validate_choice(sandbox, ALLOWED_SANDBOXES, "sandbox")
        approval_mode = validate_choice(approval_mode, ALLOWED_APPROVAL_MODES, "approval mode")
        repo_path, skip_git_repo_check, repo_label = self.resolve_repo(repo_name_or_path)
        task_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        task_dir = self.runs_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        json_output_path = task_dir / "events.jsonl"
        stderr_path = task_dir / "stderr.log"
        last_message_path = task_dir / "last_message.txt"
        normalized_add_dirs = [str(extra_dir).strip() for extra_dir in add_dirs or [] if str(extra_dir).strip()]
        command, resolved_command = self._build_command(
            prompt=cleaned_prompt,
            model=(model or self.default_model).strip() or self.default_model,
            sandbox=sandbox,
            approval_mode=approval_mode,
            repo_path=repo_path,
            last_message_path=last_message_path,
            add_dirs=normalized_add_dirs,
            skip_git_repo_check=skip_git_repo_check,
        )

        record = CodexTaskRecord(
            task_id=task_id,
            title=(title or cleaned_prompt.splitlines()[0][:80]).strip() or "Codex task",
            prompt=cleaned_prompt,
            repo_label=repo_label,
            repo_path=str(repo_path),
            command=command,
            command_path=resolved_command,
            model=(model or self.default_model).strip() or self.default_model,
            sandbox=sandbox,
            approval_mode=approval_mode,
            skip_git_repo_check=skip_git_repo_check,
            json_output_path=str(json_output_path),
            stderr_path=str(stderr_path),
            last_message_path=str(last_message_path),
            created_at=utc_now(),
        )
        self._save_record(record)

        stdout_handle = json_output_path.open("w", encoding="utf-8", buffering=1)
        stderr_handle = stderr_path.open("w", encoding="utf-8", buffering=1)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=str(repo_path),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                creationflags=creationflags,
                env=self._build_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stdout_handle.close()
            stderr_handle.close()
            record.status = "failed"
            record.finished_at = utc_now()
            record.error = str(exc)
            self._save_record(record)
            return record.to_payload()

        record.status = "running"
        record.started_at = utc_now()
        self._save_record(record)
        with self._lock:
            self._active_processes[task_id] = process
        watcher = threading.Thread(
            target=self._monitor_process,
            args=(task_id, process, stdout_handle, stderr_handle),
            daemon=True,
        )
        watcher.start()
        return self.get_task_status(task_id)

    def list_tasks(self, limit: int = 10) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for record_path in sorted(self.runs_root.glob("*/task.json"), reverse=True):
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append(payload)
            if len(records) >= max(1, limit):
                break
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return {"items": records[: max(1, limit)]}

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        record = self._load_record(task_id)
        if record.status in ACTIVE_STATUSES:
            with self._lock:
                process = self._active_processes.get(task_id)
            if process is not None:
                poll_result = process.poll()
                if poll_result is not None:
                    record.exit_code = poll_result
                    record.finished_at = utc_now()
                    record.last_message = tail_text(Path(record.last_message_path), max_chars=4000)
                    json_tail = tail_text(Path(record.json_output_path), max_chars=2500)
                    stderr_tail = tail_text(Path(record.stderr_path), max_chars=2500)
                    if poll_result == 0 or looks_like_completed_run(record.last_message, json_tail):
                        record.status = "completed"
                        record.error = None
                    else:
                        record.status = "failed"
                        record.error = stderr_tail
                    self._save_record(record)
            else:
                last_message = tail_text(Path(record.last_message_path), max_chars=4000)
                json_tail = tail_text(Path(record.json_output_path), max_chars=2500)
                stderr_tail = tail_text(Path(record.stderr_path), max_chars=2500)
                if looks_like_completed_run(last_message, json_tail):
                    record.status = "completed"
                    record.finished_at = record.finished_at or utc_now()
                    record.last_message = last_message
                    record.error = None
                    self._save_record(record)
                elif stderr_tail:
                    record.status = "failed"
                    record.finished_at = record.finished_at or utc_now()
                    record.error = stderr_tail
                    self._save_record(record)
        else:
            last_message = tail_text(Path(record.last_message_path), max_chars=4000)
            json_tail = tail_text(Path(record.json_output_path), max_chars=2500)
            if looks_like_completed_run(last_message, json_tail):
                record.status = "completed"
                record.last_message = last_message
                record.error = None
                record.finished_at = record.finished_at or utc_now()
                self._save_record(record)
        payload = record.to_payload()
        payload["json_output_tail"] = tail_text(Path(record.json_output_path), max_chars=2500)
        payload["stderr_tail"] = tail_text(Path(record.stderr_path), max_chars=2500)
        payload["last_message"] = tail_text(Path(record.last_message_path), max_chars=4000) or record.last_message
        return payload

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            process = self._active_processes.get(task_id)
        if process is None:
            payload = self.get_task_status(task_id)
            if payload["status"] in ACTIVE_STATUSES:
                payload["status"] = "failed"
                payload["finished_at"] = payload.get("finished_at") or utc_now()
                payload["error"] = payload.get("stderr_tail") or "Task is marked active but no running process was found."
                record = self._load_record(task_id)
                record.status = "failed"
                record.finished_at = payload["finished_at"]
                record.error = payload["error"]
                self._save_record(record)
            return payload
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        record = self._load_record(task_id)
        record.status = "cancelled"
        record.finished_at = utc_now()
        record.exit_code = process.returncode
        record.error = "Cancelled by user."
        record.last_message = tail_text(Path(record.last_message_path), max_chars=4000)
        self._save_record(record)
        with self._lock:
            self._active_processes.pop(task_id, None)
        return self.get_task_status(task_id)

    def wait_for_task(self, task_id: str, poll_interval: float = 1.0) -> dict[str, Any]:
        while True:
            payload = self.get_task_status(task_id)
            if payload["status"] not in ACTIVE_STATUSES:
                return payload
            time.sleep(max(0.25, poll_interval))


def print_status_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constellation bridge for Codex CLI tasks.")
    parser.add_argument(
        "--workspace-root",
        default=str(SCRIPT_DIR.parent.resolve()),
        help="Workspace root used to discover repos and resolve relative paths.",
    )
    parser.add_argument(
        "--codex-command",
        default=DEFAULT_CODEX_COMMAND,
        help=f"Codex executable or alias (default: {DEFAULT_CODEX_COMMAND}).",
    )
    parser.add_argument(
        "--default-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Default Codex model (default: {DEFAULT_CODEX_MODEL}).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=ALLOWED_PROVIDERS,
        help=f"How to launch Codex (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--wsl-distro",
        default=DEFAULT_WSL_DISTRO,
        help=f"WSL distro to use when provider is wsl (default: {DEFAULT_WSL_DISTRO}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    repos_parser = subparsers.add_parser("repos", help="List discovered Git repos.")
    repos_parser.add_argument("--limit", type=int, default=20)

    runs_parser = subparsers.add_parser("runs", help="List recent Codex tasks.")
    runs_parser.add_argument("--limit", type=int, default=10)

    subparsers.add_parser("status", help="Check whether Codex CLI is runnable from this environment.")

    show_parser = subparsers.add_parser("show", help="Show one Codex task.")
    show_parser.add_argument("task_id")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running Codex task.")
    cancel_parser.add_argument("task_id")

    run_parser = subparsers.add_parser("run", help="Queue a Codex task.")
    run_parser.add_argument("--repo", help="Repo name or absolute path. Defaults to workspace root.")
    run_parser.add_argument("--title", help="Short title for the queued task.")
    run_parser.add_argument("--model", help="Override the Codex model for this task.")
    run_parser.add_argument("--sandbox", default=DEFAULT_SANDBOX, choices=ALLOWED_SANDBOXES)
    run_parser.add_argument("--approval-mode", default=DEFAULT_APPROVAL_MODE, choices=ALLOWED_APPROVAL_MODES)
    run_parser.add_argument("--add-dir", action="append", default=[], help="Extra directory to make available.")
    run_parser.add_argument("--wait", action="store_true", help="Wait for task completion and print final status.")
    run_parser.add_argument("prompt", help="The task prompt to send to Codex CLI.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    bridge = CodexBridgeManager(
        workspace_root=Path(args.workspace_root),
        codex_command=args.codex_command,
        default_model=args.default_model,
        provider=args.provider,
        wsl_distro=args.wsl_distro,
    )

    if args.command == "repos":
        print_status_payload({"items": bridge.discover_repositories(max_repos=max(1, args.limit))})
        return 0
    if args.command == "runs":
        print_status_payload(bridge.list_tasks(limit=max(1, args.limit)))
        return 0
    if args.command == "status":
        print_status_payload(bridge.status())
        return 0
    if args.command == "show":
        print_status_payload(bridge.get_task_status(args.task_id))
        return 0
    if args.command == "cancel":
        print_status_payload(bridge.cancel_task(args.task_id))
        return 0
    if args.command == "run":
        payload = bridge.submit_task(
            prompt=args.prompt,
            repo_name_or_path=args.repo,
            title=args.title,
            model=args.model,
            sandbox=args.sandbox,
            approval_mode=args.approval_mode,
            add_dirs=args.add_dir,
        )
        if args.wait and payload["status"] in ACTIVE_STATUSES:
            payload = bridge.wait_for_task(payload["task_id"])
        print_status_payload(payload)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
