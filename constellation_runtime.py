from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = SCRIPT_DIR / "workspace"
RUNTIME_PATHS_PATH = SCRIPT_DIR / "toolbelt" / "runtime_paths.json"


def _resolve_runtime_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = SCRIPT_DIR / candidate
    return candidate.resolve()


@dataclass
class RuntimePaths:
    workspace_root: str = str(DEFAULT_WORKSPACE_ROOT)
    library_roots: list[str] = field(default_factory=list)

    def normalize(self) -> None:
        self.workspace_root = str(_resolve_runtime_path(self.workspace_root))

        normalized_libraries: list[str] = []
        seen: set[str] = set()
        for raw_root in self.library_roots:
            text = str(raw_root).strip()
            if not text:
                continue
            resolved = str(_resolve_runtime_path(text))
            normalized_key = resolved.lower()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            normalized_libraries.append(resolved)
        self.library_roots = normalized_libraries

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_root).resolve()

    @property
    def library_paths(self) -> list[Path]:
        return [Path(root).resolve() for root in self.library_roots]

    @property
    def existing_library_paths(self) -> list[Path]:
        return [root for root in self.library_paths if root.exists() and root.is_dir()]

    @property
    def repo_roots(self) -> list[Path]:
        roots = [self.workspace_path, *self.existing_library_paths]
        unique_roots: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            normalized = str(root).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_roots.append(root)
        return unique_roots

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_paths() -> RuntimePaths:
    if not RUNTIME_PATHS_PATH.exists():
        paths = RuntimePaths()
        paths.normalize()
        return paths
    try:
        payload = json.loads(RUNTIME_PATHS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        paths = RuntimePaths()
        paths.normalize()
        return paths
    if not isinstance(payload, dict):
        paths = RuntimePaths()
        paths.normalize()
        return paths

    library_roots_raw = payload.get("library_roots")
    if isinstance(library_roots_raw, list):
        library_roots = [str(item) for item in library_roots_raw]
    else:
        legacy_root = str(payload.get("legacy_library_root", "")).strip()
        library_roots = [legacy_root] if legacy_root else []

    paths = RuntimePaths(
        workspace_root=str(payload.get("workspace_root", DEFAULT_WORKSPACE_ROOT)),
        library_roots=library_roots,
    )
    paths.normalize()
    return paths


def save_runtime_paths(paths: RuntimePaths) -> None:
    paths.normalize()
    RUNTIME_PATHS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATHS_PATH.write_text(json.dumps(paths.to_payload(), indent=2), encoding="utf-8")


def ensure_runtime_layout(paths: RuntimePaths) -> RuntimePaths:
    paths.normalize()
    paths.workspace_path.mkdir(parents=True, exist_ok=True)
    for child in ("scratch", "repos", "notes", "exports"):
        (paths.workspace_path / child).mkdir(parents=True, exist_ok=True)
    save_runtime_paths(paths)
    return paths


def mount_library(paths: RuntimePaths, library_root: str) -> RuntimePaths:
    paths.normalize()
    mounted = str(_resolve_runtime_path(library_root))
    if mounted.lower() not in {root.lower() for root in paths.library_roots}:
        paths.library_roots.append(mounted)
    paths.normalize()
    save_runtime_paths(paths)
    return paths


def unmount_library(paths: RuntimePaths, library_root: str) -> RuntimePaths:
    target = str(_resolve_runtime_path(library_root))
    paths.library_roots = [root for root in paths.library_roots if root.lower() != target.lower()]
    paths.normalize()
    save_runtime_paths(paths)
    return paths


def reset_runtime_paths() -> RuntimePaths:
    paths = RuntimePaths()
    return ensure_runtime_layout(paths)
