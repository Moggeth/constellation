from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voice_notes_toolbelt import AVAILABLE_REALTIME_VOICES, normalize_voice_name

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "toolbelt" / "realtime_tray_settings.json"
LOG_PATH = SCRIPT_DIR / "toolbelt" / "realtime_voice.log"


@dataclass
class RealtimeTraySettings:
    voice: str = "verse"
    auto_start: bool = False

    def clamp(self) -> None:
        self.voice = normalize_voice_name(self.voice) or "verse"


def load_tray_settings() -> RealtimeTraySettings:
    if not SETTINGS_PATH.exists():
        settings = RealtimeTraySettings()
        settings.clamp()
        return settings
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        settings = RealtimeTraySettings()
        settings.clamp()
        return settings
    settings = RealtimeTraySettings(**payload)
    settings.clamp()
    return settings


def save_tray_settings(settings: RealtimeTraySettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def open_path(path: Path) -> None:
    target = str(path)
    if os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
        return
    subprocess.Popen(["xdg-open", target])


def build_realtime_tray_image(mode: str = "idle", frame_index: int = 0) -> Any:
    from PIL import Image, ImageDraw

    palettes = {
        "idle": {
            "bg": "#07111f",
            "halo": "#132646",
            "line": "#6db5ff",
            "star": "#f9e38b",
            "icon": "#eef6ff",
            "accent": "#4cc9f0",
        },
        "live": {
            "bg": "#08121e",
            "halo": "#0d2f48",
            "line": "#79d2ff",
            "star": "#ffe38a",
            "icon": "#ffffff",
            "accent": "#7df9ff",
        },
        "stopped": {
            "bg": "#121826",
            "halo": "#2f3c58",
            "line": "#8ea0c8",
            "star": "#d8dff5",
            "icon": "#f5f7ff",
            "accent": "#a4b0d1",
        },
        "error": {
            "bg": "#220c11",
            "halo": "#47202a",
            "line": "#ff8b9c",
            "star": "#ffd7a8",
            "icon": "#fff5f5",
            "accent": "#ff6b7f",
        },
    }
    palette = palettes.get(mode, palettes["idle"])
    image = Image.new("RGBA", (64, 64), palette["bg"])
    draw = ImageDraw.Draw(image)

    draw.ellipse((6, 6, 58, 58), fill=palette["halo"])
    draw.ellipse((12, 12, 52, 52), fill=palette["bg"])

    stars = [(13, 14), (19, 24), (14, 44), (25, 49), (37, 14), (47, 22), (51, 39), (41, 50)]
    for index, (x, y) in enumerate(stars):
        twinkle = (frame_index + index) % 6
        radius = 1 if twinkle else 2
        color = palette["star"] if twinkle else "#ffffff"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    constellation_points = [(18, 44), (28, 34), (38, 38), (46, 26)]
    draw.line(constellation_points, fill=palette["line"], width=2)
    for x, y in constellation_points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=palette["star"])

    orbit_radius = 18
    orbit_center = (32, 32)
    draw.arc(
        (
            orbit_center[0] - orbit_radius,
            orbit_center[1] - orbit_radius,
            orbit_center[0] + orbit_radius,
            orbit_center[1] + orbit_radius,
        ),
        start=210,
        end=520,
        fill=palette["accent"],
        width=2,
    )
    angle_positions = [(49, 22), (52, 33), (48, 45), (37, 50), (24, 49), (15, 39)]
    highlight = angle_positions[frame_index % len(angle_positions)]
    draw.ellipse((highlight[0] - 3, highlight[1] - 3, highlight[0] + 3, highlight[1] + 3), fill="#ffffff")

    draw.rounded_rectangle((24, 18, 40, 34), radius=8, outline=palette["icon"], width=3)
    draw.rectangle((30, 34, 34, 45), fill=palette["icon"])
    draw.rounded_rectangle((24, 44, 40, 48), radius=2, fill=palette["icon"])
    draw.arc((21, 14, 43, 39), start=200, end=340, fill=palette["accent"], width=2)
    return image


class IgnoreErrors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return True


def make_voice_action(controller: "RealtimeTrayController", voice: str) -> Any:
    def action(icon: Any, item: Any) -> None:
        controller.select_voice(icon, item, voice)

    return action


def make_voice_checked(controller: "RealtimeTrayController", voice: str) -> Any:
    def checked(item: Any) -> bool:
        return controller.settings.voice == voice

    return checked


class RealtimeTrayController:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.settings = load_tray_settings()
        self.icon: Any = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.frame_index = 0
        self.status = "Idle"
        self.detail = "Ready to launch."
        self.mode = "idle"

    def bind_icon(self, icon: Any) -> None:
        self.icon = icon
        worker = threading.Thread(target=self._animation_loop, daemon=True)
        worker.start()
        if self.settings.auto_start:
            self.start_session(icon, None)

    def _animation_loop(self) -> None:
        while not self.stop_event.wait(0.8):
            with self.lock:
                if self.process is not None:
                    return_code = self.process.poll()
                    if return_code is not None:
                        self.process = None
                        if self.log_handle is not None:
                            self.log_handle.close()
                            self.log_handle = None
                        self.mode = "stopped" if return_code == 0 else "error"
                        self.status = "Stopped" if return_code == 0 else "Error"
                        self.detail = (
                            "Voice session finished."
                            if return_code == 0
                            else f"Voice session exited with code {return_code}."
                        )
                self.frame_index = (self.frame_index + 1) % 12
                icon = self.icon
                mode = self.mode
                status = self.status
                detail = self.detail
            if icon is not None:
                icon.icon = build_realtime_tray_image(mode=mode, frame_index=self.frame_index)
                icon.title = f"Constellation Voice\n{status}: {detail}"
                with IgnoreErrors():
                    icon.update_menu()

    def is_running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def set_voice(self, voice: str) -> None:
        self.settings.voice = normalize_voice_name(voice) or self.settings.voice
        save_tray_settings(self.settings)
        self.detail = f"Default voice set to {self.settings.voice}."
        if self.icon is not None:
            with IgnoreErrors():
                self.icon.update_menu()

    def select_voice(self, icon: Any, item: Any, voice: str) -> None:
        self.set_voice(voice)
        if self.is_running():
            self.restart_session(icon, item)

    def toggle_auto_start(self, icon: Any, item: Any) -> None:
        self.settings.auto_start = not self.settings.auto_start
        save_tray_settings(self.settings)
        self.detail = "Updated tray auto-start preference."
        with IgnoreErrors():
            icon.update_menu()

    def toggle_session(self, icon: Any, item: Any) -> None:
        if self.is_running():
            self.stop_session(icon, item)
            return
        self.start_session(icon, item)

    def toggle_session_label(self) -> str:
        return "Disable Voice Mode" if self.is_running() else "Enable Voice Mode"

    def build_child_command(self) -> list[str]:
        script_path = SCRIPT_DIR / "voice_notes_realtime.py"
        command = [
            sys.executable,
            str(script_path),
            "--tray-child",
            "--voice",
            self.settings.voice,
            "--model",
            self.args.model,
            "--transcription-model",
            self.args.transcription_model,
            "--workspace-root",
            self.args.workspace_root,
            "--archive-root",
            self.args.archive_root,
            "--ingest-script",
            self.args.ingest_script,
            "--speech-speed",
            str(self.args.speech_speed),
            "--max-output-tokens",
            str(self.args.max_output_tokens),
        ]
        if self.args.temperature is not None:
            command.extend(["--temperature", str(self.args.temperature)])
        if self.args.verbose_responses:
            command.append("--verbose-responses")
        if self.args.no_greeting:
            command.append("--no-greeting")
        if self.args.no_thinking_sound:
            command.append("--no-thinking-sound")
        if self.args.input_device is not None:
            command.extend(["--input-device", str(self.args.input_device)])
        if self.args.output_device is not None:
            command.extend(["--output-device", str(self.args.output_device)])
        return command

    def notify(self, message: str) -> None:
        if self.icon is None:
            return
        try:
            self.icon.notify(message, "Constellation Voice")
        except Exception:
            return

    def start_session(self, icon: Any, item: Any) -> None:
        if self.is_running():
            self.detail = "Voice session is already running."
            return
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_handle = LOG_PATH.open("a", encoding="utf-8")
        log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting voice session.\n")
        log_handle.flush()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                self.build_child_command(),
                cwd=str(SCRIPT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log_handle.close()
            self.mode = "error"
            self.status = "Error"
            self.detail = str(exc)
            self.notify(f"Constellation could not start voice mode: {exc}")
            return

        with self.lock:
            self.process = process
            self.log_handle = log_handle
            self.mode = "live"
            self.status = "Live"
            self.detail = f"Running with {self.settings.voice}."
        self.notify(f"Voice mode started with the {self.settings.voice} voice.")

    def stop_session(self, icon: Any, item: Any) -> None:
        with self.lock:
            process = self.process
        if process is None:
            self.mode = "stopped"
            self.status = "Stopped"
            self.detail = "Voice session is not running."
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        with self.lock:
            self.process = None
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            self.mode = "stopped"
            self.status = "Stopped"
            self.detail = "Voice session stopped."
        self.notify("Voice mode stopped.")

    def restart_session(self, icon: Any, item: Any) -> None:
        if self.is_running():
            self.stop_session(icon, item)
            time.sleep(0.5)
        self.start_session(icon, item)

    def open_workspace(self, icon: Any, item: Any) -> None:
        open_path(Path(self.args.workspace_root))

    def open_log_file(self, icon: Any, item: Any) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_PATH.exists():
            LOG_PATH.write_text("", encoding="utf-8")
        open_path(LOG_PATH)

    def on_quit(self, icon: Any, item: Any) -> None:
        self.stop_event.set()
        if self.is_running():
            self.stop_session(icon, item)
        icon.stop()


def run_realtime_tray(args: Any) -> None:
    try:
        import pystray
    except ImportError as exc:
        raise SystemExit("Realtime tray mode requires 'pystray' and 'Pillow'. Install with: pip install pystray pillow") from exc

    controller = RealtimeTrayController(args)
    voice_menu = pystray.Menu(
        *[
            pystray.MenuItem(
                voice,
                make_voice_action(controller, voice),
                checked=make_voice_checked(controller, voice),
            )
            for voice in AVAILABLE_REALTIME_VOICES
        ]
    )
    menu = pystray.Menu(
        pystray.MenuItem(lambda item: f"Status: {controller.status}", None, enabled=False),
        pystray.MenuItem(lambda item: controller.detail, None, enabled=False),
        pystray.MenuItem(lambda item: controller.toggle_session_label(), controller.toggle_session, default=True),
        pystray.MenuItem("Start Voice Session", controller.start_session),
        pystray.MenuItem("Restart Voice Session", controller.restart_session),
        pystray.MenuItem("Stop Voice Session", controller.stop_session),
        pystray.MenuItem("Voice", voice_menu),
        pystray.MenuItem(
            "Launch On Tray Start",
            controller.toggle_auto_start,
            checked=lambda item: controller.settings.auto_start,
        ),
        pystray.MenuItem("Open Workspace", controller.open_workspace),
        pystray.MenuItem("Open Voice Log", controller.open_log_file),
        pystray.MenuItem("Quit", controller.on_quit),
    )
    icon = pystray.Icon(
        "constellation_voice",
        build_realtime_tray_image(),
        "Constellation Voice",
        menu,
    )
    controller.bind_icon(icon)
    try:
        icon.run()
    finally:
        controller.stop_event.set()
