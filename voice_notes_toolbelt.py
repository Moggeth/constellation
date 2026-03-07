from __future__ import annotations

from dataclasses import dataclass
from typing import Any

AVAILABLE_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)

VOICE_CHANGE_REQUIRES_RECONNECT = (
    "Realtime voice changes require a short reconnect after audio has started. "
    "The assistant can switch automatically and then continue in the new voice."
)

DEFAULT_SPEECH_SPEED = 1.2


@dataclass
class RuntimePreferences:
    speech_speed: float = DEFAULT_SPEECH_SPEED
    concise_mode: bool = True
    thinking_sound_enabled: bool = True
    voice: str = "alloy"

    def clamp(self) -> None:
        self.speech_speed = max(0.75, min(2.0, self.speech_speed))
        self.voice = normalize_voice_name(self.voice) or "alloy"


def normalize_voice_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in AVAILABLE_REALTIME_VOICES:
        raise ValueError(
            f"Unsupported voice '{text}'. Available voices: {', '.join(AVAILABLE_REALTIME_VOICES)}"
        )
    return text


def build_voice_catalog(current_voice: str | None = None) -> dict[str, Any]:
    return {
        "current_voice": current_voice,
        "available_realtime_voices": list(AVAILABLE_REALTIME_VOICES),
        "note": VOICE_CHANGE_REQUIRES_RECONNECT,
    }
