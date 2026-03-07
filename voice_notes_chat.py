#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("faq_notes_chat.py")

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
