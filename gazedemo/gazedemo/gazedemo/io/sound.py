"""Beep without blocking the main loop: winsound on Windows, terminal bell elsewhere."""
from __future__ import annotations

import sys
import threading


def beep(freq: int = 1000, ms: int = 150) -> None:
    def run():
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(int(freq), int(ms))
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass
    threading.Thread(target=run, name="beep", daemon=True).start()
