"""`setup-aoi`: drag boxes A, B, C on a frame → inscribed ellipses → aoi.json (+ aoi_ref.png).
Click (or right-click) the user's head to draw rays head → plate and check their angles."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from gazedemo.app.text import TextPainter, Txt
from gazedemo.app.ui import RED, WHITE, YELLOW, key_name, plate_color
from gazedemo.core.aoi import ellipse_from_box, find_overlaps, ray_angles
from gazedemo.io.logs import save_aoi, save_image

MIN_BOX_PX = 8


class AOIEditor:
    """Tool state without GUI calls (tested in tests/test_setup_aoi.py)."""

    def __init__(self, names, margin: float = 0.10, min_angle_deg: float = 15.0):
        self.names = list(names)
        self.margin = float(margin)
        self.min_angle = float(min_angle_deg)
        self.boxes: list[tuple[float, float, float, float]] = []
        self.head: tuple[float, float] | None = None
        self._history: list[str] = []

    @property
    def next_name(self) -> str | None:
        return self.names[len(self.boxes)] if len(self.boxes) < len(self.names) else None

    def add_box(self, x0, y0, x1, y1) -> bool:
        if self.next_name is None or abs(x1 - x0) < MIN_BOX_PX or abs(y1 - y0) < MIN_BOX_PX:
            return False
        self.boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        self._history.append("box")
        return True

    def set_head(self, x, y) -> None:
        if self.head is None:
            self._history.append("head")
        self.head = (float(x), float(y))

    def undo(self) -> None:
        if self._history:
            if self._history.pop() == "box":
                self.boxes.pop()
            else:
                self.head = None

    def change_margin(self, delta: float) -> None:
        self.margin = round(min(1.0, max(-0.5, self.margin + delta)), 2)

    def ellipses(self):
        return [ellipse_from_box(n, *b, margin=self.margin) for n, b in zip(self.names, self.boxes)]

    def angles(self) -> list[tuple[str, str, float]]:
        return ray_angles(self.head, self.ellipses()) if self.head and len(self.boxes) >= 2 else []

    def problems(self) -> tuple[list[str], list[str]]:
        """(errors that block saving, warnings)."""
        errors = []
        if self.next_name is not None:
            errors.append(f"Not all {len(self.names)} plates drawn yet: {self.next_name} is missing")
        for a, b in find_overlaps(self.ellipses()):
            errors.append(f"Ellipses {a} and {b} overlap → U to redraw or − to reduce the margin")
        warnings = [] if self.head else ["User's head not clicked yet (needed to check the angles between rays)"]
        for a, b, ang in self.angles():
            if ang < self.min_angle:
                warnings.append(f"Ray angle {a}–{b} = {ang:.0f}° < {self.min_angle:.0f}° → these two plates are easy to confuse")
        return errors, warnings

    def extra(self) -> dict:
        return {"margin": self.margin,
                "boxes": {n: [round(v, 1) for v in b] for n, b in zip(self.names, self.boxes)},
                "head_xy": list(self.head) if self.head else None,
                "ray_angles_deg": {f"{a}-{b}": round(ang, 1) for a, b, ang in self.angles()},
                "created": datetime.now().isoformat(timespec="seconds")}


def render_editor(frame, ed: AOIEditor, painter: TextPainter, drag=None, message: str = "", help_bar=True):
    img = frame.copy()
    h, w = img.shape[:2]
    fs = max(16, h // 32)
    items = []
    top = h - int(fs * 4.6)
    if help_bar:  # darken first so shapes drawn over the bar stay bright
        img[top:] = (img[top:] * 0.3).astype(np.uint8)
    for name, box, e in zip(ed.names, ed.boxes, ed.ellipses()):
        color = plate_color(ed.names, name)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1, cv2.LINE_AA)
        cv2.ellipse(img, (int(e.cx), int(e.cy)), (int(e.rx), int(e.ry)), e.angle_deg, 0, 360, color, 3, cv2.LINE_AA)
        items.append(Txt(name, (e.cx, e.cy - e.ry - 6), int(fs * 1.4), color, "ms", True, 2))
        if ed.head:
            cv2.line(img, (int(ed.head[0]), int(ed.head[1])), (int(e.cx), int(e.cy)), WHITE, 1, cv2.LINE_AA)
    if ed.head:
        hx, hy = int(ed.head[0]), int(ed.head[1])
        cv2.drawMarker(img, (hx, hy), YELLOW, cv2.MARKER_CROSS, 28, 3)
        items.append(Txt("head", (hx + 16, hy - 16), fs, YELLOW, "ls", False, 2))
    if drag and drag.get("start") and drag.get("cur") and ed.next_name:
        (x0, y0), (x1, y1) = drag["start"], drag["cur"]
        cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 2)
        items.append(Txt(ed.next_name, (min(x0, x1), min(y0, y1) - 6), fs, YELLOW, "ls", True, 2))
    if help_bar:
        nxt = f"draw box for plate {ed.next_name}" if ed.next_name else "all plates drawn"
        items.append(Txt(f"Drag: {nxt} · click/right-click: head point · +/− margin {ed.margin:+.2f} · "
                         "U undo · R recapture · Enter save · Q quit", (12, top + 8), fs, WHITE))
        angles = ed.angles()
        line2 = ("Ray angles: " + " · ".join(f"{a}–{b} {ang:.0f}°" for a, b, ang in angles)) if angles else \
            "Click the user's head to measure the angles between the head → plate rays"
        items.append(Txt(line2, (12, top + 8 + fs * 1.5), fs, WHITE))
        errors, warnings = ed.problems()
        note = message or (errors[0] if errors and ed.next_name is None else (warnings[0] if warnings else ""))
        bad = bool(message) or (errors and ed.next_name is None)
        items.append(Txt(note, (12, top + 8 + fs * 3.0), fs, RED if bad else YELLOW))
    return painter.draw(img, items)


def _grab(source, settle_s: float = 0.6):
    """Newest frame after letting auto-exposure settle."""
    frame, t_end = None, time.monotonic() + settle_s
    while frame is None or time.monotonic() < t_end:
        item = source.read(timeout=1.0)
        if item is not None:
            frame = item[2].copy()
        elif source.finished:
            break
    if frame is None:
        raise RuntimeError("Could not get a frame from the source")
    return frame


def run_setup_aoi(cfg, source_spec, out_path: Path) -> int:
    from gazedemo.io.camera import open_source
    source = open_source(source_spec, cfg).start()
    try:
        frame = _grab(source)
        ed = AOIEditor(cfg.aoi.names, cfg.aoi.margin, cfg.aoi.min_ray_angle_deg)
        painter = TextPainter(cfg.ui.font)
        win = "gazedemo setup-aoi"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        drag = {"start": None, "cur": None}

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drag["start"] = drag["cur"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
                drag["cur"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
                (x0, y0), drag["start"] = drag["start"], None
                if abs(x - x0) < MIN_BOX_PX and abs(y - y0) < MIN_BOX_PX:
                    ed.set_head(x, y)
                else:
                    ed.add_box(x0, y0, x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                ed.set_head(x, y)

        cv2.setMouseCallback(win, on_mouse)
        message = ""
        while True:
            cv2.imshow(win, render_editor(frame, ed, painter, drag, message))
            key = key_name(cv2.waitKey(20))
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 or key in ("q", "esc"):
                print("[setup-aoi] quit without saving")
                return 1
            if key:
                message = ""
            if key in ("+", "="):
                ed.change_margin(0.02)
            elif key in ("-", "_"):
                ed.change_margin(-0.02)
            elif key == "u":
                ed.undo()
            elif key == "r":
                frame = _grab(source, 0.2)
                message = "Frame recaptured"
            elif key == "enter":
                errors, warnings = ed.problems()
                if errors:
                    message = errors[0]
                    continue
                save(out_path, frame, ed, painter)
                for wmsg in warnings:
                    print(f"[setup-aoi] WARNING: {wmsg}")
                return 0
    finally:
        source.stop()
        cv2.destroyAllWindows()


def save(out_path: Path, frame, ed: AOIEditor, painter: TextPainter) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    save_aoi(out_path, ed.ellipses(), (w, h), **ed.extra())
    save_image(out_path.with_name("aoi_ref.png"), frame)
    save_image(out_path.with_name("aoi_ref_overlay.png"), render_editor(frame, ed, painter, help_bar=False))
    print(f"[setup-aoi] Saved {out_path} (+ aoi_ref.png, aoi_ref_overlay.png), image {w}×{h}, margin {ed.margin:+.2f}")
    for e in ed.ellipses():
        print(f"  {e.name}: centre ({e.cx:.0f}, {e.cy:.0f}), semi-axes ({e.rx:.0f}, {e.ry:.0f})")
    for a, b, ang in ed.angles():
        print(f"  ray angle {a}–{b}: {ang:.1f}°")
