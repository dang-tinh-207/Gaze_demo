"""Window for live / cue / collect (SPEC 5.1): video + ellipses + argmax point + heatmap (H),
status bar, per-plate table. cv2.imshow / waitKey are only called from the main thread."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gazedemo.app.text import TextPainter, Txt
from gazedemo.types import EllipseAOI, Sample, TrialResult

PLATE_COLORS = [(80, 80, 240), (80, 210, 80), (240, 160, 60), (220, 90, 220), (60, 210, 230)]  # BGR
WHITE, GREY, YELLOW, RED, BG = (240, 240, 240), (150, 150, 150), (60, 220, 250), (90, 90, 255), (28, 28, 28)
_KEYS = {13: "enter", 10: "enter", 27: "esc", 32: "space", 8: "backspace", 127: "backspace"}


def key_name(code) -> str | None:
    if code is None or code < 0:
        return None
    code &= 0xFF
    if code in _KEYS:
        return _KEYS[code]
    ch = chr(code)
    return ch.lower() if ch.isprintable() else None


def plate_color(plates, name) -> tuple[int, int, int]:
    return PLATE_COLORS[plates.index(name) % len(PLATE_COLORS)] if name in plates else GREY


@dataclass
class Screen:
    kind: str = "video"          # "video": camera + overlays | "text": dark screen, text only
    big: str | None = None       # large centred text, e.g. "LOOK AT A"
    big_color: tuple | None = None
    big_size: float = 0.2        # font height as a fraction of the frame height
    lines: list[str] = field(default_factory=list)   # centred lines (text) / bottom banner (video)
    show_table: bool = True


@dataclass
class View:
    mode: str
    frame: np.ndarray | None = None
    sample: Sample | None = None
    heatmap: np.ndarray | None = None
    aois: list[EllipseAOI] = field(default_factory=list)
    hz: float | None = None
    latency_ms: float | None = None
    trial_id: int | None = None
    trial_t: float | None = None
    result: TrialResult | None = None
    foods: dict[str, str] = field(default_factory=dict)
    p_choice: dict[str, float] | None = None
    screen: Screen = field(default_factory=Screen)
    help: str = ""
    status: list[str] = field(default_factory=list)


def _fmt(v, digits=0, empty="—"):
    if v is None:
        return empty
    return f"{v:.{digits}f}"


class UI:
    def __init__(self, ui_cfg, plates, window: str = "gazedemo"):
        self.cfg = ui_cfg
        self.plates = list(plates)
        self.painter = TextPainter(ui_cfg.font)
        self.heatmap_on = bool(ui_cfg.heatmap)
        self.window = window
        self._open = False

    def toggle_heatmap(self) -> None:
        self.heatmap_on = not self.heatmap_on

    # ---------- drawing ----------

    def render(self, v: View) -> np.ndarray:
        h, w = v.frame.shape[:2] if v.frame is not None else (720, 1280)
        fs = int(self.cfg.font_size)
        canvas = np.full((h, w + int(self.cfg.panel_width), 3), BG, np.uint8)
        video = canvas[:, :w]
        items: list[Txt] = []
        is_video = v.screen.kind == "video" and v.frame is not None
        if is_video:
            video[:] = v.frame
            if self.heatmap_on and v.heatmap is not None:
                self._blend_heatmap(video, v.heatmap)
            self._draw_aois(video, v, items, fs)
            if v.sample is not None and v.sample.gaze_xy is not None:
                x, y = (int(round(c)) for c in v.sample.gaze_xy)
                cv2.circle(video, (x, y), 11, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(video, (x, y), 8, YELLOW, -1, cv2.LINE_AA)
            self._status_bar(video, v, items, fs)
        big = int(h * v.screen.big_size)
        if v.screen.big:
            y = h / 2 - (big * 0.6 if v.screen.lines else 0)
            items.append(Txt(v.screen.big, (w / 2, y), big, v.screen.big_color or WHITE, "mm", True,
                             3 if is_video else 0))
        if v.screen.lines:
            if is_video:
                self._banner(video, v.screen.lines, items, fs)
            else:
                y0 = h / 2 + (big * 0.45 if v.screen.big else -len(v.screen.lines) * fs * 0.9)
                for i, line in enumerate(v.screen.lines):
                    items.append(Txt(line, (w / 2, y0 + i * fs * 1.9), int(fs * 1.4), WHITE, "ma"))
        self._panel(canvas, w, v, items, fs)
        return self.painter.draw(canvas, items)

    def _blend_heatmap(self, video, heatmap):
        """Alpha ∝ heatmap; only the bounding box of cells above 3 % of the peak is blended."""
        h, w = video.shape[:2]
        peak = float(heatmap.max())
        if peak <= 0:
            return
        small = (heatmap / peak).astype(np.float32)
        rows, cols = np.nonzero(small > 0.03)
        gh, gw = small.shape
        x0, x1 = int(max(0, (cols.min() - 1) * w / gw)), int(min(w, (cols.max() + 2) * w / gw))
        y0, y1 = int(max(0, (rows.min() - 1) * h / gh)), int(min(h, (rows.max() + 2) * h / gh))
        norm = np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1], 0.0, 1.0)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = norm * float(self.cfg.heatmap_alpha)
        video[y0:y1, x0:x1] = cv2.blendLinear(video[y0:y1, x0:x1], colored, 1.0 - a, a)

    def _draw_aois(self, video, v: View, items, fs):
        current = v.sample.label if v.sample is not None else None
        for aoi in v.aois:
            color = plate_color(self.plates, aoi.name)
            thick = 6 if aoi.name == current else 2
            cv2.ellipse(video, (int(aoi.cx), int(aoi.cy)), (int(aoi.rx), int(aoi.ry)), aoi.angle_deg,
                        0, 360, color, thick, cv2.LINE_AA)
            food = v.foods.get(aoi.name)
            label = f"{aoi.name} · {food}" if food else aoi.name
            items.append(Txt(label, (aoi.cx, aoi.cy - aoi.ry - 8), int(fs * 1.2), color, "ms", True, 2))

    def _status_bar(self, video, v: View, items, fs):
        bar_h = int(fs * 1.8)
        video[:bar_h] = (video[:bar_h] * 0.35).astype(np.uint8)
        s = v.sample
        parts = [f"{_fmt(v.hz, 1)} Hz", f"{_fmt(v.latency_ms)} ms",
                 f"inout {_fmt(s.inout, 2) if s is not None else '—'}",
                 f"label {s.label if s is not None else '—'}", v.mode.upper()]
        if v.trial_id is not None:
            parts += [f"trial {v.trial_id}", f"{_fmt(v.trial_t, 1)} s"]
        items.append(Txt("   ".join(parts), (12, bar_h / 2), fs, WHITE, "lm"))

    def _banner(self, video, lines, items, fs):
        h = video.shape[0]
        line_h = int(fs * 1.9)
        top = h - line_h * len(lines) - 14
        video[top:] = (video[top:] * 0.3).astype(np.uint8)
        for i, line in enumerate(lines):
            items.append(Txt(line, (video.shape[1] / 2, top + 10 + i * line_h), int(fs * 1.35), WHITE, "ma"))

    def _columns(self, v: View, cal: bool):
        cols = [("plate", 64, "l"), ("food", 104, "l"), ("TTFD", 80, "r"), ("DD", 82, "r"), ("DC", 50, "r"),
                ("p_look", 92, "r")]
        if cal:
            cols.append(("p_cal", 86, "r"))
        if v.p_choice is not None:
            cols.append(("p_choice", 112, "r"))
        return cols

    def _panel(self, canvas, x_video, v: View, items, fs):
        x0 = x_video + 18
        width = canvas.shape[1] - x0 - 18
        y = 14
        title = v.mode.upper() + (f"   Trial {v.trial_id}" if v.trial_id is not None else "")
        items.append(Txt(title, (x0, y), fs + 6, WHITE, "la", True))
        y += fs * 2
        r = v.result
        if r is not None:
            t_s = (r.t1 - r.t0) if v.trial_t is None else v.trial_t
            line = f"t = {t_s:.1f} s   {_fmt(v.hz, 1)} Hz   valid {r.valid_ratio:.0%}"
            items.append(Txt(line, (x0, y), fs, GREY))
            y += fs * 1.8
        if v.screen.show_table and r is not None:
            cal = any(m.p_look_cal is not None for m in r.per_aoi.values())
            cols = self._columns(v, cal)
            scale = min(1.0, width / sum(c[1] for c in cols))
            tfs = max(12, int(fs * scale))
            xs, x = [], x0
            for _, cw, _ in cols:
                xs.append(x)
                x += cw * scale
            for (name, cw, align), cx in zip(cols, xs):
                items.append(Txt(name, (cx + cw * scale if align == "r" else cx, y), tfs, GREY,
                                 "ra" if align == "r" else "la"))
            y += tfs * 1.7
            for p in self.plates:
                m = r.per_aoi.get(p)
                if m is None:
                    continue
                food = self._fit(v.foods.get(p, ""), cols[1][1] * scale - 6, tfs)
                vals = [p, food, _fmt(m.ttfd_ms), _fmt(m.dd_ms), str(m.dc), _fmt(m.p_look, 2)]
                if cal:
                    vals.append(_fmt(m.p_look_cal, 2))
                if v.p_choice is not None:
                    vals.append(_fmt(v.p_choice.get(p), 2))
                color = plate_color(self.plates, p)
                for i, ((_, cw, align), cx, val) in enumerate(zip(cols, xs, vals)):
                    items.append(Txt(val, (cx + cw * scale if align == "r" else cx, y), tfs,
                                     color if i == 0 else WHITE, "ra" if align == "r" else "la", i == 0))
                y += tfs * 1.6
            none_vals = {5: _fmt(r.p_look_none, 2)}
            if cal:
                none_vals[6] = _fmt(r.p_look_cal_none, 2)
            items.append(Txt("none", (xs[0], y), tfs, GREY))
            for i, val in none_vals.items():
                items.append(Txt(val, (xs[i] + cols[i][1] * scale, y), tfs, GREY, "ra"))
            y += tfs * 1.6
            note = "provisional (trial running)" if r.provisional else "trial result"
            items.append(Txt(f"TTFD/DD in ms · DC = visits · {note}", (x0, y), int(tfs * 0.85), GREY))
            y += tfs * 2
        for line in v.status:
            items.append(Txt(line, (x0, y), fs, YELLOW))
            y += fs * 1.5
        if v.help:
            items.append(Txt(v.help, (x0, canvas.shape[0] - fs * 1.6), int(fs * 0.85), GREY))

    def _fit(self, text: str, max_w: float, size: int) -> str:
        if self.painter.width(text, size) <= max_w:
            return text
        while text and self.painter.width(text + "…", size) > max_w:
            text = text[:-1]
        return text + "…"

    # ---------- window ----------

    def show(self, canvas) -> str | None:
        if not self._open:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            h, w = canvas.shape[:2]
            s = min(1.0, 1600 / w)
            cv2.resizeWindow(self.window, int(w * s), int(h * s))
            self._open = True
        cv2.imshow(self.window, canvas)
        key = key_name(cv2.waitKey(1))
        try:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                return "q"  # window closed with the mouse
        except cv2.error:
            return "q"
        return key

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.window)
            except cv2.error:
                pass
            self._open = False
