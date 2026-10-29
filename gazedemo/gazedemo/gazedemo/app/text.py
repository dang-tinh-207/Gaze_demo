"""Vietnamese text on OpenCV images: cv2.putText has no diacritics, so draw with PIL + a TrueType
font; without a font fall back to cv2.putText on accent-stripped text."""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_WIN_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_REGULAR = [_WIN_FONTS / "arial.ttf", _WIN_FONTS / "segoeui.ttf", _WIN_FONTS / "tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
_BOLD = [_WIN_FONTS / "arialbd.ttf", _WIN_FONTS / "segoeuib.ttf", _WIN_FONTS / "tahomabd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
_SYMBOLS = {"—": "-", "–": "-", "·": "|", "≥": ">=", "≤": "<=", "→": "->", "×": "x", "…": "...", "đ": "d", "Đ": "D"}


def find_font(path=None, bold: bool = False) -> str | None:
    for cand in ([path] if path else []) + (_BOLD if bold else _REGULAR):
        if cand and Path(cand).is_file():
            return str(cand)
    return None


def ascii_fold(text: str) -> str:
    for a, b in _SYMBOLS.items():
        text = text.replace(a, b)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


@dataclass
class Txt:
    text: str
    xy: tuple[float, float]
    size: int = 22
    color: tuple[int, int, int] = (235, 235, 235)   # BGR
    anchor: str = "la"                              # PIL anchors: la, ra, ma, mm, ...
    bold: bool = False
    stroke: int = 0


class TextPainter:
    def __init__(self, font_path=None):
        self.regular = find_font(font_path)
        self.bold = find_font(None, bold=True) or self.regular
        self._fonts = {}
        self._masks = {}

    @property
    def unicode(self) -> bool:
        return self.regular is not None

    def font(self, size: int, bold: bool = False):
        from PIL import ImageFont
        key = (int(size), bold)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(self.bold if bold else self.regular, int(size))
        return self._fonts[key]

    def width(self, text: str, size: int, bold: bool = False) -> float:
        if self.unicode:
            return float(self.font(size, bold).getlength(text))
        (w, _), _ = cv2.getTextSize(ascii_fold(text), cv2.FONT_HERSHEY_SIMPLEX, size / 32, 2)
        return float(w)

    def draw(self, img: np.ndarray, items) -> np.ndarray:
        """Draw all items on the BGR image in place. Each distinct text is rasterised once by PIL
        into a small alpha mask (cached) and blended only where it lands: drawing through a full
        canvas PIL round-trip every frame cost ~40 ms."""
        for it in items:
            if not it.text:
                continue
            if self.unicode:
                self._blit(img, it)
            else:
                self._draw_cv(img, it)
        return img

    def _mask(self, it: Txt):
        key = (it.text, int(it.size), it.bold, int(it.stroke), it.anchor)
        hit = self._masks.get(key)
        if hit is None:
            from PIL import Image, ImageDraw
            font = self.font(it.size, it.bold)
            x0, y0, x1, y1 = font.getbbox(it.text, anchor=it.anchor, stroke_width=int(it.stroke))
            size = (max(1, x1 - x0), max(1, y1 - y0))

            def render(stroke):
                m = Image.new("L", size, 0)
                ImageDraw.Draw(m).text((-x0, -y0), it.text, font=font, fill=255, anchor=it.anchor,
                                       stroke_width=stroke, stroke_fill=255)
                return np.asarray(m, dtype=np.float32)[..., None] / 255.0

            hit = (x0, y0, render(0), render(int(it.stroke)) if it.stroke else None)
            if len(self._masks) > 4000:
                self._masks.clear()
            self._masks[key] = hit
        return hit

    def _blit(self, img: np.ndarray, it: Txt) -> None:
        x0, y0, fill, stroke = self._mask(it)
        X, Y = int(round(it.xy[0] + x0)), int(round(it.xy[1] + y0))
        h, w = fill.shape[:2]
        ax, ay = max(X, 0), max(Y, 0)
        bx, by = min(X + w, img.shape[1]), min(Y + h, img.shape[0])
        if ax >= bx or ay >= by:
            return
        cut = (slice(ay - Y, by - Y), slice(ax - X, bx - X))
        region = img[ay:by, ax:bx].astype(np.float32)
        if stroke is not None:  # black outline under the text
            region *= 1.0 - stroke[cut]
        a = fill[cut]
        region = region * (1.0 - a) + np.asarray(it.color, dtype=np.float32) * a
        img[ay:by, ax:bx] = (region + 0.5).astype(np.uint8)

    def _draw_cv(self, img, it: Txt):
        text = ascii_fold(it.text)
        scale = it.size / 32
        thick = 3 if it.bold else 2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = it.xy
        if it.anchor[0] == "m":
            x -= w / 2
        elif it.anchor[0] == "r":
            x -= w
        y += h if it.anchor[1] in "at" else (h / 2 if it.anchor[1] == "m" else 0)
        org = (int(x), int(y))
        if it.stroke:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2 * it.stroke, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, it.color, thick, cv2.LINE_AA)
