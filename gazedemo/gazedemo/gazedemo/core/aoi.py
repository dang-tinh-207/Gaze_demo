"""Plate AOIs (ellipses in image pixels), 64x64 area masks and per-frame label assignment."""
from __future__ import annotations

import numpy as np

from gazedemo.types import NONE, OUT, EllipseAOI


def ellipse_from_box(name: str, x0: float, y0: float, x1: float, y1: float,
                     margin: float = 0.0) -> EllipseAOI:
    """Ellipse inscribed in a drawn box, radii scaled by (1 + margin)."""
    scale = 1.0 + margin
    return EllipseAOI(name=name, cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0,
                      rx=abs(x1 - x0) / 2.0 * scale, ry=abs(y1 - y0) / 2.0 * scale)


def inside(aoi: EllipseAOI, x, y) -> np.ndarray:
    """Points (x, y) in pixels inside the (possibly rotated) ellipse."""
    a = np.deg2rad(aoi.angle_deg)
    dx = np.asarray(x, dtype=float) - aoi.cx
    dy = np.asarray(y, dtype=float) - aoi.cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (u / aoi.rx) ** 2 + (v / aoi.ry) ** 2 <= 1.0


def ellipse_mask(aoi: EllipseAOI, width: float, height: float,
                 shape: tuple[int, int] = (64, 64), sub: int = 4) -> np.ndarray:
    """Fraction of each heatmap cell's area inside the ellipse (sub x sub points per cell).
    Cell (i, j) covers x in [j, j+1)/gw * width, y in [i, i+1)/gh * height."""
    gh, gw = shape
    off = (np.arange(sub) + 0.5) / sub
    xs = (np.arange(gw)[:, None] + off).ravel() / gw * width
    ys = (np.arange(gh)[:, None] + off).ravel() / gh * height
    gx, gy = np.meshgrid(xs, ys)
    hit = inside(aoi, gx, gy).astype(np.float32)
    return hit.reshape(gh, sub, gw, sub).mean(axis=(1, 3))


def build_masks(aois, width, height, shape=(64, 64), sub=4) -> dict[str, np.ndarray]:
    return {a.name: ellipse_mask(a, width, height, shape, sub) for a in aois}


def assign_label(H, inout, masks, cfg) -> tuple[str, dict[str, float]]:
    """SPEC 4.2. cfg needs gamma, tau_mass, delta, tau_inout."""
    P = np.maximum(H, 0) ** cfg.gamma
    P = P / (P.sum() + 1e-9)
    mass = {k: float((P * W).sum()) for k, W in masks.items()}
    mass[NONE] = max(0.0, 1.0 - sum(mass.values()))
    if inout is not None and inout < cfg.tau_inout:
        return OUT, mass
    if not masks:
        return NONE, mass
    ranked = sorted(masks, key=lambda k: mass[k], reverse=True)
    best = ranked[0]
    second = mass[ranked[1]] if len(ranked) > 1 else 0.0
    if mass[best] >= cfg.tau_mass and mass[best] - second >= cfg.delta:
        return best, mass
    return NONE, mass


def cell_center(i: int, j: int, width: float, height: float,
                shape: tuple[int, int] = (64, 64)) -> tuple[float, float]:
    return ((j + 0.5) / shape[1] * width, (i + 0.5) / shape[0] * height)


def heatmap_argmax_xy(H, width: float, height: float) -> tuple[float, float]:
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    x, y = cell_center(int(i), int(j), width, height, H.shape)
    return float(x), float(y)


def q_matrix(masses, inouts, plates) -> np.ndarray:
    """SPEC 1.1 per-frame look distribution for many frames: rows [q_plate..., q_none] with
    q_k = inout · mass_k, q_none = 1 − Σ q_k; no inout head (None) → inout = 1."""
    M = np.array([[float(m.get(k, 0.0)) for k in plates] for m in masses], dtype=float).reshape(-1, len(plates))
    w = np.array([1.0 if v is None else float(v) for v in inouts], dtype=float)[:, None]
    Q = w * M
    return np.hstack([Q, np.maximum(0.0, 1.0 - Q.sum(axis=1, keepdims=True))])


def frame_q(mass: dict[str, float], inout: float | None, plates) -> np.ndarray:
    return q_matrix([mass], [inout], plates)[0]


def dist_norm(aoi: EllipseAOI, xy) -> float:
    return float(np.hypot(xy[0] - aoi.cx, xy[1] - aoi.cy) / ((aoi.rx + aoi.ry) / 2.0))


def bounding_box(aoi: EllipseAOI) -> tuple[float, float, float, float]:
    a = np.deg2rad(aoi.angle_deg)
    hw = float(np.hypot(aoi.rx * np.cos(a), aoi.ry * np.sin(a)))
    hh = float(np.hypot(aoi.rx * np.sin(a), aoi.ry * np.cos(a)))
    return aoi.cx - hw, aoi.cy - hh, aoi.cx + hw, aoi.cy + hh


def ellipses_overlap(a: EllipseAOI, b: EllipseAOI, step: float = 1.0) -> bool:
    """Pixel-level test on the intersection of the two bounding boxes."""
    ax0, ay0, ax1, ay1 = bounding_box(a)
    bx0, by0, bx1, by1 = bounding_box(b)
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return False
    gx, gy = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
    return bool(np.any(inside(a, gx, gy) & inside(b, gx, gy)))


def find_overlaps(aois, step: float = 1.0) -> list[tuple[str, str]]:
    return [(a.name, b.name) for i, a in enumerate(aois) for b in aois[i + 1:]
            if ellipses_overlap(a, b, step)]


def ray_angles(head_xy, aois) -> list[tuple[str, str, float]]:
    """Angle in degrees between the rays head → plate centre, for every pair of plates."""
    out = []
    for i, a in enumerate(aois):
        va = np.array([a.cx - head_xy[0], a.cy - head_xy[1]], dtype=float)
        for b in aois[i + 1:]:
            vb = np.array([b.cx - head_xy[0], b.cy - head_xy[1]], dtype=float)
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            cos = float(va @ vb / denom) if denom > 0 else 1.0
            out.append((a.name, b.name, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))))
    return out


def scale_aois(aois, sx: float, sy: float) -> list[EllipseAOI]:
    """Rescale to another frame size (exact for axis-aligned ellipses)."""
    return [EllipseAOI(a.name, a.cx * sx, a.cy * sy, a.rx * sx, a.ry * sy, a.angle_deg) for a in aois]
