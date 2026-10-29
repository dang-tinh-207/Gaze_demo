"""Deterministic stand-ins for camera + Gazelle (tests, `synth`, demos without camera/GPU).

FakeCamera renders a scenario (which target is looked at over time) into frames. The target is
encoded as the grey level of a block in the top-left corner, which survives video compression, so
offline replay of a recorded fake run gives the same labels; a white dot marks the gaze point for
humans. FakeGaze decodes the block and returns a Gaussian heatmap at that target.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from gazedemo.core.aoi import dist_norm
from gazedemo.types import NONE, OUT, EllipseAOI

CODE_STEP = 24


def default_aois(width: int = 1280, height: int = 720) -> list[EllipseAOI]:
    """Three plates fanned out below a head at the top centre (synthetic source, synth, tests)."""
    sx, sy = width / 1280, height / 720
    return [EllipseAOI("A", 360 * sx, 540 * sy, 120 * sx, 75 * sy),
            EllipseAOI("B", 640 * sx, 600 * sy, 120 * sx, 75 * sy),
            EllipseAOI("C", 920 * sx, 540 * sy, 120 * sx, 75 * sy)]


def _far_point(aois, width, height) -> tuple[float, float]:
    """Grid point farthest (in plate radii) from every plate: where 'none' gaze lands."""
    best, best_d = (width / 2, height / 4), -1.0
    for y in np.linspace(0.15, 0.9, 16) * height:
        for x in np.linspace(0.1, 0.9, 17) * width:
            d = min(dist_norm(a, (x, y)) for a in aois) if aois else 1.0
            if d > best_d:
                best, best_d = (float(x), float(y)), d
    return best


@dataclass
class FakeWorld:
    width: int
    height: int
    targets: dict[str, tuple[float, float]]          # gaze point per target: plates + "none"
    aois: list[EllipseAOI] = field(default_factory=list)

    @classmethod
    def from_aois(cls, aois, width: int, height: int) -> "FakeWorld":
        targets = {a.name: (a.cx, a.cy) for a in aois}
        targets[NONE] = _far_point(aois, width, height)
        return cls(int(width), int(height), targets, list(aois))

    @property
    def names(self) -> list[str]:
        return [OUT, *self.targets]

    def code(self, name: str) -> int:
        return 16 + CODE_STEP * self.names.index(name)

    def render(self, name: str, idx: int = 0) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 40, np.uint8)
        for a in self.aois:
            cv2.ellipse(img, (int(a.cx), int(a.cy)), (int(a.rx), int(a.ry)), a.angle_deg, 0, 360,
                        (90, 90, 90), 2)
        if name != OUT:
            x, y = self.targets[name]
            cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), -1)
        cv2.putText(img, f"{name}  #{idx}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        img[4:28, 4:28] = self.code(name)
        return img

    def decode(self, frame: np.ndarray) -> str:
        v = float(frame[8:24, 8:24].mean())
        i = int(round((v - 16) / CODE_STEP))
        return self.names[i] if 0 <= i < len(self.names) else OUT


class FakeGaze:
    """Gazelle stand-in: decode the target from the frame, Gaussian heatmap there."""
    has_inout = True

    def __init__(self, world: FakeWorld, sigma_cells: float = 1.5, shape=(64, 64)):
        self.world = world
        self.sigma = sigma_cells
        self.shape = shape
        gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
        self._grid = (gx.astype(np.float32), gy.astype(np.float32))

    def heatmap_at(self, xy, sigma_cells: float | None = None) -> np.ndarray:
        s = self.sigma if sigma_cells is None else sigma_cells
        gx, gy = self._grid
        cx = xy[0] / self.world.width * self.shape[1] - 0.5
        cy = xy[1] / self.world.height * self.shape[0] - 0.5
        return np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s)).astype(np.float32)

    def infer(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, float | None]:
        name = self.world.decode(frame_rgb)
        if name == OUT:
            return np.full(self.shape, 0.01, np.float32), 0.05
        return self.heatmap_at(self.world.targets[name]), 0.95

    def describe(self) -> str:
        return "FakeGaze (no GPU)"


def scenario_from_spans(spans, default: str = NONE):
    """spans: [(t_from_s, t_to_s, target)], times relative to the camera start."""
    def at(t_rel: float) -> str:
        for a, b, name in spans:
            if a <= t_rel < b:
                return name
        return default
    return at


def demo_scenario(world: FakeWorld, period_spans=None):
    """Repeating look pattern over the plates, for `--source synthetic`."""
    plates = [n for n in world.targets if n != NONE]
    spans, t = [], 0.0
    for name, dur in period_spans or [(NONE, 1.0)] + [(p, 2.5) for p in plates] + [(OUT, 1.0)]:
        spans.append((t, t + dur, name))
        t += dur
    period = t
    inner = scenario_from_spans(spans)
    return lambda t_rel: inner(t_rel % period)


class FakeCamera:
    """Frames at `fps` following scenario(t_rel) -> target.
    Virtual time (default): as fast as possible, timestamps t_start + i/fps, clock() = last frame t.
    realtime=True: paced with time.monotonic() timestamps, for interactive demos."""

    def __init__(self, world: FakeWorld, scenario, fps: float = 15.0, duration: float | None = None,
                 t_start: float = 1000.0, realtime: bool = False, on_frame=None):
        self.world = world
        self.scenario = scenario
        self.fps = float(fps)
        self.duration = duration
        self.t_start = t_start
        self.realtime = realtime
        self.on_frame = on_frame
        self.finished = False
        self.error = None
        self._i = 0
        self._now = t_start
        self._t0_mono = None

    @property
    def size(self) -> tuple[int, int]:
        return self.world.width, self.world.height

    def start(self):
        self._t0_mono = time.monotonic()
        return self

    def read(self, timeout=None):
        t_rel = self._i / self.fps
        if self.duration is not None and t_rel >= self.duration:
            self.finished = True
            return None
        if self.realtime:
            wait = self._t0_mono + t_rel - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            t = time.monotonic()
        else:
            t = self.t_start + t_rel
        frame = self.world.render(self.scenario(t_rel), self._i)
        idx = self._i
        self._i += 1
        self._now = t
        if self.on_frame is not None:
            self.on_frame(idx, t, frame)
        return idx, t, frame

    def clock(self) -> float:
        return time.monotonic() if self.realtime else self._now

    def stop(self):
        self.finished = True
