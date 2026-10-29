from types import SimpleNamespace

import numpy as np
import pytest

from gazedemo.types import EllipseAOI, Sample


def dwell_cfg(**kw):
    base = dict(min_dwell_ms=100, max_gap_ms=150, max_interval_s=0.3, lookback_s=2.0, buffer_s=10.0)
    base.update(kw)
    return SimpleNamespace(**base)


def labelled_samples(spans_ms, hz=20, t_from_ms=0, t_to_ms=1000, default="none"):
    """Samples every 1000/hz ms in [t_from_ms, t_to_ms); label from (start_ms, end_ms, label) spans.
    Integer ms keep the interval boundaries exact."""
    step = 1000 // hz
    out = []
    for i, t_ms in enumerate(range(t_from_ms, t_to_ms, step)):
        label = default
        for a, b, lab in spans_ms:
            if a <= t_ms < b:
                label = lab
        mass = {} if label == "invalid" else {"A": 0.0, "B": 0.0, "C": 0.0, "none": 1.0}
        if label in ("A", "B", "C"):
            mass = {k: (0.9 if k == label else 0.0) for k in ("A", "B", "C")} | {"none": 0.1}
        out.append(Sample(t=t_ms / 1000.0, frame_idx=i, label=label, mass=mass,
                          gaze_xy=None, inout=1.0))
    return out


AOIS_ABC = [EllipseAOI("A", 320, 520, 110, 70), EllipseAOI("B", 640, 560, 110, 70),
            EllipseAOI("C", 960, 520, 110, 70)]


def gaussian_heatmap(x, y, width=1280, height=720, sigma_cells=1.5, shape=(64, 64)):
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    cx, cy = x / width * shape[1] - 0.5, y / height * shape[0] - 0.5
    return np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma_cells ** 2)).astype(np.float32)


@pytest.fixture
def assign_cfg():
    return SimpleNamespace(gamma=1.0, tau_mass=0.35, delta=0.10, tau_inout=0.5)
