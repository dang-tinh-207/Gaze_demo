"""Shared immutable data types. Times are monotonic seconds; exported metrics are ms."""
from __future__ import annotations

from dataclasses import dataclass, field

NONE = "none"
OUT = "out"
INVALID = "invalid"
SPECIAL_LABELS = (NONE, OUT, INVALID)


def is_plate(label: str | None) -> bool:
    return label is not None and label not in SPECIAL_LABELS


@dataclass(frozen=True)
class EllipseAOI:
    name: str
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float = 0.0


@dataclass(frozen=True)
class Sample:
    t: float                              # s, time.monotonic() when the frame was captured
    frame_idx: int
    label: str                            # plate name | "none" | "out" | "invalid"
    mass: dict[str, float]                # heatmap mass per plate + "none"; {} if invalid
    gaze_xy: tuple[float, float] | None   # centre of the argmax heatmap cell, image pixels
    inout: float | None                   # None if the model has no in/out head
    latency_ms: float = 0.0               # capture → sample ready


@dataclass(frozen=True)
class Segment:
    label: str
    t_start: float
    t_end: float

    @property
    def dur(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class Dwell:
    aoi: str
    t_start: float
    t_end: float
    n_samples: int

    @property
    def dur(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class AOIMetrics:
    dd_ms: float
    dc: int
    ttfd_ms: float | None
    first_dwell_ms: float | None
    longest_dwell_ms: float | None
    mean_dwell_ms: float | None
    share: float | None             # DD / total DD of all plates
    p_look: float | None            # mean q_k over valid frames (with "none", sums to 1)
    p_look_plate: float | None      # p_look renormalised over plates only
    p_look_cal: float | None        # temperature-calibrated p_look (calibration.json)
    dd_last1s_ms: float
    dd_last2s_ms: float
    is_last: int
    is_first: int
    mean_dist_norm: float | None    # mean |gaze_xy - centre| / mean semi-axis


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    t0: float
    t1: float
    per_aoi: dict[str, AOIMetrics]
    first_aoi: str | None
    last_aoi: str | None
    none_ms: float
    out_ms: float
    invalid_ms: float
    valid_ratio: float
    n_samples: int
    eff_hz: float
    p_look_none: float | None
    p_look_cal_none: float | None
    dwells: tuple[Dwell, ...] = ()
    provisional: bool = False
    label: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def dur_ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0
