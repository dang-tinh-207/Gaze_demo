"""Trial metrics in [t0, t1): the single compute() used by live, cue, collect, offline and dataset."""
from __future__ import annotations

import numpy as np

from gazedemo.core.aoi import q_matrix
from gazedemo.core.calibration import apply_temperature
from gazedemo.core.dwell import clip_segments, dwell_timeline, to_dwells
from gazedemo.types import INVALID, NONE, OUT, AOIMetrics, TrialResult


def _overlap_s(dwells, a: float, b: float) -> float:
    return sum(max(0.0, min(d.t_end, b) - max(d.t_start, a)) for d in dwells)


def compute(samples, t0: float, t1: float, cfg, aois, *, trial_id: int = 0,
            provisional: bool = False, label: str | None = None, meta: dict | None = None,
            temperature: float | None = None, cal_eps: float = 1e-4) -> TrialResult:
    """cfg: the dwell section (min_dwell_ms, max_gap_ms, max_interval_s, lookback_s).

    Only samples with t0 - lookback_s <= t < t1 are used, so the result does not depend on how
    much history the caller kept, nor on samples after t1 ("only gaze before the beep")."""
    plates = [a.name for a in aois]
    t1 = max(t1, t0)
    lo = t0 - cfg.lookback_s
    sel = sorted((s for s in samples if lo <= s.t < t1), key=lambda s: s.t)
    segs = clip_segments(dwell_timeline(sel, lo, t1, cfg), t0, t1, cfg.min_dwell_ms)
    win = [s for s in sel if s.t >= t0]
    dwells = to_dwells(segs, [s.t for s in win])

    totals = {NONE: 0.0, OUT: 0.0, INVALID: 0.0}
    for s in segs:
        if s.label in totals:
            totals[s.label] += s.dur * 1000.0
    first_aoi = dwells[0].aoi if dwells else None
    last_aoi = dwells[-1].aoi if dwells else None
    dd_total = sum(d.dur for d in dwells) * 1000.0

    # p_look (SPEC 1.1): mean of per-frame q over valid frames in the window
    valid = [s for s in win if s.label != INVALID and s.mass]
    p_look = p_cal = None
    if valid:
        Q = q_matrix([s.mass for s in valid], [s.inout for s in valid], plates)
        p_look = Q.mean(axis=0)
        if temperature:
            p_cal = apply_temperature(Q, temperature, cal_eps).mean(axis=0)
    plate_sum = float(p_look[:-1].sum()) if p_look is not None else 0.0
    gaze = np.array([s.gaze_xy for s in win if s.label != INVALID and s.gaze_xy is not None],
                    dtype=float).reshape(-1, 2)

    per = {}
    for k, aoi in enumerate(aois):
        ds = [d for d in dwells if d.aoi == aoi.name]
        durs = [d.dur * 1000.0 for d in ds]
        dd = float(sum(durs))
        per[aoi.name] = AOIMetrics(
            dd_ms=dd,
            dc=len(ds),
            ttfd_ms=(ds[0].t_start - t0) * 1000.0 if ds else None,
            first_dwell_ms=durs[0] if ds else None,
            longest_dwell_ms=max(durs) if ds else None,
            mean_dwell_ms=dd / len(ds) if ds else None,
            share=dd / dd_total if dd_total > 0 else None,
            p_look=float(p_look[k]) if p_look is not None else None,
            p_look_plate=float(p_look[k] / plate_sum) if plate_sum > 0 else None,
            p_look_cal=float(p_cal[k]) if p_cal is not None else None,
            dd_last1s_ms=_overlap_s(ds, max(t0, t1 - 1.0), t1) * 1000.0,
            dd_last2s_ms=_overlap_s(ds, max(t0, t1 - 2.0), t1) * 1000.0,
            is_last=int(last_aoi == aoi.name),
            is_first=int(first_aoi == aoi.name),
            mean_dist_norm=(float(np.hypot(gaze[:, 0] - aoi.cx, gaze[:, 1] - aoi.cy).mean())
                            / ((aoi.rx + aoi.ry) / 2.0)) if len(gaze) else None,
        )

    dur_ms = (t1 - t0) * 1000.0
    return TrialResult(
        trial_id=trial_id, t0=t0, t1=t1, per_aoi=per,
        first_aoi=first_aoi, last_aoi=last_aoi,
        none_ms=totals[NONE], out_ms=totals[OUT], invalid_ms=totals[INVALID],
        valid_ratio=1.0 - totals[INVALID] / dur_ms if dur_ms > 0 else 0.0,
        n_samples=len(win),
        eff_hz=len(win) / (t1 - t0) if t1 > t0 else 0.0,
        p_look_none=float(p_look[-1]) if p_look is not None else None,
        p_look_cal_none=float(p_cal[-1]) if p_cal is not None else None,
        dwells=tuple(dwells), provisional=provisional, label=label, meta=dict(meta or {}),
    )
