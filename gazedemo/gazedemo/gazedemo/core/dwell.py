"""Samples → labelled timeline → dwells (SPEC 4.3)."""
from __future__ import annotations

import numpy as np

from gazedemo.types import INVALID, NONE, Dwell, Segment, is_plate

EPS_MS = 1e-6                 # float slack for "≤ max_gap" / "< min_dwell" on exact boundaries
GAP_LABELS = (NONE, INVALID)  # what may be bridged between two dwells on the same plate


def merge_segments(segments) -> list[Segment]:
    """Drop empty segments and merge neighbours with the same label."""
    out: list[Segment] = []
    for s in segments:
        if s.t_end <= s.t_start:
            continue
        if out and out[-1].label == s.label:
            out[-1] = Segment(s.label, out[-1].t_start, s.t_end)
        else:
            out.append(s)
    return out


def build_timeline(samples, t_start: float, t_end: float, max_interval_s: float) -> list[Segment]:
    """Sample i covers [t_i, t_{i+1}), the last one [t_n, t_end); each cover is capped at
    max_interval_s and the excess is 'invalid'. Time before the first sample is 'invalid'."""
    if t_end <= t_start:
        return []
    pts = sorted((s for s in samples if t_start <= s.t < t_end), key=lambda s: s.t)
    if not pts:
        return [Segment(INVALID, t_start, t_end)]
    segs = []
    if pts[0].t > t_start:
        segs.append(Segment(INVALID, t_start, pts[0].t))
    for i, s in enumerate(pts):
        nxt = pts[i + 1].t if i + 1 < len(pts) else t_end
        covered = min(nxt, s.t + max_interval_s)
        segs.append(Segment(s.label, s.t, covered))
        if nxt > covered:
            segs.append(Segment(INVALID, covered, nxt))
    return merge_segments(segs)


def bridge_gaps(segments, max_gap_ms: float) -> list[Segment]:
    """X, then none/invalid segments totalling ≤ max_gap_ms, then X → one X segment.
    Never bridges across another plate or 'out'."""
    out = []
    i, n = 0, len(segments)
    while i < n:
        cur = segments[i]
        i += 1
        if is_plate(cur.label):
            while True:
                j, gap = i, 0.0
                while j < n and segments[j].label in GAP_LABELS:
                    gap += segments[j].dur
                    j += 1
                if i < j < n and segments[j].label == cur.label and gap * 1000 <= max_gap_ms + EPS_MS:
                    cur = Segment(cur.label, cur.t_start, segments[j].t_end)
                    i = j + 1
                else:
                    break
        out.append(cur)
    return merge_segments(out)


def filter_flicker(segments, min_dwell_ms: float) -> list[Segment]:
    """Plate segments shorter than min_dwell_ms become 'none'."""
    return merge_segments([Segment(NONE, s.t_start, s.t_end)
                           if is_plate(s.label) and s.dur * 1000 < min_dwell_ms - EPS_MS else s
                           for s in segments])


def dwell_timeline(samples, t_start: float, t_end: float, cfg) -> list[Segment]:
    """The 3 steps, in order: bridge gaps, filter flicker, bridge gaps again.
    cfg needs max_interval_s, max_gap_ms, min_dwell_ms."""
    segs = build_timeline(samples, t_start, t_end, cfg.max_interval_s)
    segs = bridge_gaps(segs, cfg.max_gap_ms)
    segs = filter_flicker(segs, cfg.min_dwell_ms)
    return bridge_gaps(segs, cfg.max_gap_ms)


def clip_segments(segments, t0: float, t1: float, min_dwell_ms: float) -> list[Segment]:
    """Cut to [t0, t1); a dwell left shorter than min_dwell_ms after cutting becomes 'none'."""
    out = []
    for s in segments:
        a, b = max(s.t_start, t0), min(s.t_end, t1)
        if b <= a:
            continue
        label = s.label
        if is_plate(label) and (b - a) * 1000 < min_dwell_ms - EPS_MS:
            label = NONE
        out.append(Segment(label, a, b))
    return merge_segments(out)


def to_dwells(segments, sample_times) -> list[Dwell]:
    t = np.sort(np.asarray(list(sample_times), dtype=float))
    out = []
    for s in segments:
        if is_plate(s.label):
            n = int(np.searchsorted(t, s.t_end) - np.searchsorted(t, s.t_start))
            out.append(Dwell(s.label, s.t_start, s.t_end, n))
    return out
