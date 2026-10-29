"""Per-(trial, plate) features for the choice classifier (SPEC 7.1)."""
from __future__ import annotations

import numpy as np

from gazedemo.types import TrialResult

G1 = ("ttfd_ms", "never_looked", "dd_ms", "dc")
G2 = ("share", "longest_dwell_ms", "mean_dwell_ms", "p_look")
G3 = ("dd_last1s_ms", "dd_last2s_ms", "is_last", "is_first")
G4 = ("mean_dist_norm",)
FEATURE_SETS = {
    "g1": G1,
    "g1g2": G1 + G2,
    "g1g2g3": G1 + G2 + G3,
    "all": G1 + G2 + G3 + G4,
    "liking_only": ("liking",),
    "all+liking": G1 + G2 + G3 + G4 + ("liking",),
}
FAR_DIST = 10.0  # mean_dist_norm without any valid sample: "far from every plate"


def plate_features(res: TrialResult, plate: str) -> dict[str, float]:
    """Empty TTFD → trial length + never_looked = 1; other empty values → 0."""
    m = res.per_aoi[plate]
    never = m.ttfd_ms is None
    return {
        "ttfd_ms": res.dur_ms if never else m.ttfd_ms,
        "never_looked": float(never),
        "dd_ms": m.dd_ms,
        "dc": float(m.dc),
        "share": m.share or 0.0,
        "longest_dwell_ms": m.longest_dwell_ms or 0.0,
        "mean_dwell_ms": m.mean_dwell_ms or 0.0,
        "p_look": m.p_look or 0.0,
        "dd_last1s_ms": m.dd_last1s_ms,
        "dd_last2s_ms": m.dd_last2s_ms,
        "is_last": float(m.is_last),
        "is_first": float(m.is_first),
        "mean_dist_norm": FAR_DIST if m.mean_dist_norm is None else m.mean_dist_norm,
    }


def dwell_spans(res: TrialResult, plate: str) -> str:
    """'start-end;start-end' in ms from t0, kept for dwell-time baselines."""
    return ";".join(f"{(d.t_start - res.t0) * 1000:.0f}-{(d.t_end - res.t0) * 1000:.0f}"
                    for d in res.dwells if d.aoi == plate)


def parse_spans(text) -> list[tuple[float, float]]:
    out = []
    for part in str(text or "").split(";"):
        a, sep, b = part.partition("-")
        if sep:
            out.append((float(a), float(b)))
    return out


def plate_rows(res: TrialResult, plates, liking: dict | None = None) -> list[dict]:
    rows = []
    for p in plates:
        row = {"plate": p, **plate_features(res, p)}
        if liking is not None:
            row["liking"] = liking.get(p)
        row.update(valid_ratio=res.valid_ratio, eff_hz=res.eff_hz, dwells_ms=dwell_spans(res, p))
        rows.append(row)
    return rows


def feature_matrix(rows, names) -> np.ndarray:
    """Rows of one trial → (K, F); missing → 0."""
    def val(v):
        return 0.0 if v is None or v == "" else float(v)
    return np.array([[val(r.get(n)) for n in names] for r in rows], dtype=float)
