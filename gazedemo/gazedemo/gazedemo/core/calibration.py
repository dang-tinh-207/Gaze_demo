"""p_look calibration (SPEC 1.1): temperature scaling of per-frame q, reliability table, ECE."""
from __future__ import annotations

import numpy as np


def apply_temperature(Q, T: float, eps: float) -> np.ndarray:
    """Row-wise softmax(log(q + eps) / T) over [plates..., none]."""
    L = np.log(np.asarray(Q, dtype=float) + eps) / float(T)
    L = L - L.max(axis=-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=-1, keepdims=True)


def nll(P, y) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(Q, y, eps: float, t_min: float = 0.05, t_max: float = 20.0) -> float:
    """T minimising the log-loss: coarse grid on log T, then golden-section refinement."""
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=int)

    def f(log_t):
        return nll(apply_temperature(Q, np.exp(log_t), eps), y)

    grid = np.linspace(np.log(t_min), np.log(t_max), 61)
    i = int(np.argmin([f(g) for g in grid]))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2.0))


def reliability(Q, y, n_bins: int = 5) -> tuple[list[dict], float | None]:
    """Frames binned by q_max = max over [plates..., none]; per bin: share of frames whose argmax
    is the true (cued) class. Returns (rows, ECE)."""
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    if len(y) == 0:
        return [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0, "conf": None, "acc": None}
                for b in range(n_bins)], None
    Q = np.asarray(Q, dtype=float)
    conf = Q.max(axis=1)
    correct = Q.argmax(axis=1) == y
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        conf_b = float(conf[m].mean()) if k else None
        acc_b = float(correct[m].mean()) if k else None
        if k:
            ece += k / len(y) * abs(acc_b - conf_b)
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k,
                     "conf": conf_b, "acc": acc_b})
    return rows, float(ece)
