"""Baselines, CV folds (by trial / by participant), top-1, log-loss, confusion, bootstrap CI."""
from __future__ import annotations

import numpy as np

BASELINES = ("chance", "argmax_dd", "argmax_p_look", "last_look", "first_look", "dwell_T", "argmax_liking")


def kfold_trials(n: int, k: int, seed: int) -> np.ndarray:
    """Fold id per trial (all plates of a trial share it)."""
    k = max(2, min(int(k), n))
    folds = np.empty(n, dtype=int)
    folds[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return folds


def group_folds(groups) -> np.ndarray:
    """Leave-one-group-out: fold id = participant index."""
    return np.unique(np.asarray(groups), return_inverse=True)[1]


def hard_proba(S, eps: float) -> np.ndarray:
    """Hard prediction as probabilities: top alternative(s) share 1 - eps, the rest share eps;
    ties split evenly; everything tied → uniform."""
    S = np.asarray(S, dtype=float)
    K = S.shape[1]
    top = np.isclose(S, S.max(axis=1, keepdims=True), rtol=0.0, atol=1e-9)
    n_top = top.sum(axis=1, keepdims=True)
    P = np.where(top, (1.0 - eps) / n_top, eps / np.maximum(K - n_top, 1))
    P[n_top[:, 0] == K] = 1.0 / K
    return P


def baseline_scores(name: str, data: dict, dwell_T_ms: float) -> np.ndarray:
    """data: (N, K) arrays dd_ms, p_look, is_last, is_first, liking; 'dwells': [N][K] (start, end) ms."""
    if name == "chance":
        return np.zeros_like(data["dd_ms"])
    if name == "argmax_dd":
        return data["dd_ms"]
    if name == "argmax_p_look":
        return data["p_look"]
    if name == "last_look":
        return data["is_last"]
    if name == "first_look":
        return data["is_first"]
    if name == "argmax_liking":
        return data["liking"]
    if name == "dwell_T":  # first plate holding a dwell ≥ T; none → argmax DD
        S = np.array(data["dd_ms"], dtype=float)
        for i, per_plate in enumerate(data["dwells"]):
            starts = np.array([min((a for a, b in spans if b - a >= dwell_T_ms), default=np.inf)
                               for spans in per_plate])
            if np.isfinite(starts).any():
                S[i] = -starts  # earliest long dwell wins; -inf for plates without one
        return S
    raise ValueError(f"unknown baseline: {name}")


def top1(P, y) -> np.ndarray:
    """Per-trial accuracy; ties give fractional credit (expected accuracy of a random tie-break)."""
    P = np.asarray(P, dtype=float)
    top = np.isclose(P, P.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    return top[np.arange(len(y)), y] / top.sum(axis=1)


def log_loss(P, y) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None))


def confusion(P, y, K: int) -> np.ndarray:
    """Rows = chosen plate, columns = predicted plate (ties split)."""
    P = np.asarray(P, dtype=float)
    top = np.isclose(P, P.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    C = np.zeros((K, K))
    for yi, row in zip(y, top):
        C[yi] += row / row.sum()
    return C


def bootstrap_ci(values, n_boot: int, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean, resampling trials."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(int(n_boot), len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
