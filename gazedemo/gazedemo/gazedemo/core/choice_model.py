"""Conditional logit: P(plate k | trial) = softmax_k(w · x_k). numpy only, L2, Newton's method
(exact Hessian; the objective is convex, so ~10 iterations reach the optimum)."""
from __future__ import annotations

import numpy as np


def _log_softmax(S: np.ndarray) -> np.ndarray:
    S = S - S.max(axis=1, keepdims=True)
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


def objective(w, X, y, l2: float):
    """Σ_trials -log P(y_n) + l2/2·|w|² → (value, gradient, Hessian). X: (N, K, F), y: (N,)."""
    n = np.arange(len(y))
    logp = _log_softmax(X @ w)
    P = np.exp(logp)
    f = -logp[n, y].sum() + 0.5 * l2 * float(w @ w)
    Ex = np.einsum("nk,nkf->nf", P, X)
    g = (Ex - X[n, y]).sum(axis=0) + l2 * w
    H = np.einsum("nk,nkf,nkg->fg", P, X, X) - Ex.T @ Ex + l2 * np.eye(len(w))
    return float(f), g, H


class ConditionalLogit:
    def __init__(self, feature_names, l2: float = 1.0):
        self.feature_names = list(feature_names)
        self.l2 = float(l2)
        self.mean = self.std = self.w = None
        self.n_trial = 0
        self.n_iter = 0

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit(self, X, y, max_iter: int = 100):
        """Standardise with this training set's mean/std, then Newton with backtracking."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.std = np.where(std > 1e-12, std, 1.0)
        Z = self._z(X)
        w = np.zeros(X.shape[-1])
        f, g, H = objective(w, Z, y, self.l2)
        ridge = 1e-9 * np.eye(len(w))
        for it in range(max_iter):
            step = np.linalg.solve(H + ridge, g)
            t = 1.0
            while True:
                w_new = w - t * step
                f_new, g_new, H_new = objective(w_new, Z, y, self.l2)
                if f_new <= f - 1e-4 * t * float(g @ step) or t < 1e-10:
                    break
                t *= 0.5
            moved = float(np.abs(w_new - w).max())
            w, f, g, H = w_new, f_new, g_new, H_new
            self.n_iter = it + 1
            if moved < 1e-10:
                break
        self.w = w
        self.n_trial = len(y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """X: (N, K, F) → (N, K); a single trial (K, F) → (K,)."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 2
        P = np.exp(_log_softmax(self._z(X[None] if single else X) @ self.w))
        return P[0] if single else P

    def to_json(self) -> dict:
        """JSON-ready dict (the app layer writes the file)."""
        return {"type": "conditional_logit", "feature_names": self.feature_names, "l2": self.l2,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "weights": self.w.tolist(),
                "n_trial": self.n_trial}

    @classmethod
    def from_json(cls, d: dict) -> "ConditionalLogit":
        m = cls(d["feature_names"], d.get("l2", 1.0))
        m.mean = np.asarray(d["mean"], dtype=float)
        m.std = np.asarray(d["std"], dtype=float)
        m.w = np.asarray(d["weights"], dtype=float)
        m.n_trial = int(d.get("n_trial", 0))
        return m
