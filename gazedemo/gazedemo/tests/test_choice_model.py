import numpy as np
import pytest

from gazedemo.core.choice_model import ConditionalLogit, objective
from gazedemo.core.eval import bootstrap_ci, confusion, hard_proba, kfold_trials, log_loss, top1


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3, 4))
    w_true = np.array([2.0, -1.0, 0.0, 0.5])
    S = X @ w_true
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in P])
    return X, y, w_true


def test_gradient_and_hessian_match_finite_differences():
    X, y, _ = toy(50)
    w = np.array([0.3, -0.2, 0.1, 0.4])
    f, g, H = objective(w, X, y, l2=0.7)
    h = 1e-6
    num_g = np.array([(objective(w + h * e, X, y, 0.7)[0] - objective(w - h * e, X, y, 0.7)[0]) / (2 * h)
                      for e in np.eye(4)])
    num_H = np.array([(objective(w + h * e, X, y, 0.7)[1] - objective(w - h * e, X, y, 0.7)[1]) / (2 * h)
                      for e in np.eye(4)])
    assert np.allclose(g, num_g, rtol=1e-5, atol=1e-5)
    assert np.allclose(H, num_H, rtol=1e-4, atol=1e-4)


def test_fit_recovers_weights_and_probabilities_sum_to_one():
    X, y, w_true = toy(2000)
    m = ConditionalLogit(["a", "b", "c", "d"], l2=1.0).fit(X, y)
    w = m.w / m.std  # back to raw feature scale
    assert np.sign(w[:2]).tolist() == [1.0, -1.0] and abs(w[2]) < 0.3
    assert w[0] == pytest.approx(w_true[0], rel=0.2)
    P = m.predict_proba(X)
    assert P.shape == (2000, 3) and np.allclose(P.sum(1), 1.0)
    assert np.allclose(m.predict_proba(X[0]), P[0])
    assert m.n_iter < 30


def test_json_roundtrip_same_probabilities():
    X, y, _ = toy(100)
    m = ConditionalLogit(["a", "b", "c", "d"], l2=1.0).fit(X, y)
    m2 = ConditionalLogit.from_json(m.to_json())
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))


def test_constant_feature_and_tiny_data_do_not_crash():
    X = np.zeros((2, 3, 2))
    X[:, :, 0] = [[1, 0, 0], [0, 1, 0]]
    m = ConditionalLogit(["x", "const"], l2=1.0).fit(X, np.array([0, 1]))
    assert np.isfinite(m.w).all() and np.allclose(m.predict_proba(X).sum(1), 1)


def test_eval_helpers():
    P = hard_proba(np.array([[5.0, 1.0, 1.0], [2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), eps=0.06)
    assert np.allclose(P, [[0.94, 0.03, 0.03], [0.47, 0.47, 0.06], [1 / 3] * 3])
    y = np.array([0, 1, 2])
    assert np.allclose(top1(P, y), [1.0, 0.5, 1 / 3])
    assert np.allclose(log_loss(P, y), -np.log([0.94, 0.47, 1 / 3]))
    C = confusion(P, y, 3)
    assert np.allclose(C.sum(1), 1.0) and C[1, 0] == pytest.approx(0.5)
    folds = kfold_trials(23, 5, seed=0)
    assert sorted(np.bincount(folds)) == [4, 4, 5, 5, 5]
    lo, hi = bootstrap_ci(np.r_[np.ones(30), np.zeros(10)], 1000, 0)
    assert lo < 0.75 < hi
