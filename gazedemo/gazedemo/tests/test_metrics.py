import numpy as np
import pytest

from conftest import AOIS_ABC, dwell_cfg, labelled_samples
from gazedemo.core.calibration import apply_temperature, fit_temperature, reliability
from gazedemo.core.metrics import compute
from gazedemo.types import Sample


def test_p_look_spec_example():
    # mass A 0.6, B 0.2, C 0.0, inout 0.9 → q = A .54, B .18, C 0, none .28; plate A .75, B .25
    s = Sample(0.1, 0, "A", {"A": 0.6, "B": 0.2, "C": 0.0, "none": 0.2}, (320.0, 520.0), 0.9)
    res = compute([s], 0.0, 0.2, dwell_cfg(), AOIS_ABC)
    pa = res.per_aoi
    assert pa["A"].p_look == pytest.approx(0.54)
    assert pa["B"].p_look == pytest.approx(0.18)
    assert pa["C"].p_look == pytest.approx(0.0)
    assert res.p_look_none == pytest.approx(0.28)
    assert pa["A"].p_look_plate == pytest.approx(0.75)
    assert pa["B"].p_look_plate == pytest.approx(0.25)
    assert pa["C"].p_look_plate == pytest.approx(0.0)


def test_p_look_is_mean_of_frames_and_skips_invalid():
    f1 = Sample(0.0, 0, "A", {"A": 0.6, "B": 0.2, "C": 0.0, "none": 0.2}, None, 0.9)
    f2 = Sample(0.1, 1, "C", {"A": 0.0, "B": 0.0, "C": 1.0, "none": 0.0}, None, None)
    bad = Sample(0.2, 2, "invalid", {}, None, None)
    res = compute([f1, f2, bad], 0.0, 0.3, dwell_cfg(), AOIS_ABC)
    assert res.per_aoi["A"].p_look == pytest.approx(0.27)
    assert res.per_aoi["C"].p_look == pytest.approx(0.5)
    assert res.p_look_none == pytest.approx(0.14)
    total = sum(m.p_look for m in res.per_aoi.values()) + res.p_look_none
    assert total == pytest.approx(1.0)


def test_last_seconds_and_first_last():
    # 4 s trial: A 0–1.5 s, B 1.5–3.2 s, C 3.2–4.0 s
    samples = labelled_samples([(0, 1500, "A"), (1500, 3200, "B"), (3200, 4000, "C")],
                               t_to_ms=4000)
    res = compute(samples, 0.0, 4.0, dwell_cfg(), AOIS_ABC)
    a, b, c = (res.per_aoi[k] for k in "ABC")
    assert c.dd_last1s_ms == pytest.approx(800)
    assert b.dd_last1s_ms == pytest.approx(200)
    assert c.dd_last2s_ms == pytest.approx(800) and b.dd_last2s_ms == pytest.approx(1200)
    assert a.dd_last2s_ms == 0
    assert (a.is_first, b.is_first, c.is_first) == (1, 0, 0)
    assert (a.is_last, b.is_last, c.is_last) == (0, 0, 1)
    assert a.share == pytest.approx(1500 / 4000)
    assert b.longest_dwell_ms == pytest.approx(1700) and b.mean_dwell_ms == pytest.approx(1700)


def test_mean_dist_norm():
    a = AOIS_ABC[0]  # centre (320, 520), mean semi-axis 90
    pts = [(320.0, 520.0), (320.0 + 90.0, 520.0), (320.0, 520.0 + 180.0)]
    samples = [Sample(0.1 * i, i, "A", {"A": 1.0, "none": 0.0}, p, 1.0) for i, p in enumerate(pts)]
    samples.append(Sample(0.3, 3, "invalid", {}, None, None))
    res = compute(samples, 0.0, 0.4, dwell_cfg(), AOIS_ABC)
    assert res.per_aoi["A"].mean_dist_norm == pytest.approx((0 + 1 + 2) / 3)


def test_calibrated_p_look_column():
    s = Sample(0.1, 0, "A", {"A": 0.6, "B": 0.2, "C": 0.0, "none": 0.2}, None, 0.9)
    res = compute([s], 0.0, 0.2, dwell_cfg(), AOIS_ABC, temperature=1.0, cal_eps=1e-9)
    assert res.per_aoi["A"].p_look_cal == pytest.approx(0.54, abs=1e-6)
    hot = compute([s], 0.0, 0.2, dwell_cfg(), AOIS_ABC, temperature=3.0)
    assert hot.per_aoi["A"].p_look_cal < 0.54  # T > 1 softens


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1.0, size=(4000, 4))
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in p_true])
    overconfident = apply_temperature(p_true, 1 / 3, 1e-12)  # sharpened ×3
    T = fit_temperature(overconfident, y, eps=1e-12)
    assert T == pytest.approx(3.0, rel=0.15)
    rows, ece_before = reliability(overconfident, y)
    _, ece_after = reliability(apply_temperature(overconfident, T, 1e-12), y)
    assert ece_after < ece_before
    assert sum(r["n"] for r in rows) == len(y)
