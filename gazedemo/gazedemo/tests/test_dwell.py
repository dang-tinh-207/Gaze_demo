"""SPEC 4.5 hand-computed example + variants, time invariant, empty and single-sample windows."""
import numpy as np
import pytest

from conftest import AOIS_ABC, dwell_cfg, labelled_samples
from gazedemo.core.dwell import build_timeline
from gazedemo.core.metrics import compute
from gazedemo.types import Sample

approx = pytest.approx


def run(spans, t0=0.0, t1=1.0, **kw):
    samples = labelled_samples(spans, hz=20, t_from_ms=kw.pop("t_from_ms", 0),
                               t_to_ms=kw.pop("t_to_ms", 1000))
    return compute(samples, t0, t1, dwell_cfg(**kw), AOIS_ABC)


def check(res, plate, dd, dc, ttfd):
    m = res.per_aoi[plate]
    assert m.dd_ms == approx(dd, abs=1e-6)
    assert m.dc == dc
    assert m.ttfd_ms == (None if ttfd is None else approx(ttfd, abs=1e-6))


def test_spec_example():
    # none 0–0.20 | A 0.20–0.50 | none 0.50–0.60 | A 0.60–0.70 | B 0.70–0.75 | C 0.75–1.00
    res = run([(200, 500, "A"), (600, 700, "A"), (700, 750, "B"), (750, 1000, "C")])
    check(res, "A", 500, 1, 200)
    check(res, "B", 0, 0, None)
    check(res, "C", 250, 1, 750)
    assert res.none_ms == approx(250)
    assert res.first_aoi == "A" and res.last_aoi == "C"
    total = sum(m.dd_ms for m in res.per_aoi.values()) + res.none_ms + res.out_ms + res.invalid_ms
    assert total == approx(1000)


def test_gap_too_long_gives_two_dwells():
    # A 0.20–0.50, none 0.50–0.70, A 0.70–0.80, C 0.80–1.00 → A 400/2/200; C 200/1/800
    res = run([(200, 500, "A"), (700, 800, "A"), (800, 1000, "C")])
    check(res, "A", 400, 2, 200)
    check(res, "C", 200, 1, 800)


def test_flicker_to_other_plate_is_bridged_after_filter():
    # A 0.20–0.40, B 0.40–0.45, A 0.45–0.60, none after → A 400/1
    res = run([(200, 400, "A"), (400, 450, "B"), (450, 600, "A")])
    check(res, "A", 400, 1, 200)
    check(res, "B", 0, 0, None)


def test_dwell_from_before_t0_and_cut_at_t1():
    res = run([(-1000, 1500, "A")], t_from_ms=-1000, t_to_ms=1500)
    check(res, "A", 1000, 1, 0)
    assert res.first_aoi == "A"


def test_long_interval_is_invalid():
    samples = [Sample(0.0, 0, "A", {"A": 1.0, "none": 0.0}, None, 1.0),
               Sample(1.0, 1, "A", {"A": 1.0, "none": 0.0}, None, 1.0)]
    segs = build_timeline(samples, 0.0, 1.3, 0.3)
    assert [(s.label, round(s.t_start, 6), round(s.t_end, 6)) for s in segs] == [
        ("A", 0.0, 0.3), ("invalid", 0.3, 1.0), ("A", 1.0, 1.3)]
    res = compute(samples, 0.0, 1.0, dwell_cfg(), AOIS_ABC)
    assert res.invalid_ms == approx(700)
    check(res, "A", 300, 1, 0)


def test_out_is_not_bridged():
    res = run([(200, 400, "A"), (400, 450, "out"), (450, 600, "A")])
    check(res, "A", 350, 2, 200)
    assert res.out_ms == approx(50)


def test_samples_at_or_after_t1_are_ignored():
    samples = labelled_samples([(0, 2000, "A")], t_from_ms=0, t_to_ms=2000)
    res = compute(samples, 0.0, 1.0, dwell_cfg(), AOIS_ABC)
    later = [s if s.t < 1.0 else Sample(s.t, s.frame_idx, "B", s.mass, None, 1.0) for s in samples]
    res2 = compute(later, 0.0, 1.0, dwell_cfg(), AOIS_ABC)
    assert res.per_aoi["A"].dd_ms == res2.per_aoi["A"].dd_ms == approx(1000)
    assert res2.per_aoi["B"].dd_ms == 0


def test_empty_window():
    res = compute([], 5.0, 6.0, dwell_cfg(), AOIS_ABC)
    assert res.invalid_ms == approx(1000)
    assert res.valid_ratio == 0.0 and res.n_samples == 0
    assert all(m.dd_ms == 0 and m.ttfd_ms is None and m.p_look is None for m in res.per_aoi.values())
    assert res.first_aoi is None


def test_single_sample():
    s = Sample(5.1, 0, "B", {"A": 0.0, "B": 0.9, "C": 0.0, "none": 0.1}, (640.0, 560.0), 1.0)
    res = compute([s], 5.0, 6.0, dwell_cfg(), AOIS_ABC)
    check(res, "B", 300, 1, 100)
    assert res.invalid_ms == approx(700)
    assert res.n_samples == 1


@pytest.mark.parametrize("seed", range(20))
def test_time_invariant_random(seed):
    rng = np.random.default_rng(seed)
    labels = ["A", "B", "C", "none", "out", "invalid"]
    t, samples = -0.5, []
    while t < 3.0:
        lab = labels[rng.integers(len(labels))]
        samples.append(Sample(t, len(samples), lab, {} if lab == "invalid" else {"none": 1.0}, None, 1.0))
        t += float(rng.uniform(0.01, 0.5))
    t0, t1 = 0.0, float(rng.uniform(0.5, 3.0))
    res = compute(samples, t0, t1, dwell_cfg(), AOIS_ABC)
    total = sum(m.dd_ms for m in res.per_aoi.values()) + res.none_ms + res.out_ms + res.invalid_ms
    assert total == approx((t1 - t0) * 1000)
    assert all(d.dur * 1000 >= 100 - 1e-6 for d in res.dwells)
    assert sum(m.dc for m in res.per_aoi.values()) == len(res.dwells)
