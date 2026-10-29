from types import SimpleNamespace

import pytest

from conftest import AOIS_ABC, dwell_cfg
from gazedemo.app.tracker import GazeDwellTracker
from gazedemo.types import Sample

TRIAL = SimpleNamespace(finalize_timeout_s=1.0)


def sample(t, label):
    mass = {"A": 0.0, "B": 0.0, "C": 0.0, "none": 1.0}
    if label in mass:
        mass[label] = 0.9
        mass["none"] = 0.1
    return Sample(t, int(t * 20), label, mass, None, 1.0)


def feed(tr, t_from, t_to, label, hz=20):
    n0, n1 = round(t_from * hz), round(t_to * hz)
    for i in range(n0, n1):
        tr.update(sample(i / hz, label))


def test_finalized_only_after_sample_past_t1():
    tr = GazeDwellTracker(dwell_cfg(), TRIAL, AOIS_ABC)
    feed(tr, 0.0, 1.0, "none")
    tid = tr.start_trial(1.0)
    feed(tr, 1.0, 2.0, "A")
    tr.end_trial(2.0, label="A")
    assert tr.poll_finished(now=2.2) == []          # no sample with t >= t1 yet
    tr.update(sample(2.0, "B"))
    (res,) = tr.poll_finished(now=2.2)
    assert res.trial_id == tid and res.label == "A" and not res.provisional
    assert res.per_aoi["A"].dd_ms == pytest.approx(1000)
    assert res.per_aoi["B"].dd_ms == 0               # sample at t1 is not used


def test_late_samples_after_t1_are_ignored():
    tr = GazeDwellTracker(dwell_cfg(), TRIAL, AOIS_ABC)
    tr.start_trial(0.0)
    feed(tr, 0.0, 1.0, "A")
    tr.end_trial(1.0)
    feed(tr, 1.0, 3.0, "C")
    (res,) = tr.poll_finished(now=3.0)
    assert res.per_aoi["A"].dd_ms == pytest.approx(1000)
    assert res.per_aoi["C"].dd_ms == 0 and res.per_aoi["C"].ttfd_ms is None


def test_timeout_finalizes_when_camera_stalls():
    tr = GazeDwellTracker(dwell_cfg(), TRIAL, AOIS_ABC)
    tr.start_trial(0.0)
    feed(tr, 0.0, 0.5, "B")
    tr.end_trial(1.0)
    assert tr.poll_finished(now=1.5) == []
    (res,) = tr.poll_finished(now=2.01)
    assert res.per_aoi["B"].dd_ms == pytest.approx(500 + 300 - 50)  # last sample covers max_interval
    assert res.invalid_ms == pytest.approx(250)


def test_snapshot_is_provisional_and_grows():
    tr = GazeDwellTracker(dwell_cfg(), TRIAL, AOIS_ABC)
    tr.start_trial(0.0)
    feed(tr, 0.0, 0.5, "A")
    s1 = tr.snapshot(now=0.5)
    feed(tr, 0.5, 1.5, "A")
    s2 = tr.snapshot(now=1.5)
    assert s1.provisional and s2.provisional
    assert s1.per_aoi["A"].dd_ms == pytest.approx(500)
    assert s2.per_aoi["A"].dd_ms == pytest.approx(1500)
    assert tr.snapshot(now=1.5).trial_id == s1.trial_id


def test_lookback_kept_for_dwell_before_t0():
    tr = GazeDwellTracker(dwell_cfg(buffer_s=0.5), TRIAL, AOIS_ABC)
    feed(tr, 0.0, 3.0, "C")
    tr.start_trial(3.0)
    feed(tr, 3.0, 4.0, "C")
    tr.end_trial(4.0)
    tr.update(sample(4.0, "C"))
    (res,) = tr.poll_finished(now=4.0)
    assert res.per_aoi["C"].ttfd_ms == 0
    assert res.per_aoi["C"].dd_ms == pytest.approx(1000)
    tr.update(sample(4.05, "C"))  # no trial left → keep max(buffer_s, lookback_s) of history
    assert tr.samples[0].t == pytest.approx(4.05 - 2.0)


def test_cancel_and_window():
    tr = GazeDwellTracker(dwell_cfg(), TRIAL, AOIS_ABC)
    tr.start_trial(0.0)
    feed(tr, 0.0, 1.0, "A")
    assert tr.cancel_trial() == 1
    assert tr.poll_finished(now=5.0) == [] and tr.open is None
    w = tr.window(0.5, now=1.0)
    assert w.per_aoi["A"].dd_ms == pytest.approx(500)
    assert tr.start_trial(1.0) == 2
