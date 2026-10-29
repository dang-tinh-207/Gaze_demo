"""FakeCamera + FakeGaze scenario → runner in live mode, no window → trials.csv within 1 sample."""
import numpy as np
import pytest

from gazedemo.app.modes import LiveController
from gazedemo.app.runner import Runner
from gazedemo.app.text import TextPainter
from gazedemo.app.ui import UI, Screen, View
from gazedemo.config import load_config
from gazedemo.io.logs import RunLogger, num, read_csv, read_jsonl
from gazedemo.perception.fake import FakeCamera, FakeGaze, FakeWorld, default_aois, scenario_from_spans

W, H, FPS, T0 = 640, 360, 15.0, 1000.0
ONE_SAMPLE_MS = 1000 / FPS


def fake_setup(spans, duration):
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    cam = FakeCamera(world, scenario_from_spans(spans), fps=FPS, duration=duration, t_start=T0).start()
    return aois, cam, FakeGaze(world)


def test_live_runner_headless(tmp_path):
    cfg = load_config()
    # none 0–1 s | A 1–4 s | B 4–6 s | none 6–8 s; S at 0.5 s, E at 6.5 s, label 1 (A) at 7 s
    aois, cam, gaze = fake_setup([(1.0, 4.0, "A"), (4.0, 6.0, "B")], duration=8.0)
    keys = [(T0 + 0.5, "s"), (T0 + 6.5, "e"), (T0 + 7.0, "1")]
    logger = RunLogger(tmp_path, [a.name for a in aois])
    runner = Runner(cfg, cam, gaze, aois, LiveController(), logger, keys=keys)
    results = runner.run()

    assert len(results) == 1
    (row,) = read_csv(tmp_path / "trials.csv")
    assert row["label"] == "A" and row["mode"] == "live"
    assert num(row["A_dd_ms"]) == pytest.approx(3000, abs=ONE_SAMPLE_MS)
    assert num(row["B_dd_ms"]) == pytest.approx(2000, abs=ONE_SAMPLE_MS)
    assert num(row["C_dd_ms"]) == 0 and row["C_ttfd_ms"] == ""
    assert num(row["A_ttfd_ms"]) == pytest.approx(500, abs=ONE_SAMPLE_MS)
    assert num(row["B_ttfd_ms"]) == pytest.approx(3500, abs=ONE_SAMPLE_MS)
    assert (int(row["A_dc"]), int(row["B_dc"])) == (1, 1)
    assert (row["first_aoi"], row["last_aoi"]) == ("A", "B")
    assert num(row["A_p_look"]) > num(row["C_p_look"])
    assert num(row["valid_ratio"]) == pytest.approx(1.0)

    samples = read_jsonl(tmp_path / "samples.jsonl")
    assert len(samples) == int(8.0 * FPS)
    kinds = [e["type"] for e in read_jsonl(tmp_path / "events.jsonl")]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert {"trial_start", "trial_end", "trial_update"} <= set(kinds)
    assert read_csv(tmp_path / "dwells.csv")[0]["aoi"] == "A"


def test_gazelle_errors_become_invalid_samples(tmp_path):
    cfg = load_config()
    aois, cam, gaze = fake_setup([(0.0, 3.0, "C")], duration=3.0)

    class Flaky:
        n = 0

        def infer(self, rgb):
            Flaky.n += 1
            if Flaky.n % 5 == 0:
                raise RuntimeError("CUDA hiccup")
            return gaze.infer(rgb)

    keys = [(T0 + 0.1, "s"), (T0 + 2.9, "e")]
    runner = Runner(cfg, cam, Flaky(), aois, LiveController(), RunLogger(tmp_path, "ABC"), keys=keys)
    (res,) = runner.run()
    labels = [s["label"] for s in read_jsonl(tmp_path / "samples.jsonl")]
    assert labels.count("invalid") == len(labels) // 5
    assert res.per_aoi["C"].dd_ms > 2500  # isolated invalid frames are bridged


def test_ui_renders_headless():
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    frame = world.render("B")
    gaze = FakeGaze(world)
    hm, inout = gaze.infer(frame[:, :, ::-1])
    cfg = load_config()
    from gazedemo.core.metrics import compute
    from gazedemo.types import Sample
    s = Sample(1.0, 0, "B", {"A": 0.0, "B": 0.95, "C": 0.0, "none": 0.05}, (320.0, 300.0), inout)
    res = compute([s], 1.0, 1.5, cfg.dwell, aois, provisional=True)
    ui = UI(cfg.ui, "ABC")
    view = View(mode="collect", frame=frame, sample=s, heatmap=hm, aois=aois, hz=14.8, latency_ms=80,
                trial_id=7, trial_t=0.5, result=res, foods={"A": "cơm", "B": "trứng", "C": "rau"},
                p_choice={"A": 0.2, "B": 0.7, "C": 0.1}, help="S start · Q quit")
    img = ui.render(view)
    assert img.shape == (H, W + cfg.ui.panel_width, 3) and img.dtype == np.uint8
    text_screen = ui.render(View(mode="cue", frame=frame, screen=Screen("text", big="LOOK AT A", lines=["Round 1/15"])))
    assert text_screen[:, :W].mean() < 60  # dark screen, no video
    plain = TextPainter.__new__(TextPainter)  # no-font fallback path
    plain.regular = plain.bold = None
    plain._fonts = {}
    from gazedemo.app.text import Txt
    plain.draw(img, [Txt("Hãy nhìn và chọn món bạn muốn", (10, 10))])
