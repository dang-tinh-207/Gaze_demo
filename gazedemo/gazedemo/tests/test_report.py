"""report: gathers cue score + dataset + classifier into SUMMARY.md (DEMO_SUPERVISOR layout),
with an overlay frame rebuilt from a recorded run and a label timeline."""
from gazedemo.app.dataset import build_dataset
from gazedemo.app.modes import CueController
from gazedemo.app.report import make_report
from gazedemo.app.runner import Runner
from gazedemo.app.score import calibrate_run, score_run
from gazedemo.app.session import write_run_files
from gazedemo.app.synth import run_synth
from gazedemo.app.train import run_train
from gazedemo.config import load_config
from gazedemo.io.logs import RunLogger
from gazedemo.io.recorder import Recorder
from gazedemo.perception.fake import FakeCamera, FakeGaze, FakeWorld, default_aois

W, H, T0 = 640, 360, 1000.0


def fake_cue_run(cfg, root):
    """Cue run where the fake person looks at the cued plate after 0.5 s."""
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    run_dir = root / "20260101_000000_cue"
    run_dir.mkdir(parents=True)
    ctl = CueController(cfg.cue, "ABC")

    def scenario(t_rel):
        if ctl.state != "cue" or ctl.t_cue is None:
            return "none"
        return ctl.seq[ctl.i] if T0 + t_rel - ctl.t_cue > 0.5 else "none"

    rec = Recorder(run_dir, fps=15, blocking=True)
    cam = FakeCamera(world, scenario, fps=15, duration=70.0, t_start=T0, on_frame=rec.push).start()
    gaze = FakeGaze(world)
    write_run_files(run_dir, cfg, "cue", aois, (W, H), None, gaze, None)
    Runner(cfg, cam, gaze, aois, ctl, RunLogger(run_dir, "ABC"), keys=[(T0 + 0.5, "space")],
           recorder=rec, run_dir=run_dir).run()
    return run_dir


def test_cue_score_calibrate_and_report(tmp_path):
    cfg = load_config()
    runs = tmp_path / "runs"
    cue = fake_cue_run(cfg, runs)
    res = score_run(cue, cfg, quiet=True)
    assert res["n_trials"] == 15 and res["correct_dd"] == 15 and res["n_frames"] > 500
    cfg.calibration.file = str(tmp_path / "models" / "calibration.json")
    cal = calibrate_run(cue, cfg, tmp_path / "models" / "calibration.json")
    assert cal["T"] > 0 and cal["nll_after"] <= cal["nll_before"] + 1e-9

    run_synth(cfg, participants=1, trials=30, out=runs, seed=1)
    build_dataset(cfg, runs, tmp_path / "data" / "gaze_choice.csv")
    run_train(cfg, tmp_path / "data" / "gaze_choice.csv", tmp_path / "models")
    out = make_report(cfg, tmp_path / "report", runs, tmp_path / "data" / "gaze_choice.csv", tmp_path / "models")
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. What the module does", "## 2.", "## 3. Food-choice data", "## 4. Classifier",
                    "## 5. Limitations", "## 6. Next steps", "### models/report.md"):
        assert heading in text
    assert "15 / 15" in text and "| Logit all |" in text and "Rounds: 30 (1 person" in text
    assert "Hz on GPU ?" in text  # no bench run in this folder → "?"
    assert (tmp_path / "report" / "timeline.png").exists() and (tmp_path / "report" / "overlay.png").exists()
