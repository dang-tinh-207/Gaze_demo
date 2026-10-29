"""A fake run with a short recorded video + frames.csv + events → `offline` gives the same
trials.csv as live when FakeGaze is deterministic."""
from gazedemo.app.modes import LiveController
from gazedemo.app.offline import run_offline
from gazedemo.app.runner import Runner
from gazedemo.app.session import write_run_files
from gazedemo.config import load_config
from gazedemo.io.logs import RunLogger, read_csv, read_json
from gazedemo.io.recorder import Recorder
from gazedemo.perception.fake import FakeCamera, FakeGaze, FakeWorld, default_aois, scenario_from_spans

W, H, T0 = 640, 360, 1000.0


def test_offline_reproduces_live(tmp_path):
    cfg = load_config()
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    run_dir = tmp_path / "20260101_000000_live"
    run_dir.mkdir()
    rec = Recorder(run_dir, fps=15, codec=cfg.source.codec, blocking=True)  # virtual time: no drops
    spans = [(0.8, 2.0, "A"), (2.0, 2.06, "B"), (2.06, 3.5, "A"), (3.5, 5.0, "out"), (5.0, 7.5, "C"),
             (7.5, 8.0, "B"), (8.0, 9.5, "C")]
    cam = FakeCamera(world, scenario_from_spans(spans), fps=15, duration=10.0, t_start=T0,
                     on_frame=rec.push).start()
    gaze = FakeGaze(world)
    write_run_files(run_dir, cfg, "live", aois, (W, H), None, gaze, None)
    keys = [(T0 + 0.3, "s"), (T0 + 4.4, "e"), (T0 + 4.6, "2"), (T0 + 4.8, "s"), (T0 + 9.7, "e"), (T0 + 9.8, "3")]
    Runner(cfg, cam, gaze, aois, LiveController(), RunLogger(run_dir, "ABC"), keys=keys, recorder=rec,
           run_dir=run_dir).run()

    out = run_offline(cfg, run_dir, use_fake_gaze=True)
    live, off = read_csv(run_dir / "trials.csv"), read_csv(out / "trials.csv")
    assert len(live) == 2 and [r["label"] for r in live] == ["B", "C"]
    assert list(live[0]) == list(off[0])
    assert live == off
    assert read_json(out / "run.json")["of_mode"] == "live"
    assert len(read_csv(run_dir / "frames.csv")) == 150


def test_offline_stride_and_missing_video(tmp_path, capsys):
    cfg = load_config()
    assert run_offline(cfg, tmp_path) is None
    assert "has no video.mp4" in capsys.readouterr().out
