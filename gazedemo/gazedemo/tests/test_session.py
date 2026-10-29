"""The real live / cue / collect entry point (cli → run_session) with the synthetic camera, FakeGaze
and a stubbed window that "presses" keys on a timetable (no window is opened)."""
import time

import pytest

from gazedemo import cli
from gazedemo.app import ui as ui_module
from gazedemo.io.logs import read_csv, read_json, read_jsonl


@pytest.fixture
def keypresses(monkeypatch):
    """keypresses(schedule): schedule = [(seconds after the first frame shown, key), ...]."""
    def install(schedule):
        state = {"t0": None, "todo": sorted(schedule)}

        def show(self, canvas):
            now = time.monotonic()
            state["t0"] = state["t0"] or now
            assert canvas.ndim == 3
            if state["todo"] and now - state["t0"] >= state["todo"][0][0]:
                return state["todo"].pop(0)[1]
            if now - state["t0"] > 30:
                return "q"  # safety net
            return None

        monkeypatch.setattr(ui_module.UI, "show", show)
        monkeypatch.setattr(ui_module.UI, "close", lambda self: None)
    return install


def run(cmd, tmp_path, *extra):
    argv = ["--set", f"output.root={tmp_path.as_posix()}", "--set", "calibration.file=none/calibration.json",
            cmd, "--source", "synthetic", "--fake-gaze", "--aoi", str(tmp_path / "missing.json"), *extra]
    assert cli.main(argv) == 0
    (run_dir,) = [d for d in tmp_path.iterdir() if d.name.endswith("_" + cmd)]
    return run_dir


def test_live_session_files(tmp_path, keypresses):
    keypresses([(0.2, "s"), (1.6, "e"), (1.7, "2"), (2.2, "q")])
    d = run("live", tmp_path)
    for name in ("config.yaml", "aoi.json", "run.json", "samples.jsonl", "events.jsonl", "trials.csv",
                 "dwells.csv", "video.mp4", "frames.csv", "snapshot.png"):
        assert (d / name).exists(), name
    (row,) = read_csv(d / "trials.csv")
    assert row["label"] == "B" and 1300 < float(row["dur_ms"]) < 1700
    end = read_jsonl(d / "events.jsonl")[-1]
    assert end["type"] == "run_end" and end["video_frames"] >= end["n_samples"] > 15
    assert read_json(d / "run.json")["mode"] == "live"


def test_cue_session_scores(tmp_path, keypresses):
    keypresses([(0.2, "space")])
    d = run("cue", tmp_path)  # n_rep 1 × 3 plates, short cues (see --set below)
    rows = read_csv(d / "trials.csv")
    assert [r["label"] for r in rows] == [r["cue"] for r in rows] and len(rows) == 3
    assert read_json(d / "score.json")["n_trials"] == 3 and (d / "score.md").exists()


@pytest.fixture(autouse=True)
def short_cue(monkeypatch):
    real = cli.load_config

    def load(path=None, overrides=()):
        extra = ["cue.n_rep=1", "cue.cue_s=0.6", "cue.settle_s=0.2", "collect.trial_s=0.6",
                 "collect.countdown_s=0.2", "collect.show_result_s=0.2"]
        return real(path, list(overrides) + extra)
    monkeypatch.setattr(cli, "load_config", load)


def test_collect_session(tmp_path, keypresses):
    likes = [(0.1 + 0.05 * i, k) for i, k in enumerate("345215")]
    keypresses(likes + [(0.8, "space"), (2.2, "3"), (2.8, "space"), (4.2, "x")])
    d = run("collect", tmp_path, "--participant", "PX", "--trials", "2")
    rows = read_csv(d / "trials.csv")
    assert [r["chosen_plate"] for r in rows] == ["C", ""] and [r["skipped"] for r in rows] == ["0", "1"]
    assert rows[0]["chosen_food"] == rows[0]["food_C"] and rows[0]["participant"] == "PX"
    session = read_json(d / "session.json")
    assert list(session["liking"].values()) == [3, 4, 5, 2, 1, 5]
    assert len(read_csv(d / "schedule.csv")) == 2


def test_missing_aoi_without_synthetic_source(tmp_path, capsys):
    argv = ["--set", f"output.root={tmp_path.as_posix()}", "live", "--aoi", str(tmp_path / "nope.json")]
    assert cli.main(argv) == 1
    assert "setup-aoi" in capsys.readouterr().out
