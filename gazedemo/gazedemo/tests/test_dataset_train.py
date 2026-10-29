"""synth → dataset → train end to end; logit beats chance on data with signal; model.json reloads
to the same probabilities and drives p_choice in a (fake) live run."""
import numpy as np
import pytest

from gazedemo.app.choice import ChoiceScorer
from gazedemo.app.dataset import build_dataset
from gazedemo.app.modes import LiveController
from gazedemo.app.runner import Runner
from gazedemo.app.synth import run_synth
from gazedemo.app.train import load_trials, run_train
from gazedemo.config import load_config
from gazedemo.core.choice_model import ConditionalLogit
from gazedemo.core.features import FEATURE_SETS, feature_matrix
from gazedemo.io.logs import RunLogger, num, read_csv, read_json
from gazedemo.perception.fake import FakeCamera, FakeGaze, FakeWorld, default_aois, scenario_from_spans


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = load_config()
    run_synth(cfg, participants=1, trials=60, out=tmp / "runs_synth", seed=0)
    stats = build_dataset(cfg, tmp / "runs_synth", tmp / "data" / "gaze_choice.csv")
    results = run_train(cfg, tmp / "data" / "gaze_choice.csv", tmp / "models")
    return cfg, tmp, stats, results


def test_dataset_long_format(trained):
    cfg, tmp, stats, _ = trained
    rows = read_csv(tmp / "data" / "gaze_choice.csv")
    assert stats["n_trials"] >= 55 and len(rows) == 3 * stats["n_trials"]
    assert {"run_id", "participant", "trial_id", "plate", "food", "liking", "chosen", "ttfd_ms", "never_looked",
            "dd_ms", "dc", "share", "p_look", "dd_last1s_ms", "is_last", "mean_dist_norm"} <= set(rows[0])
    for i in range(0, len(rows), 3):
        assert sum(int(r["chosen"]) for r in rows[i:i + 3]) == 1
    never = [r for r in rows if r["never_looked"] == "1"]
    assert all(num(r["ttfd_ms"]) == pytest.approx(cfg.collect.trial_s * 1000) for r in never)
    assert (tmp / "data" / "gaze_choice_trials.csv").exists()
    assert sum(stats["chosen_plate"].values()) == stats["n_trials"]


def test_logit_beats_chance_and_reports(trained):
    cfg, tmp, stats, results = trained
    assert results["chance"]["top1"] == pytest.approx(1 / 3)
    assert results["logit:all"]["top1"] > 0.6
    assert results["logit:all"]["logloss"] < np.log(3)
    assert results["argmax_dd"]["top1"] > 0.45
    ev = read_json(tmp / "models" / "eval.json")
    assert ev["cv"].startswith("5-fold") and set(ev["models"]) >= {"logit:g1", "logit:all+liking", "dwell_T"}
    report = (tmp / "models" / "report.md").read_text(encoding="utf-8")
    assert "Data from 1 person" in report and "| Logit all |" in report


def test_model_json_reloads_same_probabilities(trained):
    cfg, tmp, _, _ = trained
    trials, plates = load_trials(tmp / "data" / "gaze_choice.csv")
    names = FEATURE_SETS["all"]
    X = np.stack([feature_matrix(g, names) for g in trials])
    y = np.array([[int(r["chosen"]) for r in g].index(1) for g in trials])
    direct = ConditionalLogit(names, cfg.train.l2).fit(X, y).predict_proba(X)
    loaded = ConditionalLogit.from_json(read_json(tmp / "models" / "model.json"))
    assert np.allclose(loaded.predict_proba(X), direct, atol=1e-8)


def test_live_with_model_writes_p_choice(trained, tmp_path):
    cfg, tmp, _, _ = trained
    W, H, T0 = 640, 360, 1000.0
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    cam = FakeCamera(world, scenario_from_spans([(0.5, 3.5, "C")]), fps=15, duration=5.0, t_start=T0).start()
    choice = ChoiceScorer(tmp / "models" / "model.json", "ABC")
    runner = Runner(cfg, cam, FakeGaze(world), aois, LiveController(), RunLogger(tmp_path, "ABC"),
                    keys=[(T0 + 0.2, "s"), (T0 + 4.0, "e")], choice=choice)
    runner.run()
    (row,) = read_csv(tmp_path / "trials.csv")
    pc = [num(row[f"{p}_p_choice"]) for p in "ABC"]
    assert sum(pc) == pytest.approx(1.0, abs=1e-3) and pc[2] == max(pc)


def test_two_participants_use_lopo(tmp_path):
    cfg = load_config()
    run_synth(cfg, participants=2, trials=12, out=tmp_path / "runs", seed=3)
    build_dataset(cfg, tmp_path / "runs", tmp_path / "d.csv")
    run_train(cfg, tmp_path / "d.csv", tmp_path / "m")
    ev = read_json(tmp_path / "m" / "eval.json")
    assert ev["cv"].startswith("leave-one-participant-out") and ev["n_trial"] == 24
    assert not any("1 person" in n for n in ev["notes"])


def test_too_little_data_still_runs(tmp_path):
    cfg = load_config()
    run_synth(cfg, participants=1, trials=8, out=tmp_path / "runs", seed=5)
    build_dataset(cfg, tmp_path / "runs", tmp_path / "d.csv")
    run_train(cfg, tmp_path / "d.csv", tmp_path / "m")
    ev = read_json(tmp_path / "m" / "eval.json")
    assert any("Too little data" in n for n in ev["notes"]) and (tmp_path / "m" / "model.json").exists()
