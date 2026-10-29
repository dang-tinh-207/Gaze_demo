from collections import Counter

import pytest

from gazedemo.app.collect import CollectController, make_schedule, read_foods
from gazedemo.app.runner import Runner
from gazedemo.config import load_config, project_path
from gazedemo.io.logs import RunLogger, read_csv, read_json
from gazedemo.perception.fake import FakeCamera, FakeGaze, FakeWorld, default_aois, scenario_from_spans

FOODS = ["cơm", "trứng", "rau", "thịt", "đậu", "chuối"]
PLATES = ["A", "B", "C"]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_balanced_food_by_plate(seed):
    sched = make_schedule(FOODS, PLATES, 30, seed)
    assert len(sched) == 30
    assert all(len(set(e.values())) == 3 for e in sched)
    cells = Counter((e[p], p) for e in sched for p in PLATES)
    assert set(cells.values()) == {5}          # 30 trials × 3 plates / (6 foods × 3 plates)
    shown = Counter(f for e in sched for f in e.values())
    assert set(shown.values()) == {15}


def test_uneven_counts_stay_close_and_seed_is_fixed():
    sched = make_schedule(FOODS, PLATES, 25, 3)
    cells = Counter((e[p], p) for e in sched for p in PLATES)
    counts = [cells[(f, p)] for f in FOODS for p in PLATES]
    assert max(counts) - min(counts) <= 1
    assert make_schedule(FOODS, PLATES, 25, 3) == sched
    assert make_schedule(FOODS, PLATES, 25, 4) != sched


def test_too_few_foods():
    with pytest.raises(ValueError):
        make_schedule(["cơm", "rau"], PLATES, 5, 0)


def test_foods_yaml_is_valid():
    assert len(read_foods(project_path("configs/foods.yaml"))) >= 3


def test_collect_flow_headless(tmp_path):
    cfg = load_config()
    W, H, T0 = 640, 360, 1000.0
    aois = default_aois(W, H)
    world = FakeWorld.from_aois(aois, W, H)
    # liking 6 keys, then 3 trials of ~9 s: Space, 1 s countdown, 5 s trial, answer, 2 s result
    spans = [(1.0 + 9 * k + 1.0, 1.0 + 9 * k + 4.0, "A") for k in range(3)]
    cam = FakeCamera(world, scenario_from_spans(spans), fps=15, duration=40.0, t_start=T0).start()
    keys = [(T0 + 0.1 + 0.1 * i, k) for i, k in enumerate("435215")]
    for k, answer in enumerate(["1", "2", "x"]):
        base = T0 + 1.0 + 9 * k
        keys += [(base, "space"), (base + 6.5, answer)]
    ctl = CollectController(cfg.collect, PLATES, "PTEST", FOODS, 3, 0, tmp_path)
    runner = Runner(cfg, cam, FakeGaze(world), aois, ctl, RunLogger(tmp_path, PLATES), keys=keys)
    results = runner.run()

    assert len(results) == 3 and ctl.done
    rows = read_csv(tmp_path / "trials.csv")
    sched = ctl.schedule
    assert [r["chosen_plate"] for r in rows] == ["A", "B", ""]
    assert [r["skipped"] for r in rows] == ["0", "0", "1"]
    assert rows[0]["chosen_food"] == sched[0]["A"] and rows[1]["chosen_food"] == sched[1]["B"]
    liking = dict(zip(FOODS, [4, 3, 5, 2, 1, 5]))
    for r, e in zip(rows, sched):
        assert r["participant"] == "PTEST" and r["mode"] == "collect"
        assert [r[f"food_{p}"] for p in PLATES] == [e[p] for p in PLATES]
        assert [int(r[f"liking_{p}"]) for p in PLATES] == [liking[e[p]] for p in PLATES]
        assert abs(float(r["dur_ms"]) - cfg.collect.trial_s * 1000) <= 1000 / 15 + 1e-6
    assert float(rows[0]["A_dd_ms"]) > 2500
    assert read_json(tmp_path / "session.json")["liking"] == liking
    assert len(read_csv(tmp_path / "schedule.csv")) == 3
