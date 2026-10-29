"""`synth` (SPEC 7.3): fake `collect` runs in the real file format, made with FakeGaze, to test
dataset → train before real data exists. The chosen plate gets more and longer dwells and is
usually looked at last (gaze cascade); choices also follow liking."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from gazedemo.app.collect import make_schedule, read_foods
from gazedemo.app.session import write_run_files
from gazedemo.config import project_path
from gazedemo.core.aoi import assign_label, build_masks, heatmap_argmax_xy
from gazedemo.core.metrics import compute
from gazedemo.io.logs import RunLogger, new_run_dir, write_csv, write_json
from gazedemo.perception.fake import FakeGaze, FakeWorld, default_aois
from gazedemo.types import INVALID, NONE, OUT, Sample


def gaze_plan(rng, plates, chosen: str, trial_s: float) -> list[tuple[float, float, str]]:
    """(start_s, end_s, plate) from t0; gaps between them are 'none'."""
    spans, t = [], float(rng.uniform(0.15, 0.5))
    others = [p for p in plates if p != chosen]
    while t < trial_s:
        plate = chosen if rng.random() < 0.45 else others[int(rng.integers(len(others)))]
        dur = float(rng.lognormal(np.log(0.9 if plate == chosen else 0.5), 0.4))
        spans.append((t, min(t + dur, trial_s), plate))
        t += dur + float(rng.uniform(0.05, 0.35))
    if rng.random() < 0.75:  # last glance on the chosen plate
        cut = trial_s - float(rng.uniform(0.8, 1.5))
        spans = [(a, min(b, cut), p) for a, b, p in spans if a < cut] + [(cut, trial_s, chosen)]
    return spans


def _target(spans, t_rel: float) -> str:
    for a, b, p in spans:
        if a <= t_rel < b:
            return p
    return NONE


def run_synth(cfg, participants: int = 1, trials: int = 60, out=Path("runs_synth"), seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    W, H = int(cfg.source.width), int(cfg.source.height)
    aois = default_aois(W, H)
    plates = [a.name for a in aois]
    world = FakeWorld.from_aois(aois, W, H)
    gaze = FakeGaze(world)
    masks = build_masks(aois, W, H)
    foods = read_foods(project_path("configs/foods.yaml"))
    hz, trial_s = float(cfg.synth.hz), float(cfg.collect.trial_s)
    dirs = []
    for k in range(participants):
        pid = f"P{k + 1:02d}"
        run_dir = new_run_dir(Path(out), f"collect_{pid}")
        liking = {f: int(rng.integers(1, 6)) for f in foods}
        schedule = make_schedule(foods, plates, trials, seed + k)
        write_run_files(run_dir, cfg, "collect", aois, (W, H), None, gaze, None,
                        {"participant": pid, "synthetic": True})
        write_json(run_dir / "session.json", {"participant": pid, "foods": foods, "liking": liking,
                                              "plates": plates, "n_trials": trials, "seed": seed + k,
                                              "trial_s": trial_s, "synthetic": True,
                                              "created": datetime.now().isoformat(timespec="seconds")})
        write_csv(run_dir / "schedule.csv", [{"trial": i + 1, **e} for i, e in enumerate(schedule)],
                  ["trial", *plates])
        logger = RunLogger(run_dir, plates)
        t, idx = 1000.0 + 10000.0 * k, 0
        for i, entry in enumerate(schedule):
            like = np.array([liking[entry[p]] for p in plates], dtype=float)
            p_choose = np.exp(like - like.max())
            chosen = plates[int(rng.choice(len(plates), p=p_choose / p_choose.sum()))]
            spans = gaze_plan(rng, plates, chosen, trial_s)
            t0, t1 = t + 1.0, t + 1.0 + trial_s
            meta = {"mode": "collect", "participant": pid, "schedule_idx": i + 1,
                    **{f"food_{p}": entry[p] for p in plates}, **{f"liking_{p}": liking[entry[p]] for p in plates}}
            logger.event("trial_start", t0, trial_id=i + 1, t0=t0, meta=meta)
            samples, ts = [], t0 - 0.6
            while ts < t1 + 0.3:
                rel = ts - t0
                target = _target(spans, rel) if 0 <= rel < trial_s else NONE
                r = rng.random()
                if r < 0.01:
                    s = Sample(ts, idx, INVALID, {}, None, None, 60.0)
                else:
                    if r < 0.03 or target == OUT:
                        hm, inout = np.full((64, 64), 0.01, np.float32), 0.05
                    else:
                        x, y = world.targets[target]
                        hm, inout = gaze.heatmap_at((x + rng.normal(0, 18), y + rng.normal(0, 12))), 0.95
                    label, mass = assign_label(hm, inout, masks, cfg.assign)
                    s = Sample(ts, idx, label, mass, heatmap_argmax_xy(hm, W, H), inout, 60.0)
                samples.append(s)
                logger.sample(s)
                idx += 1
                ts += 1.0 / hz + float(rng.normal(0, 0.004))
            logger.event("trial_end", t1, trial_id=i + 1, t1=t1, label=None)
            fields = {"label": chosen, "chosen_plate": chosen, "chosen_food": entry[chosen], "skipped": 0}
            logger.event("trial_update", t1 + 1.5, trial_id=i + 1, fields=fields)
            logger.trial(compute(samples, t0, t1, cfg.dwell, aois, trial_id=i + 1, meta=meta), fields=fields)
            t = t1 + 3.0
        logger.close()
        dirs.append(run_dir)
        print(f"[synth] {pid}: {trials} rounds → {run_dir}")
    return dirs
