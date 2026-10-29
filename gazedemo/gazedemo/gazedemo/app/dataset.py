"""`dataset` (SPEC 7.1): every `collect` run (its offline/ re-run if present) → long table with one
row per (trial, plate), recomputed with the same compute() as live."""
from __future__ import annotations

import bisect
from collections import Counter
from datetime import datetime
from pathlib import Path

from gazedemo.config import load_config
from gazedemo.core.features import G1, G2, G3, G4, plate_rows
from gazedemo.core.metrics import compute
from gazedemo.io.logs import load_aoi, num, read_csv, read_json, read_samples, write_csv, write_json

LONG_COLUMNS = (["run_id", "participant", "trial_id", "plate", "food", "liking", "chosen"]
                + list(G1 + G2 + G3 + G4) + ["valid_ratio", "eff_hz", "dwells_ms"])


def find_collect_runs(root: Path) -> list[Path]:
    runs = []
    for info in sorted(Path(root).rglob("run.json")):
        if info.parent.name != "offline" and read_json(info).get("mode") == "collect":
            runs.append(info.parent)
    return runs


def source_dir(run_dir: Path) -> Path:
    """Prefer offline/ (all frames) over the live run (frames dropped while Gazelle was busy)."""
    off = run_dir / "offline"
    return off if (off / "trials.csv").exists() and (off / "samples.jsonl").exists() else run_dir


def run_rows(run_dir: Path, cfg) -> tuple[list[dict], list[dict], Counter]:
    src = source_dir(run_dir)
    run_cfg = load_config(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else cfg
    aois, _, _ = load_aoi(src / "aoi.json" if (src / "aoi.json").exists() else run_dir / "aoi.json")
    plates = [a.name for a in aois]
    samples = read_samples(src / "samples.jsonl")
    times = [s.t for s in samples]
    long_rows, trial_rows, counts = [], [], Counter()
    for r in read_csv(src / "trials.csv"):
        chosen = r.get("chosen_plate") or ""
        if r.get("skipped") == "1" or chosen not in plates:
            counts["skipped"] += 1
            continue
        t0, t1 = float(r["t0"]), float(r["t1"])
        window = samples[bisect.bisect_left(times, t0 - run_cfg.dwell.lookback_s):bisect.bisect_left(times, t1)]
        res = compute(window, t0, t1, run_cfg.dwell, aois, trial_id=int(r["trial_id"]))
        if res.valid_ratio < cfg.dataset.min_valid_ratio:
            counts["low_valid"] += 1
            continue
        counts["used"] += 1
        liking = {p: num(r.get(f"liking_{p}")) for p in plates}
        for row in plate_rows(res, plates, liking):
            p = row["plate"]
            long_rows.append({"run_id": run_dir.name, "participant": r.get("participant", ""),
                              "trial_id": res.trial_id, "food": r.get(f"food_{p}", ""),
                              "chosen": int(p == chosen), **row})
        trial_rows.append({"run_id": run_dir.name, "participant": r.get("participant", ""),
                           "trial_id": res.trial_id, "source": src.name, "t0": t0, "t1": t1,
                           "dur_ms": res.dur_ms, **{f"food_{p}": r.get(f"food_{p}", "") for p in plates},
                           **{f"liking_{p}": liking[p] for p in plates}, "chosen_plate": chosen,
                           "chosen_food": r.get("chosen_food", ""),
                           **{f"{p}_{m}": getattr(res.per_aoi[p], m) for p in plates for m in ("ttfd_ms", "dd_ms", "dc", "p_look")},
                           "first_aoi": res.first_aoi, "last_aoi": res.last_aoi,
                           "valid_ratio": res.valid_ratio, "eff_hz": res.eff_hz})
    return long_rows, trial_rows, counts


def build_dataset(cfg, runs_root: Path, out: Path) -> dict | None:
    runs = find_collect_runs(runs_root)
    if not runs:
        print(f"[dataset] No collect runs in {runs_root}")
        return None
    long_rows, trial_rows, counts = [], [], Counter()
    for run in runs:
        lr, tr, c = run_rows(run, cfg)
        long_rows += lr
        trial_rows += tr
        counts += c
        origin = "offline" if source_dir(run) != run else "live"
        print(f"[dataset] {run.name} ({origin}): {c['used']} rounds used, "
              f"{c['skipped']} skipped, {c['low_valid']} with valid_ratio < {cfg.dataset.min_valid_ratio}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, long_rows, LONG_COLUMNS)
    trials_path = out.with_name(out.stem + "_trials.csv")
    write_csv(trials_path, trial_rows, list(trial_rows[0]) if trial_rows else ["run_id"])
    stats = {"created": datetime.now().isoformat(timespec="seconds"), "data": str(out),
             "n_runs": len(runs), "n_trials": counts["used"], "n_skipped": counts["skipped"],
             "n_low_valid": counts["low_valid"],
             "participants": sorted({t["participant"] for t in trial_rows}),
             "chosen_plate": dict(sorted(Counter(t["chosen_plate"] for t in trial_rows).items())),
             "chosen_food": dict(Counter(t["chosen_food"] for t in trial_rows).most_common()),
             "mean_dur_ms": sum(t["dur_ms"] for t in trial_rows) / len(trial_rows) if trial_rows else None,
             "mean_valid_ratio": sum(t["valid_ratio"] for t in trial_rows) / len(trial_rows) if trial_rows else None}
    write_json(out.with_name(out.stem + "_stats.json"), stats)
    print(f"[dataset] {stats['n_trials']} rounds ({len(long_rows)} rows) from {len(runs)} runs, "
          f"participants: {', '.join(stats['participants']) or '—'}")
    print(f"  chosen plates: {stats['chosen_plate']}")
    print(f"  chosen foods: {stats['chosen_food']}")
    print(f"  → {out} + {trials_path.name} + {out.stem}_stats.json")
    return stats
