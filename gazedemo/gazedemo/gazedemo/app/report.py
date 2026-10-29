"""`report`: latest bench + cue score + dataset stats + classifier → report/SUMMARY.md laid out like
docs/DEMO_SUPERVISOR.md (missing numbers stay "?"), plus an overlay frame and a label timeline."""
from __future__ import annotations

import bisect
from datetime import datetime
from pathlib import Path

from gazedemo.config import load_config, project_path
from gazedemo.core.dwell import build_timeline, dwell_timeline
from gazedemo.core.metrics import compute
from gazedemo.io.logs import (load_aoi, load_image, num, read_csv, read_json, read_samples, save_image)
from gazedemo.types import INVALID, NONE, OUT

Q = "?"


def _latest(paths):
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _runs(root: Path, mode: str) -> list[Path]:
    out = []
    for info in Path(root).glob("*/run.json"):
        if read_json(info).get("mode") == mode:
            out.append(info.parent)
    return sorted(out)


def _fmt(v, d=2):
    return Q if v is None else f"{v:.{d}f}"


def _example_trial(rows, plates) -> list[str]:
    r = rows[0]
    has_food = any(r.get(f"food_{p}") for p in plates)
    has_pc = any(r.get(f"{p}_p_choice") not in (None, "") for p in plates)
    head = "| plate | " + ("food | " if has_food else "") + "TTFD (ms) | DD (ms) | DC | p_look |" + (" p_choice |" if has_pc else "")
    lines = [head, "| " + " | ".join(["---"] * (head.count("|") - 1)) + " |"]
    for p in plates:
        cells = [p] + ([r.get(f"food_{p}", "")] if has_food else [])
        ttfd = num(r.get(f"{p}_ttfd_ms"))
        cells += ["—" if ttfd is None else f"{ttfd:.0f}", f"{num(r.get(f'{p}_dd_ms'), 0):.0f}",
                  r.get(f"{p}_dc", ""), _fmt(num(r.get(f"{p}_p_look")))]
        if has_pc:
            cells.append(_fmt(num(r.get(f"{p}_p_choice"))))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    chosen = r.get("chosen_plate") or r.get("label")
    lines.append(f"\nRound {r.get('trial_id')} ({r.get('mode', '')}), {num(r.get('dur_ms'), 0) / 1000:.1f} s long, "
                 f"valid {num(r.get('valid_ratio'), 0):.0%}" + (f", chosen/label: **{chosen}**" if chosen else ""))
    return lines


def overlay_image(run_dir: Path, cfg, out_png: Path) -> bool:
    """snapshot.png saved by the live window, else rebuilt from video.mp4 + samples + aoi.json."""
    snap = run_dir / "snapshot.png"
    if snap.exists():
        save_image(out_png, load_image(snap))
        return True
    video, frames = run_dir / "video.mp4", run_dir / "frames.csv"
    trials = read_csv(run_dir / "trials.csv") if (run_dir / "trials.csv").exists() else []
    if not (video.exists() and frames.exists() and trials):
        return False
    import cv2
    from gazedemo.app.ui import UI, View
    from gazedemo.io.camera import read_frames_csv
    aois, _, _ = load_aoi(run_dir / "aoi.json")
    samples = read_samples(run_dir / "samples.jsonl")
    t0, t1 = float(trials[0]["t0"]), float(trials[0]["t1"])
    rows = read_frames_csv(frames)
    t_mid = (t0 + t1) / 2
    k = min(range(len(rows)), key=lambda i: abs(rows[i][1] - t_mid))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, frame = cap.read()
    cap.release()
    if not ok or not samples:
        return False
    s = min(samples, key=lambda x: abs(x.t - rows[k][1]))
    res = compute(samples, t0, rows[k][1], cfg.dwell, aois, trial_id=int(trials[0]["trial_id"]), provisional=True)
    plates = [a.name for a in aois]
    view = View(mode=read_json(run_dir / "run.json").get("mode", ""), frame=frame, sample=s, aois=aois,
                trial_id=res.trial_id, trial_t=rows[k][1] - t0, result=res,
                foods={p: trials[0].get(f"food_{p}", "") for p in plates if trials[0].get(f"food_{p}")})
    save_image(out_png, UI(cfg.ui, plates).render(view))
    return True


def timeline_image(run_dir: Path, cfg, out_png: Path) -> bool:
    """Per-frame labels vs. dwells for the first trial with a dwell (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    trials = read_csv(run_dir / "trials.csv") if (run_dir / "trials.csv").exists() else []
    samples = read_samples(run_dir / "samples.jsonl") if (run_dir / "samples.jsonl").exists() else []
    if not trials or not samples:
        return False
    aois, _, _ = load_aoi(run_dir / "aoi.json")
    plates = [a.name for a in aois]
    trial = next((r for r in trials if any(num(r.get(f"{p}_dc"), 0) > 0 for p in plates)), trials[0])
    t0, t1 = float(trial["t0"]), float(trial["t1"])
    times = [x.t for x in samples]
    sel = samples[bisect.bisect_left(times, t0 - cfg.dwell.lookback_s):bisect.bisect_left(times, t1)]
    from gazedemo.app.ui import plate_color
    colors = {p: tuple(c / 255 for c in plate_color(plates, p)[::-1]) for p in plates}
    colors.update({NONE: (0.85, 0.85, 0.85), OUT: (0.45, 0.45, 0.45), INVALID: (0.1, 0.1, 0.1)})
    rows = [("per-frame label", build_timeline(sel, t0, t1, cfg.dwell.max_interval_s)),
            ("after 3-step dwell", [s for s in dwell_timeline(sel, t0 - cfg.dwell.lookback_s, t1, cfg.dwell)
                                  if s.t_end > t0])]
    fig, ax = plt.subplots(figsize=(10, 2.0))
    for y, (_, segs) in enumerate(rows):
        for s in segs:
            a, b = max(s.t_start, t0) - t0, min(s.t_end, t1) - t0
            if b > a:
                ax.barh(1 - y, b - a, left=a, color=colors.get(s.label, (0.6, 0.6, 0.6)), height=0.6)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel(f"seconds from t0 (round {trial['trial_id']})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), ncol=len(colors), loc="upper center", bbox_to_anchor=(0.5, 1.45), frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def make_report(cfg, out_dir: Path, runs_root: Path, data_path: Path, models_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(runs_root)
    bench_json = _latest(list(runs_root.glob("*_bench/bench.json")))
    bench = read_json(bench_json) if bench_json else None
    cue_runs = _runs(runs_root, "cue")
    score = None
    if cue_runs:
        cue = cue_runs[-1]
        if not (cue / "score.json").exists():
            from gazedemo.app.score import score_run
            score_run(cue, cfg, quiet=True)
        score = read_json(cue / "score.json")
    data_path = Path(data_path)
    stats_path = data_path.with_name(data_path.stem + "_stats.json")
    stats = read_json(stats_path) if stats_path.exists() else None
    ev = read_json(Path(models_dir) / "eval.json") if (Path(models_dir) / "eval.json").exists() else None

    # model / speed line
    model_name, hz, gpu = cfg.gazelle.name, None, None
    if bench:
        gpu = bench.get("gpu")
        dtype = "fp16" if cfg.gazelle.fp16 else "fp32"
        row = next((r for r in bench["rows"] if r["model"] == model_name and r["dtype"] == dtype), None)
        hz = row["hz_final"] if row else None
    vit = "ViT-L" if "vitl" in model_name else "ViT-B"

    # newest collected run first, then live, then cue
    candidates = [r for mode in ("collect", "live") for r in reversed(_runs(runs_root, mode))] + cue_runs[::-1]
    candidates = [r for r in candidates if (r / "trials.csv").exists() and read_csv(r / "trials.csv")]
    example_rows, plates = None, list(cfg.aoi.names)
    if candidates:
        example_rows = read_csv(candidates[0] / "trials.csv")
        plates = read_json(candidates[0] / "run.json").get("plates", plates)
    images = []
    if any(overlay_image(r, cfg, out_dir / "overlay.png") for r in candidates):
        images.append("![Frame with overlay](overlay.png)")
    for r in candidates:
        run_cfg = load_config(r / "config.yaml") if (r / "config.yaml").exists() else cfg
        if timeline_image(r, run_cfg, out_dir / "timeline.png"):
            images.append("![Label timeline of one round](timeline.png)")
            break

    now = datetime.now().isoformat(timespec="seconds")
    md = ["# Quick report: gaze module → TTFD/DD/DC/p_look → food-choice classifier", "",
          f"_Generated by `python -m gazedemo report` at {now}. Cells without numbers yet show \"?\"._", "",
          "## 1. What the module does",
          f"External camera → Gazelle ({vit}, {_fmt(hz, 1)} Hz on GPU {gpu or Q}) → each frame is assigned to plate "
          f"{'/'.join(plates)}/none by the share of the heatmap inside each plate ellipse → grouped into dwells → each round outputs, per plate:", "",
          "| Feature | Meaning |", "| --- | --- |",
          "| TTFD | Time from the start of the round to the first look at the plate |",
          "| DD | Total time spent looking at the plate |", "| DC | Number of visits to the plate |",
          "| p_look | Probability of looking at the plate (mean heatmap share inside the plate, normalised) |",
          "| p_choice | Probability of choosing the food, from the trained classifier |", "",
          "Example of a real round (from `trials.csv`):", ""]
    md += _example_trial(example_rows, plates) if example_rows else [Q]
    md += ["", *images, ""]

    md += [f"## 2. Is \"which plate\" correct? (cue, 1 person, {score['n_trials'] if score else Q} rounds)"]
    if score and score["n_trials"]:
        cols = score["columns"]
        md += [f"- Accuracy (plate with the largest DD = cued plate): {score['correct_dd']} / {score['n_trials']}",
               f"- Accuracy by argmax p_look: {score['correct_p_look']} / {score['n_trials']}",
               f"- Mean fraction of valid time: {_fmt(score['mean_valid_ratio'])}",
               f"- Per-frame ECE of p_look: {_fmt(score['ece'], 3)}"
               + (f" (after calibration, T = {score['calibration_T']:.2f}: {_fmt(score['ece_cal'], 3)})" if "ece_cal" in score else ""),
               "- Confusion matrix (rows = cued plate):", "",
               "| cued \\ predicted | " + " | ".join(cols) + " |", "| --- " * (len(cols) + 1) + "|"]
        md += [f"| {p} | " + " | ".join(str(v) for v in row) + " |" for p, row in zip(cols, score["confusion_dd"])]
    else:
        md += [f"- Accuracy (plate with the largest DD = cued plate): {Q} / {Q}", f"- Confusion matrix: {Q}",
               f"- Mean fraction of valid time: {Q}"]

    md += ["", "## 3. Food-choice data collected"]
    if stats:
        n_foods = len(stats["chosen_food"]) if stats.get("chosen_food") else Q
        foods_total = Q
        trials_csv = data_path.with_name(data_path.stem + "_trials.csv")
        if trials_csv.exists():
            rows = read_csv(trials_csv)
            foods_total = len({r[f"food_{p}"] for r in rows for p in plates if r.get(f"food_{p}")})
        md += [f"- Rounds: {stats['n_trials']} ({len(stats['participants'])} person(s): {', '.join(stats['participants'])}), "
               f"{_fmt((stats['mean_dur_ms'] or 0) / 1000, 1)} s per round, {len(plates)} foods out of {foods_total}, "
               "food positions balanced across plates.",
               f"- Skipped: {stats['n_skipped']} rounds; dropped for valid_ratio < {cfg.dataset.min_valid_ratio}: {stats['n_low_valid']}.",
               f"- Chosen plates: {stats['chosen_plate']} · chosen foods: {stats['chosen_food']} ({n_foods} different foods).",
               "- Ground truth: the chosen plate, entered after the beep; only gaze before the beep is used."]
    else:
        md += [f"- Rounds: {Q} (1 person), {Q} s per round, 3 foods out of 6, food positions balanced across plates.",
               "- Ground truth: the chosen plate, entered after the beep; only gaze before the beep is used."]

    md += ["", f"## 4. Classifier (conditional logit, {ev['cv'] if ev else 'K-fold by round'})"]
    for note in (ev or {}).get("notes", []):
        md.append(f"> {note}")
    md += ["", "| Model | Top-1 | Log-loss | 95% CI (top-1) |", "| --- | --- | --- | --- |"]
    wanted = [("chance", "Chance"), ("argmax_dd", "Argmax DD"), ("last_look", "Last-looked plate"),
              ("argmax_liking", "Liking only"), ("logit:g1", "Logit G1 (TTFD/DD/DC)"),
              ("logit:all", "Logit all"), ("logit:all+liking", "Logit all + liking")]
    for key, label in wanted:
        r = (ev or {}).get("models", {}).get(key)
        if r:
            ci = "—" if key == "chance" else f"{r['top1_ci'][0]:.2f}–{r['top1_ci'][1]:.2f}"
            md.append(f"| {label} | {r['top1']:.2f} | {r['logloss']:.2f} | {ci} |")
        else:
            md.append(f"| {label} | {'0.33' if key == 'chance' else Q} | {'1.10' if key == 'chance' else Q} | — |")
    if ev:
        md.append("\nFull table (all baselines and feature sets): see the `models/report.md` appendix.")
    md += ["", "## 5. Limitations (stated plainly)",
           "- 1 person, who also designed the experiment → knows the purpose and may look \"on purpose\". "
           "The results show that the pipeline works; they are not general evidence.",
           "- AOIs drawn by hand, no head box, no ArUco yet.",
           "- Gazelle has not been evaluated on a feeding rig (it was trained on general scenes).", "",
           "## 6. Next steps", "- Collect data from more people (≥ ? people × 40 rounds) → LOSO.",
           "- ArUco for automatic AOIs; compare head box none vs fixed.",
           "- Integrate into the framework: p_choice → fusion with speech and prior.", "", "---", "## Appendix"]
    appendix = [("bench.md", bench_json.with_name("bench.md") if bench_json else None),
                ("score.md (cue)", cue_runs[-1] / "score.md" if cue_runs else None),
                ("models/report.md", Path(models_dir) / "report.md")]
    for title, path in appendix:
        if path is not None and path.exists():
            body = path.read_text(encoding="utf-8").replace("\n# ", "\n#### ").replace("\n## ", "\n#### ")
            body = body.lstrip("# ").split("\n", 1)
            md += ["", f"### {title} — {body[0]}", body[1] if len(body) > 1 else ""]
        else:
            md += ["", f"### {title}", Q]
    out = out_dir / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[report] → {out}" + (f" (+ {', '.join(i.split('(')[1][:-1] for i in images)})" if images else ""))
    return out
