"""`score` (accuracy, confusion, frame reliability/ECE of a labelled run) and `calibrate`
(temperature for p_look from a cue run), SPEC 1.1 and 5."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from gazedemo.config import project_path
from gazedemo.core.aoi import q_matrix
from gazedemo.core.calibration import apply_temperature, fit_temperature, nll, reliability
from gazedemo.io.logs import num, read_csv, read_json, read_samples, write_json
from gazedemo.types import INVALID, NONE


def run_plates(run_dir: Path, cfg) -> list[str]:
    info = run_dir / "run.json"
    if info.exists():
        plates = read_json(info).get("plates")
        if plates:
            return list(plates)
    return list(cfg.aoi.names)


def labelled_trials(run_dir: Path, plates) -> list[dict]:
    path = run_dir / "trials.csv"
    return [r for r in read_csv(path) if r.get("label") in plates] if path.exists() else []


def trial_frames(samples, trials, plates) -> tuple[np.ndarray, np.ndarray]:
    """Valid frames inside labelled trials → Q (n, plates + none), y = index of the label."""
    times = np.array([s.t for s in samples])
    frames, y = [], []
    for r in trials:
        lo, hi = np.searchsorted(times, float(r["t0"])), np.searchsorted(times, float(r["t1"]))
        for s in samples[lo:hi]:
            if s.label != INVALID and s.mass:
                frames.append(s)
                y.append(plates.index(r["label"]))
    return q_matrix([s.mass for s in frames], [s.inout for s in frames], plates), np.array(y, dtype=int)


def _argmax_or_none(values, plates, positive=True):
    vals = [v for v in values if v is not None]
    if not vals or (positive and max(vals) <= 0):
        return NONE
    return plates[int(np.argmax([-np.inf if v is None else v for v in values]))]


def _confusion_md(C, plates, cols) -> list[str]:
    out = ["| true \\ predicted | " + " | ".join(cols) + " |", "| --- " * (len(cols) + 1) + "|"]
    for i, p in enumerate(plates):
        out.append(f"| {p} | " + " | ".join(str(int(v)) for v in C[i]) + " |")
    return out


def _reliability_md(rows) -> list[str]:
    out = ["| q_max bin | frames | mean q_max | fraction correct |", "| --- | --- | --- | --- |"]
    for r in rows:
        conf = "—" if r["conf"] is None else f"{r['conf']:.2f}"
        acc = "—" if r["acc"] is None else f"{r['acc']:.2f}"
        out.append(f"| {r['lo']:.1f}–{r['hi']:.1f} | {r['n']} | {conf} | {acc} |")
    return out


def score_run(run_dir, cfg, quiet: bool = False) -> dict:
    run_dir = Path(run_dir)
    plates = run_plates(run_dir, cfg)
    trials = labelled_trials(run_dir, plates)
    cols = plates + [NONE]
    C_dd = np.zeros((len(plates), len(cols)), dtype=int)
    C_pl = np.zeros_like(C_dd)
    for r in trials:
        k = plates.index(r["label"])
        C_dd[k, cols.index(_argmax_or_none([num(r.get(f"{p}_dd_ms"), 0.0) for p in plates], plates))] += 1
        C_pl[k, cols.index(_argmax_or_none([num(r.get(f"{p}_p_look")) for p in plates], plates, False))] += 1
    n = len(trials)
    valid = [num(r.get("valid_ratio")) for r in trials if num(r.get("valid_ratio")) is not None]
    samples = read_samples(run_dir / "samples.jsonl")
    Q, y = trial_frames(samples, trials, plates)
    rows, ece = reliability(Q, y, int(cfg.score.n_bins))
    res = {"run": str(run_dir), "n_trials": n, "per_label": {p: int(C_dd[i].sum()) for i, p in enumerate(plates)},
           "correct_dd": int(np.trace(C_dd[:, :len(plates)])), "correct_p_look": int(np.trace(C_pl[:, :len(plates)])),
           "confusion_dd": C_dd.tolist(), "confusion_p_look": C_pl.tolist(), "columns": cols,
           "mean_valid_ratio": float(np.mean(valid)) if valid else None, "n_frames": int(len(y)),
           "reliability": rows, "ece": ece, "created": datetime.now().isoformat(timespec="seconds")}
    res["accuracy_dd"] = res["correct_dd"] / n if n else None
    res["accuracy_p_look"] = res["correct_p_look"] / n if n else None
    cal_path = project_path(cfg.calibration.file)
    if cal_path.exists() and len(y):
        cal = read_json(cal_path)
        if list(cal.get("plates") or plates) == plates:
            rows_c, ece_c = reliability(apply_temperature(Q, cal["T"], cal["eps"]), y, int(cfg.score.n_bins))
            res.update(calibration_T=cal["T"], reliability_cal=rows_c, ece_cal=ece_c)

    md = [f"# Score — {run_dir.name}", ""]
    if not n:
        md.append("No labelled trials (live: press 1/2/3 after each trial; cue labels automatically).")
    else:
        dist = " · ".join(f"{p} {c}" for p, c in res["per_label"].items())
        md += [f"- Labelled trials: {n} ({dist})",
               f"- **Accuracy (plate with the largest DD = label): {res['correct_dd']}/{n} = {res['accuracy_dd']:.2f}**",
               f"- Accuracy by argmax p_look: {res['correct_p_look']}/{n} = {res['accuracy_p_look']:.2f}",
               f"- Mean fraction of valid time (valid_ratio): "
               + ("—" if res["mean_valid_ratio"] is None else f"{res['mean_valid_ratio']:.2f}"), "",
               "## Confusion — prediction = plate with the largest DD (all 0 → none)", *_confusion_md(C_dd, plates, cols), "",
               "## Confusion — prediction = argmax p_look", *_confusion_md(C_pl, plates, cols), "",
               f"## Per-frame p_look reliability ({len(y)} valid frames inside the trials)",
               "q_max = max of q over A/B/C/none; correct = argmax q is the labelled plate.", "",
               *_reliability_md(rows), "", f"ECE = {'—' if ece is None else f'{ece:.3f}'}"]
        if "ece_cal" in res:
            md += ["", f"### After calibration (T = {res['calibration_T']:.3f})", *_reliability_md(res["reliability_cal"]),
                   "", f"ECE = {res['ece_cal']:.3f}"]
    text = "\n".join(md) + "\n"
    (run_dir / "score.md").write_text(text, encoding="utf-8")
    write_json(run_dir / "score.json", res)
    if not quiet:
        print(text)
        print(f"[score] Wrote {run_dir / 'score.md'}")
    return res


def calibrate_run(run_dir, cfg, out_path) -> dict | None:
    run_dir, out_path = Path(run_dir), Path(out_path)
    plates = run_plates(run_dir, cfg)
    trials = labelled_trials(run_dir, plates)
    Q, y = trial_frames(read_samples(run_dir / "samples.jsonl"), trials, plates)
    if len(y) < 20:
        print(f"[calibrate] only {len(y)} labelled frames in {run_dir} → needs a completed cue run")
        return None
    eps = float(cfg.calibration.eps)
    T = fit_temperature(Q, y, eps)
    P = apply_temperature(Q, T, eps)
    _, ece_before = reliability(Q, y, int(cfg.score.n_bins))
    _, ece_after = reliability(P, y, int(cfg.score.n_bins))
    res = {"T": T, "eps": eps, "plates": plates, "n_frames": int(len(y)), "n_trials": len(trials),
           "run": str(run_dir), "nll_before": nll(Q, y), "nll_after": nll(P, y),
           "ece_before": ece_before, "ece_after": ece_after,
           "created": datetime.now().isoformat(timespec="seconds")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, res)
    print(f"[calibrate] T = {T:.3f} from {len(y)} frames / {len(trials)} trials")
    print(f"  log-loss {res['nll_before']:.3f} → {res['nll_after']:.3f} · ECE {ece_before:.3f} → {ece_after:.3f}")
    print(f"  → {out_path} (live/cue/collect will add p_look_cal)")
    return res
