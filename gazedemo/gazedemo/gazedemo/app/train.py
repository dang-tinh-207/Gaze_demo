"""`train` (SPEC 7.2): baselines + conditional logit per feature set; CV by trial (1 participant,
repeated over seeds) or leave-one-participant-out; models/eval.json, report.md, model.json."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from gazedemo.core.choice_model import ConditionalLogit
from gazedemo.core.eval import (BASELINES, baseline_scores, bootstrap_ci, confusion, group_folds,
                                hard_proba, kfold_trials, log_loss, top1)
from gazedemo.core.features import FEATURE_SETS, feature_matrix, parse_spans
from gazedemo.io.logs import read_csv, write_json

WARNING_ONE_PERSON = ("Data from 1 person, who collected the data and also designed the experiment; "
                      "the results only show that the pipeline works, they are not general evidence.")
NAMES = {"chance": "Chance", "argmax_dd": "Argmax DD", "argmax_p_look": "Argmax p_look",
         "last_look": "Last-looked plate", "first_look": "First-looked plate", "dwell_T": "Dwell ≥ T",
         "argmax_liking": "Liking only (argmax)", "logit:g1": "Logit G1 (TTFD/DD/DC)",
         "logit:g1g2": "Logit G1+G2", "logit:g1g2g3": "Logit G1+G2+G3", "logit:all": "Logit all",
         "logit:liking_only": "Logit liking only", "logit:all+liking": "Logit all + liking"}


def load_trials(path: Path):
    """Long CSV → (trials as lists of plate rows, plate order). Keeps trials with every plate and
    exactly one chosen plate."""
    groups: OrderedDict = OrderedDict()
    for r in read_csv(path):
        groups.setdefault((r["run_id"], r["trial_id"]), []).append(r)
    if not groups:
        return [], []
    plates = [r["plate"] for r in next(iter(groups.values()))]
    trials = [g for g in groups.values()
              if [r["plate"] for r in g] == plates and sum(int(r["chosen"]) for r in g) == 1]
    return trials, plates


def evaluate(trials, plates, tcfg) -> tuple[dict, str]:
    N, K = len(trials), len(plates)
    y = np.array([[int(r["chosen"]) for r in g].index(1) for g in trials])
    parts = [g[0]["participant"] for g in trials]

    def X(names):
        return np.stack([feature_matrix(g, names) for g in trials])

    data = {k: X([k])[..., 0] for k in ("dd_ms", "p_look", "is_last", "is_first", "liking")}
    data["dwells"] = [[parse_spans(r.get("dwells_ms")) for r in g] for g in trials]
    if len(set(parts)) >= 2:
        scheme, fold_sets = f"leave-one-participant-out ({len(set(parts))} participants)", [group_folds(parts)]
    else:
        k = max(2, min(int(tcfg.k_folds), N))
        scheme = f"{k}-fold by round × {len(tcfg.seeds)} seeds"
        fold_sets = [kfold_trials(N, k, s) for s in tcfg.seeds]

    out = OrderedDict()

    def summarize(name, acc, ll, conf, note=""):
        out[name] = {"label": NAMES.get(name, name), "top1": float(acc.mean()),
                     "top1_ci": bootstrap_ci(acc, tcfg.bootstrap, 0), "logloss": float(ll.mean()),
                     "logloss_ci": bootstrap_ci(ll, tcfg.bootstrap, 1), "n_trial": N,
                     "confusion": conf.tolist(), "note": note}

    for name in BASELINES:
        P = (np.full((N, K), 1.0 / K) if name == "chance"
             else hard_proba(baseline_scores(name, data, tcfg.dwell_T_ms), tcfg.baseline_eps))
        summarize(name, top1(P, y), log_loss(P, y), confusion(P, y, K))
    if N >= 2:
        for fs in tcfg.feature_sets:
            names = FEATURE_SETS[fs]
            Xs = X(names)
            accs, lls, confs = [], [], []
            for folds in fold_sets:
                P = np.full((N, K), 1.0 / K)
                for f in np.unique(folds):
                    train, test = folds != f, folds == f
                    if train.any():
                        P[test] = ConditionalLogit(names, tcfg.l2).fit(Xs[train], y[train]).predict_proba(Xs[test])
                accs.append(top1(P, y))
                lls.append(log_loss(P, y))
                confs.append(confusion(P, y, K))
            summarize(f"logit:{fs}", np.mean(accs, axis=0), np.mean(lls, axis=0), np.mean(confs, axis=0),
                      ", ".join(names))
    return out, scheme


def _ci(ci):
    return "—" if ci is None or np.isnan(ci[0]) else f"{ci[0]:.2f}–{ci[1]:.2f}"


def run_train(cfg, data_path: Path, out_dir: Path) -> dict | None:
    tcfg = cfg.train
    trials, plates = load_trials(data_path)
    if not trials:
        print(f"[train] {data_path} has no usable rounds (run `dataset` first)")
        return None
    N, parts = len(trials), sorted({g[0]["participant"] for g in trials})
    results, scheme = evaluate(trials, plates, tcfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")

    names = FEATURE_SETS["all"]
    y = np.array([[int(r["chosen"]) for r in g].index(1) for g in trials])
    model = ConditionalLogit(names, tcfg.l2).fit(np.stack([feature_matrix(g, names) for g in trials]), y)
    model_json = {**model.to_json(), "feature_set": "all", "plates": plates, "trained_at": now,
                  "data": str(data_path)}
    write_json(out_dir / "model.json", model_json)

    notes = []
    if len(parts) == 1:
        notes.append(WARNING_ONE_PERSON)
    if N < tcfg.min_trials:
        notes.append(f"Too little data: {N} rounds < train.min_trials = {tcfg.min_trials}; the numbers are very uncertain.")
    if N < 2:
        notes.append("At least 2 rounds are needed to cross-validate: baselines only.")
    write_json(out_dir / "eval.json", {"created": now, "data": str(data_path), "n_trial": N,
                                      "participants": parts, "plates": plates, "cv": scheme,
                                      "baseline_eps": tcfg.baseline_eps, "dwell_T_ms": tcfg.dwell_T_ms,
                                      "l2": tcfg.l2, "notes": notes, "models": results})
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["top1"])
    md = [f"# Food-choice classifier — {now}", "",
          f"- Data: `{data_path}` — {N} rounds, {len(parts)} participant(s) ({', '.join(parts)}), plates {', '.join(plates)}.",
          f"- CV: {scheme}; features standardised with the mean/std of each fold's training set; L2 = {tcfg.l2}.",
          f"- Hard baselines: probability 1 − ε (ε = {tcfg.baseline_eps}) for the predicted plate; ties → split evenly. "
          f"Dwell ≥ T with T = {tcfg.dwell_T_ms:.0f} ms. 95% CI: bootstrap over rounds ({tcfg.bootstrap} resamples).", ""]
    md += [f"> {n}" for n in notes] + ([""] if notes else [])
    md += ["| Model | Top-1 | 95% CI | Log-loss | 95% CI | n_trial |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in ranked:
        md.append(f"| {r['label']} | {r['top1']:.2f} | {_ci(r['top1_ci'])} | {r['logloss']:.2f} | "
                  f"{_ci(r['logloss_ci'])} | {r['n_trial']} |")
    best = results.get("logit:all")
    if best:
        md += ["", "## Confusion — Logit all (out-of-fold, averaged over seeds); rows = chosen plate",
               "| chosen \\ predicted | " + " | ".join(plates) + " |", "| --- " * (len(plates) + 1) + "|"]
        md += [f"| {p} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for p, row in zip(plates, best["confusion"])]
    md += ["", f"## model.json — Logit all trained on all {N} rounds (weights on standardised features)",
           "| feature | w |", "| --- | --- |"]
    md += [f"| {n} | {w:+.3f} |" for n, w in sorted(zip(names, model.w), key=lambda nw: -abs(nw[1]))]
    text = "\n".join(md) + "\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[train] → {out_dir / 'model.json'}, eval.json, report.md")
    return results
