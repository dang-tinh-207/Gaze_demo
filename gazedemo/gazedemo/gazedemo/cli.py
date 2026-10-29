"""python -m gazedemo <command>. Imports are per command so dataset/train/report run without torch/camera."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gazedemo.config import load_config, project_path


def _source_args(p, record=True):
    p.add_argument("--source", help="camera index | synthetic | image | video (default: source.device)")
    p.add_argument("--fake-gaze", action="store_true",
                   help="FakeGaze instead of Gazelle (try the UI without a GPU; use with --source synthetic)")
    p.add_argument("--aoi", help="aoi.json (default: aoi.file in the config)")
    if record:
        p.add_argument("--no-record", action="store_true", help="do not record video.mp4 + frames.csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gazedemo",
                                description="Which plate is looked at (local Gazelle) → TTFD/DD/DC/p_look → p_choice")
    p.add_argument("--config", help="YAML file that overrides configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one key, e.g. --set source.device=1 (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check Python/torch/CUDA/GPU/camera/Gazelle")
    d.add_argument("--max-index", type=int, default=4, help="probe camera indices 0..N")
    d.add_argument("--no-model", action="store_true", help="do not load Gazelle")

    b = sub.add_parser("bench", help="time frame reading, Gazelle and plate assignment (ViT-L/B × fp32/fp16)")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--source", help="default: camera from the config; no camera → synthetic frames")
    b.add_argument("--models", help="model names, comma-separated")

    s = sub.add_parser("setup-aoi", help="draw the 3 plate ellipses → aoi.json + aoi_ref.png")
    s.add_argument("--source", help="camera index | image | video | synthetic")
    s.add_argument("--out", help="default: aoi.file in the config")

    for name, desc in [("live", "trials with keys S/E, labels 1/2/3"),
                       ("cue", "look where told → accuracy + confusion"),
                       ("collect", "collect food-choice data")]:
        q = sub.add_parser(name, help=desc)
        _source_args(q)
        q.add_argument("--model", help="models/model.json → also show p_choice")
        if name == "collect":
            q.add_argument("--participant", required=True)
            q.add_argument("--foods", default=str(project_path("configs/foods.yaml")))
            q.add_argument("--trials", type=int, default=30)
            q.add_argument("--seed", type=int, help="default: collect.seed")

    o = sub.add_parser("offline", help="re-run the pipeline on a run's video.mp4")
    o.add_argument("--run", required=True)
    o.add_argument("--stride", type=int, default=1)
    o.add_argument("--model", help="models/model.json → add p_choice")
    o.add_argument("--fake-gaze", action="store_true", help="FakeGaze (synthetic runs / tests)")

    sc = sub.add_parser("score", help="accuracy, confusion, reliability/ECE of a labelled run")
    sc.add_argument("--run", required=True)

    ca = sub.add_parser("calibrate", help="fit the p_look temperature from a cue run")
    ca.add_argument("--run", required=True)
    ca.add_argument("--out", help="default: calibration.file in the config")

    ds = sub.add_parser("dataset", help="merge collect runs into a long-format dataset")
    ds.add_argument("--runs", default="runs")
    ds.add_argument("--out", default="data/gaze_choice.csv")

    tr = sub.add_parser("train", help="CV + baselines + conditional logit → models/")
    tr.add_argument("--data", default="data/gaze_choice.csv")
    tr.add_argument("--out", default="models")

    rp = sub.add_parser("report", help="gather bench/score/dataset/model → report/SUMMARY.md")
    rp.add_argument("--out", default="report")
    rp.add_argument("--runs", default="runs")
    rp.add_argument("--data", default="data/gaze_choice.csv")
    rp.add_argument("--models", default="models")

    sy = sub.add_parser("synth", help="generate synthetic collect runs (to test dataset → train)")
    sy.add_argument("--participants", type=int, default=1)
    sy.add_argument("--trials", type=int, default=60)
    sy.add_argument("--out", default="runs_synth")
    sy.add_argument("--seed", type=int, help="default: synth.seed")
    for cmd in sub.choices.values():  # --config/--set also accepted after the command name
        cmd.add_argument("--config", dest="config_after", help=argparse.SUPPRESS)
        cmd.add_argument("--set", dest="set_after", action="append", default=[], help=argparse.SUPPRESS)
    return p


def cmd_doctor(cfg, a):
    from gazedemo.app.doctor import run_doctor
    return run_doctor(cfg, a.max_index, not a.no_model)


def cmd_bench(cfg, a):
    from gazedemo.app.bench import run_bench
    run_bench(cfg, a.n, a.source, a.models.split(",") if a.models else None)


def cmd_setup_aoi(cfg, a):
    from gazedemo.app.setup_aoi import run_setup_aoi
    return run_setup_aoi(cfg, a.source, Path(a.out) if a.out else project_path(cfg.aoi.file))


def cmd_session(cfg, a):
    from gazedemo.app.session import run_session
    return run_session(cfg, a.cmd, a)


def cmd_offline(cfg, a):
    from gazedemo.app.offline import run_offline
    return 0 if run_offline(cfg, Path(a.run), a.stride, a.model, a.fake_gaze) else 1


def cmd_score(cfg, a):
    from gazedemo.app.score import score_run
    score_run(Path(a.run), cfg)


def cmd_calibrate(cfg, a):
    from gazedemo.app.score import calibrate_run
    calibrate_run(Path(a.run), cfg, Path(a.out) if a.out else project_path(cfg.calibration.file))


def cmd_dataset(cfg, a):
    from gazedemo.app.dataset import build_dataset
    build_dataset(cfg, Path(a.runs), Path(a.out))


def cmd_train(cfg, a):
    from gazedemo.app.train import run_train
    run_train(cfg, Path(a.data), Path(a.out))


def cmd_report(cfg, a):
    from gazedemo.app.report import make_report
    make_report(cfg, Path(a.out), Path(a.runs), Path(a.data), Path(a.models))


def cmd_synth(cfg, a):
    from gazedemo.app.synth import run_synth
    run_synth(cfg, a.participants, a.trials, Path(a.out), cfg.synth.seed if a.seed is None else a.seed)


COMMANDS = {"doctor": cmd_doctor, "bench": cmd_bench, "setup-aoi": cmd_setup_aoi, "live": cmd_session,
            "cue": cmd_session, "collect": cmd_session, "offline": cmd_offline, "score": cmd_score,
            "calibrate": cmd_calibrate, "dataset": cmd_dataset, "train": cmd_train, "report": cmd_report,
            "synth": cmd_synth}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Vietnamese output survives redirection on Windows
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config_after or args.config, args.set + args.set_after)
    except (OSError, ValueError) as e:
        print(f"Config error: {e}")
        return 2
    return COMMANDS[args.cmd](cfg, args) or 0
