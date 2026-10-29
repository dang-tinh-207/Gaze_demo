"""Wiring for live / cue / collect: AOIs, Gazelle, run folder, recorder, source, window, runner."""
from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

from gazedemo.config import project_path, save_config
from gazedemo.core.aoi import scale_aois
from gazedemo.io.logs import RunLogger, load_aoi, new_run_dir, read_json, save_aoi, write_json


def resolve_aois(cfg, aoi_arg, source_spec):
    """→ (aois, image size they were drawn on, path or None)."""
    path = Path(aoi_arg) if aoi_arg else project_path(cfg.aoi.file)
    if path.exists():
        aois, size, _ = load_aoi(path)
        if [a.name for a in aois] != list(cfg.aoi.names):
            print(f"[aoi] WARNING: {path} has plates {[a.name for a in aois]}, config aoi.names = {cfg.aoi.names}")
        return aois, size, path
    if str(source_spec) == "synthetic":
        from gazedemo.perception.fake import default_aois
        s = cfg.source
        return default_aois(s.width, s.height), (s.width, s.height), None
    raise FileNotFoundError(f"{path} does not exist → run `python -m gazedemo setup-aoi` first")


def fit_aois(aois, aoi_size, frame_size):
    """Same aspect ratio → rescale with a warning; otherwise the AOIs are meaningless."""
    if aoi_size is None or tuple(aoi_size) == tuple(frame_size):
        return aois
    sx, sy = frame_size[0] / aoi_size[0], frame_size[1] / aoi_size[1]
    if abs(sx - sy) > 0.01 * max(sx, sy):
        raise RuntimeError(f"aoi.json was drawn on a {aoi_size[0]}×{aoi_size[1]} image but the source gives "
                           f"{frame_size[0]}×{frame_size[1]} (different aspect ratio) → run setup-aoi again")
    print(f"[aoi] aoi.json drawn at {aoi_size[0]}×{aoi_size[1]}, source {frame_size[0]}×{frame_size[1]} → scaled ×{sx:.3f}")
    return scale_aois(aois, sx, sy)


def load_calibration(cfg, plates):
    path = project_path(cfg.calibration.file)
    if not path.exists():
        return None
    cal = read_json(path)
    if cal.get("plates") and list(cal["plates"]) != list(plates):
        print(f"[calibration] ignoring {path}: plates {cal['plates']} differ from {list(plates)}")
        return None
    print(f"[calibration] T = {cal['T']:.3f} ({path}) → adding p_look_cal")
    return cal


def load_gazelle(cfg):
    from gazedemo.perception.gazelle_model import GazelleModel
    print(f"[gazelle] load {cfg.gazelle.name} ...", flush=True)
    model = GazelleModel.from_config(cfg.gazelle)
    print(f"[gazelle] {model.describe()}, {model.load_s:.1f} s — {model.source}")
    return model


def fake_gaze(source, aois):
    from gazedemo.perception.fake import FakeGaze, FakeWorld
    world = getattr(source, "world", None) or FakeWorld.from_aois(aois, *source.size)
    return FakeGaze(world)


def write_run_files(run_dir, cfg, mode, aois, frame_size, aoi_path, gaze, args, extra=None):
    save_config(cfg, run_dir / "config.yaml")
    save_aoi(run_dir / "aoi.json", aois, frame_size, source=str(aoi_path) if aoi_path else "default_aois")
    info = {"mode": mode, "created": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
            "plates": [a.name for a in aois], "frame_size": list(frame_size),
            "gaze": getattr(gaze, "describe", lambda: type(gaze).__name__)(),
            "python": sys.version.split()[0], "platform": platform.platform()}
    if args is not None:
        info.update(source=str(getattr(args, "source", None) or cfg.source.device),
                    participant=getattr(args, "participant", None), model=getattr(args, "model", None))
    info.update(extra or {})
    write_json(run_dir / "run.json", info)


def run_session(cfg, mode: str, args) -> int:
    from gazedemo.app.choice import ChoiceScorer
    from gazedemo.app.collect import CollectController, read_foods
    from gazedemo.app.modes import CueController, LiveController
    from gazedemo.app.runner import Runner
    from gazedemo.app.ui import UI
    from gazedemo.io.camera import open_source
    from gazedemo.io.recorder import Recorder
    from gazedemo.io.sound import beep

    try:
        aois, aoi_size, aoi_path = resolve_aois(cfg, args.aoi, args.source)
        foods = read_foods(args.foods) if mode == "collect" else None
    except (OSError, ValueError) as e:
        print(e)
        return 1
    plates = [a.name for a in aois]
    choice = ChoiceScorer(args.model, plates) if args.model else None
    calibration = load_calibration(cfg, plates)
    gaze = None if args.fake_gaze else load_gazelle(cfg)

    run_dir = new_run_dir(project_path(cfg.output.root), mode)
    recorder = Recorder(run_dir, cfg.source.fps, cfg.source.codec) if cfg.source.record and not args.no_record else None
    source = open_source(args.source, cfg, on_frame=recorder.push if recorder else None, aois=aois)
    try:
        source.start()
        aois = fit_aois(aois, aoi_size, source.size)
    except Exception as e:
        print(e)
        source.stop()
        if recorder:
            recorder.close()
        return 1
    if gaze is None:
        gaze = fake_gaze(source, aois)

    write_run_files(run_dir, cfg, mode, aois, source.size, aoi_path, gaze, args,
                    {"calibration_T": calibration["T"] if calibration else None})
    beep_fn = lambda: beep(cfg.ui.beep_hz, cfg.ui.beep_ms)  # noqa: E731
    if mode == "live":
        controller = LiveController(cfg.trial.trial_s)
    elif mode == "cue":
        controller = CueController(cfg.cue, plates, beep_fn)
    else:
        seed = cfg.collect.seed if args.seed is None else args.seed
        controller = CollectController(cfg.collect, plates, args.participant, foods, args.trials, seed,
                                       run_dir, beep_fn)
    runner = Runner(cfg, source, gaze, aois, controller, RunLogger(run_dir, plates),
                    ui=UI(cfg.ui, plates, window=f"gazedemo {mode}"), choice=choice,
                    calibration=calibration, recorder=recorder, run_dir=run_dir)
    print(f"[{mode}] writing to {run_dir} — click the window, then use the keys ({controller.help})")
    results = runner.run()
    print(f"[{mode}] finished {len(results)} trials → {run_dir}")
    if mode == "cue" and results:
        from gazedemo.app.score import score_run
        score_run(run_dir, cfg)
    print("Remember to back up: git add -A; git commit -m \"data\"; git push — or copy runs/ to USB/Drive.")
    return 0
