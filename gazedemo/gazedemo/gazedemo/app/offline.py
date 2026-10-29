"""`offline` (SPEC 5): re-run the pipeline on a run's video.mp4 at the capture times in frames.csv,
replaying events.jsonl to rebuild the trials → runs/<id>/offline/ in the same format.
Uses the current config (to try new thresholds/models) and the run's own aoi.json."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gazedemo.app.modes import OfflineController
from gazedemo.app.runner import Runner
from gazedemo.app.session import fake_gaze, fit_aois, load_calibration, load_gazelle, write_run_files
from gazedemo.io.camera import VideoFileSource, read_frames_csv
from gazedemo.io.logs import RunLogger, load_aoi, num, read_csv, read_json, read_jsonl


class _ReplayController(OfflineController):
    """OfflineController + a progress line every 30 s of video."""

    def __init__(self, events, t_first, t_last):
        super().__init__(events)
        self.t_first, self.t_last, self._next = t_first, t_last, t_first + 30.0

    def tick(self, now):
        super().tick(now)
        if now >= self._next:
            print(f"[offline] {now - self.t_first:.0f} / {self.t_last - self.t_first:.0f} s video", flush=True)
            self._next += 30.0


def compare(live_csv: Path, off_csv: Path, plates) -> list[str]:
    live = {r["trial_id"]: r for r in read_csv(live_csv)} if live_csv.exists() else {}
    lines = ["trial | label | DD live (" + "/".join(plates) + ") | DD offline | n_sample live → offline"]
    for r in read_csv(off_csv):
        lv = live.get(r["trial_id"], {})
        dd = lambda row: "/".join(f"{num(row.get(f'{p}_dd_ms'), 0):.0f}" for p in plates)  # noqa: E731
        lines.append(f"{r['trial_id']:>5} | {r.get('label') or '—':^4} | {dd(lv) if lv else '—'} | {dd(r)} | "
                     f"{lv.get('n_samples', '—')} → {r['n_samples']}")
    return lines


def run_offline(cfg, run_dir: Path, stride: int = 1, model=None, use_fake_gaze: bool = False,
                gaze=None) -> Path | None:
    run_dir = Path(run_dir)
    video, frames = run_dir / "video.mp4", run_dir / "frames.csv"
    if not video.exists() or not frames.exists():
        print(f"[offline] {run_dir} has no video.mp4 + frames.csv (recorded with source.record = false?)")
        return None
    aois, aoi_size, _ = load_aoi(run_dir / "aoi.json")
    plates = [a.name for a in aois]
    rows = read_frames_csv(frames)
    source = VideoFileSource(video, frames, stride=stride).start()
    aois = fit_aois(aois, aoi_size, source.size)
    if gaze is None:
        gaze = fake_gaze(source, aois) if use_fake_gaze else load_gazelle(cfg)
    choice = None
    if model:
        from gazedemo.app.choice import ChoiceScorer
        choice = ChoiceScorer(model, plates)

    out = run_dir / "offline"
    if out.exists():  # keep an earlier re-run instead of mixing files
        out.rename(run_dir / f"offline_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir()
    info = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    write_run_files(out, cfg, "offline", aois, source.size, run_dir / "aoi.json", gaze, None,
                    {"of_run": str(run_dir), "of_mode": info.get("mode"), "stride": stride,
                     "participant": info.get("participant")})
    controller = _ReplayController(read_jsonl(run_dir / "events.jsonl"), rows[0][1] if rows else 0.0,
                                   rows[-1][1] if rows else 0.0)
    runner = Runner(cfg, source, gaze, aois, controller, RunLogger(out, plates), choice=choice,
                    calibration=load_calibration(cfg, plates), run_dir=out)
    print(f"[offline] {len(rows)} frames (stride {stride}) from {video} → {out}")
    results = runner.run()
    print(f"[offline] {len(results)} trials, {runner.n_samples} samples "
          f"(live: {len(read_jsonl(run_dir / 'samples.jsonl'))})")
    if results:
        print("\n".join(compare(run_dir / "trials.csv", out / "trials.csv", plates)))
    return out
