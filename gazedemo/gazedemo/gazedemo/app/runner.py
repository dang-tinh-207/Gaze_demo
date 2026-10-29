"""Main loop shared by live / cue / collect / offline:
frame → Gazelle → plate label → tracker → logs + window. Modes plug in as Controllers."""
from __future__ import annotations

import cv2

from gazedemo.app.tracker import GazeDwellTracker
from gazedemo.app.ui import Screen, View
from gazedemo.config import check_max_interval
from gazedemo.core.aoi import assign_label, build_masks, heatmap_argmax_xy
from gazedemo.io.logs import save_image
from gazedemo.types import INVALID, Sample, TrialResult


def _f(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


class Controller:
    """Mode logic. Times are in the source's clock (time.monotonic() live, recorded time offline)."""
    name = "base"
    help = "H heatmap · Q quit"

    def __init__(self):
        self.done = False
        self.runner = None
        self.last_result: TrialResult | None = None

    def attach(self, runner) -> None:
        self.runner = runner

    def on_key(self, key: str, now: float) -> None:
        if key in ("q", "esc"):
            self.done = True

    def tick(self, now: float) -> None:
        pass

    def on_trial_finished(self, res: TrialResult) -> None:
        self.last_result = res

    def finish(self, now: float) -> None:
        if self.runner.tracker.open is not None:
            self.runner.cancel_trial("quit during a trial")

    def screen(self, now: float) -> Screen:
        return Screen()

    def foods(self) -> dict[str, str]:
        return {}

    def status(self) -> list[str]:
        return []

    def display_result(self) -> TrialResult | None:
        return self.last_result


class Runner:
    def __init__(self, cfg, source, gaze, aois, controller: Controller, logger, *, ui=None, keys=None,
                 choice=None, calibration: dict | None = None, recorder=None, run_dir=None):
        self.cfg = cfg
        self.source = source
        self.gaze = gaze
        self.aois = list(aois)
        self.plates = [a.name for a in self.aois]
        self.controller = controller
        self.logger = logger
        self.ui = ui
        self.keys = sorted(keys or [], key=lambda k: k[0])  # scripted (t, key) for headless runs
        self.choice = choice
        self.recorder = recorder
        self.run_dir = run_dir
        cal = calibration or {}
        self.tracker = GazeDwellTracker(cfg.dwell, cfg.trial, self.aois, cal.get("T"),
                                        cal.get("eps", cfg.calibration.eps))
        self.clock = source.clock
        self.results: list[TrialResult] = []
        self.pending_updates: dict[int, dict] = {}
        self.masks = None
        self.size = None
        self.last_sample: Sample | None = None
        self.last_heatmap = None
        self.last_frame = None
        self.hz = None
        self.latency_ms = None
        self.n_samples = 0
        self._dt = None
        self._prev_t = None
        self._errors = 0
        self._snapshot_due = False
        self._snapshot_done = False

    # ---------- trial API for controllers (every change is also an event for offline replay) ----------

    def start_trial(self, t0: float, meta: dict | None = None, trial_id: int | None = None) -> int:
        meta = {"mode": self.controller.name, **(meta or {})}
        tid = self.tracker.start_trial(t0, trial_id, meta)
        self.logger.event("trial_start", self.clock(), trial_id=tid, t0=t0, meta=meta)
        return tid

    def end_trial(self, t1: float, label: str | None = None) -> int | None:
        tid = self.tracker.end_trial(t1, label)
        if tid is not None:
            self.logger.event("trial_end", self.clock(), trial_id=tid, t1=t1, label=label)
        return tid

    def cancel_trial(self, reason: str = "") -> int | None:
        tid = self.tracker.cancel_trial()
        if tid is not None:
            self.logger.event("trial_cancel", self.clock(), trial_id=tid, reason=reason)
            print(f"[trial {tid}] cancelled ({reason})")
        return tid

    def update_trial(self, trial_id: int, **fields) -> None:
        """Label/choice arrive after the trial ends; applied when it is finalized if not yet."""
        self.logger.event("trial_update", self.clock(), trial_id=trial_id, fields=fields)
        if not self.logger.update_trial(trial_id, fields):
            self.pending_updates.setdefault(trial_id, {}).update(fields)

    # ---------- loop ----------

    def run(self) -> list[TrialResult]:
        ctl = self.controller
        describe = getattr(self.gaze, "describe", lambda: type(self.gaze).__name__)
        self.logger.event("run_start", self.clock(), mode=ctl.name, plates=self.plates, gaze=describe())
        ctl.attach(self)
        try:
            while not ctl.done:
                item = self.source.read(timeout=0.2)
                if item is not None:
                    self._process(*item)
                elif self.source.finished:
                    if self.source.error:
                        print(f"[runner] {self.source.error}")
                    break
                now = self.clock()
                ctl.tick(now)
                while self.keys and self.keys[0][0] <= now:
                    self.handle_key(self.keys.pop(0)[1], now)
                for res in self.tracker.poll_finished(now):
                    self._finished(res)
                if self.ui is not None:
                    canvas = self.ui.render(self.view(now))
                    if self._snapshot_due and not self._snapshot_done and self.run_dir is not None:
                        save_image(self.run_dir / "snapshot.png", canvas)
                        self._snapshot_done = True
                    key = self.ui.show(canvas)
                    if key:
                        self.handle_key(key, self.clock())
        except KeyboardInterrupt:
            print("\n[runner] Ctrl+C → saving and quitting")
        finally:
            self.close()
        return self.results

    def handle_key(self, key: str, now: float) -> None:
        self.logger.event("key", now, key=key)
        if key == "h" and self.ui is not None:
            self.ui.toggle_heatmap()
        else:
            self.controller.on_key(key, now)

    def close(self) -> None:
        now = self.clock()
        try:
            self.controller.finish(now)
        finally:
            for res in self.tracker.flush():
                self._finished(res)
            self.source.stop()
            rec = {}
            if self.recorder is not None:
                self.recorder.close()
                rec = {"video_frames": self.recorder.written, "video_dropped": self.recorder.dropped}
            self.logger.event("run_end", now, n_trials=len(self.results), n_samples=self.n_samples,
                              errors=self._errors, **rec)
            self.logger.close()
            if self.ui is not None:
                self.ui.close()

    def _process(self, idx: int, t: float, frame) -> None:
        if self.masks is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            self.masks = build_masks(self.aois, w, h)
        w, h = self.size
        hm = inout = None
        try:
            hm, inout = self.gaze.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, mass = assign_label(hm, inout, self.masks, self.cfg.assign)
            xy = heatmap_argmax_xy(hm, w, h)
        except Exception as e:  # never stop the session for one bad frame
            label, mass, xy = INVALID, {}, None
            self._error(t, e)
        latency = max(0.0, (self.clock() - t) * 1000.0)
        s = Sample(t=t, frame_idx=idx, label=label, mass=mass, gaze_xy=xy, inout=inout, latency_ms=latency)
        self.tracker.update(s)
        self.logger.sample(s)
        self.last_sample, self.last_heatmap, self.last_frame = s, hm, frame
        self._rate(t, latency)

    def _rate(self, t: float, latency: float) -> None:
        self.n_samples += 1
        if self._prev_t is not None and t > self._prev_t:
            dt = t - self._prev_t
            self._dt = dt if self._dt is None else 0.9 * self._dt + 0.1 * dt
            self.hz = 1.0 / self._dt
        self._prev_t = t
        self.latency_ms = latency if self.latency_ms is None else 0.9 * self.latency_ms + 0.1 * latency
        if self.n_samples == 30 and self._dt:
            warning = check_max_interval(self.cfg.dwell, self._dt)
            if warning:
                print(warning)

    def _error(self, t: float, e: Exception) -> None:
        self._errors += 1
        self.logger.event("error", t, where="gazelle", msg=f"{type(e).__name__}: {e}")
        if self._errors <= 3 or self._errors % 100 == 0:
            print(f"[runner] Gazelle error #{self._errors} → sample invalid: {type(e).__name__}: {e}")

    def _finished(self, res: TrialResult) -> None:
        pc = self.choice.predict(res) if self.choice is not None else None
        fields = self.pending_updates.pop(res.trial_id, {})
        self.logger.trial(res, p_choice=pc, fields=fields)
        self.results.append(res)
        print(self.summary(res, fields.get("label", res.label), pc), flush=True)
        self._snapshot_due = True
        self.controller.on_trial_finished(res)

    def summary(self, res: TrialResult, label=None, p_choice=None) -> str:
        parts = []
        for p in self.plates:
            m = res.per_aoi[p]
            s = f"{p} TTFD {_f(m.ttfd_ms, 0)} DD {m.dd_ms:.0f} DC {m.dc} p_look {_f(m.p_look)}"
            if p_choice:
                s += f" p_choice {_f(p_choice.get(p))}"
            parts.append(s)
        tag = f" label {label}" if label else ""
        return (f"[trial {res.trial_id}{tag}] " + " | ".join(parts)
                + f" | valid {res.valid_ratio:.0%} · {res.eff_hz:.1f} Hz")

    def view(self, now: float) -> View:
        ctl = self.controller
        tr = self.tracker.open
        result = self.tracker.snapshot(now) if tr is not None else ctl.display_result()
        pc = self.choice.predict(result) if self.choice is not None and result is not None else None
        return View(mode=ctl.name, frame=self.last_frame, sample=self.last_sample, heatmap=self.last_heatmap,
                    aois=self.aois, hz=self.hz, latency_ms=self.latency_ms,
                    trial_id=tr.trial_id if tr else (result.trial_id if result else None),
                    trial_t=(now - tr.t0) if tr else None, result=result, foods=ctl.foods(),
                    p_choice=pc, screen=ctl.screen(now), help=ctl.help, status=ctl.status())
