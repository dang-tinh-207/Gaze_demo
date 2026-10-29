"""Controllers: live (keys S/E/1-3), cue (look where told), offline (replay events.jsonl)."""
from __future__ import annotations

import numpy as np

from gazedemo.app.runner import Controller
from gazedemo.app.ui import Screen, plate_color


class LiveController(Controller):
    name = "live"
    help = "S start · E end · 1/2/3 label last trial · H heatmap · Q quit"

    def __init__(self, trial_s: float | None = None):
        super().__init__()
        self.trial_s = trial_s
        self.last_tid = None
        self.message = "Press S to start a trial"

    def on_key(self, key, now):
        r = self.runner
        if key == "s":
            if r.tracker.open is None:
                self.message = f"Trial {r.start_trial(now)} running — E to end"
        elif key == "e":
            self._end(now)
        elif key.isdigit() and 1 <= int(key) <= len(r.plates):
            target = r.tracker.open.trial_id if r.tracker.open else self.last_tid
            if target is not None:
                plate = r.plates[int(key) - 1]
                r.update_trial(target, label=plate)
                self.message = f"Trial {target}: label {plate}"
        else:
            super().on_key(key, now)

    def _end(self, t1):
        tid = self.runner.end_trial(t1)
        if tid is not None:
            self.last_tid = tid
            self.message = f"Trial {tid} done — 1/2/3 to label, S for a new trial"

    def tick(self, now):
        tr = self.runner.tracker.open
        if self.trial_s and tr is not None and now >= tr.t0 + self.trial_s:
            self._end(tr.t0 + self.trial_s)

    def status(self):
        return [self.message] if self.message else []


class CueController(Controller):
    """n_rep cues per plate in seeded random order; trial = [t_cue + settle_s, t_cue + cue_s),
    label = cued plate. The screen shows only the instruction (no gaze feedback to chase)."""
    name = "cue"
    help = "Space start · P pause/redo round · Q quit"

    def __init__(self, cue_cfg, plates, beep=lambda: None):
        super().__init__()
        self.c = cue_cfg
        self.plates = list(plates)
        seq = [p for p in self.plates for _ in range(int(cue_cfg.n_rep))]
        order = np.random.default_rng(cue_cfg.seed).permutation(len(seq))
        self.seq = [seq[i] for i in order]
        self.beep = beep
        self.i = 0
        self.state = "wait"
        self.t_cue = None
        self.trial_open = False

    def _cue(self, now):
        self.state, self.t_cue, self.trial_open = "cue", now, False
        self.beep()
        self.runner.logger.event("cue", now, plate=self.seq[self.i], index=self.i + 1)

    def on_key(self, key, now):
        if key == "space" and self.state == "wait":
            self._cue(now)
        elif key == "p" and self.state == "cue":
            if self.trial_open:
                self.runner.cancel_trial("paused")
            self.state = "paused"
        elif key == "p" and self.state == "paused":
            self._cue(now)
        else:
            super().on_key(key, now)

    def tick(self, now):
        if self.state == "cue":
            plate = self.seq[self.i]
            t0, t1 = self.t_cue + self.c.settle_s, self.t_cue + self.c.cue_s
            if not self.trial_open and now >= t0 and t1 > t0:
                self.runner.start_trial(t0, meta={"cue": plate, "cue_index": self.i + 1})
                self.trial_open = True
            if now >= t1:
                if self.trial_open:
                    self.runner.end_trial(t1, label=plate)
                self.i += 1
                if self.i < len(self.seq):
                    self._cue(now)
                else:
                    self.state = "finishing"
        elif self.state == "finishing" and not self.runner.tracker.pending:
            self.done = True

    def screen(self, now):
        n = len(self.seq)
        if self.state == "wait":
            return Screen("video", lines=[f"Look where told: {n} rounds ({' · '.join(self.plates)})",
                                          "Sit in position, then press Space"], show_table=False)
        if self.state == "cue":
            plate = self.seq[self.i]
            return Screen("text", big=f"LOOK AT {plate}", big_color=plate_color(self.plates, plate),
                          lines=[f"Round {self.i + 1}/{n}"], show_table=False)
        if self.state == "paused":
            return Screen("text", lines=["PAUSED", f"Press P to redo round {self.i + 1}"], show_table=False)
        return Screen("text", lines=["Done — computing results…"], show_table=False)


class OfflineController(Controller):
    """Replays trial events of a recorded run at their logged times."""
    name = "offline"
    help = ""
    TRIAL_EVENTS = ("trial_start", "trial_end", "trial_cancel", "trial_update")

    def __init__(self, events):
        super().__init__()
        self.events = sorted((e for e in events if e.get("type") in self.TRIAL_EVENTS), key=lambda e: e["t"])
        self.i = 0

    def tick(self, now):
        while self.i < len(self.events) and self.events[self.i]["t"] <= now:
            self._apply(self.events[self.i])
            self.i += 1

    def _apply(self, e):
        r = self.runner
        kind = e["type"]
        if kind == "trial_start":
            if r.tracker.open is not None:
                r.cancel_trial("new trial started before the previous one ended")
            r.start_trial(e["t0"], meta=e.get("meta"), trial_id=e["trial_id"])
        elif kind == "trial_end":
            r.end_trial(e["t1"], label=e.get("label"))
        elif kind == "trial_cancel":
            r.cancel_trial(e.get("reason", ""))
        elif kind == "trial_update":
            r.update_trial(e["trial_id"], **e.get("fields", {}))

    def finish(self, now):
        while self.i < len(self.events):  # answers given after the last frame
            self._apply(self.events[self.i])
            self.i += 1
        super().finish(now)
