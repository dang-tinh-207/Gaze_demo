"""`collect` (SPEC 6): liking per food, balanced food × plate schedule, keyboard-only trial flow
for one person working alone: Space start · 1/2/3 choose · X skip · P pause · Q save and quit."""
from __future__ import annotations

import itertools
from datetime import datetime

import numpy as np
import yaml

from gazedemo.app.runner import Controller
from gazedemo.app.ui import Screen
from gazedemo.io.logs import write_csv, write_json

LIKING_KEYS = ("1", "2", "3", "4", "5")


def read_foods(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        foods = [str(x) for x in (yaml.safe_load(f) or {}).get("foods", [])]
    if len(set(foods)) != len(foods):
        raise ValueError(f"{path}: duplicate food names")
    return foods


def make_schedule(foods, plates, n_trials: int, seed: int) -> list[dict[str, str]]:
    """Each trial puts len(plates) different foods on the plates. Greedy over all choices: least
    shown foods first, then the food × plate cells used least, avoiding repeating the previous
    set; seeded random tie-break. Keeps every food × plate count within about one of the others."""
    k = len(plates)
    if len(foods) < k:
        raise ValueError(f"need at least {k} different foods, foods.yaml has {len(foods)}")
    rng = np.random.default_rng(seed)
    order = [foods[i] for i in rng.permutation(len(foods))]
    shown = {f: 0 for f in foods}
    at = {(f, p): 0 for f in foods for p in plates}
    prev, out = None, []
    for _ in range(n_trials):
        best, best_key = None, None
        for combo in itertools.combinations(order, k):
            for perm in itertools.permutations(combo):
                cells = [at[(f, p)] for f, p in zip(perm, plates)]
                key = (sum(shown[f] for f in perm), max(cells), sum(cells),
                       frozenset(perm) == prev, rng.random())
                if best_key is None or key < best_key:
                    best, best_key = perm, key
        for f, p in zip(best, plates):
            shown[f] += 1
            at[(f, p)] += 1
        prev = frozenset(best)
        out.append(dict(zip(plates, best)))
    return out


class CollectController(Controller):
    name = "collect"

    def __init__(self, collect_cfg, plates, participant: str, foods, n_trials: int, seed: int,
                 run_dir, beep=lambda: None):
        super().__init__()
        self.c = collect_cfg
        self.plates = list(plates)
        self.participant = participant
        self.food_list = list(foods)
        self.seed = seed
        self.schedule = make_schedule(self.food_list, self.plates, n_trials, seed)
        self.run_dir = run_dir
        self.beep = beep
        self.liking: dict[str, int] = {}
        self.state = "liking"
        self.t_state = 0.0
        self.i = 0
        self.tid = None
        self.results = {}
        self.answer = None
        self.n_chosen = 0
        self.n_skipped = 0

    @property
    def help(self) -> str:
        return {"liking": "1–5 liking · Backspace fix · Q quit",
                "ready": "Space start round · P pause · H heatmap · Q save and quit",
                "ask": "1/2/3 chosen plate · X skip round · Q quit",
                "result": "Space next round · Q save and quit",
                "paused": "P resume · Q save and quit"}.get(self.state, "P pause · Q save and quit")

    def attach(self, runner):
        super().attach(runner)
        rows = [{"trial": i + 1, **entry} for i, entry in enumerate(self.schedule)]
        write_csv(self.run_dir / "schedule.csv", rows, ["trial", *self.plates])

    def _set(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    def _save_session(self) -> None:
        write_json(self.run_dir / "session.json", {
            "participant": self.participant, "foods": self.food_list, "liking": self.liking,
            "plates": self.plates, "n_trials": len(self.schedule), "seed": self.seed,
            "trial_s": self.c.trial_s, "created": datetime.now().isoformat(timespec="seconds")})

    def _meta(self) -> dict:
        entry = self.schedule[self.i]
        meta = {"participant": self.participant, "schedule_idx": self.i + 1}
        meta.update({f"food_{p}": entry[p] for p in self.plates})
        meta.update({f"liking_{p}": self.liking.get(entry[p]) for p in self.plates})
        return meta

    def on_key(self, key, now):
        st = self.state
        if key in ("q", "esc"):
            self.done = True
        elif st == "liking":
            if key in LIKING_KEYS:
                self.liking[self.food_list[len(self.liking)]] = int(key)
                if len(self.liking) == len(self.food_list):
                    self._save_session()
                    self._set("ready", now)
            elif key == "backspace" and self.liking:
                self.liking.pop(next(reversed(self.liking)))
        elif st == "ready":
            if key == "space":
                self._set("countdown", now)
            elif key == "p":
                self._set("paused", now)
        elif st in ("countdown", "trial") and key == "p":
            if st == "trial":
                self.runner.cancel_trial("paused")
            self._set("paused", now)
        elif st == "ask":
            if key.isdigit() and 1 <= int(key) <= len(self.plates):
                self._answer(self.plates[int(key) - 1], now)
            elif key == "x":
                self._answer(None, now)
        elif st == "result" and key == "space":
            self._next(now)
        elif st == "paused" and key == "p":
            self._set("ready", now)

    def _answer(self, plate, now):
        if plate:
            food = self.schedule[self.i][plate]
            self.runner.update_trial(self.tid, label=plate, chosen_plate=plate, chosen_food=food, skipped=0)
            self.n_chosen += 1
        else:
            self.runner.update_trial(self.tid, label="", chosen_plate="", chosen_food="", skipped=1)
            self.n_skipped += 1
        self.answer = plate
        self._set("result", now)

    def tick(self, now):
        st = self.state
        if st == "countdown" and now - self.t_state >= self.c.countdown_s:
            self.beep()
            self.runner.logger.event("beep", now, kind="t0")
            self.tid = self.runner.start_trial(now, meta=self._meta())
            self._set("trial", now)
        elif st == "trial" and now - self.t_state >= self.c.trial_s:
            self.beep()
            self.runner.logger.event("beep", now, kind="t1")
            self.runner.end_trial(now)  # t1 = beep; gaze after it is not used
            self._set("ask", now)
        elif st == "result" and now - self.t_state >= self.c.show_result_s:
            self._next(now)
        elif st == "finishing" and not self.runner.tracker.pending:
            self.done = True

    def _next(self, now):
        self.i += 1
        self._set("ready" if self.i < len(self.schedule) else "finishing", now)

    def on_trial_finished(self, res):
        super().on_trial_finished(res)
        self.results[res.trial_id] = res

    def finish(self, now):
        if self.state == "ask" and self.tid is not None:  # quit before answering
            self.runner.update_trial(self.tid, label="", chosen_plate="", chosen_food="", skipped=1)
        super().finish(now)
        print(f"[collect] {self.n_chosen} rounds with a choice, {self.n_skipped} skipped → {self.run_dir}")

    def foods(self):
        if self.state in ("ready", "countdown", "trial", "ask", "result") and self.i < len(self.schedule):
            return dict(self.schedule[self.i])
        return {}

    def display_result(self):
        return self.results.get(self.tid) if self.state == "result" else None

    def screen(self, now):
        n, st = len(self.schedule), self.state
        if st == "liking":
            lines = ["How much do you like each food? 1 = not at all … 5 = very much"]
            for i, f in enumerate(self.food_list):
                mark = "→ " if i == len(self.liking) else "   "
                lines.append(f"{mark}{f}: {self.liking.get(f, '_')}")
            return Screen("text", lines=lines, show_table=False)
        if st == "ready":
            entry = self.schedule[self.i]
            place = " · ".join(f"{p} = {entry[p]}" for p in self.plates)
            return Screen("video", lines=[f"Round {self.i + 1}/{n} — Place: {place}",
                                          "Place the cards, sit back in position, then press Space"], show_table=False)
        if st == "countdown":
            return Screen("text", big="Get ready…", big_size=0.12, show_table=False)
        if st == "trial":
            return Screen("text", big="Look and choose the food you want", big_size=0.075, show_table=False)
        if st == "ask":
            entry = self.schedule[self.i]
            keys = " · ".join(f"{k + 1} = {p} ({entry[p]})" for k, p in enumerate(self.plates))
            return Screen("text", big="Which plate did you choose?", big_size=0.1, lines=[keys, "X = skip round"],
                          show_table=False)
        if st == "result":
            entry = self.schedule[self.i]
            text = (f"Round {self.i + 1}: chose {self.answer} ({entry[self.answer]})" if self.answer
                    else f"Round {self.i + 1}: skipped")
            return Screen("video", lines=[text], show_table=True)
        if st == "paused":
            return Screen("text", big="PAUSED", big_size=0.12,
                          lines=["Press P to resume (the unfinished round will be redone)"], show_table=False)
        return Screen("text", lines=[f"Done {self.n_chosen + self.n_skipped}/{n} rounds — saving…"],
                      show_table=False)

    def status(self):
        if self.state == "liking":
            return []
        return [f"{self.participant}: {self.n_chosen} chosen · {self.n_skipped} skipped · round {min(self.i + 1, len(self.schedule))}/{len(self.schedule)}"]
