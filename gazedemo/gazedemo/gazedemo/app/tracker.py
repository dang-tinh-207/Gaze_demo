"""GazeDwellTracker: keeps recent samples and turns trials [t0, t1) into TrialResults via
core.metrics.compute (recomputed from samples each time, never accumulated)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gazedemo.core.metrics import compute
from gazedemo.types import Sample, TrialResult


@dataclass
class OpenTrial:
    trial_id: int
    t0: float
    t1: float | None = None
    label: str | None = None
    meta: dict = field(default_factory=dict)


class GazeDwellTracker:
    def __init__(self, dwell_cfg, trial_cfg, aois, temperature: float | None = None, cal_eps: float = 1e-4):
        self.dwell_cfg = dwell_cfg
        self.finalize_timeout_s = float(trial_cfg.finalize_timeout_s)
        self.aois = list(aois)
        self.temperature = temperature
        self.cal_eps = cal_eps
        self.samples: deque[Sample] = deque()
        self.open: OpenTrial | None = None
        self.pending: list[OpenTrial] = []   # ended, waiting for a sample with t >= t1
        self._next_id = 1

    @property
    def last_t(self) -> float | None:
        return self.samples[-1].t if self.samples else None

    def update(self, sample: Sample) -> None:
        self.samples.append(sample)
        keep_from = sample.t - max(self.dwell_cfg.buffer_s, self.dwell_cfg.lookback_s)
        for tr in ([self.open] if self.open else []) + self.pending:
            keep_from = min(keep_from, tr.t0 - self.dwell_cfg.lookback_s)
        while self.samples and self.samples[0].t < keep_from:
            self.samples.popleft()

    def start_trial(self, t0: float, trial_id: int | None = None, meta: dict | None = None) -> int:
        if self.open is not None:
            raise RuntimeError(f"trial {self.open.trial_id} is still open")
        tid = self._next_id if trial_id is None else int(trial_id)
        self._next_id = max(self._next_id, tid + 1)
        self.open = OpenTrial(tid, float(t0), meta=dict(meta or {}))
        return tid

    def end_trial(self, t1: float, label: str | None = None) -> int | None:
        if self.open is None:
            return None
        tr, self.open = self.open, None
        tr.t1, tr.label = max(float(t1), tr.t0), label
        self.pending.append(tr)
        return tr.trial_id

    def cancel_trial(self) -> int | None:
        tr, self.open = self.open, None
        return tr.trial_id if tr else None

    def poll_finished(self, now: float) -> list[TrialResult]:
        """A trial is final once a sample with t >= t1 arrived (only gaze before t1 counts) or
        finalize_timeout_s after t1 (camera stalled)."""
        last = self.last_t
        done = [tr for tr in self.pending
                if (last is not None and last >= tr.t1) or now - tr.t1 >= self.finalize_timeout_s]
        return self._finish(done)

    def flush(self) -> list[TrialResult]:
        """Finalize every ended trial (end of a video / quitting)."""
        return self._finish(list(self.pending))

    def _finish(self, done) -> list[TrialResult]:
        self.pending = [tr for tr in self.pending if tr not in done]
        return [self._compute(tr.t0, tr.t1, tr.trial_id, False, tr.label, tr.meta) for tr in done]

    def snapshot(self, now: float) -> TrialResult | None:
        """Provisional result of the open trial over [t0, now)."""
        tr = self.open
        if tr is None:
            return None
        return self._compute(tr.t0, max(now, tr.t0), tr.trial_id, True, tr.label, tr.meta)

    def window(self, window_s: float, now: float) -> TrialResult:
        """Sliding window [now - window_s, now) (robot runtime)."""
        return self._compute(now - window_s, now, 0, True, None, {})

    def _compute(self, t0, t1, trial_id, provisional, label, meta) -> TrialResult:
        return compute(self.samples, t0, t1, self.dwell_cfg, self.aois, trial_id=trial_id,
                       provisional=provisional, label=label, meta=meta,
                       temperature=self.temperature, cal_eps=self.cal_eps)
