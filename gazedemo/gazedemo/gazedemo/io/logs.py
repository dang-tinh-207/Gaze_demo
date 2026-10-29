"""Run folders and file formats: samples.jsonl, events.jsonl, trials.csv, dwells.csv, aoi.json.
Writing happens on a separate thread so the main loop never waits for the disk."""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from gazedemo.types import AOIMetrics, EllipseAOI, Sample, TrialResult

CSV_ENCODING = "utf-8-sig"  # BOM so Excel shows Vietnamese food names correctly
PER_PLATE = [f.name for f in fields(AOIMetrics)]
HEAD = ["trial_id", "mode", "t0", "t1", "dur_ms", "label"]
TAIL = ["p_look_none", "p_look_cal_none", "first_aoi", "last_aoi", "none_ms", "out_ms", "invalid_ms",
        "valid_ratio", "eff_hz", "n_samples"]
DWELL_COLUMNS = ["trial_id", "aoi", "t_start", "t_end", "dur_ms", "start_ms", "n_samples"]


# ---------- generic file helpers ----------

def new_run_dir(root, mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(root) / f"{stamp}_{mode}"
    d, n = base, 2
    while d.exists():
        d = base.with_name(f"{base.name}_{n}")
        n += 1
    d.mkdir(parents=True)
    return d


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written last line after a crash
    return out


def save_image(path, img) -> None:
    """cv2.imwrite fails on non-ASCII Windows paths (e.g. a user folder with diacritics)."""
    import cv2
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def load_image(path):
    import cv2
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding=CSV_ENCODING) as f:
        return list(csv.DictReader(f))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


def write_csv(path, rows, columns, exact=("t0", "t1", "t_start", "t_end")) -> None:
    """Atomic write (temp file + replace). Raises PermissionError if Excel holds the file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") if c in exact and r.get(c) is not None else _cell(r.get(c))
                        for c in columns])
    os.replace(tmp, path)


def num(s, default=None):
    """CSV cell → float (None/default if empty)."""
    if s is None or s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


# ---------- samples / trials ----------

def sample_to_dict(s: Sample) -> dict:
    return {"t": s.t, "frame_idx": s.frame_idx, "label": s.label,
            "mass": {k: round(v, 5) for k, v in s.mass.items()},
            "gaze_xy": [round(s.gaze_xy[0], 1), round(s.gaze_xy[1], 1)] if s.gaze_xy else None,
            "inout": None if s.inout is None else round(s.inout, 4),
            "latency_ms": round(s.latency_ms, 1)}


def sample_from_dict(d: dict) -> Sample:
    xy = d.get("gaze_xy")
    return Sample(t=float(d["t"]), frame_idx=int(d.get("frame_idx", 0)), label=d["label"],
                  mass={k: float(v) for k, v in (d.get("mass") or {}).items()},
                  gaze_xy=(float(xy[0]), float(xy[1])) if xy else None,
                  inout=None if d.get("inout") is None else float(d["inout"]),
                  latency_ms=float(d.get("latency_ms") or 0.0))


def read_samples(path) -> list[Sample]:
    return sorted((sample_from_dict(d) for d in read_jsonl(path)), key=lambda s: s.t)


def trial_row(res: TrialResult, plates, p_choice: dict | None = None) -> dict:
    row = {"trial_id": res.trial_id, "t0": res.t0, "t1": res.t1, "dur_ms": res.dur_ms, "label": res.label}
    for p in plates:
        m = res.per_aoi[p]
        for name in PER_PLATE:
            row[f"{p}_{name}"] = getattr(m, name)
        if p_choice:
            row[f"{p}_p_choice"] = p_choice.get(p)
    row.update(p_look_none=res.p_look_none, p_look_cal_none=res.p_look_cal_none,
               first_aoi=res.first_aoi, last_aoi=res.last_aoi, none_ms=res.none_ms, out_ms=res.out_ms,
               invalid_ms=res.invalid_ms, valid_ratio=res.valid_ratio, eff_hz=res.eff_hz,
               n_samples=res.n_samples)
    row.update(res.meta)
    return row


def trial_columns(rows, plates) -> list[str]:
    present = set().union(*(r.keys() for r in rows)) if rows else set()
    filled = {c for r in rows for c, v in r.items() if v not in (None, "")}
    per = [f"{p}_{m}" for p in plates for m in PER_PLATE + ["p_choice"]]
    collect = (["participant", "schedule_idx"] + [f"food_{p}" for p in plates]
               + ["chosen_plate", "chosen_food"] + [f"liking_{p}" for p in plates] + ["skipped"])
    known = HEAD + per + TAIL + collect

    def keep(c):
        if c in HEAD:
            return True
        if c == "p_look_cal_none" or c.endswith(("_p_look_cal", "_p_choice")):
            return c in filled  # only when calibration.json / --model was used
        return c in present
    return [c for c in known if keep(c)] + sorted(present - set(known))


def dwell_rows(res: TrialResult) -> list[dict]:
    return [{"trial_id": res.trial_id, "aoi": d.aoi, "t_start": d.t_start, "t_end": d.t_end,
             "dur_ms": d.dur * 1000.0, "start_ms": (d.t_start - res.t0) * 1000.0, "n_samples": d.n_samples}
            for d in res.dwells]


class RunLogger:
    """samples.jsonl / events.jsonl appended; trials.csv rewritten whenever a row changes
    (labels arrive after the trial ends); dwells.csv appended per finished trial."""

    def __init__(self, run_dir, plates):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.plates = list(plates)
        self.rows: dict[int, dict] = {}
        self._q: queue.Queue = queue.Queue()
        self._warned = False
        self._thread = threading.Thread(target=self._loop, name="logger", daemon=True)
        self._thread.start()

    def sample(self, s: Sample) -> None:
        self._q.put(("samples.jsonl", json.dumps(sample_to_dict(s), ensure_ascii=False)))

    def event(self, etype: str, t: float, **fields) -> None:
        self._q.put(("events.jsonl", json.dumps({"type": etype, "t": t, **fields}, ensure_ascii=False)))

    def trial(self, res: TrialResult, p_choice: dict | None = None, fields: dict | None = None) -> dict:
        row = trial_row(res, self.plates, p_choice)
        row.update(fields or {})
        self.rows[res.trial_id] = row
        self._q.put(("dwells", dwell_rows(res)))
        self._q.put(("trials", [dict(r) for r in self.rows.values()]))
        return row

    def update_trial(self, trial_id: int, fields: dict) -> bool:
        if trial_id not in self.rows:
            return False
        self.rows[trial_id].update(fields)
        self._q.put(("trials", [dict(r) for r in self.rows.values()]))
        return True

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30)

    def _write_trials(self, rows) -> bool:
        try:
            write_csv(self.dir / "trials.csv", rows, trial_columns(rows, self.plates))
            return True
        except PermissionError:
            if not self._warned:
                print("[log] trials.csv is open in another program (Excel?) → will write it at the next update")
                self._warned = True
            return False

    def _loop(self):
        files = {}
        pending_trials = None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "trials":
                    pending_trials = None if self._write_trials(payload) else payload
                elif kind == "dwells":
                    path = self.dir / "dwells.csv"
                    new = not path.exists()
                    with open(path, "a", newline="", encoding=CSV_ENCODING) as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(DWELL_COLUMNS)
                        for r in payload:
                            w.writerow([r[c] if c in ("t_start", "t_end") else _cell(r[c]) for c in DWELL_COLUMNS])
                else:
                    f = files.get(kind)
                    if f is None:
                        f = files[kind] = open(self.dir / kind, "a", encoding="utf-8")
                    f.write(payload + "\n")
                if self._q.empty():
                    for f in files.values():
                        f.flush()
        finally:
            for f in files.values():
                f.close()
            if pending_trials is not None:
                self._write_trials(pending_trials)


# ---------- aoi.json ----------

def save_aoi(path, aois, image_size, **extra) -> None:
    data = {"image_size": [int(image_size[0]), int(image_size[1])],
            "aois": [asdict(a) for a in aois], **extra}
    write_json(path, data)


def load_aoi(path) -> tuple[list[EllipseAOI], tuple[int, int], dict]:
    data = read_json(path)
    keys = ("name", "cx", "cy", "rx", "ry", "angle_deg")
    aois = [EllipseAOI(**{k: a[k] for k in keys if k in a}) for a in data["aois"]]
    w, h = data["image_size"]
    return aois, (int(w), int(h)), data
