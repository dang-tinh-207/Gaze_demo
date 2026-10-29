"""`bench`: mean/p95 of frame read, Gazelle and plate assignment for ViT-L/ViT-B × fp32/fp16."""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from gazedemo.config import check_max_interval, project_path
from gazedemo.core.aoi import assign_label, build_masks, heatmap_argmax_xy
from gazedemo.io.logs import load_aoi, new_run_dir


def _stats(ms) -> tuple[float, float]:
    return float(np.mean(ms)), float(np.percentile(ms, 95))


def _model_pair(name: str) -> list[str]:
    other = name.replace("vitl14", "vitb14") if "vitl14" in name else name.replace("vitb14", "vitl14")
    return sorted({name, other}, key=lambda n: "vitb14" in n)  # ViT-L first


def _grab_frames(cfg, source, n):
    """Returns (frames to infer on, read times ms, description, camera fps)."""
    from gazedemo.io.camera import open_capture, open_source, quiet_opencv
    s = cfg.source
    if source is None or str(source).isdigit():
        device = s.device if source is None else int(source)
        with quiet_opencv():
            cap = open_capture(device, s.backend, s.width, s.height, s.fps, s.fourcc)
        if cap is not None:
            frames, read_ms = [], []
            try:
                for i in range(n):
                    t = time.perf_counter()
                    ok, frame = cap.read()
                    read_ms.append((time.perf_counter() - t) * 1000)
                    if ok and i % 10 == 0 and len(frames) < 20:
                        frames.append(frame)
            finally:
                cap.release()
            if frames:
                w, h = frames[0].shape[1], frames[0].shape[0]
                fps = 1000.0 / np.mean(read_ms[1:] or read_ms)
                return frames, read_ms, f"camera {device} ({w}×{h}, {s.fourcc or 'default'})", fps
        print(f"[bench] Cannot read camera {device} → using synthetic frames (only Gazelle is timed)")
        source = "synthetic"
    src = open_source(source, cfg).start()
    frames, read_ms = [], []
    try:
        for i in range(min(n, 60)):
            t = time.perf_counter()
            item = src.read(timeout=1.0)
            read_ms.append((time.perf_counter() - t) * 1000)
            if item is not None and len(frames) < 20:
                frames.append(item[2].copy())
    finally:
        src.stop()
    w, h = src.size
    return frames, None, f"{source} ({w}×{h}, not a camera → frame read time skipped)", None


def run_bench(cfg, n: int = 200, source=None, models=None):
    import torch
    from gazedemo.perception.gazelle_model import GazelleModel

    frames, read_ms, src_desc, cam_fps = _grab_frames(cfg, source, n)
    h, w = frames[0].shape[:2]
    aoi_path = project_path(cfg.aoi.file)
    if aoi_path.exists():
        aois, _, _ = load_aoi(aoi_path)
    else:
        from gazedemo.perception.fake import default_aois
        aois = default_aois(w, h)
    masks = build_masks(aois, w, h)

    names = models or _model_pair(cfg.gazelle.name)
    rows = []
    for name in names:
        print(f"[bench] load {name} ...", flush=True)
        model = GazelleModel.from_config(cfg.gazelle, name=name, fp16=False, warmup=0)
        for fp16 in ([False, True] if model.device.type == "cuda" else [False]):
            model.fp16 = fp16
            model.warmup(cfg.gazelle.warmup, size=(h, w))
            g_ms, a_ms = [], []
            for i in range(n):
                frame = frames[i % len(frames)]
                t0 = time.perf_counter()
                hm, inout = model.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                t1 = time.perf_counter()
                assign_label(hm, inout, masks, cfg.assign)
                heatmap_argmax_xy(hm, w, h)
                t2 = time.perf_counter()
                g_ms.append((t1 - t0) * 1000)
                a_ms.append((t2 - t1) * 1000)
            g_mean, g_p95 = _stats(g_ms)
            a_mean, a_p95 = _stats(a_ms)
            hz = 1000.0 / (g_mean + a_mean)
            row = dict(model=name, dtype="fp16" if model.use_fp16 else "fp32", device=model.device.type,
                       preprocess=model.preprocess, gazelle_mean_ms=g_mean, gazelle_p95_ms=g_p95,
                       assign_mean_ms=a_mean, assign_p95_ms=a_p95, hz_proc=hz,
                       hz_final=min(hz, cam_fps) if cam_fps else hz)
            rows.append(row)
            print(f"  {name} {row['dtype']}: Gazelle {g_mean:.1f}/{g_p95:.1f} ms, plate assignment "
                  f"{a_mean:.2f}/{a_p95:.2f} ms → {row['hz_final']:.1f} Hz", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = new_run_dir(project_path(cfg.output.root), "bench")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
    read_stats = _stats(read_ms) if read_ms else None
    info = dict(created=datetime.now().isoformat(timespec="seconds"), n=n, source=src_desc, gpu=gpu,
                python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
                read_mean_ms=read_stats[0] if read_stats else None,
                read_p95_ms=read_stats[1] if read_stats else None, camera_fps=cam_fps, rows=rows)
    current = [r for r in rows if r["model"] == cfg.gazelle.name
               and r["dtype"] == ("fp16" if cfg.gazelle.fp16 and r["device"] == "cuda" else "fp32")]
    warning = check_max_interval(cfg.dwell, 1.0 / current[0]["hz_final"]) if current else None
    info["advice"] = _advice(cfg, rows)
    info["warning"] = warning
    (out_dir / "bench.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    md = _markdown(info)
    (out_dir / "bench.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[bench] Wrote {out_dir / 'bench.md'}")
    return out_dir / "bench.md"


def _advice(cfg, rows) -> str:
    def best(prefix):
        cand = [r for r in rows if prefix in r["model"]]
        return max(cand, key=lambda r: r["hz_final"]) if cand else None
    vitl, vitb = best("vitl14"), best("vitb14")
    if vitl and vitl["hz_final"] >= 5:
        return (f"ViT-L {vitl['dtype']} reaches {vitl['hz_final']:.1f} Hz (≥ 5) → keep gazelle.name = "
                f"{vitl['model']}, gazelle.fp16 = {vitl['dtype'] == 'fp16'}.")
    if vitb:
        return (f"ViT-L is slow (< 5 Hz) → switch gazelle.name = {vitb['model']} "
                f"({vitb['dtype']}, {vitb['hz_final']:.1f} Hz).")
    return "Not enough data for a recommendation."


def _markdown(info) -> str:
    lines = [f"# Bench — {info['created']}", "",
             f"- Machine: {info['platform']} · Python {info['python']} · torch {info['torch']} · GPU {info['gpu']}",
             f"- Frame source: {info['source']}"]
    if info["read_mean_ms"] is not None:
        lines.append(f"- Frame read: mean {info['read_mean_ms']:.1f} ms, p95 {info['read_p95_ms']:.1f} ms "
                     f"(≈ {info['camera_fps']:.1f} fps camera)")
    lines += [f"- n = {info['n']} frames per configuration. Gazelle = BGR→RGB + preprocessing + model + copy to CPU; "
              "plate assignment = assign_label + argmax. Final Hz = min(processing Hz, camera fps).", "",
              "| Model | dtype | Gazelle mean (ms) | Gazelle p95 (ms) | Assign mean (ms) | Assign p95 (ms) | Processing Hz | Final Hz |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in info["rows"]:
        tag = "ViT-L" if "vitl14" in r["model"] else "ViT-B"
        lines.append(f"| {tag} (`{r['model']}`) | {r['dtype']} | {r['gazelle_mean_ms']:.1f} | "
                     f"{r['gazelle_p95_ms']:.1f} | {r['assign_mean_ms']:.2f} | {r['assign_p95_ms']:.2f} | "
                     f"{r['hz_proc']:.1f} | {r['hz_final']:.1f} |")
    lines += ["", f"**Recommendation:** {info['advice']}",
              "Time resolution of DD/TTFD ≈ 1/Hz."]
    if info.get("warning"):
        lines += ["", info["warning"]]
    return "\n".join(lines) + "\n"
