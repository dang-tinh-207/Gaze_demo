"""`doctor`: first thing to run on a new machine — libraries, CUDA/GPU, cameras, Gazelle."""
from __future__ import annotations

import importlib
import platform
import sys
import time

from gazedemo.config import project_path


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "?")
    except Exception:
        return None


def fourcc_str(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "?"


def probe_camera(index: int, s) -> dict | None:
    import cv2
    from gazedemo.io.camera import open_capture
    cap = open_capture(index, s.backend, s.width, s.height, s.fps, s.fourcc)
    if cap is None:
        return None
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"index": index, "ok": False}
        n, t0 = 0, time.perf_counter()
        while n < 30 and time.perf_counter() - t0 < 3.0:
            n += bool(cap.read()[0])
        fps = n / (time.perf_counter() - t0)
        return {"index": index, "ok": True, "size": (frame.shape[1], frame.shape[0]), "fps": fps,
                "fps_prop": cap.get(cv2.CAP_PROP_FPS), "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}
    finally:
        cap.release()


def run_doctor(cfg, max_index: int = 4, check_model: bool = True) -> int:
    print("== gazedemo doctor ==")
    print(f"Python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"OS          {platform.platform()}")
    for mod, label in [("numpy", "numpy"), ("cv2", "opencv"), ("torch", "torch"), ("torchvision", "torchvision"),
                       ("timm", "timm"), ("PIL", "pillow"), ("yaml", "pyyaml"),
                       ("scipy", "scipy (optional)"), ("matplotlib", "matplotlib (optional)")]:
        v = _version(mod)
        print(f"  {label:<22} {v if v else 'NOT INSTALLED'}")

    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA        {'YES' if cuda_ok else 'NO'} (torch CUDA {torch.version.cuda})")
        if cuda_ok:
            p = torch.cuda.get_device_properties(0)
            print(f"GPU         {p.name}, {p.total_memory / 2**30:.1f} GB")
    except ImportError:
        print("CUDA        torch not installed")

    from gazedemo.app.text import find_font
    font = find_font(cfg.ui.font)
    print(f"Font UI     {font or 'NOT FOUND → on-screen text will lose diacritics (food names)'}")

    from gazedemo.io.camera import backend_id, quiet_opencv
    s = cfg.source
    print(f"Camera      backend {s.backend} (id {backend_id(s.backend)}), xin {s.width}×{s.height} @ {s.fps} fps, fourcc {s.fourcc}")
    cams = []
    for idx in range(max_index + 1):
        with quiet_opencv():
            info = probe_camera(idx, s)
        if info is None:
            continue
        cams.append(info)
        if info["ok"]:
            w, h = info["size"]
            print(f"  index {idx}: {w}×{h}, measured {info['fps']:.1f} fps (reported {info['fps_prop']:.0f}, {info['fourcc']})")
        else:
            print(f"  index {idx}: opens but returns no frames")
    if not cams:
        print("  no camera could be opened (camera plugged in? try another source.backend?)")
    cam_ok = any(c["ok"] and c["index"] == s.device for c in cams)
    if cams and not cam_ok:
        good = [c["index"] for c in cams if c["ok"]]
        print(f"  → source.device = {s.device} is not usable; try --set source.device={good[0] if good else 0}")

    g = cfg.gazelle
    repo = project_path(g.repo) if g.repo else None
    print(f"Gazelle     {g.name}; source: {repo if repo and repo.exists() else 'torch.hub fkryan/gazelle'}")
    gazelle_ok = False
    if check_model:
        try:
            import numpy as np
            from gazedemo.perception.gazelle_model import GazelleModel
            model = GazelleModel.from_config(g, warmup=1)
            t = time.perf_counter()
            hm, inout = model.infer(np.zeros((s.height, s.width, 3), np.uint8))
            ms = (time.perf_counter() - t) * 1000
            print(f"  load OK ({model.load_s:.1f} s): {model.describe()}")
            print(f"  weights: {model.source}")
            print(f"  1 frame: heatmap {hm.shape}, inout {inout if inout is None else round(inout, 3)}, {ms:.0f} ms")
            gazelle_ok = hm.shape == (64, 64)
        except Exception as e:
            print(f"  ERROR loading Gazelle: {type(e).__name__}: {e}")
    else:
        print("  (model load skipped: --no-model)")

    aoi = project_path(cfg.aoi.file)
    print(f"AOI         {aoi if aoi.exists() else 'no aoi.json yet → run setup-aoi'}")
    print("== Summary ==")
    print(f"  CUDA {'OK' if cuda_ok else 'NO'} · camera {s.device} {'OK' if cam_ok else 'NO'} · "
          f"Gazelle {'OK' if gazelle_ok else ('skipped' if not check_model else 'ERROR')}")
    return 0 if cuda_ok and cam_ok and (gazelle_ok or not check_model) else 1
