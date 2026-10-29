"""Gazelle (Gaze-LLE), run locally: infer(frame_rgb) -> (heatmap 64x64 float32 in [0, 1], inout | None).

Loading follows the user's old FLAIR bridge (scripts/gazelle_bridge/server.py): local source
checkout + get_gazelle_model(name) + load_gazelle_state_dict(torch.load(ckpt)). Without a local
checkout it falls back to torch.hub "fkryan/gazelle" (needs internet the first time).
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
import time
import types as pytypes
from pathlib import Path

import numpy as np
import torch

HUB_REPO = "fkryan/gazelle"
CKPT_URL = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/{name}.pt"
# gazelle/utils.py imports these at module level although inference never calls them.
_EVAL_ONLY = {"matplotlib": ("matplotlib", "matplotlib.pyplot"), "sklearn": ("sklearn", "sklearn.metrics")}


def pick_device(name: str = "auto") -> str:
    if name in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@contextlib.contextmanager
def _stub_eval_only_deps():
    """Stand in for matplotlib/sklearn while gazelle is imported, only if they are not installed,
    so neither becomes a hard dependency (scikit-learn is not in the allowed library list)."""
    added = []
    for top, names in _EVAL_ONLY.items():
        if top in sys.modules or importlib.util.find_spec(top) is not None:
            continue
        parent = None
        for name in names:
            mod = pytypes.ModuleType(name)

            def _missing(attr, _name=name):
                def unavailable(*args, **kwargs):
                    raise ImportError(f"{_name}.{attr} is not installed (only needed to evaluate Gazelle)")
                return unavailable

            mod.__getattr__ = _missing
            sys.modules[name] = mod
            added.append(name)
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[1], mod)
            parent = mod
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


def _import_gazelle_model(repo: Path):
    """Import gazelle.model from this checkout even if another gazelle is pip-installed
    (sys.path order alone would pick the installed regular package)."""
    for name in [n for n in sys.modules if n == "gazelle" or n.startswith("gazelle.")]:
        del sys.modules[name]
    spec = importlib.machinery.ModuleSpec("gazelle", None, is_package=True)
    spec.submodule_search_locations = [str(repo / "gazelle")]
    sys.modules["gazelle"] = importlib.util.module_from_spec(spec)
    with _stub_eval_only_deps():
        return importlib.import_module("gazelle.model")


def load_gazelle(name: str, repo=None, ckpt=None):
    """Returns (model, transform, source description)."""
    repo = Path(repo) if repo else None
    if repo is not None and (repo / "gazelle" / "model.py").is_file():
        gazelle_model = _import_gazelle_model(repo)
        model, transform = gazelle_model.get_gazelle_model(name)
        ckpt_path = Path(ckpt) if ckpt else repo / "checkpoints" / f"{name}.pt"
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            source = f"{repo} + {ckpt_path}"
        else:
            url = CKPT_URL.format(name=name)
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            source = f"{repo} + {url}"
        expected = {k for k in model.state_dict() if not k.startswith("backbone")}
        missing = expected - set(state)
        if missing:
            raise RuntimeError(f"checkpoint does not match {name}: missing {sorted(missing)[:5]}")
        model.load_gazelle_state_dict(state)
        return model, transform, source
    if repo is not None:
        print(f"[gazelle] no source found at {repo} → using torch.hub {HUB_REPO}")
    with _stub_eval_only_deps():
        model, transform = torch.hub.load(HUB_REPO, name, trust_repo=True)
    return model, transform, f"torch.hub {HUB_REPO}"


def _tensor_steps(transform):
    """(Normalize, Resize) if transform is Gazelle's Compose([ToTensor, Normalize, Resize])."""
    import torchvision.transforms as T
    steps = getattr(transform, "transforms", None)
    if (steps and len(steps) == 3 and isinstance(steps[0], T.ToTensor)
            and isinstance(steps[1], T.Normalize) and isinstance(steps[2], T.Resize)):
        return steps[1], steps[2]
    return None


class GazelleModel:
    def __init__(self, name: str, device: str = "auto", fp16: bool = True, warmup: int = 5,
                 repo=None, ckpt=None, preprocess: str = "gpu"):
        self.name = name
        self.device = torch.device(pick_device(device))
        t = time.perf_counter()
        self.model, self.transform, self.source = load_gazelle(name, repo, ckpt)
        self.model.eval().to(self.device)
        self.fp16 = bool(fp16)
        self.has_inout = bool(getattr(self.model, "inout", False))
        steps = _tensor_steps(self.transform) if preprocess == "gpu" else None
        self._steps = steps
        self.preprocess = "gpu" if steps else "pil"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.load_s = time.perf_counter() - t
        self.warmup(warmup)

    @classmethod
    def from_config(cls, gcfg, root=None, **overrides):
        from gazedemo.config import project_path
        kw = dict(name=gcfg.name, device=gcfg.device, fp16=gcfg.fp16, warmup=gcfg.warmup,
                  repo=project_path(gcfg.repo) if gcfg.repo else None,
                  ckpt=project_path(gcfg.ckpt) if gcfg.ckpt else None,
                  preprocess=gcfg.preprocess)
        kw.update(overrides)
        return cls(**kw)

    @property
    def use_fp16(self) -> bool:
        return self.fp16 and self.device.type == "cuda"

    def to_input(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """[1, 3, 448, 448]. 'gpu' runs Gazelle's own Normalize + Resize on the device after
        ToTensor (/255, HWC→CHW); 'pil' is the reference path BGR→RGB→PIL→transform."""
        if self._steps is not None:
            norm, resize = self._steps
            x = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(self.device, non_blocking=True)
            x = x.permute(2, 0, 1).float().div_(255.0)
            return resize(norm(x)).unsqueeze(0)
        from PIL import Image
        return self.transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)

    def infer(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, float | None]:
        autocast = (torch.autocast("cuda", dtype=torch.float16) if self.use_fp16
                    else contextlib.nullcontext())
        with torch.inference_mode():
            x = self.to_input(frame_rgb)
            with autocast:
                out = self.model({"images": x, "bboxes": [[None]]})
            heatmap = out["heatmap"][0][0].float().cpu().numpy()
            inout = float(out["inout"][0][0]) if out.get("inout") is not None else None
        return heatmap.astype(np.float32, copy=False), inout

    def warmup(self, n: int, size=(720, 1280)) -> None:
        dummy = np.zeros((*size, 3), np.uint8)
        for _ in range(max(0, int(n))):
            self.infer(dummy)

    def describe(self) -> str:
        dtype = "fp16" if self.use_fp16 else "fp32"
        return f"{self.name} ({self.device.type}, {dtype}, preprocess {self.preprocess})"
