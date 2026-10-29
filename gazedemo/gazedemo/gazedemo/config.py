"""Config: configs/default.yaml, optionally merged with a user file and `--set a.b=value`."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Section(dict):
    """dict with attribute access (cfg.dwell.min_dwell_ms)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"config has no key '{name}'") from None

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Section) else v for k, v in self.items()}


def _wrap(obj):
    if isinstance(obj, dict):
        return Section({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None, overrides=()) -> Section:
    data = read_yaml(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        data = _merge(data, read_yaml(path))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set needs the form section.key=value, got '{item}'")
        *parents, leaf = key.strip().split(".")
        node = data
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise ValueError(f"--set: no section '{key}' in the config")
            node = node[p]
        if leaf not in node:
            raise ValueError(f"--set: no key '{key}' in the config")
        node[leaf] = yaml.safe_load(raw)
    return _wrap(data)


def save_config(cfg: Section, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)


def project_path(p) -> Path:
    """Paths inside the config are relative to the project folder (where configs/ lives)."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def check_max_interval(dwell_cfg, sample_interval_s: float) -> str | None:
    """Warn when max_interval_s would cut normal sample gaps into 'invalid'."""
    if sample_interval_s > 0 and dwell_cfg.max_interval_s < 2 * sample_interval_s:
        return (f"WARNING: dwell.max_interval_s = {dwell_cfg.max_interval_s} < 2 × the actual sample "
                f"interval ({sample_interval_s * 1000:.0f} ms) → many stretches will count as invalid; "
                f"set it to ≥ {2 * sample_interval_s:.2f}")
    return None
