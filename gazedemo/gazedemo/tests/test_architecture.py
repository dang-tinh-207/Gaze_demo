"""core/ only imports numpy and gazedemo.types (plus other core modules); one compute()."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "gazedemo"
FORBIDDEN = {"cv2", "torch", "time", "os", "pathlib", "json", "csv"}


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: use absolute imports in core/"
            yield node.module


def test_core_imports_only_numpy_and_types():
    files = sorted((PKG / "core").glob("*.py"))
    assert len(files) >= 4
    for f in files:
        for mod in imported_modules(f):
            top = mod.split(".")[0]
            assert top not in FORBIDDEN, f"{f.name} import {mod}"
            ok = (mod == "numpy" or mod == "__future__" or mod == "gazedemo.types"
                  or mod.startswith("gazedemo.core."))
            assert ok, f"{f.name} imports {mod} (core/ may only import numpy and gazedemo.types)"


def test_single_compute_function():
    defs = [f for f in PKG.rglob("*.py") if "def compute(" in f.read_text(encoding="utf-8")]
    assert [f.relative_to(PKG).as_posix() for f in defs] == ["core/metrics.py"]
