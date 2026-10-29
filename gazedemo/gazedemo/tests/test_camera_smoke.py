"""The configured camera delivers frames with increasing capture times (skipped by -m "not camera")."""
import pytest

from gazedemo.config import load_config
from gazedemo.io.camera import open_source

pytestmark = pytest.mark.camera


def test_camera_delivers_frames():
    cfg = load_config()
    src = open_source(None, cfg)
    try:
        src.start()
    except RuntimeError as e:
        pytest.skip(str(e))
    try:
        items = [src.read(timeout=2.0) for _ in range(10)]
    finally:
        src.stop()
    assert all(it is not None for it in items)
    ts = [t for _, t, _ in items]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    w, h = src.size
    assert items[-1][2].shape == (h, w, 3)
