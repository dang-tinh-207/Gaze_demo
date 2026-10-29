import cv2
import numpy as np
import pytest

from gazedemo.app import setup_aoi
from gazedemo.app.setup_aoi import AOIEditor, render_editor, save
from gazedemo.app.text import TextPainter
from gazedemo.config import load_config
from gazedemo.io.logs import load_aoi, load_image, save_image


def drawn_editor():
    ed = AOIEditor(["A", "B", "C"], margin=0.10, min_angle_deg=15)
    ed.add_box(240, 470, 480, 610)
    ed.add_box(520, 530, 760, 670)
    ed.add_box(800, 470, 1040, 610)
    return ed


def test_editor_boxes_head_undo_margin():
    ed = drawn_editor()
    assert ed.next_name is None and not ed.add_box(0, 0, 50, 50)
    assert ed.problems()[0] == []
    assert "head" in ed.problems()[1][0]
    ed.set_head(640, 200)
    assert len(ed.angles()) == 3 and min(a for *_, a in ed.angles()) > 15
    ed.undo()
    assert ed.head is None and len(ed.boxes) == 3
    ed.undo()
    assert ed.next_name == "C" and ed.problems()[0]
    ed.change_margin(+0.02)
    assert ed.margin == pytest.approx(0.12)
    assert not ed.add_box(10, 10, 12, 12)  # too small: treated as a click


def test_overlap_blocks_saving_and_small_angle_warns():
    ed = AOIEditor(["A", "B"], margin=0.0)
    ed.add_box(100, 100, 300, 200)
    ed.add_box(280, 100, 480, 200)
    assert any("overlap" in e for e in ed.problems()[0])
    ed2 = AOIEditor(["A", "B"], margin=0.0)
    ed2.add_box(100, 400, 200, 450)
    ed2.add_box(100, 600, 200, 650)
    ed2.set_head(150, 100)  # both plates on almost the same ray
    assert any("Ray angle" in w for w in ed2.problems()[1])


def test_save_and_reload(tmp_path):
    ed = drawn_editor()
    ed.set_head(640, 200)
    frame = np.full((720, 1280, 3), 60, np.uint8)
    painter = TextPainter()
    assert render_editor(frame, ed, painter).shape == frame.shape
    out = tmp_path / "món" / "aoi.json"  # non-ASCII folder on purpose
    save(out, frame, ed, painter)
    aois, size, data = load_aoi(out)
    assert size == (1280, 720)
    assert [a.name for a in aois] == ["A", "B", "C"]
    assert aois[0].rx == pytest.approx(120 * 1.1)
    assert data["head_xy"] == [640.0, 200.0] and set(data["ray_angles_deg"]) == {"A-B", "A-C", "B-C"}
    assert load_image(out.with_name("aoi_ref.png")).shape == frame.shape


def test_interactive_loop_scripted(tmp_path, monkeypatch):
    """Drive run_setup_aoi with simulated mouse/keys (no real window): drag A, B, C, click the
    head, Enter → aoi.json saved; overlapping boxes are refused first."""
    img = np.full((720, 1280, 3), 80, np.uint8)
    save_image(tmp_path / "frame.png", img)
    L, UP, MOVE, R = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE, cv2.EVENT_RBUTTONDOWN

    def drag(x0, y0, x1, y1):
        return [("mouse", L, x0, y0), ("mouse", MOVE, x1, y1), ("mouse", UP, x1, y1)]

    script = (drag(240, 470, 480, 610) + drag(400, 470, 640, 610)   # B overlaps A
              + [("key", 13), ("key", ord("u"))]                       # Enter refused, undo B
              + drag(520, 530, 760, 670) + drag(800, 470, 1040, 610)
              + [("mouse", R, 640, 200), ("key", ord("-")), ("key", 13)])
    state = {"cb": None, "shown": 0}
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: 1.0)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda win, cb, *a: state.update(cb=cb))
    monkeypatch.setattr(cv2, "imshow", lambda win, im: state.update(shown=state["shown"] + 1))

    def wait_key(ms):
        if not script:
            return ord("q")
        kind, *rest = script.pop(0)
        if kind == "mouse":
            state["cb"](rest[0], rest[1], rest[2], 0, None)
            return -1
        return rest[0]

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    cfg = load_config()
    out = tmp_path / "aoi.json"
    assert setup_aoi.run_setup_aoi(cfg, str(tmp_path / "frame.png"), out) == 0
    aois, size, data = load_aoi(out)
    assert size == (1280, 720) and [a.name for a in aois] == ["A", "B", "C"]
    assert aois[1].cx == pytest.approx(640) and data["margin"] == pytest.approx(cfg.aoi.margin - 0.02)
    assert data["head_xy"] == [640.0, 200.0]
    assert (tmp_path / "aoi_ref.png").exists() and state["shown"] > 5
