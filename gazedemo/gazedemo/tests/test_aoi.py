import numpy as np
import pytest

from conftest import AOIS_ABC, gaussian_heatmap
from gazedemo.core.aoi import (assign_label, build_masks, ellipse_from_box, find_overlaps,
                               heatmap_argmax_xy, ray_angles)
from gazedemo.types import EllipseAOI

W, H = 1280, 720


def test_masks_in_unit_range_and_sum_le_1():
    masks = build_masks(AOIS_ABC, W, H)
    total = sum(masks.values())
    for m in masks.values():
        assert m.shape == (64, 64)
        assert m.min() >= 0 and m.max() <= 1
    assert total.max() <= 1.0 + 1e-6


def test_mask_area_matches_ellipse():
    a = AOIS_ABC[1]
    cell_area = (W / 64) * (H / 64)
    area = build_masks([a], W, H)["B"].sum() * cell_area
    assert area == pytest.approx(np.pi * a.rx * a.ry, rel=0.03)


def test_gaussian_at_center_of_b(assign_cfg):
    masks = build_masks(AOIS_ABC, W, H)
    label, mass = assign_label(gaussian_heatmap(640, 560), 0.9, masks, assign_cfg)
    assert label == "B"
    assert mass["B"] > 0.8
    assert sum(mass.values()) == pytest.approx(1.0)


def test_between_a_and_b_is_none(assign_cfg):
    masks = build_masks(AOIS_ABC, W, H)
    hm = gaussian_heatmap(320, 520) + gaussian_heatmap(640, 560)
    label, mass = assign_label(hm, 0.9, masks, assign_cfg)
    assert label == "none"
    assert abs(mass["A"] - mass["B"]) < assign_cfg.delta


def test_low_inout_is_out(assign_cfg):
    masks = build_masks(AOIS_ABC, W, H)
    label, mass = assign_label(gaussian_heatmap(640, 560), 0.2, masks, assign_cfg)
    assert label == "out"
    assert mass["B"] > 0.8  # mass is still logged


def test_no_inout_head_means_in_frame(assign_cfg):
    masks = build_masks(AOIS_ABC, W, H)
    assert assign_label(gaussian_heatmap(960, 520), None, masks, assign_cfg)[0] == "C"


def test_heatmap_to_pixels():
    hm = np.zeros((64, 64), np.float32)
    hm[10, 20] = 1.0
    x, y = heatmap_argmax_xy(hm, W, H)
    assert x == pytest.approx((20 + 0.5) / 64 * W)
    assert y == pytest.approx((10 + 0.5) / 64 * H)


def test_ellipse_from_box_and_margin():
    e = ellipse_from_box("A", 100, 200, 300, 300, margin=0.1)
    assert (e.cx, e.cy) == (200, 250)
    assert e.rx == pytest.approx(110) and e.ry == pytest.approx(55)


def test_overlap_detection():
    assert find_overlaps(AOIS_ABC) == []
    touching = [EllipseAOI("A", 100, 100, 60, 40), EllipseAOI("B", 200, 100, 60, 40)]
    assert find_overlaps(touching) == [("A", "B")]


def test_ray_angles():
    head = (640, 0)
    aois = [EllipseAOI("A", 340, 300, 50, 50), EllipseAOI("B", 640, 300, 50, 50),
            EllipseAOI("C", 940, 300, 50, 50)]
    angles = {(a, b): ang for a, b, ang in ray_angles(head, aois)}
    assert angles[("A", "B")] == pytest.approx(45.0)
    assert angles[("A", "C")] == pytest.approx(90.0)
