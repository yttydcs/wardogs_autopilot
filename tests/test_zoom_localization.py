"""Zoomed minimaps must retain scale checks and pass local verification."""

from types import SimpleNamespace

import cv2
import numpy as np

from autopilot.common.config import LocatorConfig
from autopilot.vision.featureindex import FeatureIndex
from autopilot.vision.locator import MapLocator


def example(tmp_path, scale=0.386, matches=30):
    rng = np.random.default_rng(7)
    src = rng.uniform([15, 15], [320, 285], (matches, 2)).astype(np.float32)
    dst = src * scale + [1000, 2000]
    desc = rng.uniform(0, 255, (matches, 128)).astype(np.float32)
    path = tmp_path / "index.npz"
    np.savez(
        path,
        ms=2.6544,
        mu_h=5000,
        mu_w=5000,
        tile=512,
        levels=[1.7],
        pts_lv0=dst.astype(np.float32),
        desc_lv0=desc,
        tile_lv0=(np.floor(dst[:, 1] * 1.7 / 512) * 17 + np.floor(dst[:, 0] * 1.7 / 512)).astype(
            np.int32
        ),
    )
    with np.load(path) as data:
        index = FeatureIndex("test", data)
    index.prepare_global_matcher()
    engine = MapLocator(SimpleNamespace(loc_cfg=lambda: LocatorConfig().model_dump()))
    feats = ([cv2.KeyPoint(float(x), float(y), 3) for x, y in src], desc)
    return engine, index, feats


def test_zoomed_global_match_is_locally_verified(tmp_path):
    engine, index, feats = example(tmp_path)
    result = engine._index_find(
        np.zeros((300, 337), np.uint8), None, index, None, None, feats=feats
    )
    pose, diag = result[:2]
    assert pose is not None
    assert abs(pose["s"] - 0.386) < 1e-4
    assert pose["inl"] == 30
    assert diag["search_global"] is True
    np.testing.assert_allclose(
        engine._mm_center_to_map(pose, np.zeros((300, 337))), [1065.041, 2057.9], atol=0.01
    )


def test_weak_global_candidate_cannot_skip_verification(tmp_path):
    engine, index, feats = example(tmp_path, matches=6)
    result = engine._index_find(
        np.zeros((300, 337), np.uint8), None, index, None, None, feats=feats
    )
    assert result[0] is None
    assert "local verification" in result[1]["detail"]


def test_degenerate_scale_still_rejected_and_counts_reported(tmp_path):
    engine, index, feats = example(tmp_path, scale=0.01)
    pose, diag = engine._pose_via_index(
        np.zeros((300, 337), np.uint8), None, index, 0, 0, None, feats=feats
    )
    assert pose is None
    assert "scale" in diag["detail"]
    assert diag["good1"] == 30
    assert diag["inl1"] == 30
