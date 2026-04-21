import sys
from pathlib import Path

import numpy as np


def _import_utils():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    import utils
    return utils


def _gaussian2d(shape, sigma):
    y, x = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def _gaussian3d(shape, sigma):
    z, y, x = np.indices(shape)
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def test_get_weighted_characteristic_scale_constant_returns_zero():
    utils = _import_utils()
    const = np.ones((16, 16), dtype=float)
    assert utils.get_weighted_characteristic_scale(const) == 0.0


def test_estimate_channel_scales_for_2d_and_nd():
    utils = _import_utils()

    g_small = _gaussian2d((64, 64), sigma=2.0)
    g_large = _gaussian2d((64, 64), sigma=6.0)

    scales_2d = utils.estimate_channel_scales(g_small)
    assert len(scales_2d) == 1
    assert scales_2d[0] > 0

    cube2d = np.stack([g_small, g_large], axis=0)
    scales_c = utils.estimate_channel_scales(cube2d, channel_axis=0)
    assert len(scales_c) == 2
    assert scales_c[1] > scales_c[0]

    a = _gaussian3d((20, 20, 20), sigma=2.0)
    b = _gaussian3d((20, 20, 20), sigma=4.0)
    nd = np.stack([a, b], axis=0)
    scales_nd = utils.estimate_channel_scales(nd, channel_axis=0)
    assert len(scales_nd) == 2
    assert scales_nd[1] > scales_nd[0]


def test_load_data_npy(tmp_path):
    utils = _import_utils()
    path = tmp_path / "arr.npy"
    arr = np.random.RandomState(42).randn(4, 5, 6).astype(np.float32)
    np.save(path, arr)
    loaded = utils.load_data(path)
    assert loaded.shape == arr.shape
    assert loaded.dtype == float
