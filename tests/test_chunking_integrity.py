import sys
from pathlib import Path

import numpy as np


def _import_chunking():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    import chunking
    return chunking


def test_calculate_overlap_pixels():
    chunking = _import_chunking()
    assert chunking.calculate_overlap_pixels(3.0, overlap_factor=2.0) == 6
    assert chunking.calculate_overlap_pixels(3.1, overlap_factor=1.5) == 5


def test_plan_vram_chunks_returns_expected_schema():
    chunking = _import_chunking()
    plan = chunking.plan_vram_chunks(
        data_shape=(128, 128, 64),
        max_scale=16.0,
        vram_limit_gb=0.05,
        overlap_factor=2.0,
        dtype=np.float32,
        memory_factor=4.0,
    )
    assert "chunks" in plan
    assert "n_chunk" in plan
    assert "overlap_pixels" in plan
    assert int(plan["n_chunk"]) >= 1
    assert int(plan["overlap_pixels"]) > 0
    assert len(plan["chunks"]) == int(plan["n_chunk"])
