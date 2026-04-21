import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _import_main():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from morph_informed_decomposition import main
    return main


def test_morph_informed_cli_writes_lxy_cube(tmp_path, monkeypatch):
    main = _import_main()

    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output_scales.fits"
    data = np.random.RandomState(0).randn(32, 40).astype(np.float32)
    fits.writeto(input_path, data, overwrite=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morph_informed_decomposition.py",
            str(input_path),
            str(output_path),
            "--cpu",
            "--max-level",
            "4",
            "--max-scale",
            "16",
        ],
    )
    main()

    with fits.open(output_path) as hdul:
        cube = hdul[0].data
        assert cube.ndim == 3
        assert cube.shape[1:] == data.shape
        assert hdul[0].header["GPU"] is False


def test_morphological_alias_writes_labels_extension(tmp_path, monkeypatch):
    main = _import_main()

    input_path = tmp_path / "input2.fits"
    output_path = tmp_path / "output2_scales.fits"
    data = np.random.RandomState(1).randn(24, 24).astype(np.float32)
    fits.writeto(input_path, data, overwrite=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "morph_informed_decomposition.py",
            str(input_path),
            str(output_path),
            "--cpu",
            "--morphology-aware",
            "--max-level",
            "3",
            "--max-scale",
            "10",
        ],
    )
    main()

    with fits.open(output_path) as hdul:
        assert len(hdul) >= 2
        assert hdul[1].name == "MORPH_LABELS"
        assert hdul[1].data.shape == hdul[0].data.shape
        assert hdul[0].header["MORPH"] is True

    clump_path = output_path.with_name(output_path.stem + "_clump.fits")
    filament_path = output_path.with_name(output_path.stem + "_filament.fits")
    diffuse_path = output_path.with_name(output_path.stem + "_diffuse.fits")
    assert clump_path.exists()
    assert filament_path.exists()
    assert diffuse_path.exists()

    with fits.open(output_path) as base_hdul:
        expected_shape = base_hdul[0].data.shape

    with fits.open(clump_path) as hdul:
        assert hdul[0].data.shape == expected_shape

    with fits.open(filament_path) as hdul:
        assert hdul[0].data.shape == expected_shape

    with fits.open(diffuse_path) as hdul:
        assert hdul[0].data.shape == expected_shape
