#!/usr/bin/env python3
"""Utility helpers for channel-scale estimation."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_laplace

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None


def load_data(file_path):
    """Load .npy or FITS data as float ndarray, squeezing singleton dimensions."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".npy":
        data = np.load(path)
    elif ext in [".fits", ".fit", ".fts"]:
        if fits is None:
            raise ImportError("astropy is required to read FITS files")
        with fits.open(path) as hdul:
            data = hdul[0].data if hdul[0].data is not None else hdul[1].data
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return np.squeeze(np.asarray(data, dtype=float))


def get_weighted_characteristic_scale(channel_data):
    """
    Estimate a characteristic scale from integrated gradient/laplacian ratio.

    Works for N-D channel arrays and is robust to NaN/Inf and zero-point shifts.
    """
    img = np.nan_to_num(np.asarray(channel_data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    laplacian = np.abs(gaussian_laplace(img, sigma=1.0))
    gradient = gaussian_gradient_magnitude(img, sigma=1.0)

    sum_laplacian = float(np.sum(laplacian))
    sum_gradient = float(np.sum(gradient))

    if sum_laplacian == 0.0:
        return 0.0

    raw_scale = sum_gradient / sum_laplacian
    return raw_scale * 1.55


def estimate_channel_scales(data, channel_axis=0):
    """
    Estimate one characteristic scale per channel.

    For 2D input, treats data as a single channel.
    For N-D input (N >= 3), treats `channel_axis` as the channel dimension.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2:
        raise ValueError("Input data must be at least 2D")

    if arr.ndim == 2:
        channels = arr[np.newaxis, ...]
    else:
        channels = np.moveaxis(arr, channel_axis, 0)

    return [get_weighted_characteristic_scale(channels[i, ...]) for i in range(channels.shape[0])]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate weighted characteristic scale for each channel from .npy or FITS input."
    )
    parser.add_argument("input", help="Input file path (.npy/.fits/.fit/.fts)")
    parser.add_argument(
        "--channel-axis",
        type=int,
        default=0,
        help="Channel axis for N-D arrays (ignored for 2D inputs). Default: 0",
    )
    args = parser.parse_args(argv)

    try:
        data = load_data(args.input)
        scales = estimate_channel_scales(data, channel_axis=args.channel_axis)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(scales)} channels...")
    print("Method: Laplacian/Gradient Weighted Ratio")
    print("\n" + "=" * 45)
    print(f"{'Channel':<10} | {'Weighted Scale (σ)':<25}")
    print("-" * 45)
    for idx, scale in enumerate(scales):
        print(f"Ch {idx:<7} | {scale:<25.6f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
