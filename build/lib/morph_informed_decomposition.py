#!/usr/bin/env python3
"""
CDD 2D-to-3D FITS wrapper with optional morphology-aware filtering.

Input:
    2D FITS (X, Y)
Output:
    3D FITS (L, X, Y), where L is scale/channel index
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

import constrained_diffusion as cdd


def _hessian_eigenvalues_2d(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D Hessian eigenvalues after Gaussian smoothing."""
    smoothed = ndimage.gaussian_filter(image, sigma=max(float(sigma), 1e-3), mode="reflect")
    dxx = ndimage.gaussian_filter(smoothed, sigma=0.0, order=(2, 0), mode="reflect")
    dxy = ndimage.gaussian_filter(smoothed, sigma=0.0, order=(1, 1), mode="reflect")
    dyy = ndimage.gaussian_filter(smoothed, sigma=0.0, order=(0, 2), mode="reflect")

    trace = dxx + dyy
    det_term = (dxx - dyy) ** 2 + 4.0 * (dxy ** 2)
    root = np.sqrt(np.maximum(det_term, 0.0))
    l1 = 0.5 * (trace + root)
    l2 = 0.5 * (trace - root)
    return l1, l2


def _morphology_keep_mask(channel: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep clumps/filaments/sheets and reject concave regions.
    Label codes:
      0 background or ambiguous
      1 clump
      2 filament
      3 sheet
      4 concave
    """
    labels = np.zeros_like(channel, dtype=np.uint8)

    if channel.ndim != 2:
        # ND fallback: use Laplacian sign to reject concave regions.
        lap = ndimage.laplace(ndimage.gaussian_filter(channel, sigma=max(float(sigma), 1e-3), mode="reflect"))
        strength = np.abs(lap)
        thr = 0.25 * float(np.std(strength))
        strong = strength > thr
        concave = (lap > 0) & strong
        keep = (lap <= 0) & strong
        labels[concave] = 4
        labels[keep] = 2
        return keep, labels

    l1, l2 = _hessian_eigenvalues_2d(channel, sigma=sigma)
    strength = np.maximum(np.abs(l1), np.abs(l2))
    thr = 0.25 * float(np.std(strength))
    strong = strength > thr
    neg1 = l1 < 0
    neg2 = l2 < 0

    clump = (neg1 & neg2) & strong
    ridge_like = (neg1 ^ neg2) & strong
    filament = ridge_like
    # In 2D there is no volumetric "sheet" class; keep key for API compatibility.
    sheet = np.zeros_like(ridge_like, dtype=bool)
    concave = ((~neg1) & (~neg2)) & strong

    labels[clump] = 1
    labels[filament] = 2
    labels[sheet] = 3
    labels[concave] = 4

    keep = clump | filament | sheet
    return keep, labels


def _suffix_path(path: Path, suffix: str) -> Path:
    """Return sibling FITS path with suffix appended before extension."""
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def morph_informed_diffusion_decomposition(
    data,
    morphology_aware=True,
    allow_diffuse=True,
    return_details=False,
    return_scales=False,
    **kwargs,
):
    """
    Programmatic API for morphology-informed decomposition.

    Backward-compatible behavior:
      - default return: (results, residual) or (results, residual, scales)
      - with return_details=True: return a dict with cube outputs

    Details dictionary keys:
      - result_cube: filtered decomposition (or raw decomposition if morphology_aware=False)
      - labels_cube: morphology labels, or None
      - clump_cube: clump-only cube, or None
      - filament_cube: filament-only cube, or None
      - diffuse_cube: diffuse-only cube, or None
      - residual: CDD residual
      - scales: channel scales
    """
    results, residual, scales = cdd.constrained_diffusion_decomposition(
        data,
        **kwargs,
        return_scales=True,
    )

    labels_cube = None
    clump_cube = None
    filament_cube = None
    sheet_cube = None
    diffuse_cube = None
    channel_results = None

    if morphology_aware:
        filtered = []
        labels = []
        clump_list = []
        filament_list = []
        sheet_list = []
        diffuse_list = []
        channel_results = []
        for idx, channel in enumerate(results):
            sigma = float(scales[idx]) if idx < len(scales) else 1.0
            keep_mask, label_map = _morphology_keep_mask(np.asarray(channel), sigma=max(sigma / 2.0, 1.0))
            clump_mask = label_map == 1
            filament_mask = label_map == 2
            sheet_mask = label_map == 3
            concave_mask = label_map == 4
            diffuse_mask = ~(clump_mask | filament_mask | sheet_mask | concave_mask)
            filtered_mask = (clump_mask | filament_mask | sheet_mask | diffuse_mask) if allow_diffuse else keep_mask

            filtered.append(np.where(filtered_mask, channel, 0.0))
            labels.append(label_map)
            clump_list.append(np.where(clump_mask, channel, 0.0))
            filament_list.append(np.where(filament_mask, channel, 0.0))
            sheet_list.append(np.where(sheet_mask, channel, 0.0))
            diffuse_list.append(np.where(diffuse_mask, channel, 0.0))
            channel_results.append(
                {
                    "clump": clump_list[-1],
                    "filament": filament_list[-1],
                    "sheet": sheet_list[-1],
                    "diffuse": diffuse_list[-1],
                    "filtered": filtered[-1],
                    "labels": label_map,
                }
            )
        results = filtered
        labels_cube = np.stack(labels, axis=0).astype(np.uint8)
        clump_cube = np.stack(clump_list, axis=0).astype(np.float32)
        filament_cube = np.stack(filament_list, axis=0).astype(np.float32)
        sheet_cube = np.stack(sheet_list, axis=0).astype(np.float32)
        diffuse_cube = np.stack(diffuse_list, axis=0).astype(np.float32)

    result_cube = np.stack(results, axis=0).astype(np.float32)

    out = {
        "results": results,
        "result_cube": result_cube,
        "labels_cube": labels_cube,
        "clump_cube": clump_cube,
        "filament_cube": filament_cube,
        "sheet_cube": sheet_cube,
        "diffuse_cube": diffuse_cube,
        "residual": residual,
        "scales": scales,
    }
    if return_details:
        if return_scales:
            return out, scales
        return out
    if morphology_aware:
        if return_scales:
            return channel_results, residual, scales
        return channel_results, residual
    if return_scales:
        return results, residual, scales
    return results, residual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose 2D FITS into 3D (L, X, Y) scales with optional morphology-aware filtering."
    )
    parser.add_argument("input", type=Path, help="Input 2D FITS file")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output 3D FITS path")

    parser.add_argument("--max-level", type=int, default=None, help="Max number of channels (default: auto)")
    parser.add_argument("--max-scale", type=float, default=None, help="Largest scale in pixels (default: auto)")
    parser.add_argument("--unconstrained", action="store_true", help="Disable sign-constrained diffusion")

    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode (default behavior)")

    parser.add_argument("--upsample", dest="upsample", action="store_true", help="Enable high-res upsampling (default)")
    parser.add_argument("--no-upsample", dest="upsample", action="store_false", help="Disable high-res upsampling")
    parser.set_defaults(upsample=True)

    parser.add_argument(
        "--morphological",
        "--morphology-aware",
        dest="morphology_aware",
        action="store_true",
        help="Enable morphology-aware post-filtering and save clump/filament/diffuse cubes",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_scales.fits")

    try:
        with fits.open(args.input) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            header = hdul[0].header.copy()
    except Exception as exc:
        print(f"Read error: {exc}", file=sys.stderr)
        sys.exit(1)

    if data.ndim != 2:
        print(f"Warning: expected 2D image, got {data.ndim}D. Proceeding with generic ND decomposition.", file=sys.stderr)

    constrained = not args.unconstrained
    use_gpu = True
    if args.cpu:
        use_gpu = False
    if args.gpu:
        use_gpu = True

    print(
        f"Decomposing {data.shape} | GPU: {use_gpu} | Up-sample: {args.upsample} | "
        f"Max-scale: {args.max_scale if args.max_scale is not None else 'auto'} | "
        f"Morphology-aware: {args.morphology_aware}"
    )

    try:
        out = morph_informed_diffusion_decomposition(
            data,
            morphology_aware=args.morphology_aware,
            return_details=True,
            num_channels=args.max_level,
            max_scale=args.max_scale,
            constrained=constrained,
            use_gpu=use_gpu,
            up_sample=args.upsample,
        )
    except Exception as exc:
        print(f"Decomposition failed: {exc}", file=sys.stderr)
        sys.exit(1)

    result_cube = out["result_cube"]
    labels_cube = out["labels_cube"]
    clump_cube = out["clump_cube"]
    filament_cube = out["filament_cube"]
    diffuse_cube = out["diffuse_cube"]
    header["NCHAN"] = int(result_cube.shape[0])
    header["CONSTR"] = (constrained, "Constrained mode enabled")
    header["GPU"] = (use_gpu, "GPU acceleration requested")
    header["UPSAMP"] = (bool(args.upsample), "Hybrid upsampling")
    header["MAXSCL"] = (float(args.max_scale) if args.max_scale is not None else -1.0, "Max scale; -1=auto")
    header["MORPH"] = (bool(args.morphology_aware), "Morphology-aware filtering")

    try:
        if labels_cube is None:
            fits.writeto(args.output, result_cube, header, overwrite=True)
        else:
            hdu0 = fits.PrimaryHDU(data=result_cube, header=header)
            hdu1 = fits.ImageHDU(data=labels_cube, name="MORPH_LABELS")
            fits.HDUList([hdu0, hdu1]).writeto(args.output, overwrite=True)
            fits.writeto(_suffix_path(args.output, "clump"), clump_cube, header, overwrite=True)
            fits.writeto(_suffix_path(args.output, "filament"), filament_cube, header, overwrite=True)
            fits.writeto(_suffix_path(args.output, "diffuse"), diffuse_cube, header, overwrite=True)
    except Exception as exc:
        print(f"Save error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Success! Saved 3D array {result_cube.shape} (L, X, Y) to {args.output}")
    if labels_cube is not None:
        print("Added extension MORPH_LABELS with code map: 1=clump, 2=filament, 3=sheet, 4=concave")
        print(f"Saved clump cube: {_suffix_path(args.output, 'clump')}")
        print(f"Saved filament cube: {_suffix_path(args.output, 'filament')}")
        print(f"Saved diffuse cube: {_suffix_path(args.output, 'diffuse')}")


if __name__ == "__main__":
    main()
