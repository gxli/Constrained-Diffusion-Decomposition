#!/usr/bin/env python3
"""
3D-surface tiling performance benchmark on a 1000x1000 Gaussian-mixture field.

Runs two modes:
1) GPU baseline without forced tiling (n_chunk=1)
2) GPU with enforced tiling (for example n_chunk=16)

Both runs save a verification plot to tests/outputs/.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = CURRENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

from constrained_diffusion import constrained_diffusion_decomposition, TORCH_AVAILABLE  # noqa: E402


def _has_accelerator():
    if not TORCH_AVAILABLE:
        return False
    import torch  # noqa: WPS433

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return bool(torch.cuda.is_available() or has_mps)


def _resolve_device(user_device=None):
    if not TORCH_AVAILABLE:
        return None
    import torch  # noqa: WPS433

    if user_device:
        return str(user_device)
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def _create_2d_gaussian(shape, center, sigma, amplitude=1.0):
    y, x = np.indices(shape)
    y0, x0 = center
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def create_mixed_scale_field(shape=(1000, 1000)):
    """Build a 1000x1000 field with Gaussian structures at multiple scales."""
    image = np.zeros(shape, dtype=np.float32)
    entries = [
        ((150, 180), 8, 1.0),
        ((300, 700), 18, 0.9),
        ((750, 250), 40, 1.2),
        ((620, 620), 75, -0.7),
        ((500, 500), 150, 0.8),
        ((850, 820), 220, -0.5),
    ]
    for center, sigma, amp in entries:
        image += _create_2d_gaussian(shape, center=center, sigma=sigma, amplitude=amp)
    return image


def _reconstruct(results, residual):
    stacked = np.stack([np.asarray(ch, dtype=np.float32) for ch in results], axis=0)
    return np.sum(stacked, axis=0) + np.asarray(residual, dtype=np.float32)


def _save_verification_plot(original, recon, case_name, elapsed_s, out_path):
    import matplotlib.pyplot as plt  # noqa: WPS433

    err = np.abs(original - recon)
    rmse = float(np.sqrt(np.mean((original - recon) ** 2)))
    vmax = float(np.max(np.abs(original)))

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"{case_name} | elapsed={elapsed_s:.2f}s | RMSE={rmse:.3e}",
        fontsize=12,
        weight="bold",
    )

    stride = 5
    y = np.arange(0, original.shape[0], stride)
    x = np.arange(0, original.shape[1], stride)
    xx, yy = np.meshgrid(x, y)
    zz_orig = original[::stride, ::stride]
    zz_recon = recon[::stride, ::stride]

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(xx, yy, zz_orig, cmap="viridis", linewidth=0, antialiased=False)
    ax1.set_title("Original (3D Surface)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(xx, yy, zz_recon, cmap="viridis", linewidth=0, antialiased=False)
    ax2.set_title("Reconstruction (3D Surface)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(err, cmap="magma")
    ax3.set_title("Absolute Error")
    ax3.set_xticks([])
    ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    for ax in (ax1, ax2):
        ax.set_zlim(-vmax, vmax)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_case(data, use_gpu, n_chunk, n_overlap=2.0, device=None):
    start = time.perf_counter()
    results, residual = constrained_diffusion_decomposition(
        data=data,
        min_scale=2,
        max_scale=192,
        mode="log",
        log_scale_base=2.0,
        constrained=True,
        up_sample=False,
        use_gpu=use_gpu,
        device=device,
        n_chunk=int(n_chunk),
        n_overlap=float(n_overlap),
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    recon = _reconstruct(results, residual)
    return elapsed, recon


def main():
    parser = argparse.ArgumentParser(description="1000x1000 tiling performance benchmark.")
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Require CUDA/MPS; fail instead of CPU fallback when unavailable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device (e.g. 'cuda', 'cuda:0', or 'mps').",
    )
    args = parser.parse_args()

    data = create_mixed_scale_field(shape=(1000, 1000))
    has_accelerator = _has_accelerator()
    requested_device = _resolve_device(args.device)
    use_gpu = bool(requested_device is not None)

    if args.gpu_only and not use_gpu:
        raise RuntimeError("GPU-only run requested but no CUDA/MPS accelerator is available.")

    if not use_gpu:
        print("No CUDA/MPS accelerator found. Running CPU fallback for both modes.")
    else:
        print(f"Using accelerator device: {requested_device}")

    elapsed_base, recon_base = run_case(
        data=data,
        use_gpu=use_gpu,
        device=requested_device,
        n_chunk=1,
    )
    base_plot = OUTPUT_DIR / "tiling_3d_gpu_baseline_1000x1000.png"
    _save_verification_plot(
        original=data,
        recon=recon_base,
        case_name="GPU baseline (n_chunk=1)" if use_gpu else "CPU fallback baseline (n_chunk=1)",
        elapsed_s=elapsed_base,
        out_path=base_plot,
    )

    elapsed_tiled, recon_tiled = run_case(
        data=data,
        use_gpu=use_gpu,
        device=requested_device,
        n_chunk=16,
    )
    tiled_plot = OUTPUT_DIR / "tiling_3d_gpu_enforced_n16_1000x1000.png"
    _save_verification_plot(
        original=data,
        recon=recon_tiled,
        case_name="GPU enforced tiling (n_chunk=16)" if use_gpu else "CPU fallback tiled (n_chunk=16)",
        elapsed_s=elapsed_tiled,
        out_path=tiled_plot,
    )

    print("Saved verification plots:")
    print(f"  - {base_plot}")
    print(f"  - {tiled_plot}")
    print(f"Elapsed baseline: {elapsed_base:.2f}s")
    print(f"Elapsed tiled:    {elapsed_tiled:.2f}s")


if __name__ == "__main__":
    main()
