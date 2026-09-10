"""Reproduce README figures through the accelerated PyTorch backend."""

from pathlib import Path

import pytest

from constrained_diffusion import TORCH_AVAILABLE


def _accelerator_device():
    if not TORCH_AVAILABLE:
        return None

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.gpu
def test_readme_figures_gpu():
    from readme_plot_utils import assert_pngs_readable, generate_readme_plots

    device = _accelerator_device()
    if device is None:
        pytest.skip("No CUDA/MPS accelerator available for README GPU figure reproduction.")

    output_dir = Path(__file__).resolve().parent / "outputs" / "readme_gpu"
    paths = generate_readme_plots(output_dir=output_dir, use_gpu=True, device=device)
    assert_pngs_readable(paths)
