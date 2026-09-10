"""Reproduce README figures through the CPU backend."""

from pathlib import Path

import pytest

from readme_plot_utils import assert_pngs_readable, generate_readme_plots


@pytest.mark.plot
def test_readme_figures_cpu():
    output_dir = Path(__file__).resolve().parent / "outputs" / "readme_cpu"
    paths = generate_readme_plots(output_dir=output_dir, use_gpu=False)
    assert_pngs_readable(paths)
