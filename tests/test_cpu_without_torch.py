"""Regression coverage for CPU-only installations."""

import os
from pathlib import Path
import subprocess
import sys


def test_default_cpu_path_does_not_import_torch():
    """A normal NumPy decomposition must work even if importing torch is forbidden."""
    source_dir = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins
import sys

original_import = builtins.__import__

def reject_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('CPU path attempted to import PyTorch')
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_torch

import numpy as np
from constrained_diffusion import constrained_diffusion_decomposition

data = np.linspace(0.0, 1.0, 32, dtype=np.float32)
components, residual = constrained_diffusion_decomposition(
    data, max_scale=8, up_sample=False, verbose=False
)

assert components
assert residual.shape == data.shape
assert 'torch' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_dir), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
