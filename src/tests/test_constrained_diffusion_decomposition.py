import pytest
import numpy as np
from src.constrained_diffusion_decomposition import constrained_diffusion_decomposition

def test_constrained_diffusion_decomposition():
    # Create a small 2D test array
    data = np.ones((16, 16))
    result, residual = constrained_diffusion_decomposition(data, e_rel=3e-2, max_n=2)
    
    # Check that result is a list of arrays
    assert isinstance(result, list)
    assert len(result) <= 2  # max_n=2
    assert all(isinstance(r, np.ndarray) for r in result)
    
    # Check residual is an array
    assert isinstance(residual, np.ndarray)
    assert residual.shape == data.shape
    
 THOUGH# Check that the sum of results and residual approximates the input
    reconstructed = np.sum(result, axis=0) + residual
    np.testing.assert_allclose(reconstructed, data, rtol=1e-5)

def test_invalid_input():
    with pytest.raises(ValueError):
        constrained_diffusion_decomposition(np.array([]))  # Empty array
