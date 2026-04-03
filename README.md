
# Constrained Diffusion Decomposition: A PDE-based Image Decomposition Method

## General Design
A natural image often contains components of different scales. This project provides a powerful tool to decompose N-dimensional `numpy.ndarray` or `torch.Tensor` data into its constituent scale components.

The decomposition is highly flexible, allowing for **logarithmically** or **linearly** spaced scales that can be **automatically determined** from the data's shape or specified manually by the user.

The code is based on the principles described in
<a href="https://arxiv.org/abs/2201.05484">Li 2022, Multi-Scale Decomposition of Astronomical Maps -- Constrained Diffusion Method</a>.

Assuming an input of $I(x, y)$, the decomposition is achieved by solving the equation:

$$\frac{\partial I_t }{\partial t} ={\rm sgn}(I_t) \mathcal{H}({- \rm sgn}(I_t) \nabla^2 I_t) \nabla^2 I_t$$

where the diffusion time $t$ is related to the characteristic scale $l$ by $t = l^2/2$.

### Connection to ReLU (Structural Rectification)
While the equation above uses the Heaviside step function $\mathcal{H}$, it is insightful to view this operation through the lens of modern Deep Learning. The constraint is mathematically equivalent to applying a **Rectified Linear Unit (ReLU)** logic to the curvature of the image.

For positive structures ($I > 0$), the evolution simplifies to:

$$\frac{\partial I}{\partial t} = -\text{ReLU}(-\nabla^2 I) \equiv \min(0, \nabla^2 I)$$

## Key Features

* **🚀 GPU Acceleration**: Native support for **CUDA** (NVIDIA) and **MPS** (Apple Silicon) via PyTorch. Decomposing large 3D volumes is significantly faster than CPU-only methods.
* **Automatic Parameter Detection**: Scale boundaries and the number of channels are automatically inferred from the input data's shape if not provided.
* **Hybrid Upsampling Strategy**: Uses a high-resolution (4x upsampled) decomposition for small scales ($\le 5$ pixels) to capture fine details accurately, then switches to a standard grid for larger scales.
* **Constrained vs. Unconstrained Modes**:
    * **Default (`constrained=True`)**: Artifact-free, sign-preserving decomposition that prevents the creation of artificial peaks or valleys.
    * **Optional (`constrained=False`)**: Standard linear diffusion (faster, but may introduce "ringing" artifacts).
* **Inverted Decomposition**: A special `inverted=True` mode allows the algorithm to decompose negative features ("holes" or absorption dips) within a positive background.
* **VRAM Efficient**: The GPU engine uses an "Offload-to-CPU" strategy—processing happens in VRAM, but finished channels are stored in system RAM immediately to prevent Out-of-Memory errors.

## Installation

1.  **From source via git clone:**
    ```bash
    git clone [https://github.com/gxli/Constrained-Diffusion-Decomposition.git](https://github.com/gxli/Constrained-Diffusion-Decomposition.git)
    cd Constrained-Diffusion-Decomposition 
    pip install .
    ```

2.  **Via `pip`:**
    ```bash
    pip install constrained-diffusion
    ```

> **Note**: For GPU support, ensure [PyTorch](https://pytorch.org/get-started/locally/) is installed in your environment.

## Usage

The main entry point is the automated `constrained_diffusion_decomposition` function. It handles both `numpy` arrays and `torch` tensors.

### Function Signature
```python
constrained_diffusion_decomposition(
    data, 
    num_channels=None, 
    max_scale=None, 
    min_scale=1,
    mode='log', 
    log_scale_base=2.0,
    linear_scale_step=None,
    up_sample=True, 
    constrained=True,
    inverted=False,
    use_gpu=False,    # Toggle hardware acceleration
    device=None,      # Target 'cuda' or 'mps'
    return_scales=False
)

Key Parameters
Parameter	Description	Default
data	Input N-dimensional array (NumPy or PyTorch).	(Required)
use_gpu	If True, uses PyTorch backend for hardware acceleration.	False
device	Target device (e.g., 'cuda', 'mps'). If None, auto-selects.	None
mode	Scale spacing: 'log' (powers of base) or 'lin' (linear).	'log'
up_sample	If True, uses the efficient hybrid upsampling strategy.	True
constrained	If True, uses the artifact-free algorithm.	True
inverted	If True, decomposes depressions ("holes") instead of peaks.	False
num_channels	Number of channels. Calculated automatically if None.	None
max_scale	Largest scale to analyze. Default is max(data.shape)/2.	None
return_scales	If True, returns the list of scale boundaries.	False
Performance Comparison

The GPU implementation is highly effective for 3D data where standard CPU-based Gaussian convolutions become a bottleneck.
Data Shape	CPU Time (s)	GPU Time (s)	Speedup
1D (1000)	0.04s	0.08s	< 1x (Overhead)
2D (512x512)	1.20s	0.20s	~6x
3D (100x100x100)	15.40s	0.80s	~19x
Quickstart (GPU Enabled)
Python

import constrained_diffusion as cdd
import numpy as np

# Create a sample 3D volume
data = np.random.rand(64, 64, 64).astype(np.float32)

# Perform decomposition on the GPU (results returned as NumPy)
results, residual = cdd.constrained_diffusion_decomposition(
    data, 
    use_gpu=True, 
    mode='log'
)

print(f"Decomposed into {len(results)} scale channels using GPU.")

Examples
1D Signal Decomposition

The constrained diffusion method separates signals without introducing negative ripples (ringing).
<img src="images/notebook_image_3.png" width="700"/>
Inverted Mode (Hole Detection)

Setting inverted=True allows the algorithm to fill gaps, identifying absorption dips or depressions.
<img src="images/notebook_image_4.png" width="700"/>
High-Resolution Detail

Using up_sample=True ensures that fine structures are not lost during the initial diffusion steps.
<img src="images/notebook_image_5.png" width="700"/>
Reference

Li, G. (2022). Multi-Scale Decomposition of Astronomical Maps -- Constrained Diffusion Method. arXiv:2201.05484
License

See the LICENSE file for details.
convert this to markdown format, for pasting