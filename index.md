## Welcome to GitHub Pages
Decompose images into components of different sizes by solving a modified version of the diffusion equation.

Input:

numpy nd array, of shape e.g. (nx, ny, nz)

Output:

result: numpy nd array, of shape (m, nx, ny, nz). The mth commponent contain structures of sizes 2**(m-1) to 2**m pixels. residual: numpy nd array, of shape (nx, ny, nz) the input data will be recovered as input = sum_i result[i] + residual

Usage:

(a) under the shell,

python constrained_diffusion_decomposition.py input.fits

the output file will be named as input.fits_scale.fits

(b) inside python

import constrained_diffusion_decomposition

result, residual = constrained_diffusion_decomposition.dcnstrained_diffusion_decomposition(data)

How it is done:

Assuuming an input of I(x, y),t he decomposition is achieved by solving the equation

\frac{\partial I_t }{\partial t} ={\rm sgn}(I_t) \mathcal{H}({- \rm sgn}(I_t) \nabla^2 I_t) \nabla^2 I_t ;,

where t is related to the scale l by t = l**2

References:

Li 2022, Multi-Scale Decomposition of Astronomical Maps -- Constrained Diffusion Method
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
