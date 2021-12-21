<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

# Constrain diffusion: A new PDE-based image decomposition
Decompose images into components of different sizes by solving a modified version of the diffusion equation.

Input:

  numpy nd array, of shape e.g. (nx, ny, nz)

Output:
  
  numpy nd array, of shape (m, nx, ny, nz). The mth commponent contain structures of sizes 2**(m-1) to 2**m pixels.
  

Usage:

(a) under the shell,


  python constrained_diffusion_decomposition.py input.fits 
  the output file will be named as input.fits_scale.fits

(b) inside python
  
  import constrained_diffusion_decomposition
  result = constrained_diffusion_decomposition.dcnstrained_diffusion_decomposition(data)
 

  
How it is done:

Assuuming an input of I(x, y),t he decomposition is achieved by solving the equation 

  \frac{\partial I_t }{\partial t} ={\rm sgn}(I_t) \mathcal{H}({- \rm sgn}(I_t)  \nabla^2  I_t)   \nabla^2 I_t \;,
 
where t is related to the scale l by t = l**2
 

  
  
References: 

Li 2022, Multi-Scale Decomposition of Astronomical Maps -- Constrained Diffusion Method
