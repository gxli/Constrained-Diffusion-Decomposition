# Constrain diffusion: A new PDE-based image decomposition
Decompose images into components of different sizes by solving a modified version of the diffusion equation.

Usage:

  import 


Input:

  numpy nd array, of shape e.g. (nx, ny, nz)

Output:
  
  numpy nd array, of shape (m, nx, ny, nz).
  the mth component represent structures of sizes ranging form 2**m to 2**(m+1) pixels. 
  
  
References: 

