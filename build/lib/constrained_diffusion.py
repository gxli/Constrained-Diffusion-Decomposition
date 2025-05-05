from scipy import ndimage
import numpy as np
from math import log

def constrained_diffusion_decomposition(data,
                                        e_rel=3e-2,
                                        max_n=None, 
                                        sm_mode='reflect'):
    """
    Perform constrained diffusion decomposition on n-dimensional data.
    
    Args:
        data: n-dimensional array
        e_rel: Relative error (smaller e_rel increases accuracy but computational cost)
        max_n: Maximum number of channels (if None, calculated automatically)
        sm_mode: Mode for array boundary extension in convolution ('reflect', 'constant', 
                'nearest', 'mirror', 'wrap'). Default is 'reflect'.
    
    Returns:
        tuple: (results, residual)
            - results: n+1 dimensional array where results[i] contains structures of 
                      sizes between 2**i and 2**(i+1) pixels
            - residual: Structures too large to be contained in results
    """
    if data.size == 0:
        raise ValueError("Input data array is empty")
        
    ntot = int(log(min(data.shape)) / log(2) - 1)
    if max_n is not None:
        ntot = min(ntot, max_n)
    print("total number of scales", ntot)

    result = []
    diff_image = data.copy() * 0

    for i in range(ntot):
        # print("number of scale", i+1)
        channel_image = data.copy() * 0
        scale_end = float(pow(2, i + 1))
        scale_beginning = float(pow(2, i))
        t_end = scale_end**2 / 2
        t_beginning = scale_beginning**2 / 2

        if i == 0:
            delta_t_max = t_beginning * 0.1
        else:
            delta_t_max = t_beginning * e_rel

        niter = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)

        print("current channel", i, "current scale", 2**i)
        
        for kk in range(niter):
            smooth_image = ndimage.gaussian_filter(data, kernel_size, mode=sm_mode)
            sm_image_1 = np.minimum(data, smooth_image)
            sm_image_2 = np.maximum(data, smooth_image)

            diff_image_1 = data - sm_image_1
            diff_image_2 = data - sm_image_2

            diff_image = diff_image * 0

            positions_1 = np.where(np.logical_and(diff_image_1 > 0, data > 0))
            positions_2 = np.where(np.logical_and(diff_image_2 < 0, data < 0))

            diff_image[positions_1] = diff_image_1[positions_1]
            diff_image[positions_2] = diff_image_2[positions_2]

            channel_image = channel_image + diff_image
            data = data - diff_image
        
        result.append(channel_image)
    
    residual = data
    return result, residual
