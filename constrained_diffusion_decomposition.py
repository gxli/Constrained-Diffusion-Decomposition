#!/usr/bin/env python
from astropy.io import fits
import sys
import numpy as np
from math import log
from scipy import ndimage


def constrained_diffusion_decomposition(data,
                                      e_rel=3e-2,
                                      max_n=None, sm_mode='reflect'):

    """
        perform constrained diffusion decomposition
        inputs:
            data: 
                n-dimensional array
            e_rel:
                relative error, a smaller e_rel means a better
                accuracy yet a larger computational cost
            max_n: 
                maximum number of channels. Channel number
                ranges from 0 to max_n
                if None, the program will calculate it automatically
            sm_mode: 
                {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional The mode
                parameter determines how the input array is extended beyond its
                boundaries in the convolution operation. Default is 'reflect'.
        output:
            results of constained diffusion decomposition. Assuming that the input
            is a n-dimensional array, then the output would be a n+1 dimensional
            array. The added dimension is the scale. Component maps can be accessed
            via output[n], where n is the channel number.

                output[i] contains structures of sizes larger than 2**i pixels
                yet smaller than 2**(i+1) pixels.
                
    """
    
    ntot = int(log(min(data.shape))/log(2) - 1)    
    # the total number of scale map
    
    result = []
    # residual = []
    # residual.append(data)
    if  max_n is not None:
        ntot = np.min(ntot, max_n)        
    print("ntot", ntot)

    diff_image = data.copy() * 0

    for i in range(ntot):
        print("i =", i)
        channel_image = data.copy() * 0  

        # computing the step size
        scale_end = float(pow(2, i + 1))
        scale_begining = float(pow(2, i))
        t_end = scale_end**2  / 2  # t at the end of this scale
        t_beginning = scale_begining**2  / 2 # t at the beginning of this scale

        if i == 0:
            delta_t_max = t_beginning * 0.1
        else:
            delta_t_max = t_beginning * e_rel



        niter = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)    # size of gaussian kernel
        print("kernel_size", kernel_size)
        for kk in range(niter):
            smooth_image = ndimage.gaussian_filter(data, kernel_size,
                                                   mode=sm_mode)
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
            # data = ndimage.gaussian_filter(data, kernel_size)     # !!!!
        result.append(channel_image)
        # residual.append(data)
    residual = data
    return result, residual


if __name__ == "__main__":
    fname = sys.argv[1] 
    hdulist = fits.open(fname)
    data = hdulist[0].data
    data[np.isnan(data)] = 0
    result, residual = constrained_diffusion_decomposition(data)

    nhdulist = fits.PrimaryHDU(result)
    nhdulist.header = hdulist[0].header
    nhdulist.header['DMIN'] = np.nanmin(result)
    nhdulist.header['DMAX'] = np.nanmin(result)

    nhdulist.writeto(sys.argv[1] + '_scale.fits', overwrite=True)
