import numpy as np
from scipy import ndimage
from math import log



# =============================================================================
# CORE ENGINE (UNCHANGED)
# =============================================================================
def constrained_multiscale_decomposition(data, scales, e_rel=3e-2, sm_mode='reflect', constrained=True, inverted=False):
    """
    (Core Engine) Perform diffusion decomposition on n-dimensional data.
    """
    if data.size == 0:
        raise ValueError("Input data array is empty")
    if not np.all(np.diff(scales) > -1e-9):
        raise ValueError("The 'scales' array must be sorted in increasing order.")

    ntot = len(scales)
    # print(f"Decomposing across {ntot} user-specified scales: {np.round(scales, 2)}")

    if constrained:
        if inverted: print("Running in CONSTRAINED (inverted) mode.")
        else: print("Running in CONSTRAINED (standard) mode.")
    else:
        print("Running in UNCONSTRAINED (linear diffusion) mode.")

    current_data = data.copy()
    result = []
    t_beginning = 0.0

    for i, scale_end in enumerate(scales):
        channel_image = np.zeros_like(current_data)
        t_end = scale_end**2 / 2
        if t_beginning > 0: delta_t_max = t_beginning * e_rel
        else: delta_t_max = t_end * e_rel
        if delta_t_max <= 0: delta_t_max = (t_end - t_beginning) * 0.1
        niter = int((t_end - t_beginning) / delta_t_max + 0.5)
        niter = max(1, niter)
        delta_t = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)
        print(f"Channel {i}: Scale < {scale_end:.2f} pixels, Iterations: {niter}")

        for _ in range(niter):
            smooth_image = ndimage.gaussian_filter(current_data, kernel_size, mode=sm_mode)
            diff_image = None
            if constrained:
                diff_image_pos = current_data - np.minimum(current_data, smooth_image)
                diff_image_neg = current_data - np.maximum(current_data, smooth_image)
                diff_image = np.zeros_like(current_data)
                if not inverted:
                    pos1 = np.where(np.logical_and(diff_image_pos > 0, current_data > 0))
                    pos2 = np.where(np.logical_and(diff_image_neg < 0, current_data < 0))
                    diff_image[pos1] = diff_image_pos[pos1]; diff_image[pos2] = diff_image_neg[pos2]
                else:
                    pos1 = np.where(np.logical_and(diff_image_neg < 0, current_data > 0))
                    pos2 = np.where(np.logical_and(diff_image_pos > 0, current_data < 0))
                    diff_image[pos1] = diff_image_neg[pos1]; diff_image[pos2] = diff_image_pos[pos2]
            else:
                diff_image = current_data - smooth_image
            channel_image += diff_image
            current_data -= diff_image
        result.append(channel_image)
        t_beginning = t_end
    residual = current_data
    return result, residual


# Note: This function requires the core engine 'constrained_multiscale_decomposition'
# to be defined in the same file or imported.

def constrained_diffusion_decomposition(
    data,
    num_channels=None,
    max_scale=None,
    min_scale=1,
    mode='log',
    log_scale_base=2.0,
    linear_scale_step=None,
    e_rel=3e-2,
    sm_mode='reflect',
    up_sample=True,
    constrained=True,
    inverted=False,
    return_scales=False
):
    """
    Overall wrapper for diffusion decomposition with a highly automated interface.

    Args:
        data (np.ndarray): n-dimensional array.
        num_channels (int, optional): The number of channels. If None, calculated automatically.
                                      For 'lin' mode, this argument is ignored.
        max_scale (float, optional): The largest scale size. If None, set to max(data.shape) / 2.
        min_scale (float, optional): The smallest scale size. Defaults to 1.
        mode (str): Scale generation mode: 'log' (default) or 'lin'.
        log_scale_base (float): The base for logarithmic scale generation. Defaults to 2.0.
        linear_scale_step (float, optional): If mode is 'lin', this defines the step size
                                             between scales. This argument is required for 'lin' mode.
        e_rel (float): Relative error for diffusion step size.
        sm_mode (str): Mode for array boundary extension in convolution.
        up_sample (bool): If True (default), uses the hybrid upsampling strategy.
        constrained (bool): If True (default), uses the sign-based constrained algorithm.
        inverted (bool): If True, decomposes depressions ("holes") instead of peaks.
        return_scales (bool): If True, the array of scale boundaries (upper edge of each channel)
                              is also returned. Defaults to False.

    Returns:
        tuple: By default, returns a tuple of `(results, residual)`.
               If `return_scales=True`, returns `(results, residual, scale_edges)`.
    """
    # --- Step 1: Validate inputs and determine scale range ---
    if mode == 'log' and log_scale_base <= 1:
        raise ValueError("log_scale_base must be greater than 1 for logarithmic mode.")
    if mode == 'lin' and linear_scale_step is None:
        raise ValueError("`linear_scale_step` must be provided when mode is 'lin'.")
    if mode == 'lin' and linear_scale_step is not None and linear_scale_step <= 0:
        raise ValueError("linear_scale_step must be a positive number.")

    effective_min_scale = float(min_scale)
    if max_scale is None:
        effective_max_scale = float(max(data.shape) / 2)
        print(f"Automatically determined max_scale = {effective_max_scale:.2f} (from data shape {data.shape})")
    else:
        effective_max_scale = float(max_scale)
    if effective_max_scale <= effective_min_scale:
        raise ValueError(f"max_scale ({effective_max_scale:.2f}) must be greater than min_scale ({effective_min_scale:.2f}).")

    # --- Step 2: Generate Scale Edges ---
    scale_edges = None
    if mode == 'log':
        if num_channels is None:
            print("Automatically determining num_channels for log mode.")
            if effective_max_scale <= effective_min_scale: effective_num_channels = 0
            else:
                log_diff = np.log(effective_max_scale * (1 + 1e-9)) - np.log(effective_min_scale)
                effective_num_channels = int(log_diff / np.log(log_scale_base)) + 1
            print(f"--> Determined num_channels = {effective_num_channels}")
        else:
            effective_num_channels = int(num_channels)

        if effective_num_channels < 1:
            raise ValueError(f"Number of channels must be at least 1. Calculated value was {effective_num_channels}.")

        start_power = np.log(effective_min_scale) / np.log(log_scale_base)
        stop_power = start_power + (effective_num_channels - 1)
        scale_edges = np.logspace(start_power, stop_power, num=effective_num_channels, base=log_scale_base)
        
        adjusted_max_scale = scale_edges[-1]
        if abs(adjusted_max_scale - effective_max_scale) > 1e-6:
             print(f"NOTE: Adjusted max_scale from {effective_max_scale:.2f} to {adjusted_max_scale:.2f} to align with log_scale_base.")

    elif mode == 'lin':
        print(f"Generating linear scales with a step of {linear_scale_step}.")
        if effective_max_scale <= effective_min_scale: effective_num_channels = 0
        else:
            scale_range = effective_max_scale - effective_min_scale
            effective_num_channels = int(np.floor(scale_range / linear_scale_step)) + 1
        print(f"--> Determined num_channels = {effective_num_channels}")
        
        if effective_num_channels < 1:
            raise ValueError(f"Number of channels must be at least 1. Calculated value was {effective_num_channels}.")
            
        scale_edges = effective_min_scale + np.arange(effective_num_channels) * linear_scale_step

        adjusted_max_scale = scale_edges[-1]
        if (effective_max_scale - adjusted_max_scale) > 1e-6:
             print(f"NOTE: Adjusted max_scale from {effective_max_scale:.2f} to {adjusted_max_scale:.2f} to align with linear_scale_step.")

    if scale_edges is None or len(scale_edges) == 0:
        raise ValueError("Scale generation failed. Check your min/max_scale and step/num_channels parameters.")

    # --- Step 3: Perform decomposition ---
    if not constrained and inverted:
        print("Warning: 'inverted=True' has no effect when 'constrained=False'. Ignoring.")

    core_kwargs = {'e_rel': e_rel, 'sm_mode': sm_mode, 'constrained': constrained, 'inverted': inverted if constrained else False}

    results = []
    residual = data
    if up_sample:
        switch_scale = 5.0
        zoom_factor = 4
        scales_small = scale_edges[scale_edges <= switch_scale]
        scales_large = scale_edges[scale_edges > switch_scale]
        
        current_data = data
        if len(scales_small) > 0:
            print(f"\n--- STAGE 1: Performing high-resolution decomposition for scales <= {switch_scale} ---")
            upsampled_data = ndimage.zoom(current_data, zoom_factor, order=1)
            upsampled_scales = scales_small * zoom_factor
            upsampled_scales = np.maximum(upsampled_scales, zoom_factor)
            upsampled_scales = np.unique(upsampled_scales)
            # NOTE: We need to import the core engine for this call to work
            results_small_up, residual_up = constrained_multiscale_decomposition(upsampled_data, upsampled_scales, **core_kwargs)
            print('Downsampling small-scale results...')
            results_small = [ndimage.zoom(res, 1/zoom_factor, order=1) for res in results_small_up]
            results.extend(results_small)
            current_data = ndimage.zoom(residual_up, 1/zoom_factor, order=1)

        if len(scales_large) > 0:
            print(f"\n--- STAGE 2: Performing fixed-grid decomposition on residual for scales > {switch_scale} ---")
            # NOTE: We need to import the core engine for this call to work
            results_large, residual_large = constrained_multiscale_decomposition(current_data, scales_large, **core_kwargs)
            results.extend(results_large)
            residual = residual_large
        else:
            residual = current_data
    else:
        print(f'\n--- Performing standard fixed-grid decomposition for all scales ---')
        # NOTE: We need to import the core engine for this call to work
        results, residual = constrained_multiscale_decomposition(data, scale_edges, **core_kwargs)
        
    # --- Step 4: Return the results ---
    if return_scales:
        # Return the actual scale boundaries used in the decomposition for consistency.
        return results, residual, scale_edges
    else:
        return results, residual
    
