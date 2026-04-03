import numpy as np
from scipy import ndimage

# =============================================================================
# OPTIONAL GPU SUPPORT (Graceful Fallback)
# =============================================================================
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _get_optimal_device(device_arg=None):
    """Determine the best available compute device (CUDA, MPS, or CPU)."""
    if device_arg is not None: return torch.device(device_arg)
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def _gaussian_filter_nd_pytorch(tensor, sigma, mode='reflect'):
    """N-dimensional Gaussian filter using PyTorch separable convolutions."""
    if sigma <= 0: return tensor
    radius = int(4.0 * sigma + 0.5)
    if radius == 0: return tensor
    
    k = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1) 

    out = tensor
    dims = out.dim()
    pad_mode = mode if mode in ['reflect', 'replicate', 'circular'] else 'reflect'

    for dim in range(dims):
        out = out.transpose(dim, -1)
        orig_shape = out.shape
        out = out.reshape(-1, 1, orig_shape[-1])
        out = F.pad(out, (radius, radius), mode=pad_mode)
        out = F.conv1d(out, kernel)
        out = out.reshape(orig_shape)
        out = out.transpose(dim, -1)
        
    return out

def _zoom_tensor(tensor, zoom_factor):
    """Mimics scipy.ndimage.zoom for N-dimensional up/down sampling."""
    dims = tensor.dim()
    out = tensor.unsqueeze(0).unsqueeze(0)
    
    if dims == 1: interp_mode = 'linear'
    elif dims == 2: interp_mode = 'bilinear'
    elif dims == 3: interp_mode = 'trilinear'
    else: interp_mode = 'nearest'

    out = F.interpolate(out, scale_factor=zoom_factor, mode=interp_mode, align_corners=False)
    return out.squeeze(0).squeeze(0)

# =============================================================================
# CORE ENGINE
# =============================================================================
def constrained_multiscale_decomposition(data, scales, e_rel=3e-2, sm_mode='reflect', 
                                         constrained=True, inverted=False, use_gpu=False, verbose=True):
    """
    (Core Engine) Perform diffusion decomposition on n-dimensional data.
    """
    # Type check based on mode
    if use_gpu and not TORCH_AVAILABLE:
        if verbose: print("Warning: PyTorch not found. Falling back to CPU execution.")
        use_gpu = False

    is_tensor = use_gpu and isinstance(data, torch.Tensor)
    if not is_tensor and hasattr(data, 'size') and data.size == 0:
        raise ValueError("Input data array is empty")
    elif is_tensor and data.numel() == 0:
        raise ValueError("Input data tensor is empty")
        
    # --- Safely handle 'scales' whether it's a Tensor, List, or NumPy array ---
    if isinstance(scales, torch.Tensor):
        scales_array = scales.cpu().numpy()
    else:
        scales_array = np.array(scales)
        
    if not np.all(np.diff(scales_array) > -1e-9):
        raise ValueError("The 'scales' array must be sorted in increasing order.")

    if verbose:
        mode_str = "CONSTRAINED " + ("(inverted)" if inverted else "(standard)") if constrained else "UNCONSTRAINED"
        device_str = str(data.device).upper() if is_tensor else "CPU"
        print(f"Running Engine in {mode_str} mode on {device_str}.")

    # ---------------------------------------------------------
    # GPU (PyTorch) Execution with VRAM offloading
    # ---------------------------------------------------------
    if use_gpu:
        current_data = data.clone()
        result = []
        t_beginning = 0.0

        for i, scale_end in enumerate(scales_array): # Use the safe numpy array
            channel_image = torch.zeros_like(current_data)
            t_end = scale_end**2 / 2
            
            delta_t_max = t_beginning * e_rel if t_beginning > 0 else t_end * e_rel
            if delta_t_max <= 0: delta_t_max = (t_end - t_beginning) * 0.1
                
            niter = int((t_end - t_beginning) / delta_t_max + 0.5)
            niter = max(1, niter)
            delta_t = (t_end - t_beginning) / niter
            kernel_size = (2 * delta_t) ** 0.5
            
            if verbose: print(f"Channel {i}: Scale < {scale_end:.2f} pixels, Iterations: {niter}")

            for _ in range(niter):
                smooth_image = _gaussian_filter_nd_pytorch(current_data, kernel_size, mode=sm_mode)
                
                if constrained:
                    diff_pos = current_data - torch.minimum(current_data, smooth_image)
                    diff_neg = current_data - torch.maximum(current_data, smooth_image)
                    diff_image = torch.zeros_like(current_data)
                    
                    if not inverted:
                        mask_pos = (diff_pos > 0) & (current_data > 0)
                        mask_neg = (diff_neg < 0) & (current_data < 0)
                        diff_image = torch.where(mask_pos, diff_pos, diff_image)
                        diff_image = torch.where(mask_neg, diff_neg, diff_image)
                    else:
                        mask_pos = (diff_neg < 0) & (current_data > 0)
                        mask_neg = (diff_pos > 0) & (current_data < 0)
                        diff_image = torch.where(mask_pos, diff_neg, diff_image)
                        diff_image = torch.where(mask_neg, diff_pos, diff_image)
                else:
                    diff_image = current_data - smooth_image
                    
                channel_image += diff_image
                current_data -= diff_image
                
            # MEMORY SAFEGUARD: Move finished channel to CPU memory immediately
            result.append(channel_image.cpu())
            t_beginning = t_end
            
        return result, current_data.cpu()

    # ---------------------------------------------------------
    # CPU (NumPy/SciPy) Execution
    # ---------------------------------------------------------
    else:
        current_data = data.copy()
        result = []
        t_beginning = 0.0

        for i, scale_end in enumerate(scales_array): # Use the safe numpy array
            channel_image = np.zeros_like(current_data)
            t_end = scale_end**2 / 2
            
            if t_beginning > 0: delta_t_max = t_beginning * e_rel
            else: delta_t_max = t_end * e_rel
            if delta_t_max <= 0: delta_t_max = (t_end - t_beginning) * 0.1
            
            niter = int((t_end - t_beginning) / delta_t_max + 0.5)
            niter = max(1, niter)
            delta_t = (t_end - t_beginning) / niter
            kernel_size = np.sqrt(2 * delta_t)
            
            if verbose: print(f"Channel {i}: Scale < {scale_end:.2f} pixels, Iterations: {niter}")

            for _ in range(niter):
                smooth_image = ndimage.gaussian_filter(current_data, kernel_size, mode=sm_mode)
                
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
            
        return result, current_data


# =============================================================================
# WRAPPER
# =============================================================================
def constrained_diffusion_decomposition(
    data,
    num_channels=None, max_scale=None, min_scale=1,
    mode='log', log_scale_base=2.0, linear_scale_step=None,
    e_rel=3e-2, sm_mode='reflect', up_sample=True,
    switch_scale=5.0, zoom_factor=4.0,
    constrained=True, inverted=False, return_scales=False,
    use_gpu=False, device=None, verbose=True
):
    """
    Overall wrapper for diffusion decomposition with a highly automated interface.
    """
    # Check GPU availability and handle data loading
    if use_gpu and not TORCH_AVAILABLE:
        if verbose: print("Warning: PyTorch not found. Falling back to CPU execution.")
        use_gpu = False

    is_numpy = isinstance(data, np.ndarray)
    
    if use_gpu:
        target_device = _get_optimal_device(device)
        if is_numpy:
            current_data = torch.from_numpy(data).float().to(target_device)
        elif isinstance(data, torch.Tensor):
            current_data = data.to(target_device)
        else:
            raise TypeError("Data must be a numpy array or a PyTorch tensor.")
    else:
        if isinstance(data, torch.Tensor):
            current_data = data.cpu().numpy()
        else:
            current_data = data

    # --- Scale Logic ---
    if mode == 'log' and log_scale_base <= 1:
        raise ValueError("log_scale_base must be greater than 1 for logarithmic mode.")
    if mode == 'lin' and (linear_scale_step is None or linear_scale_step <= 0):
        raise ValueError("`linear_scale_step` must be a positive number for 'lin' mode.")

    effective_min_scale = float(min_scale)
    effective_max_scale = float(max_scale) if max_scale is not None else float(max(current_data.shape) / 2)
    
    if effective_max_scale <= effective_min_scale:
        raise ValueError(f"max_scale ({effective_max_scale:.2f}) > min_scale ({effective_min_scale:.2f}) required.")

    if mode == 'log':
        if num_channels is None:
            log_diff = np.log(effective_max_scale * (1 + 1e-9)) - np.log(effective_min_scale)
            num_channels = int(log_diff / np.log(log_scale_base)) + 1
            
        start_power = np.log(effective_min_scale) / np.log(log_scale_base)
        stop_power = start_power + (num_channels - 1)
        scale_edges = np.logspace(start_power, stop_power, num=num_channels, base=log_scale_base)

    elif mode == 'lin':
        scale_range = effective_max_scale - effective_min_scale
        num_channels = int(np.floor(scale_range / linear_scale_step)) + 1
        scale_edges = effective_min_scale + np.arange(num_channels) * linear_scale_step

    if len(scale_edges) == 0:
        raise ValueError("Scale generation failed.")

    # --- Perform decomposition ---
    core_kwargs = {
        'e_rel': e_rel, 'sm_mode': sm_mode, 
        'constrained': constrained, 'inverted': inverted if constrained else False,
        'use_gpu': use_gpu, 'verbose': verbose
    }

    results = []
    residual = current_data
    
    if up_sample:
        scales_small = scale_edges[scale_edges <= switch_scale]
        scales_large = scale_edges[scale_edges > switch_scale]
        
        if len(scales_small) > 0:
            if verbose: print(f"\n--- STAGE 1: High-res decomposition (scales <= {switch_scale}) ---")
            upsampled_scales = np.unique(np.maximum(scales_small * zoom_factor, zoom_factor))
            
            if use_gpu:
                upsampled_data = _zoom_tensor(current_data, zoom_factor)
                results_small_up, residual_up = constrained_multiscale_decomposition(upsampled_data, upsampled_scales, **core_kwargs)
                
                # results_small_up and residual_up are already moved to CPU VRAM by the engine
                # We need to move them back to GPU for interpolation, then back to CPU.
                results_small = [_zoom_tensor(res.to(target_device), 1.0/zoom_factor).cpu() for res in results_small_up]
                current_data = _zoom_tensor(residual_up.to(target_device), 1.0/zoom_factor) # Keep active data on GPU
            else:
                upsampled_data = ndimage.zoom(current_data, zoom_factor, order=1)
                results_small_up, residual_up = constrained_multiscale_decomposition(upsampled_data, upsampled_scales, **core_kwargs)
                results_small = [ndimage.zoom(res, 1/zoom_factor, order=1) for res in results_small_up]
                current_data = ndimage.zoom(residual_up, 1/zoom_factor, order=1)
                
            results.extend(results_small)

        if len(scales_large) > 0:
            if verbose: print(f"\n--- STAGE 2: Fixed-grid decomposition (scales > {switch_scale}) ---")
            results_large, residual_large = constrained_multiscale_decomposition(current_data, scales_large, **core_kwargs)
            results.extend(results_large)
            residual = residual_large
        else:
            residual = current_data
    else:
        if verbose: print(f'\n--- Performing standard fixed-grid decomposition ---')
        results, residual = constrained_multiscale_decomposition(current_data, scale_edges, **core_kwargs)

    # --- Convert back to original type (NumPy) if requested ---
    if use_gpu and is_numpy:
        # Tensors returned from the GPU engine are already on the CPU memory
        results = [r.numpy() for r in results]
        if isinstance(residual, torch.Tensor):
            residual = residual.cpu().numpy()

    if return_scales:
        return results, residual, scale_edges
    return results, residual