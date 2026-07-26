import numpy as np
from scipy import ndimage
try:
    from .chunking import plan_vram_chunks, calculate_overlap_pixels
except ImportError:
    from chunking import plan_vram_chunks, calculate_overlap_pixels
import math
import os
from itertools import product

# =============================================================================
# OPTIONAL GPU SUPPORT (Graceful Fallback)
# =============================================================================
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MONAI_AVAILABLE = False


def _ensure_monai():
    """Lazy-import MONAI to avoid import-time deadlocks on some platforms."""
    global MONAI_AVAILABLE
    if MONAI_AVAILABLE:
        return True
    try:
        global sliding_window_inference
        from monai.inferers import sliding_window_inference  # noqa: F811
        MONAI_AVAILABLE = True
        return True
    except ImportError:
        return False


class _MonaiScaleCutoff(RuntimeError):
    """Raised when MONAI ROI/window cannot safely support the requested scale."""
    pass


def _get_optimal_device(device_arg=None):
    """
    Determine the best available compute device (CUDA, MPS, or CPU).
    """
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _gaussian_filter_nd_pytorch(tensor, sigma, mode='reflect'):
    """
    Highly optimized N-dimensional Gaussian filter using PyTorch separable convolutions.
    Uses fast-paths for 1D, 2D, and 3D data to avoid costly memory transpositions.
    """
    if sigma <= 0:
        return tensor
    radius = int(4.0 * sigma + 0.5)
    if radius == 0:
        return tensor

    # Pre-calculate 1D kernel
    k = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel = kernel / kernel.sum()

    out = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D1, D2, ...]
    dims = tensor.dim()
    pad_mode = mode if mode in ['reflect', 'replicate', 'circular'] else 'reflect'

    # Fast paths for 1D, 2D, and 3D (Avoids .transpose and .contiguous memory overhead)
    if dims == 1:
        out = F.pad(out, (radius, radius), mode=pad_mode)
        out = F.conv1d(out, kernel.view(1, 1, -1))

    elif dims == 2:
        k_y = kernel.view(1, 1, -1, 1)
        k_x = kernel.view(1, 1, 1, -1)
        out = F.pad(out, (radius, radius, 0, 0), mode=pad_mode)
        out = F.conv2d(out, k_x)
        out = F.pad(out, (0, 0, radius, radius), mode=pad_mode)
        out = F.conv2d(out, k_y)

    elif dims == 3:
        # F.pad does not support 'reflect' for 3D natively; fallback to replicate
        if pad_mode == 'reflect':
            pad_mode = 'replicate'
        k_z = kernel.view(1, 1, -1, 1, 1)
        k_y = kernel.view(1, 1, 1, -1, 1)
        k_x = kernel.view(1, 1, 1, 1, -1)
        out = F.pad(out, (radius, radius, 0, 0, 0, 0), mode=pad_mode)
        out = F.conv3d(out, k_x)
        out = F.pad(out, (0, 0, radius, radius, 0, 0), mode=pad_mode)
        out = F.conv3d(out, k_y)
        out = F.pad(out, (0, 0, 0, 0, radius, radius), mode=pad_mode)
        out = F.conv3d(out, k_z)

    else:
        # Generic N-dimensional fallback (requires transposing)
        kernel = kernel.view(1, 1, -1)
        out = out.squeeze(0).squeeze(0)
        for dim in range(dims):
            out = out.transpose(dim, -1).contiguous()
            orig_shape = out.shape
            out = out.view(-1, 1, orig_shape[-1])
            out = F.pad(out, (radius, radius), mode=pad_mode)
            out = F.conv1d(out, kernel)
            out = out.view(orig_shape)
            out = out.transpose(dim, -1)
        return out

    return out.squeeze(0).squeeze(0)


def _gaussian_filter_3d_pytorch_batched(tensor_5d, sigma, mode='reflect'):
    """Gaussian filter for 5D BCHWD tensors via separable 3D convs."""
    if sigma <= 0:
        return tensor_5d
    radius = int(4.0 * sigma + 0.5)
    if radius == 0:
        return tensor_5d

    k = torch.arange(-radius, radius + 1, device=tensor_5d.device, dtype=tensor_5d.dtype)
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel = kernel / kernel.sum()
    c = int(tensor_5d.shape[1])
    pad_mode = mode if mode in ['reflect', 'replicate', 'circular'] else 'reflect'
    if pad_mode == 'reflect':
        pad_mode = 'replicate'

    out = tensor_5d
    k_x = kernel.view(1, 1, 1, 1, -1).repeat(c, 1, 1, 1, 1)
    k_y = kernel.view(1, 1, 1, -1, 1).repeat(c, 1, 1, 1, 1)
    k_z = kernel.view(1, 1, -1, 1, 1).repeat(c, 1, 1, 1, 1)
    out = F.pad(out, (radius, radius, 0, 0, 0, 0), mode=pad_mode)
    out = F.conv3d(out, k_x, groups=c)
    out = F.pad(out, (0, 0, radius, radius, 0, 0), mode=pad_mode)
    out = F.conv3d(out, k_y, groups=c)
    out = F.pad(out, (0, 0, 0, 0, radius, radius), mode=pad_mode)
    out = F.conv3d(out, k_z, groups=c)
    return out


def _gaussian_filter_nd_pytorch_backend(
    tensor,
    sigma,
    mode='reflect',
    gaussian_backend='cuda',  # Default to cuda to avoid MONAI grid stripes
    monai_roi_size=96,
    monai_overlap=0.25,
    monai_sw_batch_size=1,
):
    """GPU Gaussian filter backend selector for 3D decomposition."""
    if gaussian_backend == 'cuda' or tensor.ndim != 3:
        return _gaussian_filter_nd_pytorch(tensor, sigma, mode=mode)
    if gaussian_backend != 'monai':
        raise ValueError("gaussian_backend must be one of {'cuda', 'monai'}")
    if not _ensure_monai():
        return _gaussian_filter_nd_pytorch(tensor, sigma, mode=mode)

    radius = int(4.0 * float(sigma) + 0.5)
    min_support = 2 * radius + 1
    roi = monai_roi_size
    if isinstance(roi, int):
        roi = (roi, roi, roi)
    roi = tuple(max(1, min(int(r), int(s))) for r, s in zip(roi, tensor.shape))
    roi = tuple(max(r, min(int(s), int(min_support))) for r, s in zip(roi, tensor.shape))
    volume = tensor.unsqueeze(0).unsqueeze(0)

    if any(min_support >= int(r) for r in roi) or all(int(r) == int(s) for r, s in zip(roi, tensor.shape)):
        raise _MonaiScaleCutoff(
            "Requested Gaussian support is too large for MONAI ROI/window blending under memory limits."
        )

    def _predictor(patch):
        return _gaussian_filter_3d_pytorch_batched(patch, sigma=sigma, mode=mode)

    out = sliding_window_inference(
        inputs=volume,
        roi_size=roi,
        sw_batch_size=int(monai_sw_batch_size),
        predictor=_predictor,
        overlap=float(monai_overlap),
        mode='gaussian',
    )
    return out.squeeze(0).squeeze(0)


def _zoom_tensor(tensor, zoom_factor):
    dims = tensor.dim()
    out = tensor.unsqueeze(0).unsqueeze(0)

    if dims == 1:
        interp_mode = 'linear'
    elif dims == 2:
        interp_mode = 'bilinear'
    elif dims == 3:
        interp_mode = 'trilinear'
    else:
        interp_mode = 'nearest'

    kwargs = {'align_corners': False} if interp_mode != 'nearest' else {}
    out = F.interpolate(out, scale_factor=zoom_factor, mode=interp_mode, **kwargs)
    return out.squeeze(0).squeeze(0)


def _downsample_mean_numpy(arr, factor):
    if factor <= 1:
        return arr
    if abs(factor - int(round(factor))) > 1e-9:
        raise ValueError("Average-pool downsampling requires an integer zoom_factor.")
    f = int(round(factor))
    spatial_shape = tuple(int(s) for s in arr.shape)
    pooled_shape = tuple((s // f) for s in spatial_shape)
    if any(p == 0 for p in pooled_shape):
        raise ValueError("zoom_factor is too large for at least one input dimension.")
    cropped_shape = tuple(p * f for p in pooled_shape)
    cropped = arr[tuple(slice(0, c) for c in cropped_shape)]
    reshape_dims = []
    for p in pooled_shape:
        reshape_dims.extend([p, f])
    pooled = cropped.reshape(tuple(reshape_dims)).mean(axis=tuple(range(1, 2 * arr.ndim, 2)))
    return pooled.astype(arr.dtype, copy=False)


def _downsample_mean_tensor(tensor, factor):
    if factor <= 1:
        return tensor
    if abs(factor - int(round(factor))) > 1e-9:
        raise ValueError("Average-pool downsampling requires an integer zoom_factor.")
    f = int(round(factor))
    ndim = int(tensor.ndim)
    if ndim not in (1, 2, 3):
        raise ValueError("Average-pool downsampling currently supports 1D/2D/3D tensors.")

    pooled_shape = tuple((int(s) // f) for s in tensor.shape)
    if any(p == 0 for p in pooled_shape):
        raise ValueError("zoom_factor is too large for at least one input dimension.")
    cropped_shape = tuple(p * f for p in pooled_shape)
    x = tensor[tuple(slice(0, c) for c in cropped_shape)].unsqueeze(0).unsqueeze(0)
    if ndim == 1:
        out = F.avg_pool1d(x, kernel_size=f, stride=f)
    elif ndim == 2:
        out = F.avg_pool2d(x, kernel_size=f, stride=f)
    else:
        out = F.avg_pool3d(x, kernel_size=f, stride=f)
    return out.squeeze(0).squeeze(0)


def _map_low_scale_to_stage2_channel(stage2_scales, low_scale):
    if len(stage2_scales) == 0:
        raise ValueError("Stage-2 scales are empty; cannot map Stage-1 channels.")
    idx = int(np.searchsorted(stage2_scales, float(low_scale), side='left'))
    if idx >= len(stage2_scales):
        idx = len(stage2_scales) - 1
    return idx


def _prime_factors(n):
    factors = []
    d = 2
    x = int(n)
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def _split_counts_for_n_chunk(shape, n_chunk):
    ndim = len(shape)
    counts = [1] * ndim
    for f in sorted(_prime_factors(n_chunk), reverse=True):
        idx = int(np.argmax([shape[i] / float(counts[i]) for i in range(ndim)]))
        counts[idx] *= int(f)
    return tuple(counts)


def _build_overlap_chunks(data_shape, n_chunk, overlap):
    shape = tuple(int(d) for d in data_shape)
    total_voxels = int(np.prod(shape))
    if n_chunk <= 1 or total_voxels <= 1:
        whole = tuple(slice(0, d) for d in shape)
        return [{"chunk_slices": whole, "valid_slices": whole, "output_slices": whole}]
    if n_chunk > total_voxels:
        raise ValueError(f"n_chunk ({n_chunk}) cannot exceed total number of elements ({total_voxels}).")

    split_counts = _split_counts_for_n_chunk(shape, int(n_chunk))
    bins_per_dim = []
    for dim, count in zip(shape, split_counts):
        if count > dim:
            raise ValueError(f"n_chunk={n_chunk} creates too many splits for shape {shape}.")
        idx_splits = np.array_split(np.arange(dim), count)
        bins = [(int(seg[0]), int(seg[-1]) + 1) for seg in idx_splits if len(seg) > 0]
        bins_per_dim.append(bins)

    chunks = []
    for bins in product(*bins_per_dim):
        output_starts = tuple(b[0] for b in bins)
        output_ends = tuple(b[1] for b in bins)
        chunk_starts = tuple(max(0, s - overlap) for s in output_starts)
        chunk_ends = tuple(min(dim, e + overlap) for e, dim in zip(output_ends, shape))
        valid_starts = tuple(s - cs for s, cs in zip(output_starts, chunk_starts))
        valid_ends = tuple(vs + (oe - os) for vs, oe, os in zip(valid_starts, output_ends, output_starts))
        chunks.append(
            {
                "chunk_slices": tuple(slice(cs, ce) for cs, ce in zip(chunk_starts, chunk_ends)),
                "valid_slices": tuple(slice(vs, ve) for vs, ve in zip(valid_starts, valid_ends)),
                "output_slices": tuple(slice(os, oe) for os, oe in zip(output_starts, output_ends)),
            }
        )
    return chunks


def _resolve_chunk_count_for_scale(data_shape, requested_n_chunk, overlap_pixels):
    n_chunk = int(requested_n_chunk)
    if n_chunk <= 1:
        return 1

    shape = tuple(int(d) for d in data_shape)
    split_counts = _split_counts_for_n_chunk(shape, n_chunk)
    for dim, n_split in zip(shape, split_counts):
        block_size = int(math.ceil(float(dim) / float(n_split)))
        if block_size <= 2 * int(overlap_pixels):
            return 1
    return n_chunk


def _available_bytes_fallback():
    page_size = os.sysconf("SC_PAGE_SIZE")
    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * avail_pages)


def _get_available_accelerator_bytes(device):
    if not TORCH_AVAILABLE:
        return None

    if device.type == "cuda":
        idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        return int(free_bytes)

    if device.type == "mps":
        if hasattr(torch.mps, "recommended_max_memory"):
            rec = int(torch.mps.recommended_max_memory())
            if hasattr(torch.mps, "driver_allocated_memory"):
                used = int(torch.mps.driver_allocated_memory())
                return max(0, rec - used)
            return rec
        try:
            return _available_bytes_fallback()
        except Exception:
            return None

    return None


def _auto_n_chunk_from_budget(data_shape, max_scale, n_overlap, bytes_budget, dtype):
    if bytes_budget is None or bytes_budget <= 0:
        return 1

    vram_limit_gb = float(bytes_budget) / float(1024 ** 3)
    np_dtype = np.float32
    if TORCH_AVAILABLE and isinstance(dtype, torch.dtype):
        if dtype == torch.float16:
            np_dtype = np.float16
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            np_dtype = np.float32
    else:
        np_dtype = np.dtype(dtype)

    plan = plan_vram_chunks(
        data_shape=tuple(int(d) for d in data_shape),
        max_scale=float(max_scale),
        vram_limit_gb=float(vram_limit_gb),
        overlap_factor=float(n_overlap),
        dtype=np_dtype,
        memory_factor=4.0,
    )
    return max(1, int(len(plan["chunks"])))


def _chunked_diffusion_step(
    current_data,
    kernel_size,
    overlap_pixels,
    n_chunk,
    use_gpu,
    sm_mode,
    gaussian_backend,
    monai_roi_size,
    monai_overlap,
    monai_sw_batch_size,
):
    chunks = _build_overlap_chunks(current_data.shape, int(n_chunk), int(overlap_pixels))

    def _linear_feather_weights(chunk_shape, valid_slices):
        """Create an N-D separable linear ramp mask for overlap feathering."""
        w = np.ones(tuple(int(v) for v in chunk_shape), dtype=np.float32)
        ndim = len(chunk_shape)
        for axis in range(ndim):
            axis_len = int(chunk_shape[axis])
            vsl = valid_slices[axis]
            left = int(vsl.start)
            right = int(axis_len - int(vsl.stop))
            axis_w = np.ones(axis_len, dtype=np.float32)
            if left > 0:
                axis_w[:left] = np.linspace(0.0, 1.0, num=left, endpoint=False, dtype=np.float32)
            if right > 0:
                axis_w[-right:] = np.linspace(1.0, 0.0, num=right, endpoint=False, dtype=np.float32)
            reshape = [1] * ndim
            reshape[axis] = axis_len
            w *= axis_w.reshape(reshape)
        return w

    if use_gpu:
        diff_accum = torch.zeros_like(current_data)
        weight_accum = torch.zeros_like(current_data)
        for ch in chunks:
            chunk_data = current_data[ch["chunk_slices"]]
            smooth = _gaussian_filter_nd_pytorch_backend(
                chunk_data,
                kernel_size,
                mode=sm_mode,
                gaussian_backend=gaussian_backend,
                monai_roi_size=monai_roi_size,
                monai_overlap=monai_overlap,
                monai_sw_batch_size=monai_sw_batch_size,
            )
            diff_chunk = chunk_data - smooth
            weight_np = _linear_feather_weights(chunk_data.shape, ch["valid_slices"])
            weight = torch.from_numpy(weight_np).to(device=chunk_data.device, dtype=diff_chunk.dtype)
            diff_accum[ch["chunk_slices"]] += diff_chunk * weight
            weight_accum[ch["chunk_slices"]] += weight
        eps = torch.finfo(diff_accum.dtype).eps if diff_accum.dtype.is_floating_point else 1e-6
        return diff_accum / torch.clamp(weight_accum, min=eps)

    diff_accum = np.zeros_like(current_data)
    weight_accum = np.zeros_like(current_data)
    for ch in chunks:
        chunk_data = current_data[ch["chunk_slices"]]
        smooth = ndimage.gaussian_filter(chunk_data, kernel_size, mode=sm_mode)
        diff_chunk = chunk_data - smooth
        weight = _linear_feather_weights(chunk_data.shape, ch["valid_slices"]).astype(diff_chunk.dtype, copy=False)
        diff_accum[ch["chunk_slices"]] += diff_chunk * weight
        weight_accum[ch["chunk_slices"]] += weight
    eps = np.finfo(diff_accum.dtype).eps if np.issubdtype(diff_accum.dtype, np.floating) else 1e-6
    return diff_accum / np.maximum(weight_accum, eps)


# =============================================================================
# CORE ENGINE
# =============================================================================
def constrained_multiscale_decomposition(data, scales, e_rel=3e-2, sm_mode='reflect',
                                         constrained=True, inverted=False, use_gpu=False, verbose=True,
                                         gaussian_backend='cuda',  # Use CUDA mathematically exact chunks instead of MONAI
                                         monai_roi_size=96, monai_overlap=0.25,
                                         monai_sw_batch_size=1,
                                         n_chunk=1, n_overlap=4.0):

    if use_gpu and not TORCH_AVAILABLE:
        if verbose:
            print("Warning: PyTorch not found. Falling back to CPU execution.")
        use_gpu = False
    if int(n_chunk) < 1:
        raise ValueError("n_chunk must be >= 1.")
    if float(n_overlap) <= 0:
        raise ValueError("n_overlap must be > 0.")

    is_tensor = use_gpu and isinstance(data, torch.Tensor)
    if not is_tensor and hasattr(data, 'size') and data.size == 0:
        raise ValueError("Input data array is empty")
    elif is_tensor and data.numel() == 0:
        raise ValueError("Input data tensor is empty")

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
    # GPU (PyTorch) Execution
    # ---------------------------------------------------------
    if use_gpu:
        current_data = data.clone()
        result = []
        t_beginning = 0.0

        for i, scale_end in enumerate(scales_array):
            channel_image = torch.zeros_like(current_data)
            t_end = scale_end**2 / 2.0

            delta_t_max = t_beginning * e_rel if t_beginning > 0 else t_end * e_rel
            if delta_t_max <= 0:
                delta_t_max = (t_end - t_beginning) * 0.1

            niter = max(1, int((t_end - t_beginning) / delta_t_max + 0.5))
            delta_t = (t_end - t_beginning) / niter
            kernel_size = (2 * delta_t) ** 0.5

            if verbose:
                print(f"Channel {i}: Scale < {scale_end:.2f} pixels, Iterations: {niter}")

            # --- FIX FOR GRID ARTIFACTS: Ensure exact chunking completely covers the gaussian filter radius ---
            raw_overlap = int(math.ceil(float(n_overlap) * float(scale_end)))
            kernel_radius = int(4.0 * kernel_size + 1.0)
            overlap_pixels = max(raw_overlap, kernel_radius)  # Strictly guarantee mathematical safety
            # ------------------------------------------------------------------------------------------------

            effective_n_chunk = _resolve_chunk_count_for_scale(
                current_data.shape, int(n_chunk), overlap_pixels
            )
            if verbose and int(n_chunk) > 1 and effective_n_chunk == 1:
                print(
                    f"Channel {i}: disabling chunking at large scale "
                    f"(requested n_chunk={int(n_chunk)}, overlap={overlap_pixels})."
                )

            for _ in range(niter):
                try:
                    if effective_n_chunk > 1:
                        diff_raw = _chunked_diffusion_step(
                            current_data=current_data,
                            kernel_size=kernel_size,
                            overlap_pixels=overlap_pixels,
                            n_chunk=effective_n_chunk,
                            use_gpu=True,
                            sm_mode=sm_mode,
                            gaussian_backend=gaussian_backend,
                            monai_roi_size=monai_roi_size,
                            monai_overlap=monai_overlap,
                            monai_sw_batch_size=monai_sw_batch_size,
                        )
                    else:
                        smooth_image = _gaussian_filter_nd_pytorch_backend(
                            current_data,
                            kernel_size,
                            mode=sm_mode,
                            gaussian_backend=gaussian_backend,
                            monai_roi_size=monai_roi_size,
                            monai_overlap=monai_overlap,
                            monai_sw_batch_size=monai_sw_batch_size,
                        )
                        diff_raw = current_data - smooth_image
                except _MonaiScaleCutoff:
                    if verbose:
                        print(
                            f"Channel {i}: cutoff triggered at scale {scale_end:.2f} "
                            f"(MONAI ROI/window limit reached)."
                        )
                    return result, current_data.cpu()

                if constrained:
                    if not inverted:
                        mask = (diff_raw * current_data) > 0
                    else:
                        mask = (diff_raw * current_data) < 0
                    diff_raw.mul_(mask)

                channel_image.add_(diff_raw)
                current_data.sub_(diff_raw)

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

        for i, scale_end in enumerate(scales_array):
            channel_image = np.zeros_like(current_data)
            t_end = scale_end**2 / 2.0

            if t_beginning > 0:
                delta_t_max = t_beginning * e_rel
            else:
                delta_t_max = t_end * e_rel
            if delta_t_max <= 0:
                delta_t_max = (t_end - t_beginning) * 0.1

            niter = max(1, int((t_end - t_beginning) / delta_t_max + 0.5))
            delta_t = (t_end - t_beginning) / niter
            kernel_size = np.sqrt(2 * delta_t)

            if verbose:
                print(f"Channel {i}: Scale < {scale_end:.2f} pixels, Iterations: {niter}")

            # CPU safety fix for grid
            raw_overlap = int(math.ceil(float(n_overlap) * float(scale_end)))
            kernel_radius = int(4.0 * kernel_size + 1.0)
            overlap_pixels = max(raw_overlap, kernel_radius)

            effective_n_chunk = _resolve_chunk_count_for_scale(
                current_data.shape, int(n_chunk), overlap_pixels
            )

            for _ in range(niter):
                if effective_n_chunk > 1:
                    diff_raw = _chunked_diffusion_step(
                        current_data=current_data,
                        kernel_size=kernel_size,
                        overlap_pixels=overlap_pixels,
                        n_chunk=effective_n_chunk,
                        use_gpu=False,
                        sm_mode=sm_mode,
                        gaussian_backend=gaussian_backend,
                        monai_roi_size=monai_roi_size,
                        monai_overlap=monai_overlap,
                        monai_sw_batch_size=monai_sw_batch_size,
                    )
                else:
                    smooth_image = ndimage.gaussian_filter(current_data, kernel_size, mode=sm_mode)
                    diff_raw = current_data - smooth_image

                if constrained:
                    if not inverted:
                        mask = (diff_raw * current_data) > 0
                    else:
                        mask = (diff_raw * current_data) < 0
                    diff_raw *= mask

                channel_image += diff_raw
                current_data -= diff_raw

            result.append(channel_image)
            t_beginning = t_end

        return result, current_data


# =============================================================================
# WRAPPER
# =============================================================================
def constrained_diffusion_decomposition(
    data,
    num_channels=None, max_scale=None, min_scale=1,
    max_n_channels=None,
    mode='log', log_scale_base=2.0, linear_scale_step=None,
    e_rel=3e-2, sm_mode='reflect', up_sample=True,
    switch_scale=5.0, zoom_factor=2.0,
    upsample_scale=None,
    constrained=True, inverted=False, return_scales=False,
    use_gpu=False, device=None, verbose=True,
    gaussian_backend='cuda',  # Default to CUDA for native seamless VRAM chunking
    monai_roi_size=96, monai_overlap=0.25, monai_sw_batch_size=1,
    n_chunk=None, n_upsample_chunk=None, n_overlap=4.0
):
    if use_gpu and not TORCH_AVAILABLE:
        if verbose:
            print("Warning: PyTorch not found. Falling back to CPU execution.")
        use_gpu = False
    if upsample_scale is not None:
        zoom_factor = float(upsample_scale)
    if zoom_factor <= 0:
        raise ValueError("upsample_scale/zoom_factor must be > 0.")
    if gaussian_backend not in {'cuda', 'monai'}:
        raise ValueError("gaussian_backend must be one of {'cuda', 'monai'}")
    if n_chunk is not None and int(n_chunk) < 1:
        raise ValueError("n_chunk must be >= 1.")
    if n_upsample_chunk is not None and int(n_upsample_chunk) < 1:
        raise ValueError("n_upsample_chunk must be >= 1.")
    if float(n_overlap) <= 0:
        raise ValueError("n_overlap must be > 0.")

    is_numpy = isinstance(data, np.ndarray)

    if use_gpu:
        target_device = _get_optimal_device(device)
        if is_numpy:
            # Normalise byte order: FITS files are often big-endian;
            # torch.from_numpy rejects non-native byte order.
            if data.dtype.byteorder not in ('=', '|'):
                data = data.astype(data.dtype.newbyteorder('='), copy=False)
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

    if max_n_channels is not None:
        scale_edges = scale_edges[: int(max_n_channels)]

    if len(scale_edges) == 0:
        raise ValueError("Scale generation failed.")

    stage1_scale_edges = scale_edges.copy()
    stage2_scale_edges = scale_edges.copy()
    if up_sample:
        stage2_scale_edges = stage2_scale_edges[stage2_scale_edges >= 2.0]
        if len(stage2_scale_edges) == 0:
            raise ValueError("Stage-2 scales are empty after enforcing minimum 2-pixel start.")

    # --- THIS BLOCK SAVES YOUR VRAM WITHOUT NEEDING MONAI ---
    n_dim = int(current_data.ndim)
    bytes_budget = None
    if use_gpu and target_device.type in {"cuda", "mps"}:
        available_bytes = _get_available_accelerator_bytes(target_device)
        if available_bytes is not None:
            bytes_budget = int(max(1, available_bytes // 2))
            if verbose:
                gb = bytes_budget / float(1024 ** 3)
                print(f"Auto-chunk memory budget: {gb:.2f} GiB (50% of available {target_device.type.upper()} memory).")

    resolved_n_chunk = int(n_chunk) if n_chunk is not None else 1
    if n_chunk is None and use_gpu and bytes_budget is not None:
        resolved_n_chunk = _auto_n_chunk_from_budget(
            data_shape=current_data.shape,
            max_scale=float(np.max(stage2_scale_edges if up_sample else scale_edges)),
            n_overlap=float(n_overlap),
            bytes_budget=int(bytes_budget),
            dtype=current_data.dtype,
        )

    resolved_n_upsample_chunk = int(n_upsample_chunk) if n_upsample_chunk is not None else (2 ** n_dim)
    if verbose:
        if n_chunk is None:
            print(f"Resolved n_chunk={resolved_n_chunk} (auto natively slicing arrays for RAM safety).")
        if n_upsample_chunk is None:
            mode = "auto on GPU" if use_gpu else "default CPU"
            print(f"Resolved n_upsample_chunk={resolved_n_upsample_chunk} ({mode}).")
    # -------------------------------------------------------

    core_kwargs = {
        'e_rel': e_rel, 'sm_mode': sm_mode,
        'constrained': constrained, 'inverted': inverted if constrained else False,
        'use_gpu': use_gpu, 'verbose': verbose,
        'gaussian_backend': gaussian_backend,
        'monai_roi_size': monai_roi_size,
        'monai_overlap': monai_overlap,
        'monai_sw_batch_size': monai_sw_batch_size,
        'n_overlap': n_overlap,
    }

    results = []
    residual = current_data

    if up_sample:
        scales_small = stage1_scale_edges[stage1_scale_edges <= switch_scale]
        n_small_scales = int(len(scales_small))
        results_small = []
        low_equiv_scales = None

        if n_small_scales > 0:
            if verbose:
                print(f"\n--- STAGE 1: High-res decomposition (scales <= {switch_scale}) ---")
            upsampled_scales = np.unique(np.maximum(scales_small * zoom_factor, zoom_factor))
            low_equiv_scales = upsampled_scales / float(zoom_factor)

            if use_gpu:
                upsampled_data = _zoom_tensor(current_data, zoom_factor)
                if n_upsample_chunk is None and bytes_budget is not None:
                    resolved_n_upsample_chunk = _auto_n_chunk_from_budget(
                        data_shape=upsampled_data.shape,
                        max_scale=float(np.max(upsampled_scales)),
                        n_overlap=float(n_overlap),
                        bytes_budget=int(bytes_budget),
                        dtype=upsampled_data.dtype,
                    )
                    if verbose:
                        print(f"Resolved n_upsample_chunk={resolved_n_upsample_chunk} (auto for upsampled stage).")
                results_small_up, residual_up = constrained_multiscale_decomposition(
                    upsampled_data, upsampled_scales, n_chunk=int(resolved_n_upsample_chunk), **core_kwargs
                )

                results_small = [_downsample_mean_tensor(res.to(target_device), zoom_factor).cpu() for res in results_small_up]
                current_data = _downsample_mean_tensor(residual_up.to(target_device), zoom_factor)
            else:
                upsampled_data = ndimage.zoom(current_data, zoom_factor, order=1)
                results_small_up, residual_up = constrained_multiscale_decomposition(
                    upsampled_data, upsampled_scales, n_chunk=int(resolved_n_upsample_chunk), **core_kwargs
                )
                results_small = [_downsample_mean_numpy(res, zoom_factor) for res in results_small_up]
                current_data = _downsample_mean_numpy(residual_up, zoom_factor)

        if len(stage2_scale_edges) > 0:
            if verbose:
                print(f"\n--- STAGE 2: Fixed-grid decomposition (full low-res grid) ---")
            results_large, residual_large = constrained_multiscale_decomposition(
                current_data, stage2_scale_edges, n_chunk=int(resolved_n_chunk), **core_kwargs
            )
            if n_small_scales > 0:
                for i in range(n_small_scales):
                    j = _map_low_scale_to_stage2_channel(stage2_scale_edges, low_equiv_scales[i])
                    if verbose:
                        print(
                            f"Merge Stage-1 channel {i} (low-equivalent scale {low_equiv_scales[i]:.2f}) "
                            f"-> Stage-2 channel {j} (scale {stage2_scale_edges[j]:.2f})."
                        )
                    results_large[j] = results_large[j] + results_small[i]
            results.extend(results_large)
            residual = residual_large
        else:
            residual = current_data
    else:
        if verbose:
            print('\n--- Performing standard fixed-grid decomposition ---')
        results, residual = constrained_multiscale_decomposition(
            current_data, scale_edges, n_chunk=int(resolved_n_chunk), **core_kwargs
        )

    if use_gpu and is_numpy:
        results = [r.numpy() for r in results]
        if isinstance(residual, torch.Tensor):
            residual = residual.cpu().numpy()

    if return_scales:
        return results, residual, (stage2_scale_edges if up_sample else scale_edges)
    return results, residual
