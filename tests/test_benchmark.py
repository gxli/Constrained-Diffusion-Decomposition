#!/usr/bin/env python3
"""
test_benchmark.py

Comprehensive test suite and benchmark for the CDD engine.
Performs CPU vs GPU benchmarks on 1D, 2D, and 3D data.
Includes specific testing for the 'inverted' mode (extracting depressions).
Saves ALL visualizations safely as PNG files.
"""

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PATH SETUP
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(os.path.normpath(src_path))

try:
    from constrained_diffusion import constrained_diffusion_decomposition, TORCH_AVAILABLE
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import constrained_diffusion from {src_path}")
    print(e)
    sys.exit(1)

# Ensure output directory exists for plots
OUTPUT_DIR = os.path.join(current_dir, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS: DATA GENERATION
# =============================================================================
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def create_2d_gaussian(shape, center, sigma, amplitude=1.0):
    y, x = np.indices(shape)
    y_c, x_c = center
    return amplitude * np.exp(-((x - x_c)**2 + (y - y_c)**2) / (2 * sigma**2))

def create_3d_gaussian(shape, center, sigma, amplitude=1.0):
    z, y, x = np.indices(shape)
    z_c, y_c, x_c = center
    return amplitude * np.exp(-((x - x_c)**2 + (y - y_c)**2 + (z - z_c)**2) / (2 * sigma**2))

def format_speedup(time_cpu, time_gpu):
    if time_gpu < time_cpu:
        return f"GPU was {time_cpu / time_gpu:.2f}x faster"
    else:
        return f"CPU was {time_gpu / time_cpu:.2f}x faster (Overhead dominated)"

# =============================================================================
# HELPER FUNCTIONS: PLOTTING
# =============================================================================
def plot_decomposition_1d(original, results, residual, scales, title, filename):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(title, fontsize=18, weight='bold')

    ax.plot(original, 'k-', label='Original Signal', linewidth=2.5)

    v_shift = (np.max(original) - np.min(original)) * 1.1
    # Handle signals with small ranges or zeros
    if v_shift == 0: v_shift = 1.0
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (channel, scale) in enumerate(zip(results, scales)):
        y_pos = -(i + 1) * v_shift
        ax.plot(channel + y_pos, color=colors[i], linewidth=2, label=f'Channel {i}')
        ax.text(5, y_pos, f'Scale ≈ {scale:.2f}', verticalalignment='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))

    y_pos_resid = -(len(results) + 1) * v_shift
    ax.plot(residual + y_pos_resid, color='firebrick', linestyle='--', label='Residual', linewidth=2)
    ax.text(5, y_pos_resid, 'Residual', verticalalignment='bottom',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title('Decomposition Channels (Vertically Shifted)')
    ax.set_yticks([]) 
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved 1D Plot: {filename}")

def plot_decomposition_2d(original, results, residual, title, filename):
    n_plots = 2 + len(results) 
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4))
    fig.suptitle(title, fontsize=16, weight='bold')
    
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]
    
    # 1. Plot Original
    im0 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Signal')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Plot Channels
    for i, channel in enumerate(results):
        im = axes[i+1].imshow(channel, cmap='inferno')
        axes[i+1].set_title(f'Channel {i}')
        axes[i+1].axis('off')
        fig.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
        
    # 3. Plot Residual
    res_idx = len(results) + 1
    im_res = axes[res_idx].imshow(residual, cmap='viridis')
    axes[res_idx].set_title('Residual')
    axes[res_idx].axis('off')
    fig.colorbar(im_res, ax=axes[res_idx], fraction=0.046, pad=0.04)
    
    # Hide any unused subplots
    for j in range(res_idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved 2D Plot: {filename}")

def plot_decomposition_3d_slice(original_3d, results_3d, residual_3d, title, filename):
    """Extracts the center slice of a 3D volume and routes it to the 2D plotter."""
    mid_z = original_3d.shape[0] // 2
    
    orig_slice = original_3d[mid_z, :, :]
    res_slices = [ch[mid_z, :, :] for ch in results_3d]
    resid_slice = residual_3d[mid_z, :, :]
    
    full_title = f"{title} (Central Z-Slice: {mid_z})"
    plot_decomposition_2d(orig_slice, res_slices, resid_slice, full_title, filename)

# =============================================================================
# BENCHMARKS
# =============================================================================
def warmup_gpu():
    if not TORCH_AVAILABLE:
        return
    print("Initializing PyTorch CUDA/MPS engine (Warmup)...")
    _ = constrained_diffusion_decomposition(np.zeros((32, 32)), max_scale=4, use_gpu=True, verbose=False)
    print("GPU Warmup Complete.\n")

def benchmark_1d():
    print("="*60)
    print("BENCHMARK 1: 1D Signal Decomposition (Standard)")
    print("="*60)
    
    x = np.linspace(0, 1000, 1000)
    signal = (1.0 * gaussian(x, 200, 8) - 0.8 * gaussian(x, 350, 25) + 0.9 * gaussian(x, 650, 80))
    params = {'data': signal, 'min_scale': 5, 'max_scale': 100, 'mode': 'log', 'log_scale_base': 2.0, 'return_scales': True, 'verbose': False}
    
    start_cpu = time.perf_counter()
    res_cpu, res_resid, scales = constrained_diffusion_decomposition(**params, use_gpu=False)
    t_cpu = time.perf_counter() - start_cpu
    print(f"  CPU Time: {t_cpu:.4f} sec")
    plot_decomposition_1d(signal, res_cpu, res_resid, scales, "1D Log Decomposition (CPU)", "1d_log_cpu.png")

    if TORCH_AVAILABLE:
        start_gpu = time.perf_counter()
        res_gpu, res_resid_gpu, scales = constrained_diffusion_decomposition(**params, use_gpu=True)
        t_gpu = time.perf_counter() - start_gpu
        print(f"  GPU Time: {t_gpu:.4f} sec")
        print(f"  {format_speedup(t_cpu, t_gpu)}")
        plot_decomposition_1d(signal, res_gpu, res_resid_gpu, scales, "1D Log Decomposition (GPU)", "1d_log_gpu.png")

def benchmark_1d_inverted():
    print("\n" + "="*60)
    print("BENCHMARK 1.1: 1D Signal Inverted Mode (Extracting Holes)")
    print("="*60)
    
    x = np.linspace(0, 1000, 1000)
    # Signal: Wide positive Gaussian (Background) with a Narrow negative Gaussian (The Hole)
    background = 1.0 * gaussian(x, 500, 150)
    hole = -0.5 * gaussian(x, 500, 15)
    signal = background + hole

    params = {
        'data': signal, 
        'min_scale': 5, 
        'max_scale': 50, 
        'mode': 'log', 
        'log_scale_base': 2.0, 
        'return_scales': True, 
        'verbose': False,
        'constrained': True
    }

    # Compare Standard vs Inverted
    print("Running Inverted vs Standard comparison...")
    
    # 1. Standard Mode (Should focus on the background peak)
    res_std, resid_std, scales = constrained_diffusion_decomposition(**params, inverted=False, use_gpu=False)
    plot_decomposition_1d(signal, res_std, resid_std, scales, "1D Standard Mode (Inverted=False)", "1d_inverted_off.png")
    
    # 2. Inverted Mode (Should focus on the hole)
    start_cpu = time.perf_counter()
    res_inv, resid_inv, scales = constrained_diffusion_decomposition(**params, inverted=True, use_gpu=False)
    t_cpu = time.perf_counter() - start_cpu
    print(f"  CPU Time (Inverted): {t_cpu:.4f} sec")
    plot_decomposition_1d(signal, res_inv, resid_inv, scales, "1D Inverted Mode (Inverted=True)", "1d_inverted_on_cpu.png")

    if TORCH_AVAILABLE:
        start_gpu = time.perf_counter()
        res_gpu, resid_gpu, _ = constrained_diffusion_decomposition(**params, inverted=True, use_gpu=True)
        t_gpu = time.perf_counter() - start_gpu
        print(f"  GPU Time (Inverted): {t_gpu:.4f} sec")
        print(f"  {format_speedup(t_cpu, t_gpu)}")
        plot_decomposition_1d(signal, res_gpu, resid_gpu, scales, "1D Inverted Mode (GPU)", "1d_inverted_on_gpu.png")

def benchmark_2d():
    print("\n" + "="*60)
    print("BENCHMARK 2: 2D Image Decomposition (256x256)")
    print("="*60)

    shape = (256, 256)
    img = np.zeros(shape, dtype=np.float32)
    img += create_2d_gaussian(shape, (128, 128), 25, 1.0)
    img += create_2d_gaussian(shape, (50, 50), 5, 0.8)
    img += create_2d_gaussian(shape, (200, 150), 40, -0.6)
    
    params = {'data': img, 'min_scale': 2, 'max_scale': 32, 'mode': 'log', 'log_scale_base': 2.0, 'verbose': False}

    start_cpu = time.perf_counter()
    cpu_res, cpu_resid = constrained_diffusion_decomposition(**params, use_gpu=False)
    t_cpu = time.perf_counter() - start_cpu
    print(f"  CPU Time: {t_cpu:.3f} sec")
    plot_decomposition_2d(img, cpu_res, cpu_resid, "2D Decomposition (CPU)", "2d_log_cpu.png")

    if TORCH_AVAILABLE:
        start_gpu = time.perf_counter()
        gpu_res, gpu_resid = constrained_diffusion_decomposition(**params, use_gpu=True)
        t_gpu = time.perf_counter() - start_gpu
        print(f"  GPU Time: {t_gpu:.3f} sec")
        print(f"  {format_speedup(t_cpu, t_gpu)}")
        plot_decomposition_2d(img, gpu_res, gpu_resid, "2D Decomposition (GPU)", "2d_log_gpu.png")

def benchmark_3d(n_cube=50):
    print("\n" + "="*60)
    print(f"BENCHMARK 3: 3D Volume Decomposition ({n_cube}x{n_cube}x{n_cube})")
    print("="*60)

    shape = (n_cube, n_cube, n_cube)
    vol = np.zeros(shape, dtype=np.float32)
    vol += create_3d_gaussian(shape, (n_cube//2, n_cube//2, n_cube//2), n_cube*0.15, 1.0)
    vol += create_3d_gaussian(shape, (n_cube//4, n_cube//4, n_cube//4), n_cube*0.05, 0.8)
    vol += create_3d_gaussian(shape, (int(n_cube*0.7), int(n_cube*0.7), int(n_cube*0.3)), n_cube*0.25, -0.6)
    
    max_scale_val = n_cube // 3
    params = {'data': vol, 'min_scale': 2, 'max_scale': max_scale_val, 'mode': 'log', 'log_scale_base': 2.0, 'verbose': False}

    print(f"Running 3D CPU (n={n_cube})...")
    start_cpu = time.perf_counter()
    cpu_res, cpu_resid = constrained_diffusion_decomposition(**params, use_gpu=False)
    t_cpu = time.perf_counter() - start_cpu
    print(f"  CPU Time: {t_cpu:.3f} sec")
    plot_decomposition_3d_slice(vol, cpu_res, cpu_resid, f"3D Decomposition CPU {n_cube}^3", f"3d_log_cpu_{n_cube}.png")

    if TORCH_AVAILABLE:
        print(f"Running 3D GPU (n={n_cube})...")
        start_gpu = time.perf_counter()
        gpu_res, gpu_resid = constrained_diffusion_decomposition(**params, use_gpu=True)
        t_gpu = time.perf_counter() - start_gpu
        print(f"  GPU Time: {t_gpu:.3f} sec")
        print(f"  {format_speedup(t_cpu, t_gpu)}")
        plot_decomposition_3d_slice(vol, gpu_res, gpu_resid, f"3D Decomposition GPU {n_cube}^3", f"3d_log_gpu_{n_cube}.png")

if __name__ == "__main__":
    warmup_gpu()
    benchmark_1d()
    benchmark_1d_inverted()  # NEW TEST CASE
    benchmark_2d()
    benchmark_3d(n_cube=50)
    print("\nAll benchmarks and plots generated successfully in /outputs/.")