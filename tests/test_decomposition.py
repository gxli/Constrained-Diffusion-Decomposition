"""
This script provides a comprehensive set of examples for the
`constrained_diffusion_decomposition` function.

The examples are ordered to highlight key features:
1.  Constrained vs. Unconstrained (1D): Shows how the constraint removes artifacts.
2.  Inverted vs. Standard (1D): Shows how to extract negative features ("holes").
3.  Upsampled vs. Fixed-Grid (1D): Shows how upsampling improves detail resolution.
4.  Unconstrained (2D): Visualizes the ringing artifacts in 2D.
5.  Constrained (2D): Shows the clean, recommended 2D decomposition result.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os

# =============================================================================
# DYNAMIC IMPORT BLOCK
# =============================================================================
try:
    # Assuming the function is in a file named `constrained_diffusion.py`
    from constrained_diffusion import constrained_diffusion_decomposition
except ModuleNotFoundError:
    # If the script is in an 'examples' folder, go up one level and into 'src'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, '../src/')
    sys.path.append(src_path)
    print(f"Added to path: {src_path}")
    from constrained_diffusion import constrained_diffusion_decomposition

# =============================================================================
# Helper Functions for Plotting
# =============================================================================

def plot_comparison_1d(original, res1, res2, resid1, resid2, title, label1, label2):
    """
    Generic 1D comparison plotter. Result 1 is SOLID, Result 2 is DASHED.
    (This function does not use the scales array for plotting).
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(title, fontsize=16)
    ax.plot(original, 'k-', label='Original Signal', linewidth=2.5, zorder=10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(res1) + 1))
    shift_amount = np.max(np.abs(original)) * 1.0

    for i in range(len(res1)):
        shift = (i + 1) * shift_amount
        color = colors[i]
        ax.plot(res1[i] - shift, color=color, linestyle='-', label=f'Ch {i} ({label1})', linewidth=1.5)
        ax.plot(res2[i] - shift, color=color, linestyle='--', label=f'Ch {i} ({label2})', linewidth=1.5)

    resid_shift = (len(res1) + 1) * shift_amount
    ax.plot(resid1 - resid_shift, color='blue', linestyle='-', label=f'Residual ({label1})')
    ax.plot(resid2 - resid_shift, color='blue', linestyle='--', label=f'Residual ({label2})')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('Decomposition Channels (Vertically Shifted for Clarity)')
    ax.set_xlabel('Signal Index')
    ax.set_yticks([])
    ax.set_ylabel('Component Amplitude (Shifted)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_decomposition_2d(original, results, residual, scales, title):
    """
    Plots a single 2D decomposition result using the 'seismic' colormap
    and the representative scale for each channel in the titles.
    """
    all_components = [original] + results + [residual]
    v_abs = max(abs(arr.min()) for arr in all_components if arr.size > 0)
    v_abs = max(v_abs, max(abs(arr.max()) for arr in all_components if arr.size > 0))
    vmin, vmax = -v_abs, v_abs
    
    num_plots = len(results) + 2
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    # --- Create plot info with new, more descriptive titles ---
    plot_info = [('Original Image', original)]

    # CHANGED: The `scales` variable now contains the representative scale for each channel.
    # The title should reflect this directly.
    for i, channel in enumerate(results):
        if i < len(scales):
            scale_val = scales[i]
            channel_title = f'Channel {i} (Scale ≈ {scale_val:.2f})'
            plot_info.append((channel_title, channel))
        else:
            # Fallback title if scales array doesn't match results
            plot_info.append((f'Channel {i}', channel))
    
    plot_info.append(('Residual', residual))

    # --- Plotting loop ---
    im = None
    for i, (plot_title, data) in enumerate(plot_info):
        im = axes[i].imshow(data, cmap='seismic', vmin=vmin, vmax=vmax)
        axes[i].set_title(plot_title)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    # Add a colorbar, attaching it to a specific axes to avoid layout issues
    if im and num_plots > 0:
        fig.colorbar(im, ax=axes[:num_plots].tolist(), shrink=0.8, location='right', pad=0.05)

    # Hide unused axes
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# =============================================================================
# Signal and Image Generation
# =============================================================================
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x_1d = np.linspace(0, 500, 500)
signal_1d = (gaussian(x_1d, 100, 5) + gaussian(x_1d, 250, 15) + gaussian(x_1d, 400, 30))
signal_inverted = gaussian(x_1d, 250, 40) - 0.7 * gaussian(x_1d, 250, 8)
y_2d, x_2d = np.mgrid[0:128, 0:128]
image_2d = (gaussian(np.sqrt((x_2d-40)**2 + (y_2d-40)**2), 0, 3) + 
            gaussian(np.sqrt((x_2d-90)**2 + (y_2d-90)**2), 0, 20))

# =============================================================================
# TEST CASES
# =============================================================================

# --- Test Case 1: Constrained vs. Unconstrained Decomposition (1D) ---
print("--- Running Test Case 1: Constrained vs. Unconstrained (1D Log) ---")
res_con, resid_con, sc = constrained_diffusion_decomposition(
    data=signal_1d, max_scale=40, mode='log', constrained=True, up_sample=True, return_scales=True)
# NEW: Print the returned scales
print(f"--> Representative Scales: {np.round(sc, 2)}")
res_uncon, resid_uncon, _ = constrained_diffusion_decomposition(
    data=signal_1d, max_scale=40, mode='log', constrained=False, up_sample=True, return_scales=True)
plot_comparison_1d(signal_1d, res_con, res_uncon, resid_con, resid_uncon,
                   'Constrained vs. Unconstrained (1D Log)', 'Constrained', 'Unconstrained')

# --- Test Case 2: Inverted vs. Standard Decomposition ---
print("\n--- Running Test Case 2: Inverted vs. Standard (1D Log) ---")
res_std, resid_std, sc = constrained_diffusion_decomposition(
    data=signal_inverted, max_scale=50, mode='log', inverted=False, up_sample=True, return_scales=True)
# NEW: Print the returned scales
print(f"--> Representative Scales: {np.round(sc, 2)}")
res_inv, resid_inv, _ = constrained_diffusion_decomposition(
    data=signal_inverted, max_scale=50, mode='log', inverted=True, up_sample=True, return_scales=True)
plot_comparison_1d(signal_inverted, res_inv, res_std, resid_inv, resid_std, 
                   'Inverted vs. Standard (1D Log)', 'Inverted', 'Standard')

# --- Test Case 3: Upsampled vs. Fixed-Grid Decomposition ---
print("\n--- Running Test Case 3: Upsampled vs. Fixed-Grid (1D Log) ---")
res_up, resid_up, sc = constrained_diffusion_decomposition(
    data=signal_1d, max_scale=40, mode='log', up_sample=True, return_scales=True)
# NEW: Print the returned scales
print(f"--> Representative Scales: {np.round(sc, 2)}")
res_noup, resid_noup, _ = constrained_diffusion_decomposition(
    data=signal_1d, max_scale=40, mode='log', up_sample=False, return_scales=True)
plot_comparison_1d(signal_1d, res_up, res_noup, resid_up, resid_noup, 
                   'Upsampled vs. Fixed-Grid (1D Log)', 'Upsampled', 'Fixed Grid')

# --- Test Case 4: 2D Unconstrained Decomposition ---
print("\n--- Running Test Case 4: 2D Unconstrained Decomposition ---")
res, resid, sc = constrained_diffusion_decomposition(
    data=image_2d, max_scale=32, mode='log', constrained=False, up_sample=True, return_scales=True)
# NEW: Print the returned scales
print(f"--> Representative Scales: {np.round(sc, 2)}")
plot_decomposition_2d(image_2d, res, resid, sc, '2D Decomposition (Unconstrained)')

# --- Test Case 5: 2D Constrained Decomposition ---
print("\n--- Running Test Case 5: 2D Constrained Decomposition ---")
res, resid, sc = constrained_diffusion_decomposition(
    data=image_2d, max_scale=32, mode='log', constrained=True, up_sample=True, return_scales=True)
# NEW: Print the returned scales
print(f"--> Representative Scales: {np.round(sc, 2)}")
plot_decomposition_2d(image_2d, res, resid, sc, '2D Decomposition (Constrained)')