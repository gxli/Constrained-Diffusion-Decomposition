"""
test_decomposition.py

A testing program to demonstrate and compare the 'log' and 'lin' modes of the
constrained_diffusion_decomposition function.

The script generates a complex 1D signal with overlapping positive and negative
features at different scales and visualizes the output for both modes.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# =============================================================================
# DYNAMIC IMPORT BLOCK
# =============================================================================
try:
    # Assuming the function is in a file named `constrained_diffusion.py` in a 'src' folder
    from src.constrained_diffusion import constrained_diffusion_decomposition
except ModuleNotFoundError:
    # If the script is in an 'examples' folder, go up one level and into 'src'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Corrected path joining for robustness
    src_path = os.path.join(current_dir, '../src/')
    sys.path.append(os.path.normpath(src_path))
    print(f"Added to path: {src_path}")
    from constrained_diffusion import constrained_diffusion_decomposition

# =============================================================================
# Helper Functions
# =============================================================================

def gaussian(x, mu, sig):
    """Generates a Gaussian function."""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plot_decomposition_1d(original, results, residual, scales, title):
    """
    Helper function to plot the 1D decomposition results.
    Each channel is plotted below the previous one for clarity.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(title, fontsize=18, weight='bold')

    # Plot original signal at the top
    ax.plot(original, 'k-', label='Original Signal', linewidth=2.5)

    # Calculate a vertical shift to separate the channels visually
    v_shift = (np.max(original) - np.min(original)) * 1.1
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot each decomposition channel
    for i, (channel, scale) in enumerate(zip(results, scales)):
        y_pos = -(i + 1) * v_shift
        ax.plot(channel + y_pos, color=colors[i], linewidth=2,
                label=f'Channel {i}')
        # Add a text label for the representative scale
        ax.text(5, y_pos, f'Scale ≈ {scale:.2f}', verticalalignment='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, pad=2))

    # Plot the residual at the bottom
    y_pos_resid = -(len(results) + 1) * v_shift
    ax.plot(residual + y_pos_resid, color='firebrick', linestyle='--',
            label='Residual', linewidth=2)
    ax.text(5, y_pos_resid, 'Residual', verticalalignment='bottom',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.5, pad=2))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title('Decomposition Channels (Vertically Shifted for Clarity)')
    ax.set_xlabel('Signal Index')
    ax.set_yticks([]) # Hide y-axis ticks as amplitudes are relative
    ax.set_ylabel('Component Amplitude -->')
    ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# 1. Generate Complex Test Signal
# =============================================================================
print("--- Generating test signal with overlapping features ---")
x = np.linspace(0, 1000, 1000)

# A composite signal with features at different scales and signs
# - Small, sharp positive peak (sigma=8)
# - Medium, broader negative peak (sigma=25)
# - Large, very broad positive peak (sigma=80)
signal = (
    1.0 * gaussian(x, 200, 8) +
    -0.8 * gaussian(x, 350, 25) +
    0.9 * gaussian(x, 650, 80)
)
print("Signal generated.\n")


# =============================================================================
# 2. Test Case: Logarithmic Decomposition
# =============================================================================
print("--- Running Test Case 1: Logarithmic Decomposition ---")
log_results, log_residual, log_scales = constrained_diffusion_decomposition(
    data=signal,
    min_scale=5,
    max_scale=100,
    mode='log',
    log_scale_base=2.0,
    constrained=True,
    return_scales=True
)

print(f"\n--> Returned Representative Log Scales: {np.round(log_scales, 2)}")
plot_decomposition_1d(
    signal,
    log_results,
    log_residual,
    log_scales,
    "Test Case 1: Logarithmic Decomposition (log_scale_base=2.0)"
)


# =============================================================================
# 3. Test Case: Linear Decomposition
# =============================================================================
print("\n--- Running Test Case 2: Linear Decomposition ---")
lin_results, lin_residual, lin_scales = constrained_diffusion_decomposition(
    data=signal,
    min_scale=5,
    max_scale=100,
    mode='lin',
    linear_scale_step=20, # Create channels of width 20 pixels
    constrained=True,
    return_scales=True
)

print(f"\n--> Returned Representative Linear Scales: {np.round(lin_scales, 2)}")
plot_decomposition_1d(
    signal,
    lin_results,
    lin_residual,
    lin_scales,
    "Test Case 2: Linear Decomposition (linear_scale_step=20)"
)