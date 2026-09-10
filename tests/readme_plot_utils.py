"""Helpers for regenerating and validating README plots in tests."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from constrained_diffusion import constrained_diffusion_decomposition


README_PLOT_NAMES = [f"notebook_image_{i}.png" for i in range(1, 8)]


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2.0 * np.power(sig, 2.0)))


def _savefig(output_dir, name):
    path = Path(output_dir) / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def _plot_component_grid(output_dir, original, results, residual, scales, title, filename, cmap="seismic"):
    components = [("Original", original)]
    components.extend(
        (f"Channel {idx}\nscale {scale:.2f}", channel)
        for idx, (channel, scale) in enumerate(zip(results, scales))
    )
    components.append(("Residual", residual))

    n_plots = len(components)
    cols = min(4, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows), squeeze=False)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=13, weight="bold")

    v_abs = max(float(np.max(np.abs(np.asarray(data)))) for _, data in components)
    vmin, vmax = (-v_abs, v_abs) if cmap == "seismic" else (None, None)

    im = None
    for ax, (label, data) in zip(axes, components):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_plots:]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes[:n_plots].tolist(), shrink=0.72, pad=0.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _savefig(output_dir, filename)


def _plot_1d_comparison(output_dir, original, left, right, left_resid, right_resid, title, left_label, right_label, filename):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(title, fontsize=13, weight="bold")
    ax.plot(original, color="black", label="Original", linewidth=2.0)

    shift = max(float(np.ptp(original)) * 1.05, 1.0)
    colors = plt.cm.viridis(np.linspace(0, 1, len(left)))
    for idx, color in enumerate(colors):
        y_pos = -(idx + 1) * shift
        ax.plot(left[idx] + y_pos, color=color, linewidth=1.4, label=f"Ch {idx} {left_label}")
        ax.plot(right[idx] + y_pos, color=color, linestyle="--", linewidth=1.4, label=f"Ch {idx} {right_label}")

    y_pos = -(len(left) + 1) * shift
    ax.plot(left_resid + y_pos, color="firebrick", linewidth=1.6, label=f"Residual {left_label}")
    ax.plot(right_resid + y_pos, color="firebrick", linestyle="--", linewidth=1.6, label=f"Residual {right_label}")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.set_yticks([])
    ax.set_xlabel("Sample")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _savefig(output_dir, filename)


def _decompose(data, use_gpu, **kwargs):
    return constrained_diffusion_decomposition(
        data,
        use_gpu=use_gpu,
        verbose=False,
        return_scales=True,
        **kwargs,
    )


def generate_readme_plots(output_dir, use_gpu=False, device=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_kwargs = {"device": device} if device is not None else {}
    y, x = np.mgrid[0:128, 0:128]
    image = (
        gaussian(np.sqrt((x - 40) ** 2 + (y - 40) ** 2), 0, 3)
        + gaussian(np.sqrt((x - 90) ** 2 + (y - 90) ** 2), 0, 20)
    ).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im = axes[0].imshow(image, cmap="viridis")
    axes[0].set_title("Two-scale input image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].plot(image[40, :], label="row 40")
    axes[1].plot(image[90, :], label="row 90")
    axes[1].set_title("Representative cuts")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    _savefig(output_dir, "notebook_image_1.png")

    results, residual, scales = _decompose(
        image,
        use_gpu=use_gpu,
        max_scale=32,
        mode="log",
        constrained=True,
        up_sample=True,
        **device_kwargs,
    )
    _plot_component_grid(output_dir, image, results, residual, scales, "2D constrained decomposition", "notebook_image_2.png")

    x_1d = np.linspace(0, 500, 500, dtype=np.float32)
    signal = gaussian(x_1d, 100, 5) + gaussian(x_1d, 250, 15) + gaussian(x_1d, 400, 30)
    con, con_resid, _ = _decompose(
        signal,
        use_gpu=use_gpu,
        max_scale=40,
        mode="log",
        constrained=True,
        up_sample=True,
        **device_kwargs,
    )
    uncon, uncon_resid, _ = _decompose(
        signal,
        use_gpu=use_gpu,
        max_scale=40,
        mode="log",
        constrained=False,
        up_sample=True,
        **device_kwargs,
    )
    _plot_1d_comparison(
        output_dir,
        signal,
        con,
        uncon,
        con_resid,
        uncon_resid,
        "Constrained vs unconstrained 1D decomposition",
        "constrained",
        "unconstrained",
        "notebook_image_3.png",
    )

    inverted_signal = gaussian(x_1d, 250, 40) - 0.7 * gaussian(x_1d, 250, 8)
    inv, inv_resid, _ = _decompose(
        inverted_signal,
        use_gpu=use_gpu,
        max_scale=50,
        mode="log",
        inverted=True,
        up_sample=True,
        **device_kwargs,
    )
    std, std_resid, _ = _decompose(
        inverted_signal,
        use_gpu=use_gpu,
        max_scale=50,
        mode="log",
        inverted=False,
        up_sample=True,
        **device_kwargs,
    )
    _plot_1d_comparison(
        output_dir,
        inverted_signal,
        inv,
        std,
        inv_resid,
        std_resid,
        "Inverted mode for hole detection",
        "inverted",
        "standard",
        "notebook_image_4.png",
    )

    up, up_resid, _ = _decompose(
        signal,
        use_gpu=use_gpu,
        max_scale=40,
        mode="log",
        up_sample=True,
        **device_kwargs,
    )
    fixed, fixed_resid, _ = _decompose(
        signal,
        use_gpu=use_gpu,
        max_scale=40,
        mode="log",
        up_sample=False,
        **device_kwargs,
    )
    _plot_1d_comparison(
        output_dir,
        signal,
        up,
        fixed,
        up_resid,
        fixed_resid,
        "Upsampled vs fixed-grid small-scale recovery",
        "upsampled",
        "fixed",
        "notebook_image_5.png",
    )

    uncon, uncon_resid, uncon_scales = _decompose(
        image,
        use_gpu=use_gpu,
        max_scale=32,
        mode="log",
        constrained=False,
        up_sample=True,
        **device_kwargs,
    )
    _plot_component_grid(output_dir, image, uncon, uncon_resid, uncon_scales, "2D unconstrained decomposition", "notebook_image_6.png")

    con, con_resid, con_scales = _decompose(
        image,
        use_gpu=use_gpu,
        max_scale=32,
        mode="log",
        constrained=True,
        up_sample=True,
        **device_kwargs,
    )
    _plot_component_grid(output_dir, image, con, con_resid, con_scales, "2D constrained decomposition", "notebook_image_7.png")

    return [output_dir / name for name in README_PLOT_NAMES]


def assert_pngs_readable(paths):
    for path in paths:
        assert path.exists(), f"Missing generated plot: {path}"
        image = mpimg.imread(path)
        assert image.ndim in (2, 3), f"Unexpected image rank for {path}: {image.shape}"
        assert image.shape[0] > 16 and image.shape[1] > 16, f"Generated plot is too small: {path}"
        assert np.isfinite(image).all(), f"Generated plot has non-finite pixels: {path}"
        assert float(np.ptp(image)) > 0.0, f"Generated plot appears blank: {path}"
