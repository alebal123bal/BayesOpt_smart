"""
Module for plotting heatmaps of optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt


def heatmap_plot(
    x_vector,
    y_vector,
    bounds,
    mu_objectives,
    variance_objectives,
    acquisition_values,
    x_next,
):
    """
    Plot mean and uncertainty (standard deviation) for each objective in a 2D design space.
    One row per objective: [Mean Plot | Uncertainty Plot | Acquisition Plot].

    Uses exact bounds without +1 (Python-exclusive upper limit: e.g., (0, 30) → 0 to 29 inclusive).
    Overlays evaluated points (colored by true objective value on mean plot).
    Highlights next candidate points with red stars only in the Acquisition Plot.

    Uses imshow for discrete grid plotting — no frame/artifacts.

    Args:
        x_vector (np.ndarray): Evaluated points, shape (n_eval, dim)
        y_vector (np.ndarray): Objective values, shape (n_eval, n_objectives)
        bounds (list of tuples): [(x_min, x_max), (y_min, y_max)] -- exclusive upper bound
        mu_objectives (np.ndarray): Mean predictions, shape (n_objectives, n_grid_points)
        variance_objectives (np.ndarray): Variance predictions, shape (n_objectives, n_grid_points)
        acquisition_values (np.ndarray): Acquisition values per grid point, shape (n_grid_points,)
        x_next (np.ndarray): Next candidate points to evaluate, shape (n_next, dim)
    """
    dim = x_vector.shape[1]
    n_objectives = y_vector.shape[1]

    if dim != 2:
        print("⚠️ heatmap_plot only supports 2D input space. Skipping plot.")
        return

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Generate grid using exclusive upper bounds (Pythonic: 0 to 30 → 0,1,...,29)
    x_grid = np.arange(x_min, x_max)  # shape: (30,)
    y_grid = np.arange(y_min, y_max)  # shape: (30,)
    nx, ny = len(x_grid), len(y_grid)

    # Reshape predictions to 2D grid - input_space uses "ij" indexing, so reshape as (nx, ny) then transpose
    mu_grids = [mu.reshape(nx, ny).T for mu in mu_objectives]
    var_grids = [var.reshape(nx, ny).T for var in variance_objectives]
    sigma_grids = [np.sqrt(var) for var in var_grids]
    acquisition_grid = acquisition_values.reshape(nx, ny).T

    # Create subplots with adaptive figure size
    base_width = 7  # base width per subplot
    base_height = 4  # base height per subplot
    fig_width = base_width * 3  # 3 columns
    fig_height = base_height * n_objectives

    fig, axs = plt.subplots(
        n_objectives, 3, figsize=(fig_width, fig_height), constrained_layout=True
    )
    if n_objectives == 1:
        axs = axs.reshape(1, -1)

    # Add grid size info to title
    fig.suptitle(
        f"Bayesian Optimization Surrogate Model (2D) - Grid: {nx}×{ny}",
        fontsize=16,
        fontweight="bold",
    )

    for obj_idx in range(n_objectives):
        mu = mu_grids[obj_idx]
        sigma = sigma_grids[obj_idx]
        acq = acquisition_grid

        # Define extent: [left, right, bottom, top] — matches grid exactly
        extent = [x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5]

        # Left: Mean prediction
        ax_mean = axs[obj_idx, 0]
        im1 = ax_mean.imshow(
            mu,
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="auto",
            interpolation="none",
        )
        ax_mean.set_title(f"Objective {obj_idx}: Mean Prediction (μ)", fontsize=13)
        # Scatter evaluated points (x,y) — already in integer coordinates
        n_eval_points = len(x_vector)
        scatter_size = max(
            20, min(50, 500 // max(n_eval_points, 10))
        )  # Adaptive point size
        _ = ax_mean.scatter(
            x_vector[:, 0],
            x_vector[:, 1],
            c=y_vector[:, obj_idx],
            cmap="plasma",
            s=scatter_size,
            edgecolors="black",
            linewidth=0.7,
            label="Evaluated Points",
        )

        # Add point numbers if not too many points
        if n_eval_points <= 20:
            for i, (x, y) in enumerate(x_vector):
                ax_mean.annotate(
                    str(i),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )
        ax_mean.set_xlabel("x")
        ax_mean.set_ylabel("y")
        ax_mean.legend(loc="upper right", fontsize=10)
        ax_mean.grid(True, linestyle=":", alpha=0.5)
        plt.colorbar(im1, ax=ax_mean, shrink=0.8, label="Objective Value")

        # Middle: Uncertainty (σ)
        ax_uncert = axs[obj_idx, 1]
        im2 = ax_uncert.imshow(
            sigma,
            origin="lower",
            extent=extent,
            cmap="hot",
            aspect="auto",
            interpolation="none",
        )
        ax_uncert.set_title(f"Objective {obj_idx}: Uncertainty (σ)", fontsize=13)
        ax_uncert.scatter(
            x_vector[:, 0],
            x_vector[:, 1],
            c="white",
            edgecolors="black",
            s=50,
            linewidth=0.7,
            label="Evaluated Points",
        )
        ax_uncert.set_xlabel("x")
        ax_uncert.set_ylabel("y")
        ax_uncert.legend(loc="upper right", fontsize=10)
        ax_uncert.grid(True, linestyle=":", alpha=0.5)
        plt.colorbar(im2, ax=ax_uncert, shrink=0.8, label="Standard Deviation")

        # Right: Acquisition Function
        ax_acq = axs[obj_idx, 2]
        im3 = ax_acq.imshow(
            acq,
            origin="lower",
            extent=extent,
            cmap="cividis",
            aspect="auto",
            interpolation="none",
        )
        ax_acq.set_title(
            f"Objective {obj_idx}: Acquisition Function (HVI)", fontsize=13
        )
        ax_acq.scatter(
            x_vector[:, 0],
            x_vector[:, 1],
            c="white",
            edgecolors="black",
            s=50,
            linewidth=0.7,
            label="Evaluated Points",
        )
        # Only show next points in acquisition plot
        if x_next is not None and len(x_next) > 0:
            ax_acq.scatter(
                x_next[:, 0],
                x_next[:, 1],
                marker="*",
                color="red",
                s=120,
                edgecolors="darkred",
                linewidth=2,
                label="Next Samples",
            )
        ax_acq.set_xlabel("x")
        ax_acq.set_ylabel("y")
        ax_acq.legend(loc="upper right", fontsize=10)
        ax_acq.grid(True, linestyle=":", alpha=0.5)
        plt.colorbar(im3, ax=ax_acq, shrink=0.8, label="Acquisition Value")

        # Set intelligent tick spacing based on range size
        def calculate_tick_spacing(range_size, max_ticks=10):
            """Calculate appropriate tick spacing to avoid overcrowding."""
            if range_size <= max_ticks:
                return 1
            elif range_size <= 50:
                return 5
            elif range_size <= 100:
                return 10
            elif range_size <= 200:
                return 20
            elif range_size <= 500:
                return 50
            else:
                return 100

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_tick_spacing = calculate_tick_spacing(x_range)
        y_tick_spacing = calculate_tick_spacing(y_range)

        # Generate tick positions
        x_ticks = np.arange(x_min, x_max, x_tick_spacing)
        y_ticks = np.arange(y_min, y_max, y_tick_spacing)

        # Set ticks with intelligent spacing
        for ax in [ax_mean, ax_uncert, ax_acq]:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xlim(x_min - 0.5, x_max - 0.5)
            ax.set_ylim(y_min - 0.5, y_max - 0.5)
            # Rotate x-axis labels if needed for better readability
            if x_range > 50:
                ax.tick_params(axis="x", rotation=45)
            # Set tick label font size
            ax.tick_params(axis="both", which="major", labelsize=8)

    plt.show()
