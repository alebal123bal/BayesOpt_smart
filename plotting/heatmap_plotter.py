"""
Module for plotting heatmaps of optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt


class HeatmapPlotterStatic:
    """
    Static heatmap plotter that shows one plot at a time.

    Each plot blocks execution until closed by pressing 'Q' or closing the window.
    This is the default plotter for simple use cases.
    """

    def __init__(self, bounds, n_objectives):
        """
        Initialize the static heatmap plotter.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max)] -- exclusive upper bound
            n_objectives (int): Number of objectives
        """
        self.bounds = bounds
        self.n_objectives = n_objectives

        # Grid parameters
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        x_grid = np.arange(x_min, x_max)
        y_grid = np.arange(y_min, y_max)
        self.nx, self.ny = len(x_grid), len(y_grid)
        self.extent = [x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5]

    def plot(
        self,
        x_vector,
        y_vector,
        mu_objectives,
        variance_objectives,
        acquisition_values,
        x_next=None,
    ):
        """
        Create and show a static plot (blocks until closed with 'Q' or window close).

        Args:
            x_vector (np.ndarray): Evaluated points, shape (n_eval, dim)
            y_vector (np.ndarray): Objective values, shape (n_eval, n_objectives)
            mu_objectives (np.ndarray): Mean predictions, shape (n_objectives, n_grid_points)
            variance_objectives (np.ndarray): Variance predictions, shape (n_objectives, n_grid_points)
            acquisition_values (np.ndarray): Acquisition values per grid point, shape (n_grid_points,)
            x_next (np.ndarray): Next candidate points to evaluate, shape (n_next, dim)
        """
        if x_vector.shape[1] != 2:
            print("⚠️ HeatmapPlotterStatic only supports 2D input space.")
            return

        # Create new figure for each plot
        base_width = 7
        base_height = 4
        fig_width = base_width * 3
        fig_height = base_height * self.n_objectives

        fig, axs = plt.subplots(
            self.n_objectives,
            3,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )
        if self.n_objectives == 1:
            axs = axs.reshape(1, -1)

        fig.suptitle(
            f"Bayesian Optimization Surrogate Model (2D) - Grid: {self.nx}×{self.ny}\nPress 'Q' to close",
            fontsize=16,
            fontweight="bold",
        )

        # Set up key press handler
        def on_key(event):
            if event.key.lower() == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        # Reshape predictions to 2D grid
        mu_grids = [mu.reshape(self.nx, self.ny).T for mu in mu_objectives]
        var_grids = [var.reshape(self.nx, self.ny).T for var in variance_objectives]
        sigma_grids = [np.sqrt(var) for var in var_grids]
        acquisition_grid = acquisition_values.reshape(self.nx, self.ny).T

        n_eval_points = len(x_vector)
        scatter_size = max(20, min(50, 500 // max(n_eval_points, 10)))

        for obj_idx in range(self.n_objectives):
            mu = mu_grids[obj_idx]
            sigma = sigma_grids[obj_idx]
            acq = acquisition_grid

            # --- Mean Plot ---
            ax_mean = axs[obj_idx, 0]
            im1 = ax_mean.imshow(
                mu,
                origin="lower",
                extent=self.extent,
                cmap="viridis",
                aspect="auto",
                interpolation="none",
            )
            ax_mean.set_title(f"Objective {obj_idx}: Mean Prediction (μ)", fontsize=13)
            ax_mean.scatter(
                x_vector[:, 0],
                x_vector[:, 1],
                c=y_vector[:, obj_idx],
                cmap="plasma",
                s=scatter_size,
                edgecolors="black",
                linewidth=0.7,
                label="Evaluated Points",
            )

            # Add point numbers if not too many
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
            fig.colorbar(im1, ax=ax_mean, shrink=0.8, label="Objective Value")

            # --- Uncertainty Plot ---
            ax_uncert = axs[obj_idx, 1]
            im2 = ax_uncert.imshow(
                sigma,
                origin="lower",
                extent=self.extent,
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
            fig.colorbar(im2, ax=ax_uncert, shrink=0.8, label="Standard Deviation")

            # --- Acquisition Plot ---
            ax_acq = axs[obj_idx, 2]
            im3 = ax_acq.imshow(
                acq,
                origin="lower",
                extent=self.extent,
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

            # Show next candidate points
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
            fig.colorbar(im3, ax=ax_acq, shrink=0.8, label="Acquisition Value")

            # Set ticks
            x_range = self.x_max - self.x_min
            y_range = self.y_max - self.y_min
            x_tick_spacing = self._calculate_tick_spacing(x_range)
            y_tick_spacing = self._calculate_tick_spacing(y_range)

            x_ticks = np.arange(self.x_min, self.x_max, x_tick_spacing)
            y_ticks = np.arange(self.y_min, self.y_max, y_tick_spacing)

            for ax in [ax_mean, ax_uncert, ax_acq]:
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xlim(self.x_min - 0.5, self.x_max - 0.5)
                ax.set_ylim(self.y_min - 0.5, self.y_max - 0.5)
                if x_range > 50:
                    ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="both", which="major", labelsize=8)

        plt.show(block=True)

    def _calculate_tick_spacing(self, range_size, max_ticks=10):
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

    def close(self):
        """Close any open plot windows."""
        plt.close("all")


class HeatmapPlotterDaemon:
    """
    Interactive heatmap plotter for real-time visualization during optimization.

    Maintains a live plot window that can be updated with new data
    without blocking the main optimization loop.
    """

    def __init__(self, bounds, n_objectives):
        """
        Initialize the interactive heatmap plotter.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max)] -- exclusive upper bound
            n_objectives (int): Number of objectives
        """
        self.bounds = bounds
        self.n_objectives = n_objectives

        # Data storage
        self.x_vector = None
        self.y_vector = None
        self.mu_objectives = None
        self.variance_objectives = None
        self.acquisition_values = None
        self.x_next = None

        # Grid parameters
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        x_grid = np.arange(x_min, x_max)
        y_grid = np.arange(y_min, y_max)
        self.nx, self.ny = len(x_grid), len(y_grid)
        self.extent = [x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5]

        # Enable interactive mode
        plt.ion()

        # Figure will be created on first plot() call
        self.fig = None
        self.axs = None

        # Flag to track if data has been updated
        self.data_updated = False

    def _create_figure(self):
        """Create the figure and axes layout."""
        base_width = 7
        base_height = 4
        fig_width = base_width * 3
        fig_height = base_height * self.n_objectives

        self.fig, self.axs = plt.subplots(
            self.n_objectives,
            3,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )
        if self.n_objectives == 1:
            self.axs = self.axs.reshape(1, -1)

        self.fig.suptitle(
            f"Bayesian Optimization Surrogate Model (2D) - Grid: {self.nx}×{self.ny}",
            fontsize=16,
            fontweight="bold",
        )

        # Set the window to normal size (not fullscreen)
        manager = plt.get_current_fig_manager()
        try:
            if hasattr(manager, "window"):
                if hasattr(manager.window, "state"):
                    manager.window.state("normal")
                elif hasattr(manager.window, "showNormal"):
                    manager.window.showNormal()
        except Exception:
            pass

        plt.show(block=False)
        plt.pause(0.01)

    def plot(
        self,
        x_vector,
        y_vector,
        mu_objectives,
        variance_objectives,
        acquisition_values,
        x_next=None,
    ):
        """
        Update the plot with new data (non-blocking).

        Args:
            x_vector (np.ndarray): Evaluated points, shape (n_eval, dim)
            y_vector (np.ndarray): Objective values, shape (n_eval, n_objectives)
            mu_objectives (np.ndarray): Mean predictions, shape (n_objectives, n_grid_points)
            variance_objectives (np.ndarray): Variance predictions, shape (n_objectives, n_grid_points)
            acquisition_values (np.ndarray): Acquisition values per grid point, shape (n_grid_points,)
            x_next (np.ndarray): Next candidate points to evaluate, shape (n_next, dim)
        """
        if x_vector.shape[1] != 2:
            print("⚠️ HeatmapPlotterDaemon only supports 2D input space.")
            return

        self.x_vector = x_vector
        self.y_vector = y_vector
        self.mu_objectives = mu_objectives
        self.variance_objectives = variance_objectives
        self.acquisition_values = acquisition_values
        self.x_next = x_next
        self.data_updated = True

        self._redraw()

    def _redraw(self):
        """Redraw the plots with current data."""
        if not self.data_updated:
            return

        # Create figure on first call
        if self.fig is None:
            self._create_figure()

        # Clear the entire figure
        self.fig.clf()

        # Recreate subplots
        self.axs = self.fig.subplots(self.n_objectives, 3)
        if self.n_objectives == 1:
            self.axs = self.axs.reshape(1, -1)

        self.fig.suptitle(
            f"Bayesian Optimization Surrogate Model (2D) - Grid: {self.nx}×{self.ny}",
            fontsize=16,
            fontweight="bold",
        )

        # Reshape predictions to 2D grid
        mu_grids = [mu.reshape(self.nx, self.ny).T for mu in self.mu_objectives]
        var_grids = [
            var.reshape(self.nx, self.ny).T for var in self.variance_objectives
        ]
        sigma_grids = [np.sqrt(var) for var in var_grids]
        acquisition_grid = self.acquisition_values.reshape(self.nx, self.ny).T

        n_eval_points = len(self.x_vector)
        scatter_size = max(20, min(50, 500 // max(n_eval_points, 10)))

        for obj_idx in range(self.n_objectives):
            mu = mu_grids[obj_idx]
            sigma = sigma_grids[obj_idx]
            acq = acquisition_grid

            # --- Mean Plot ---
            ax_mean = self.axs[obj_idx, 0]
            im1 = ax_mean.imshow(
                mu,
                origin="lower",
                extent=self.extent,
                cmap="viridis",
                aspect="auto",
                interpolation="none",
            )
            ax_mean.set_title(f"Objective {obj_idx}: Mean Prediction (μ)", fontsize=13)
            ax_mean.scatter(
                self.x_vector[:, 0],
                self.x_vector[:, 1],
                c=self.y_vector[:, obj_idx],
                cmap="plasma",
                s=scatter_size,
                edgecolors="black",
                linewidth=0.7,
                label="Evaluated Points",
            )

            # Add point numbers if not too many
            if n_eval_points <= 20:
                for i, (x, y) in enumerate(self.x_vector):
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
            self.fig.colorbar(im1, ax=ax_mean, shrink=0.8, label="Objective Value")

            # --- Uncertainty Plot ---
            ax_uncert = self.axs[obj_idx, 1]
            im2 = ax_uncert.imshow(
                sigma,
                origin="lower",
                extent=self.extent,
                cmap="hot",
                aspect="auto",
                interpolation="none",
            )
            ax_uncert.set_title(f"Objective {obj_idx}: Uncertainty (σ)", fontsize=13)
            ax_uncert.scatter(
                self.x_vector[:, 0],
                self.x_vector[:, 1],
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
            self.fig.colorbar(im2, ax=ax_uncert, shrink=0.8, label="Standard Deviation")

            # --- Acquisition Plot ---
            ax_acq = self.axs[obj_idx, 2]
            im3 = ax_acq.imshow(
                acq,
                origin="lower",
                extent=self.extent,
                cmap="cividis",
                aspect="auto",
                interpolation="none",
            )
            ax_acq.set_title(
                f"Objective {obj_idx}: Acquisition Function (HVI)", fontsize=13
            )
            ax_acq.scatter(
                self.x_vector[:, 0],
                self.x_vector[:, 1],
                c="white",
                edgecolors="black",
                s=50,
                linewidth=0.7,
                label="Evaluated Points",
            )

            # Show next candidate points
            if self.x_next is not None and len(self.x_next) > 0:
                ax_acq.scatter(
                    self.x_next[:, 0],
                    self.x_next[:, 1],
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
            self.fig.colorbar(im3, ax=ax_acq, shrink=0.8, label="Acquisition Value")

            # Set ticks
            x_range = self.x_max - self.x_min
            y_range = self.y_max - self.y_min
            x_tick_spacing = self._calculate_tick_spacing(x_range)
            y_tick_spacing = self._calculate_tick_spacing(y_range)

            x_ticks = np.arange(self.x_min, self.x_max, x_tick_spacing)
            y_ticks = np.arange(self.y_min, self.y_max, y_tick_spacing)

            for ax in [ax_mean, ax_uncert, ax_acq]:
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xlim(self.x_min - 0.5, self.x_max - 0.5)
                ax.set_ylim(self.y_min - 0.5, self.y_max - 0.5)
                if x_range > 50:
                    ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="both", which="major", labelsize=8)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def _calculate_tick_spacing(self, range_size, max_ticks=10):
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

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)

    def show(self, block=True):
        """
        Show the plot window.

        Args:
            block (bool): If True, blocks execution until window is closed.
        """
        plt.show(block=block)
