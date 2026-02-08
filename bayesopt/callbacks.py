"""
Callback observers for Bayesian Optimization monitoring.

This module provides pre-built callback classes for common monitoring tasks:
- Real-time plotting
- Progress logging
- Performance monitoring
- Convergence detection
- File-based logging
- Graph saving
"""

import numpy as np
from typing import Optional, TextIO
from datetime import datetime
from pathlib import Path


class PlotterCallback:
    """Callback for real-time plotting using PyQtPlotter."""

    def __init__(self, plotter):
        """
        Initialize plotter callback.

        Args:
            plotter: PyQtPlotter instance
        """
        self.plotter = plotter

    def __call__(self, state):
        """Update plot with current optimization state."""
        if state["x_vector"].shape[1] == 2:
            self.plotter.plot(
                x_vector=state["x_vector"],
                y_vector=state["y_vector"],
                mu_objectives=state["mu_objectives"],
                variance_objectives=state["variance_objectives"],
                acquisition_values=state["acquisition_values"],
                x_next=state.get("x_next"),
            )


class ProgressLogger:
    """Callback that logs optimization progress to console and/or file."""

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize progress logger.

        Args:
            log_file: Optional file path for logging
            verbose: Whether to print to console
        """
        self.log_file = log_file
        self.verbose = verbose
        self.best_per_objective = None  # Best value for each objective
        self.best_x_per_objective = []  # Best x for each objective
        self.prev_n_evaluations = (
            0  # Track previous evaluation count for batch detection
        )
        self._file_handle: Optional[TextIO] = None

        # Initialize log file with header
        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("iteration,n_evaluations,time_total\n")

    def __call__(self, state):
        """Log current iteration progress."""
        # Get current data
        y_values = state["y_vector"]
        x_values = state["x_vector"]
        n_objectives = y_values.shape[1]
        n_evaluations = state["n_evaluations"]

        # Determine batch size from previous iteration
        if self.best_per_objective is None:
            # First iteration - all points are new
            batch_start = 0
            self.best_per_objective = np.full(n_objectives, -float("inf"))
            self.best_x_per_objective = [None] * n_objectives
            self.prev_n_evaluations = 0
        else:
            batch_start = self.prev_n_evaluations

        batch_size = n_evaluations - batch_start
        self.prev_n_evaluations = n_evaluations

        # Track best for each objective separately (across all evaluations)
        improvements = []
        for obj_idx in range(n_objectives):
            max_idx = np.argmax(y_values[:, obj_idx])
            current_best = y_values[max_idx, obj_idx]

            if current_best > self.best_per_objective[obj_idx]:
                self.best_per_objective[obj_idx] = current_best
                self.best_x_per_objective[obj_idx] = x_values[max_idx].copy()
                improvements.append(obj_idx)

        # Console logging
        if self.verbose:
            timings = state["timings"]

            # Show this batch's evaluations
            print(f"  └─ Iter {state['iteration']}: Evaluated {batch_size} point(s)")
            for i in range(batch_start, n_evaluations):
                x_str = np.array2string(
                    x_values[i], precision=1, suppress_small=True, separator=","
                )
                y_str = np.array2string(
                    y_values[i], precision=4, suppress_small=True, separator=","
                )
                print(f"      Point {i+1}: x={x_str} → objectives={y_str}")

            # Show global bests
            best_strs = []
            for obj_idx in range(n_objectives):
                x_str = np.array2string(
                    self.best_x_per_objective[obj_idx],
                    precision=1,
                    suppress_small=True,
                    separator=",",
                )
                best_strs.append(
                    f"obj{obj_idx}={self.best_per_objective[obj_idx]:.4f} at {x_str}"
                )

            msg = f"     Best so far: {', '.join(best_strs)}"
            if improvements:
                improved_str = ",".join([f"obj{i}" for i in improvements])
                msg += f" ⭐ NEW BEST [{improved_str}]"
            msg += f" ({timings['total']:.3f}s)"
            print(msg)

        # File logging
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{state['iteration']},{state['n_evaluations']},"
                    f"{state['timings']['total']:.4f}\n"
                )


class OptimizationLogger:
    """Detailed logger for optimization events (replaces print statements)."""

    def __init__(self, verbose: bool = True, show_initial: bool = False):
        """
        Initialize optimization logger.

        Args:
            verbose: Whether to print detailed logs
            show_initial: Whether to log initial points
        """
        self.verbose = verbose
        self.show_initial = show_initial
        self.iteration_count = 0

    def log_initial_evaluations(self, x_vector, y_vector, n_evaluations):
        """Log initial LHS evaluations."""
        if not self.verbose:
            return

        print(f"🚀 Starting optimization with {n_evaluations} initial evaluations.")

        if self.show_initial:
            for i in range(n_evaluations):
                print(f"🔍 Initial point {x_vector[i]} | Objectives = {y_vector[i]}")

    def __call__(self, state):
        """Log optimization iteration details."""
        if not self.verbose:
            return

        self.iteration_count += 1
        timings = state["timings"]

        # Iteration summary
        print(
            f"\n🔄 Iteration {state['iteration']} "
            f"(n_evaluations={state['n_evaluations']})"
        )

        # Next batch info
        if "x_next" in state and state["x_next"] is not None:
            print(f"🔍 Selected {len(state['x_next'])} points for next batch")

        # Timing breakdown
        print(
            f"[Timing] "
            f"Hyperparams: {timings['hyperparams']:.4f}s | "
            f"Kernels: {timings['kernels']:.4f}s | "
            f"Acquisition: {timings['acquisition']:.4f}s | "
            f"Eval: {timings['eval']:.4f}s | "
            f"TOTAL: {timings['total']:.4f}s"
        )


class PerformanceMonitor:
    """Callback that tracks and reports computation times."""

    def __init__(self):
        """Initialize performance monitor."""
        self.timings = []
        self.hyperparams_times = []
        self.kernels_times = []
        self.acquisition_times = []
        self.eval_times = []

    def __call__(self, state):
        """Track timing information."""
        timings = state["timings"]
        self.timings.append(timings["total"])
        self.hyperparams_times.append(timings["hyperparams"])
        self.kernels_times.append(timings["kernels"])
        self.acquisition_times.append(timings["acquisition"])
        self.eval_times.append(timings["eval"])

    def summary(self):
        """Print performance summary."""
        print("\n📊 Performance Summary:")
        print(f"  Total iterations: {len(self.timings)}")
        print(f"  Average time per iteration: {np.mean(self.timings):.3f}s")
        print(f"  Total time: {np.sum(self.timings):.2f}s")
        print("\n  Breakdown (average):")
        print(
            f"    Hyperparams: {np.mean(self.hyperparams_times):.4f}s "
            f"({100*np.mean(self.hyperparams_times)/np.mean(self.timings):.1f}%)"
        )
        print(
            f"    Kernels: {np.mean(self.kernels_times):.4f}s "
            f"({100*np.mean(self.kernels_times)/np.mean(self.timings):.1f}%)"
        )
        print(
            f"    Acquisition: {np.mean(self.acquisition_times):.4f}s "
            f"({100*np.mean(self.acquisition_times)/np.mean(self.timings):.1f}%)"
        )
        print(
            f"    Evaluation: {np.mean(self.eval_times):.4f}s "
            f"({100*np.mean(self.eval_times)/np.mean(self.timings):.1f}%)"
        )


class ConvergenceChecker:
    """Callback that monitors convergence criteria."""

    def __init__(
        self, tolerance: float = 1e-4, patience: int = 3, verbose: bool = True
    ):
        """
        Initialize convergence checker.

        Args:
            tolerance: Minimum improvement threshold per objective
            patience: Number of iterations without improvement before signaling
            verbose: Whether to print convergence messages
        """
        self.tolerance = tolerance
        self.patience = patience
        self.verbose = verbose
        self.prev_best_per_obj = None
        self.no_improvement_counts = None
        self.converged = False

    def __call__(self, state):
        """Check for convergence on each objective."""
        y_values = state["y_vector"]
        n_objectives = y_values.shape[1]

        # Initialize tracking on first iteration
        if self.prev_best_per_obj is None:
            self.prev_best_per_obj = np.full(n_objectives, float("inf"))
            self.no_improvement_counts = np.zeros(n_objectives, dtype=int)

        # Check each objective separately
        for obj_idx in range(n_objectives):
            current_best = np.min(y_values[:, obj_idx])
            improvement = self.prev_best_per_obj[obj_idx] - current_best

            if improvement < self.tolerance:
                self.no_improvement_counts[obj_idx] += 1
            else:
                self.no_improvement_counts[obj_idx] = 0

            self.prev_best_per_obj[obj_idx] = current_best

        # Signal convergence if all objectives have stalled
        if np.all(self.no_improvement_counts >= self.patience) and not self.converged:
            self.converged = True
            if self.verbose:
                print(
                    f"  ⚠️  Convergence detected: "
                    f"no improvement > {self.tolerance} for {self.patience} iterations (all objectives)"
                )

    def reset(self):
        """Reset convergence checker state."""
        self.prev_best_per_obj = None
        self.no_improvement_counts = None
        self.converged = False


class GraphSaverCallback:
    """
    Callback that saves optimization plots to disk at each iteration.

    Automatically creates output directory and saves snapshots of the
    optimization state including mean predictions, uncertainties, and
    acquisition function visualizations.
    """

    def __init__(
        self,
        plotter_class,
        bounds,
        n_objectives,
        output_dir: Optional[str] = None,
        save_every: int = 1,
        save_format: str = "png",
        add_timestamp: bool = True,
        create_gif: bool = True,
        gif_duration: int = 500,
    ):
        """
        Initialize graph saver callback.

        Args:
            plotter_class: Static plotter class to use (e.g., StaticPlotter)
            bounds: Bounds for the input space
            n_objectives: Number of objectives
            output_dir: Output directory path. If None, uses 'outputs/figures'
            save_every: Save every N iterations (default: 1 = save all)
            save_format: Image format ('png', 'jpg', 'svg')
            add_timestamp: Whether to add timestamp to folder name
            create_gif: Whether to create an animated GIF at the end (default: True)
            gif_duration: Duration per frame in milliseconds (default: 500ms)
        """
        self.plotter_class = plotter_class
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.save_every = save_every
        self.save_format = save_format
        self.create_gif = create_gif
        self.gif_duration = gif_duration

        # Setup output directory
        if output_dir is None:
            base_dir = Path("outputs/figures")
        else:
            base_dir = Path(output_dir)

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = base_dir / f"run_{timestamp}"
        else:
            self.output_dir = base_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Graph saver initialized. Saving to: {self.output_dir}")

    def __call__(self, state):
        """Save current optimization state as plot."""
        iteration = state["iteration"]

        # Check if we should save this iteration
        if iteration % self.save_every != 0:
            return

        # Only save for 2D problems
        if state["x_vector"].shape[1] != 2:
            return

        # Create static plotter instance
        plotter = self.plotter_class(self.bounds, self.n_objectives)

        # Generate filename
        filename = self.output_dir / f"iteration_{iteration:04d}.{self.save_format}"

        # Save the plot
        plotter.save_to_file(
            x_vector=state["x_vector"],
            y_vector=state["y_vector"],
            mu_objectives=state["mu_objectives"],
            variance_objectives=state["variance_objectives"],
            acquisition_values=state["acquisition_values"],
            x_next=state.get("x_next"),
            filename=str(filename),
        )

        print(f"  💾 Saved plot: {filename.name}")

    def finalize(self):
        """
        Create animated GIF from all saved images.

        Call this method after optimization completes to generate
        the animation.
        """
        if not self.create_gif:
            return

        try:
            self.plotter_class.create_gif(
                image_folder=str(self.output_dir),
                output_filename="optimization.gif",
                duration=self.gif_duration,
                loop=0,
            )
        except AttributeError:
            print("⚠️ Plotter class does not support GIF creation. Skipping.")
        except Exception as e:
            print(f"⚠️ Failed to create GIF: {e}")
