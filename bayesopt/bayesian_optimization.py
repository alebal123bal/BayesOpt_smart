"""
Multi-Objective Bayesian Optimization.

This module contains the main BayesianOptimization class for performing
multi-objective optimization using Gaussian Processes and acquisition functions.
"""

import time
from typing import Callable, List, Tuple, Optional, Any
import numpy as np

# Try to import plotting modules (optional dependency)
try:
    from plotting import HeatmapPlotterDaemon, HeatmapPlotterStatic
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    HeatmapPlotterDaemon = None  # type: ignore
    HeatmapPlotterStatic = None  # type: ignore

# Import configuration
from .config import (
    DEFAULT_PRIOR_MEAN,
    DEFAULT_PRIOR_VARIANCE,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_BETA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INITIAL_SAMPLES,
    DEFAULT_PLOT_ENABLED,
)

# Import kernel functions
from .numba_kernels import (
    initialize_lhs_integer,
    compute_prior_mean,
    compute_prior_variance,
    optimize_hyperparams_mll,
    update_k,
    invert_k,
    update_k_star,
    update_mean,
    update_variance,
    standardize_objectives,
)

# Import acquisition functions
from .acquisition import (
    update_ucb,
    update_hypervolume_improvement,
    select_next_batch,
)

# Import utility functions
from .pareto import (
    compute_pareto_front,
    print_pareto_analysis,
)
def optimize(
    x_vector: np.ndarray,
    y_vector: np.ndarray,
    kernel_matrices: np.ndarray,
    k_star: np.ndarray,
    mu_objectives: np.ndarray,
    variance_objectives: np.ndarray,
    std_mu_objectives: np.ndarray,
    std_variance_objectives: np.ndarray,
    ucb: np.ndarray,
    acquisition_values: np.ndarray,
    input_space: np.ndarray,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    reference_point: np.ndarray,  # pylint: disable=unused-argument
    n_evaluations: int,
    total_samples: int,
    n_objectives: int,  # pylint: disable=unused-argument
    function: Callable[[np.ndarray], np.ndarray],
    betas: np.ndarray,
    length_scales: np.ndarray,
    batch_size: int,
    bounds: List[Tuple[int, int]],  # pylint: disable=unused-argument
    plot: bool,
    plotter_daemon: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform the Multi-Objective Bayesian optimization.

    Args:
        x_vector (np.ndarray): Array to store evaluated points.
        y_vector (np.ndarray): Array to store objective values at evaluated points.
        kernel_matrices (np.ndarray): Kernel matrices for each objective.
        mu_objectives (np.ndarray): Mean predictions for each objective.
        variance_objectives (np.ndarray): Variance predictions for each objective.
        acquisition_values (np.ndarray): Acquisition function values for each point.
        input_space (np.ndarray): Discretized input space.
        prior_mean (np.ndarray): Prior mean for each objective.
        prior_variance (np.ndarray): Prior variance for each objective.
        reference_point (np.ndarray): Reference point for hypervolume calculation.
        n_evaluations (int): Number of evaluations already performed.
        total_samples (int): Total number of samples.
        n_objectives (int): Number of objectives.
        function (callable): The function to optimize.
        betas (np.ndarray): Exploration-exploitation trade-off parameters.
        length_scales (np.ndarray): Length scale parameters for the kernel.
        batch_size (int): Number of points to evaluate in each batch.
        bounds (list of tuples): Bounds for the optimization space [(x_min, x_max), ...].
        plot (bool): Flag to enable/disable plotting.

    Returns:
        Tuple of (x_vector, y_vector, n_evaluations) after optimization.
    """

    print(f"🚀 Starting optimization with {n_evaluations} initial evaluations.")

    # Print initial evaluated points
    for i in range(n_evaluations):
        print(f"🔍 Debug: Initial point {x_vector[i]} | Objectives = {y_vector[i]}")

    # Total number of evaluations
    last_eval = 0

    for current_eval in range(n_evaluations, total_samples, batch_size):
        # Profile iteration start time
        iter_start = time.perf_counter()

        print(
            "\n"
            f"🔄 Debug: Starting iteration {current_eval}, n_evaluations={current_eval}"
        )

        # Profile initial time
        t0 = time.perf_counter()

        optimized_hyperparams = optimize_hyperparams_mll(
            x_vector=x_vector,
            y_vector=y_vector,
            kernel_matrix=kernel_matrices,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            length_scales=length_scales,
            current_eval=current_eval,
        )

        # Profile hyperparameter optimization time
        t1 = time.perf_counter()

        print(
            "🔄 Debug: Optimized hyperparameters:",
            np.array2string(optimized_hyperparams.x, precision=2, suppress_small=True),
        )

        # Update kernel matrices for each objective
        update_k(
            kernel_matrix=kernel_matrices,
            x_vector=x_vector,
            last_eval=0,  # Unfortunately, we need to recompute the full kernel matrix
            current_eval=current_eval,
            prior_variance=prior_variance,
            length_scales=length_scales,
        )

        # Invert matrix
        kernel_matrix_inv = invert_k(
            current_eval=current_eval,
            kernel_matrix=kernel_matrices,
        )

        # Update k star for each objective
        update_k_star(
            k_star=k_star,
            x_vector=x_vector,
            input_space=input_space,
            last_eval=0,  # Unfortunately, we need to recompute the full k star
            current_eval=current_eval,
            prior_variance=prior_variance,
            length_scales=length_scales,
        )

        # Profile kernel matrix and k star update time
        t2 = time.perf_counter()

        # Update mean predictions for each objective
        update_mean(
            mu_objectives=mu_objectives,
            k_star=k_star,
            inverted_kernel_matrix=kernel_matrix_inv,
            y_vector=y_vector,
            prior_mean=prior_mean,
            current_eval=current_eval,
        )

        # Update variance predictions for each objective
        update_variance(
            variance_objectives=variance_objectives,
            k_star=k_star,
            inverted_kernel_matrix=kernel_matrix_inv,
            prior_variance=prior_variance,
            current_eval=current_eval,
        )

        # Standardize the mean and variance
        standardize_objectives(
            std_mu_objectives=std_mu_objectives,
            std_variance_objectives=std_variance_objectives,
            mu_objectives=mu_objectives,
            variance_objectives=variance_objectives,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )

        # Update Upper Confidence Bound (UCB)
        update_ucb(
            ucb=ucb,
            mu_objectives=std_mu_objectives,
            variance_objectives=std_variance_objectives,
            betas=betas,
        )

        # Update hypervolume improvement acquisition function
        update_hypervolume_improvement(
            acquisition_values=acquisition_values,
            ucb=ucb,
        )

        # Select the next batch of points to evaluate
        x_next = select_next_batch(
            input_space=input_space,
            acquisition_values=acquisition_values,
            evaluated_points=x_vector[:current_eval],
            batch_size=batch_size,
        )

        # Profile prediction and acquisition time
        t3 = time.perf_counter()

        print("🔍 Debug: Selected next batch:")
        for point in x_next:
            print(f" - {point}")

        # Plot
        if x_vector.shape[1] == 2 and plot and plotter_daemon is not None:
            plotter_daemon.plot(
                x_vector=x_vector[:current_eval],
                y_vector=y_vector[:current_eval],
                mu_objectives=mu_objectives,
                variance_objectives=variance_objectives,
                acquisition_values=acquisition_values,
                x_next=x_next,
            )

        # Evaluate the function at the new points
        for b_idx, point in enumerate(x_next):
            # Evaluate the function at the new point
            x_vector[current_eval + b_idx] = point
            y_vector[current_eval + b_idx] = function(point)

            print(
                f"🔍 Debug: Evaluating point {point} "
                f"| Objectives = {y_vector[current_eval + b_idx]}"
            )

        # Update the total number of evaluations
        last_eval = current_eval

        # Profile evaluation time
        t4 = time.perf_counter()

        print(
            f"[Iter {current_eval}] "
            f"Hyperparams: {t1 - t0:.4f}s | "
            f"Kernels: {t2 - t1:.4f}s | "
            f"Acquisition: {t3 - t2:.4f}s | "
            f"Eval: {t4 - t3:.4f}s | "
            f"TOTAL: {t4 - iter_start:.4f}s"
        )

    return x_vector, y_vector, last_eval + 1


class BayesianOptimization:
    """
    Multi-Objective Bayesian Optimization using Gaussian Processes.

    This class implements a Bayesian optimization algorithm for multi-objective
    problems using Gaussian Process surrogates and Upper Confidence Bound
    acquisition functions.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        bounds: List[Tuple[int, int]],
        n_objectives: int = 3,
        n_iterations: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize the Multi-Objective Bayesian optimization class.

        Args:
            function: The objective function to optimize, taking a 1D array
                     and returning an array of objective values.
            bounds: The bounds for each input dimension [(min, max), ...].
                   Upper bound is exclusive (e.g., (0, 30) means 0-29 inclusive).
            n_objectives: Number of objectives.
            n_iterations: The number of optimization iterations.
            **kwargs: Additional parameters:
                - plotter: Optional plotter instance (HeatmapPlotterDaemon or HeatmapPlotterStatic).
                - plot (bool): Enable/disable plotting. Default: from config.
                - prior_mean (List[float]): Prior mean for each objective.
                - prior_variance (List[float]): Prior variance for each objective.
                - length_scales (List[float]): Length scales for each objective.
                - betas (List[float]): Exploration-exploitation parameters.
                - batch_size (int): Number of points to evaluate per iteration.
                - initial_samples (int): Number of initial LHS samples.
        """

        self.function = function
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.n_iterations = n_iterations

        # Get plotter from kwargs or use default
        self.plotter = kwargs.get("plotter", None)

        # Check plotting availability
        if kwargs.get("plot", DEFAULT_PLOT_ENABLED) and not PLOTTING_AVAILABLE:
            print(
                "⚠️  Warning: Plotting requested but matplotlib/heatmap_plotter not available. "
                "Continuing without plotting."
            )

        # If prior mean and variance are not provided, calculate them later from initial samples
        self.prior_mean = np.array(
            kwargs.get("prior_mean", [DEFAULT_PRIOR_MEAN] * n_objectives),
        )
        self.prior_variance = np.array(
            kwargs.get("prior_variance", [DEFAULT_PRIOR_VARIANCE] * n_objectives),
        )

        # If length_scales is not provided, set defaults
        self.length_scales = np.array(
            kwargs.get("length_scales", [DEFAULT_LENGTH_SCALE] * n_objectives),
        )

        # If betas is not provided, set defaults
        self.betas = np.array(
            kwargs.get("betas", [DEFAULT_BETA] * n_objectives),
        )

        # If batch_size is not provided, set default
        self.batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)

        # Initial number of samples to evaluate
        self.initial_samples = kwargs.get("initial_samples", DEFAULT_INITIAL_SAMPLES)

        # Dimensionality of the input space
        self.dim = len(bounds)

        # Preallocate the input space as a unique array of cartesian points
        ranges = [np.arange(b[0], b[1]) for b in bounds]
        mesh = np.meshgrid(*ranges, indexing="ij")
        self.input_space = np.stack([m.ravel() for m in mesh], axis=-1)

        # Calculate total samples
        self.total_samples = self.initial_samples + self.n_iterations * self.batch_size

        # Preallocate the function evaluations
        self.x_vector = np.zeros((self.total_samples, self.dim))
        self.y_vector = np.zeros(
            (self.total_samples, n_objectives)
        )  # Multiple objectives

        # Preallocate the kernel matrices for each objective
        self.kernel_matrices = np.zeros(
            (self.n_objectives, self.total_samples, self.total_samples),
            dtype=np.float64,
        )

        # Preallocate the kernel vector kstar for each objective
        self.k_star = np.zeros(
            (self.n_objectives, self.total_samples, len(self.input_space)),
            dtype=np.float64,
        )

        # Preallocate the mean for each objective's Gaussian process
        self.mu_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the variance for each objective's Gaussian process
        self.variance_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the standardized mean for each objective
        self.std_mu_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the standardized variance for each objective
        self.std_variance_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the upper confidence bound acquisition function values for each objective
        self.ucb = np.zeros((n_objectives, len(self.input_space)), dtype=np.float64)

        # Preallocate the acquisition function values for each point
        self.acquisition_values = np.zeros(len(self.input_space), dtype=np.float64)

        # Initial guesses
        self.n_evaluations = initialize_lhs_integer(
            x_vector=self.x_vector,
            y_vector=self.y_vector,
            bounds=np.array(self.bounds, dtype=np.int32),
            function=self.function,
            n_samples=self.initial_samples,
        )

        # If prior mean is not provided, calculate it from initial samples
        if np.all(self.prior_mean == DEFAULT_PRIOR_MEAN):
            self.prior_mean = compute_prior_mean(
                self.y_vector, self.n_evaluations, n_objectives
            )

        # If prior variance is not provided, calculate it from initial samples
        if np.all(self.prior_variance == DEFAULT_PRIOR_VARIANCE):
            self.prior_variance = compute_prior_variance(
                self.y_vector, self.n_evaluations, n_objectives
            )

        # Reference point for hypervolume (should be worse than any expected objective value)
        self.reference_point = np.array([0.0] * n_objectives)

        # Plot flag
        self.plot = kwargs.get("plot", DEFAULT_PLOT_ENABLED) and PLOTTING_AVAILABLE

    def optimize(self) -> None:
        """
        Perform the Multi-Objective Bayesian optimization.

        This method runs the main optimization loop, evaluating the objective
        function and updating the Gaussian Process surrogate models iteratively.
        """

        # Initialize plotter if needed and 2D
        if self.plot and self.dim == 2:
            if self.plotter is None:
                # Create default static plotter
                print(
                    "📊 Initializing static plot window (press 'Q' to close each plot)..."
                )
                if HeatmapPlotterStatic is not None:
                    self.plotter = HeatmapPlotterStatic(
                        bounds=self.bounds,
                        n_objectives=self.n_objectives,
                    )
            else:
                # User provided custom plotter
                print("📊 Using custom plotter...")

        # Optimize
        self.x_vector, self.y_vector, self.n_evaluations = optimize(
            x_vector=self.x_vector,
            y_vector=self.y_vector,
            kernel_matrices=self.kernel_matrices,
            k_star=self.k_star,
            mu_objectives=self.mu_objectives,
            variance_objectives=self.variance_objectives,
            std_mu_objectives=self.std_mu_objectives,
            std_variance_objectives=self.std_variance_objectives,
            ucb=self.ucb,
            acquisition_values=self.acquisition_values,
            input_space=self.input_space,
            prior_mean=self.prior_mean,
            prior_variance=self.prior_variance,
            reference_point=self.reference_point,
            n_evaluations=self.n_evaluations,
            total_samples=self.total_samples,
            n_objectives=self.n_objectives,
            function=self.function,
            betas=self.betas,
            length_scales=self.length_scales,
            batch_size=self.batch_size,
            bounds=self.bounds,
            plot=self.plot,
            plotter_daemon=self.plotter,
        )

    def pareto_analysis(self) -> np.ndarray:
        """
        Perform Pareto analysis on the results of the optimization.

        Identifies and returns the Pareto-efficient points from all
        evaluated solutions. Handles early stopping where not all
        iterations are used.

        Returns:
            Array of Pareto-efficient objective values (n_pareto, n_objectives).
        """
        # Work only with evaluated points
        evaluated_y = self.y_vector[: self.n_evaluations]
        evaluated_x = self.x_vector[: self.n_evaluations]

        # Compute Pareto front
        pareto_inputs, pareto_objectives = compute_pareto_front(evaluated_x, evaluated_y)

        # Print results
        print_pareto_analysis(pareto_inputs, pareto_objectives)

        return pareto_objectives
