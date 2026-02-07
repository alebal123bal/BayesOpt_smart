"""
Multi-Objective Bayesian optimization optimized class.
"""

import time
import numpy as np
from scipy.optimize import minimize
from heatmap_plotter import HeatmapPlotterDaemon, HeatmapPlotterStatic

# Import all numba-accelerated kernel functions
from numba_kernels import (
    toy_function,
    initialize_lhs_integer,
    compute_prior_mean,
    compute_prior_variance,
    compute_marginal_log_likelihood_parallel,
    update_k_parallel,
    invert_k,
    update_k_star_parallel,
    update_mean_parallel,
    update_variance_parallel,
    standardize_objectives,
    update_ucb,
    update_hypervolume_improvement,
)

np.random.seed(42)


def optimize_hyperparams_mll(
    x_vector,
    y_vector,
    kernel_matrix,
    prior_mean,
    prior_variance,
    length_scales,
    current_eval,
):
    """
    Optimize GP hyperparameters (length_scales, prior_variance) by maximizing
    the marginal log-likelihood (MLL). Uses a derivative-free method (Powell).

    Updates `length_scales` and `prior_variance` IN PLACE.

    Args:
        x_vector (np.ndarray): (n_eval, n_features) training inputs.
        y_vector (np.ndarray): (n_eval, n_objectives) training targets.
        kernel_matrix (np.ndarray): (n_objectives, n_eval, n_eval) preallocated kernel buffer.
        prior_mean (np.ndarray): (n_objectives,) prior means (unchanged).
        prior_variance (np.ndarray): (n_objectives,) prior variances (updated in place).
        length_scales (np.ndarray): (n_objectives,) length scales (updated in place).
        current_eval (int): Number of evaluated points to consider.

    Returns:
        res (OptimizeResult): Result object from scipy.optimize.minimize.
    """
    n_objectives = y_vector.shape[1]

    # Initial guess: concatenate length scales and variances
    initial_guess = np.concatenate([length_scales, prior_variance])

    # Bounds: enforce positivity
    bounds = [(1e-5, None)] * (2 * n_objectives)

    # Objective function: negative MLL (since minimize() is used)
    def objective(params):
        ls = params[:n_objectives]
        var = params[n_objectives:]

        mll = compute_marginal_log_likelihood_parallel(
            x_vector=x_vector,
            y_vector=y_vector,
            kernel_matrix=kernel_matrix,
            prior_mean=prior_mean,
            prior_variance=var,
            length_scales=ls,
            current_eval=current_eval,
        )

        return -mll

    # Run Powell optimization (derivative-free)
    optim_result = minimize(
        objective,
        initial_guess,
        method="Powell",
        bounds=bounds,
        options={
            "xtol": 1e-3,  # stop tolerance for parameters
            "ftol": 1e-4,  # stop tolerance for function value
            "maxiter": 1000,  # safeguard against long runs
        },
    )

    # Update hyperparameters in place
    length_scales[:] = optim_result.x[:n_objectives]
    prior_variance[:] = optim_result.x[n_objectives:]

    return optim_result


def select_next_batch(
    input_space,
    acquisition_values,
    evaluated_points,
    batch_size=3,
):
    """
    Select a batch of points to evaluate based on acquisition values.

    Args:
        input_space (np.ndarray): All candidate input points.
        acquisition_values (np.ndarray): Acquisition values for each candidate.
        evaluated_points (np.ndarray): Points already evaluated.
        batch_size (int): Number of points to select.

    Returns:
        np.ndarray: Array of shape (batch_size, n_dimensions) with new points.
    """
    sorted_indices = np.argsort(acquisition_values)[::-1]  # best → worst
    batch = []

    for idx in sorted_indices:
        candidate = input_space[idx]
        if not np.any(np.all(candidate == evaluated_points, axis=1)):
            batch.append(candidate)
            if len(batch) == batch_size:
                break

    return np.array(batch)


# @njit
def optimize(
    x_vector,
    y_vector,
    kernel_matrices,
    k_star,
    mu_objectives,
    variance_objectives,
    std_mu_objectives,
    std_variance_objectives,
    ucb,
    acquisition_values,
    input_space,
    prior_mean,
    prior_variance,
    reference_point,  # pylint: disable=unused-argument
    n_evaluations,
    total_samples,
    n_objectives,  # pylint: disable=unused-argument
    function,
    betas,
    length_scales,
    batch_size,
    bounds,
    plot,
    plotter_daemon=None,
):
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
        tuple: Updated x_vector, y_vector, and number of evaluations.
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
        update_k_parallel(
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
        update_k_star_parallel(
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
        update_mean_parallel(
            mu_objectives=mu_objectives,
            k_star=k_star,
            inverted_kernel_matrix=kernel_matrix_inv,
            y_vector=y_vector,
            prior_mean=prior_mean,
            current_eval=current_eval,
        )

        # Update variance predictions for each objective
        update_variance_parallel(
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


def is_pareto_efficient(y_vector):
    """
    Find the Pareto-efficient points from the evaluations.
    """

    # Invert the y_vector to find the Pareto-efficient points
    y_vector_neg = -y_vector

    is_efficient = np.ones(y_vector_neg.shape[0], dtype=bool)

    for i in range(y_vector_neg.shape[0]):
        if not is_efficient[i]:
            continue
        for j in range(i + 1, y_vector_neg.shape[0]):
            if np.all(y_vector_neg[j] <= y_vector_neg[i]) and np.any(
                y_vector_neg[j] < y_vector_neg[i]
            ):
                is_efficient[i] = False
                break
            if np.all(y_vector_neg[i] <= y_vector_neg[j]) and np.any(
                y_vector_neg[i] < y_vector_neg[j]
            ):
                is_efficient[j] = False

    return is_efficient


class BayesianOptimization:
    """
    A class for Multi-Objective Bayesian optimization.
    """

    def __init__(
        self,
        function,
        bounds,
        n_objectives=3,
        n_iterations=10,
        **kwargs,
    ):
        """
        Initialize the Multi-Objective Bayesian optimization class.

        Args:
            function (callable): The function to optimize (returns array of objectives).
            bounds (tuple): The bounds for the input variable (min, max).
            n_objectives (int): Number of objectives.
            n_iterations (int): The number of iterations for the optimization.
            **kwargs: Additional parameters including:
                - plotter: Optional plotter instance (HeatmapPlotterDaemon or HeatmapPlotterStatic).
                           If None and plot=True, defaults to HeatmapPlotterStatic.
                - plot (bool): Enable/disable plotting. Default: True.
        """

        self.function = function
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.n_iterations = n_iterations

        # Get plotter from kwargs or use default
        self.plotter = kwargs.get("plotter", None)

        # If prior mean and variance are not provided, calculate them later from initial samples
        self.prior_mean = np.array(
            kwargs.get("prior_mean", [0.0] * n_objectives),
        )
        self.prior_variance = np.array(
            kwargs.get("prior_variance", [1.0] * n_objectives),
        )

        # If length_scales is not provided, set it to 1.0
        self.length_scales = np.array(
            kwargs.get("length_scales", [1.0] * n_objectives),
        )

        # If betas is not provided, set it to 1.0
        self.betas = np.array(
            kwargs.get("betas", [1.0] * n_objectives),
        )

        # If batch_size is not provided, set it to 3
        self.batch_size = kwargs.get("batch_size", 3)

        # Initial number of samples to evaluate
        self.initial_samples = kwargs.get("initial_samples", 3)

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
        if np.all(self.prior_mean == 0.0):
            self.prior_mean = compute_prior_mean(
                self.y_vector, self.n_evaluations, n_objectives
            )

        # If prior variance is not provided, calculate it from initial samples
        if np.all(self.prior_variance == 1.0):
            self.prior_variance = compute_prior_variance(
                self.y_vector, self.n_evaluations, n_objectives
            )

        # Reference point for hypervolume (should be worse than any expected objective value)
        self.reference_point = np.array([0.0] * n_objectives)

        # Plot flag
        self.plot = kwargs.get("plot", True)

    def optimize(self):
        """
        Perform the Multi-Objective Bayesian optimization.
        """

        # Initialize plotter if needed and 2D
        if self.plot and self.dim == 2:
            if self.plotter is None:
                # Create default static plotter
                print(
                    "📊 Initializing static plot window (press 'Q' to close each plot)..."
                )
                self.plotter = HeatmapPlotterStatic(
                    bounds=self.bounds,
                    n_objectives=self.n_objectives,
                )
            else:
                # User provided custom plotter
                print("📊 Using custom plotter...")

        # Optimize with numba
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

    def pareto_analysis(self):
        """
        Perform Pareto analysis on the results of the optimization.
        Handles early stopping where not all iterations are used.
        """
        # Work only with evaluated points
        evaluated_y = self.y_vector[: self.n_evaluations]
        evaluated_x = self.x_vector[: self.n_evaluations]

        # Compute Pareto efficiency on evaluated points
        is_efficient = is_pareto_efficient(evaluated_y)

        # Extract Pareto-efficient points and inputs
        pareto_points = evaluated_y[is_efficient]
        pareto_input_points = evaluated_x[is_efficient]

        print("📊 Pareto Analysis Results:")
        for i, point in enumerate(pareto_points):
            print(f"Input: {pareto_input_points[i]}, Pareto Point {i + 1}: {point}")

        return pareto_points


if __name__ == "__main__":
    X_MAX = 300
    Y_MAX = 300
    Z_MAX = 300

    # Example usage
    _bounds = [
        (0, X_MAX),
        (0, Y_MAX),
        # (0, Z_MAX),
    ]

    # Initialize Dynamic Plotter (Daemon)
    dynamic_plotter = HeatmapPlotterDaemon(
        bounds=_bounds,
        n_objectives=len(_bounds),
    )

    start_time = time.time()
    print("\n⚡ Starting optimization...\n")

    optimizer = BayesianOptimization(
        toy_function,
        _bounds,
        n_objectives=len(_bounds),
        initial_samples=(X_MAX + Y_MAX) // 100,  # 1% of grid size
        n_iterations=15,
        batch_size=X_MAX // 100,  # 1% of grid size
        betas=np.array([2.0] * len(_bounds)),
        plot=True,
        plotter=dynamic_plotter,
    )

    optimizer.optimize()

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    optimizer.pareto_analysis()
