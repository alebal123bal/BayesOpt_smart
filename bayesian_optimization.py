"""
Multi-Objective Bayesian optimization optimized class.
"""

import os
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Debug flag - setup from launch configuration or environment variable
DEBUG_MODE = os.environ.get("BAYESIAN_DEBUG", "False").lower() in ("true", "1", "yes")

# Conditional imports and setup
if DEBUG_MODE:
    print("🐛 DEBUG MODE - Numba disabled")

    # Dummy njit decorator
    def njit(*args, **kwargs):  # pylint: disable=unused-argument
        """Dummy njit decorator for debugging."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    # Dummy prange matching built-in range signature
    def prange(*args):
        """Dummy prange function for debugging purposes."""
        return range(*args)

else:
    print("🚀 PRODUCTION MODE - Numba enabled")
    from numba import njit, prange

X_MAX = 30
Y_MAX = 30
Z_MAX = 30


@njit
def toy_function(x):
    """
    A multi-objective toy function to optimize.

    Args:
        x (np.ndarray): Input array (e.g., [x1, x2, ..., xd]).

    Returns:
        np.ndarray: Output array containing [f(x), g(x), h(x)].
    """
    f_x = -((x[0] - 12) ** 2) + 100
    g_x = -((x[1] - 20) ** 2) + 20
    # h_x = -((x[2] - 5) ** 2) + 120

    return np.array(
        [
            f_x,
            g_x,
            # h_x,
        ]
    )


@njit
def initialize_lhs_integer(x_vector, y_vector, bounds, function, n_samples=8):
    """
    Initialize sample points using Latin Hypercube Sampling (LHS) for integer design spaces.

    This method divides the range of each dimension into `n_samples` equally sized bins
    and ensures that exactly one sample is drawn from each bin per dimension.
    The samples are placed randomly within each bin and rounded to integers to match
    discrete search spaces.

    Args:
        x_vector (np.ndarray): (n_samples, n_dimensions) Array to store evaluated points.
        y_vector (np.ndarray): (n_samples, n_objectives) Array to store objective values.
        bounds (np.ndarray): (n_dimensions, 2) Array of (min, max) integer bounds for each dimension.
        function (callable): The function to evaluate, taking a 1D integer array as input.
        n_samples (int): Number of initial samples to generate.

    Returns:
        int: Number of evaluations performed (equals n_samples).
    """
    dim = len(bounds)
    samples = np.empty((n_samples, dim), dtype=np.int32)

    for d in range(dim):
        # Create a random permutation of bin indices
        perm = np.random.permutation(n_samples)
        min_val, max_val = bounds[d, 0], bounds[d, 1]
        step = (max_val - min_val + 1) / n_samples  # bin width in integer space

        for i in range(n_samples):
            # Random offset within the bin
            low = min_val + perm[i] * step
            high = min_val + (perm[i] + 1) * step
            # Random integer in [low, high)
            samples[i, d] = int(np.random.uniform(low, high))

    # Evaluate all sampled points
    for i in range(n_samples):
        x_vector[i] = samples[i]
        y_vector[i] = function(x_vector[i])

    return n_samples


@njit
def compute_prior_mean(y_vector, n_evaluations, n_objectives):
    """
    Compute the prior mean for each objective based on initial samples.

    Args:
        y_vector (np.ndarray): Array of objective values at evaluated points.
        n_evaluations (int): Number of evaluations performed.
        n_objectives (int): Number of objectives.

    Returns:
        np.ndarray: Prior mean for each objective.
    """

    prior_mean = np.zeros(n_objectives, dtype=np.float64)
    for obj_idx in range(n_objectives):
        prior_mean[obj_idx] = np.mean(y_vector[:n_evaluations, obj_idx])
    return prior_mean


@njit
def compute_prior_variance(y_vector, n_evaluations, n_objectives):
    """
    Compute the prior variance for each objective based on initial samples.

    Args:
        y_vector (np.ndarray): Array of objective values at evaluated points.
        n_evaluations (int): Number of evaluations performed.
        n_objectives (int): Number of objectives.

    Returns:
        np.ndarray: Prior variance for each objective.
    """

    prior_variance = np.zeros(n_objectives, dtype=np.float64)
    for obj_idx in range(n_objectives):
        prior_variance[obj_idx] = np.var(y_vector[:n_evaluations, obj_idx])
    return prior_variance


@njit(parallel=True)
def compute_marginal_log_likelihood_parallel(
    x_vector,
    y_vector,
    kernel_matrix,
    prior_mean,
    prior_variance,
    length_scales,
    current_eval,
):
    """
    Compute the marginal log likelihood for the Gaussian process in parallel.

    Args:
        x_vector (np.ndarray): (n_eval, n_features) training inputs.
        y_vector (np.ndarray): (n_eval, n_objectives) objective values at evaluated points.
        kernel_matrix (np.ndarray): Preallocated (n_objectives, n_eval, n_eval) matrix to store kernels.
        prior_mean (np.ndarray): (n_objectives,) prior mean for each objective.
        prior_variance (np.ndarray): (n_objectives,) prior variance for each objective.
        length_scales (np.ndarray): (n_objectives,) Length scale for each objective.
        current_eval (int): Number of evaluated points to consider.

    Returns:
        total_mll (float): Sum of MLL over all objectives.
    """
    # Compute kernel matrix for given hyperparameters (already parallelized)
    update_k_parallel(
        kernel_matrix=kernel_matrix,
        x_vector=x_vector,
        last_eval=0,
        current_eval=current_eval,
        prior_variance=prior_variance,
        length_scales=length_scales,
    )

    n_objectives = y_vector.shape[1]
    n_points = current_eval

    # Store per-objective MLL to avoid race conditions
    mll_values = np.empty(n_objectives, dtype=np.float64)

    for obj_idx in prange(n_objectives):
        # Ensure contiguous for Cholesky speed
        K = np.ascontiguousarray(
            kernel_matrix[obj_idx, :n_points, :n_points] / prior_variance[obj_idx]
        )

        # Center and normalize outputs
        y_centered = np.ascontiguousarray(
            y_vector[:n_points, obj_idx] - prior_mean[obj_idx]
        )
        std_y = np.std(y_centered)
        if std_y > 0.0:
            y_centered /= std_y

        # Cholesky decomposition
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n_points))

        # Solve alpha = K^{-1} y_centered
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))

        # Term 1: data fit
        data_fit_term = -0.5 * np.dot(y_centered, alpha)

        # Term 2: complexity penalty
        logdetK = 2.0 * np.sum(np.log(np.diag(L)))
        complexity_term = -0.5 * logdetK

        # Term 3: constant penalty
        constant_term = -0.5 * n_points * np.log(2.0 * np.pi)

        # Store per-objective contribution
        mll_values[obj_idx] = data_fit_term + complexity_term + constant_term

    # Sum outside the parallel loop
    return np.sum(mll_values)


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


@njit(parallel=True, fastmath=True)
def update_k_parallel(
    kernel_matrix,
    x_vector,
    last_eval,
    current_eval,
    prior_variance,
    length_scales,
):
    """
    Compute the kernel matrix for the training points in parallel.

    Args:
        kernel_matrix (np.ndarray): (n_objectives, n_points, n_points) Preallocated kernel matrix to fill.
        x_vector (np.ndarray): (n_points, n_features) Training points.
        last_eval (int): Last evaluation number.
        current_eval (int): Current number of evaluations.
        prior_variance (np.ndarray): (n_objectives,) Variance parameter for each objective's kernel.
        length_scales (np.ndarray): (n_objectives,) Length scale parameters for each objective's kernel.
    """
    n_objectives = kernel_matrix.shape[0]

    # Compute upper triangle in parallel
    for i in prange(last_eval, current_eval):
        for j in range(i, current_eval):
            diff = x_vector[i] - x_vector[j]
            sq_dist = np.dot(diff, diff)

            for obj_idx in range(n_objectives):
                k_val = prior_variance[obj_idx] * np.exp(
                    -0.5 * sq_dist / (length_scales[obj_idx] ** 2)
                )
                kernel_matrix[obj_idx, i, j] = k_val

    # Mirror upper triangle to lower triangle (sequential, small cost)
    for obj_idx in range(n_objectives):
        for i in range(last_eval, current_eval):
            for j in range(i + 1, current_eval):
                kernel_matrix[obj_idx, j, i] = kernel_matrix[obj_idx, i, j]


@njit
def invert_k(current_eval, kernel_matrix, prior_variance):
    """
    Invert the kernel matrix for each objective.

    Args:
        current_eval (int): Current number of evaluations.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.
        prior_variance (np.ndarray): Prior variance for each objective.

    Returns:
        np.ndarray: Inverted kernel matrix for each objective.
    """

    n_objectives = kernel_matrix.shape[0]

    # Allocate a contiguous array for the inverted kernel matrix
    kernel_matrix_inv = np.zeros(
        (n_objectives, current_eval, current_eval), dtype=np.float64
    )

    # Extract and ensure contiguous memory layout
    base_matrix = np.ascontiguousarray(
        kernel_matrix[0, :current_eval, :current_eval] / prior_variance[0]
    )

    # Add jitter for numerical stability
    jitter = 1e-8
    base_matrix += jitter * np.eye(base_matrix.shape[0])

    # Compute the inverse once
    base_matrix_inv = np.ascontiguousarray(np.linalg.inv(base_matrix))

    for obj_idx in range(n_objectives):
        # Scale with prior variance
        kernel_matrix_inv[obj_idx] = np.ascontiguousarray(
            base_matrix_inv / prior_variance[obj_idx]
        )

    return kernel_matrix_inv


@njit(parallel=True, fastmath=True)
def update_k_star_parallel(
    k_star,
    x_vector,
    input_space,
    last_eval,
    current_eval,
    prior_variance,
    length_scales,
):
    """
    Update the kernel vector for new points based on the training points.

    Args:
        k_star (np.ndarray): (n_objectives, n_points, n_candidates) Preallocated kernel vector.
        x_vector (np.ndarray): (n_points, n_features) Training points.
        input_space (np.ndarray): (n_candidates, n_features) Discretized input space.
        last_eval (int): Last evaluation index.
        current_eval (int): Current number of evaluations.
        prior_variance (np.ndarray): (n_objectives,) Variance parameter for each objective's kernel.
        length_scales (np.ndarray): (n_objectives,) Length scale parameter for each objective's kernel.
    """
    n_objectives = k_star.shape[0]
    n_candidates = input_space.shape[0]
    total_pairs = (current_eval - last_eval) * n_candidates

    for idx in prange(total_pairs):
        e = last_eval + idx // n_candidates
        i = idx % n_candidates

        diff = x_vector[e] - input_space[i]
        sq_dist = np.dot(diff, diff)

        for obj_idx in range(n_objectives):
            k_star[obj_idx, e, i] = prior_variance[obj_idx] * np.exp(
                -0.5 * sq_dist / (length_scales[obj_idx] ** 2)
            )


@njit
def update_mean_parallel(
    mu_objectives,
    k_star,
    inverted_kernel_matrix,
    y_vector,
    prior_mean,
    current_eval,
):
    """
    Update the mean predictions for each objective based on the kernel vector and training points.

    Args:
        mu_objectives (np.ndarray): (n_objectives, n_candidates) Preallocated mean predictions.
        k_star (np.ndarray): (n_objectives, n_points, n_candidates) Kernel vector for the new point.
        inverted_kernel_matrix (np.ndarray): (n_objectives, n_points, n_points) Inverted kernel matrices.
        y_vector (np.ndarray): (n_points, n_objectives) Objective values at evaluated points.
        prior_mean (np.ndarray): (n_objectives,) Prior mean for each objective.
        current_eval (int): Current number of evaluations.
    """
    n_objectives = mu_objectives.shape[0]

    for obj_idx in range(n_objectives):
        # Ensure C‑contiguity for fast BLAS in '@'
        kernel_matrix_inv_c = np.ascontiguousarray(
            inverted_kernel_matrix[obj_idx, :current_eval, :current_eval]
        )
        delta_y_c = np.ascontiguousarray(
            y_vector[:current_eval, obj_idx] - prior_mean[obj_idx]
        )
        k_star_obj_c = np.ascontiguousarray(k_star[obj_idx, :current_eval, :])

        # First multiply: (n_points, n_points) @ (n_points,)
        partial_result = kernel_matrix_inv_c @ delta_y_c

        # Second multiply: (n_candidates, n_points) @ (n_points,)
        mu_objectives[obj_idx, :] = (
            prior_mean[obj_idx] + np.ascontiguousarray(k_star_obj_c.T) @ partial_result
        )


@njit
def update_variance_parallel(
    variance_objectives,
    k_star,
    inverted_kernel_matrix,
    prior_variance,
    current_eval,
):
    """
    Update the variance predictions for each objective based on the kernel vector
    and training points.

    Args:
        variance_objectives (np.ndarray): (n_objectives, n_candidates) Preallocated variance predictions.
        k_star (np.ndarray): (n_objectives, n_points, n_candidates) Kernel vector for the new point.
        inverted_kernel_matrix (np.ndarray): (n_objectives, n_points, n_points) Inverted kernel matrices.
        prior_variance (np.ndarray): (n_objectives,) Prior variance for each objective.
        current_eval (int): Current number of evaluations.
    """
    n_objectives = variance_objectives.shape[0]
    n_candidates = k_star.shape[2]

    for obj_idx in range(n_objectives):
        # Ensure C‑contiguity for fast '@'
        kernel_matrix_inv_c = np.ascontiguousarray(
            inverted_kernel_matrix[obj_idx, :current_eval, :current_eval]
        )
        k_star_obj_c = np.ascontiguousarray(k_star[obj_idx, :current_eval, :])

        # First multiply: (n_points, n_points) @ (n_points, n_candidates)
        intermediate = kernel_matrix_inv_c @ k_star_obj_c

        # Compute quadratic form without unsupported einsum or transposes
        quadratic_form = np.empty(n_candidates, dtype=np.float64)
        for j in range(n_candidates):
            s = 0.0
            for i in range(current_eval):
                s += k_star_obj_c[i, j] * intermediate[i, j]
            quadratic_form[j] = s

        # Compute variances
        variance = prior_variance[obj_idx] - quadratic_form
        variance_objectives[obj_idx, :] = np.maximum(
            variance, 1e-10
        )  # Prevent negative or zero variance


@njit
def standardize_objectives(
    std_mu_objectives,
    std_variance_objectives,
    mu_objectives,
    variance_objectives,
    prior_mean,
    prior_variance,
):
    """
    Standardize objectives to have mean=0, std=1 for fair comparison.

    Args:
        std_mu_objectives (np.ndarray): Preallocated standardized mean for each objective.
        std_variance_objectives (np.ndarray): Preallocated standardized variance for each objective.
        mu_objectives (np.ndarray): Mean predictions for each objective.
        variance_objectives (np.ndarray): Variance predictions for each objective.
        prior_mean (np.ndarray): Prior mean for each objective.
        prior_variance (np.ndarray): Prior variance for each objective.
    """

    n_objectives = mu_objectives.shape[0]

    for obj_idx in range(n_objectives):
        # Standardize mean: (mu - prior_mean) / sqrt(prior_variance)
        std_mu_objectives[obj_idx] = (
            mu_objectives[obj_idx] - prior_mean[obj_idx]
        ) / np.sqrt(prior_variance[obj_idx])

        # Standardize variance: variance / prior_variance
        std_variance_objectives[obj_idx] = (
            variance_objectives[obj_idx] / prior_variance[obj_idx]
        )


@njit
def upper_confidence_bound(mu, variance, betas):
    """
    Compute the upper confidence bound for a single objective, vectorized.

    Args:
        mu (np.ndarray): Mean predictions for the objective.
        variance (np.ndarray): Variance predictions for the objective.
        betas (np.ndarray): Exploration-exploitation trade-off parameters for each objective.

    Returns:
        np.ndarray: Upper confidence bound values for the objective.
    """

    return mu + betas * np.sqrt(np.abs(variance))


@njit
def update_ucb(
    ucb,
    mu_objectives,
    variance_objectives,
    betas,
):
    """
    Update the upper confidence bound acquisition function values.

    Args:
        ucb (np.ndarray): Preallocated upper confidence bound values.
        mu_objectives (np.ndarray): Mean predictions for each objective.
        variance_objectives (np.ndarray): Variance predictions for each objective.
        betas (np.ndarray): Exploration-exploitation trade-off parameters for each objective.
    """

    n_objectives = mu_objectives.shape[0]

    # Compute the UCB for each point vectorized
    for obj_idx in range(n_objectives):
        ucb[obj_idx] = upper_confidence_bound(
            mu_objectives[obj_idx],
            variance_objectives[obj_idx],
            betas[obj_idx],
        )


@njit
def update_hypervolume_improvement(
    acquisition_values,
    ucb,
):
    """
    Update the hypervolume improvement acquisition function values.

    Args:
        ucb (np.ndarray): Upper confidence bound values for each point.
    """
    n_points = len(acquisition_values)

    # Compute the hypervolume improvement acquisition function values
    for i in range(n_points):
        acquisition_values[i] = np.sum(ucb[:, i])


def select_next_point(
    input_space,
    acquisition_values,
    evaluated_points,
    max_candidates=3,
):
    """
    Select the next point to evaluate based on acquisition values.

    Args:
        input_space (np.ndarray): The input space to sample from.
        acquisition_values (np.ndarray): The acquisition values for each point in the input space.
        evaluated_points (np.ndarray): The points that have already been evaluated.
        max_candidates (int): The maximum number of candidates to consider.

    Returns:
        np.ndarray: The selected point to evaluate next, or None if no valid point is found.
    """

    # Sort candidates by acquisition value
    sorted_indices = np.argsort(acquisition_values)[::-1]  # best → worst

    for idx in sorted_indices[:max_candidates]:
        candidate = input_space[idx]
        if not np.any(np.all(candidate == evaluated_points, axis=1)):
            return candidate  # Found a new point

    # All top candidates already evaluated
    return None


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
            prior_variance=prior_variance,
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
        if x_vector.shape[1] == 2:
            heatmap_plot(
                x_vector=x_vector[: current_eval + batch_size],
                y_vector=y_vector[: current_eval + batch_size],
                bounds=((0, X_MAX), (0, Y_MAX)),
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

    # Reshape predictions to 2D grid (ny, nx) because imshow expects [y, x]
    mu_grids = [mu.reshape(ny, nx) for mu in mu_objectives]
    var_grids = [var.reshape(ny, nx) for var in variance_objectives]
    sigma_grids = [np.sqrt(var) for var in var_grids]
    acquisition_grid = acquisition_values.reshape(ny, nx)

    # Create subplots: one row per objective, three columns
    fig, axs = plt.subplots(
        n_objectives, 3, figsize=(21, 5 * n_objectives), constrained_layout=True
    )
    if n_objectives == 1:
        axs = axs.reshape(1, -1)

    fig.suptitle(
        "Bayesian Optimization Surrogate Model (2D)", fontsize=18, fontweight="bold"
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
        scatter_mean = ax_mean.scatter(
            x_vector[:, 0],
            x_vector[:, 1],
            c=y_vector[:, obj_idx],
            cmap="plasma",
            s=50,
            edgecolors="black",
            linewidth=0.7,
            label="Evaluated Points",
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

        # Set ticks to integer positions
        for ax in [ax_mean, ax_uncert, ax_acq]:
            ax.set_xticks(range(x_min, x_max))
            ax.set_yticks(range(y_min, y_max))
            ax.set_xlim(x_min - 0.5, x_max - 0.5)
            ax.set_ylim(y_min - 0.5, y_max - 0.5)

    plt.show()


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
        """

        self.function = function
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.n_iterations = n_iterations

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

    def optimize(self):
        """
        Perform the Multi-Objective Bayesian optimization.
        """

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
    # Example usage
    _bounds = [
        (0, X_MAX),
        (0, Y_MAX),
        # (0, Z_MAX),
    ]

    start_time = time.time()
    print("\n⚡ Starting optimization...\n")

    optimizer = BayesianOptimization(
        toy_function,
        _bounds,
        n_objectives=len(_bounds),
        initial_samples=10,
        n_iterations=5,
        batch_size=3,
        betas=np.array([2.0] * len(_bounds)),
    )

    optimizer.optimize()

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    optimizer.pareto_analysis()
