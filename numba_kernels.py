"""
Numba-accelerated kernel functions for Bayesian Optimization.

This module contains Gaussian Process kernel operations and related
computationally intensive functions accelerated using Numba's JIT compilation.
"""

import numpy as np
from scipy.optimize import minimize
from config import (
    DEBUG_MODE,
    KERNEL_JITTER,
    CHOLESKY_JITTER,
    MIN_VARIANCE,
    HYPERPARAM_METHOD,
    HYPERPARAM_XTOL,
    HYPERPARAM_FTOL,
    HYPERPARAM_MAXITER,
    HYPERPARAM_MIN_BOUND,
)

# Conditional imports
if DEBUG_MODE:
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
    from numba import njit, prange


# =============================================================================
# INITIALIZATION
# =============================================================================


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
                            Upper bound is exclusive (e.g., (0, 30) means 0-29 inclusive).
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
        step = (
            max_val - min_val
        ) / n_samples  # bin width in integer space (exclusive upper bound)

        for i in range(n_samples):
            # Random offset within the bin
            low = min_val + perm[i] * step
            high = min_val + (perm[i] + 1) * step
            # Random integer in [low, high), ensuring we don't exceed max_val-1
            sample_val = int(np.random.uniform(low, high))
            samples[i, d] = min(sample_val, max_val - 1)

    # Evaluate all sampled points
    for i in range(n_samples):
        x_vector[i] = samples[i]
        y_vector[i] = function(x_vector[i])

    return n_samples


# =============================================================================
# PRIOR COMPUTATIONS
# =============================================================================


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


# =============================================================================
# MARGINAL LOG LIKELIHOOD & HYPERPARAMETER OPTIMIZATION
# =============================================================================


@njit(parallel=True)
def compute_mll(
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
    update_k(
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
        L = np.linalg.cholesky(K + CHOLESKY_JITTER * np.eye(n_points))

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
    bounds = [(HYPERPARAM_MIN_BOUND, None)] * (2 * n_objectives)

    # Objective function: negative MLL (since minimize() is used)
    def objective(params):
        ls = params[:n_objectives]
        var = params[n_objectives:]

        mll = compute_mll(
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
        method=HYPERPARAM_METHOD,
        bounds=bounds,
        options={
            "xtol": HYPERPARAM_XTOL,
            "ftol": HYPERPARAM_FTOL,
            "maxiter": HYPERPARAM_MAXITER,
        },
    )

    # Update hyperparameters in place
    length_scales[:] = optim_result.x[:n_objectives]
    prior_variance[:] = optim_result.x[n_objectives:]

    return optim_result


# =============================================================================
# KERNEL MATRIX OPERATIONS
# =============================================================================


@njit(parallel=True, fastmath=True)
def update_k(
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
def invert_k(current_eval, kernel_matrix):
    """
    Invert the kernel matrix for each objective using direct matrix inversion.
    Numba-compatible implementation with numerical stability via jitter.

    Args:
        current_eval (int): Current number of evaluations.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.

    Returns:
        np.ndarray: Inverted kernel matrix for each objective.
    """

    n_objectives = kernel_matrix.shape[0]

    # Allocate output array
    kernel_matrix_inv = np.zeros(
        (n_objectives, current_eval, current_eval), dtype=np.float64
    )

    for obj_idx in range(n_objectives):
        # Extract kernel matrix slice and add jitter for numerical stability
        K = np.zeros((current_eval, current_eval), dtype=np.float64)
        for i in range(current_eval):
            for j in range(current_eval):
                K[i, j] = kernel_matrix[obj_idx, i, j]
                if i == j:  # Add jitter to diagonal for numerical stability
                    K[i, j] += KERNEL_JITTER

        # Direct matrix inversion
        kernel_matrix_inv[obj_idx] = np.linalg.inv(K)

    return kernel_matrix_inv


@njit(parallel=True, fastmath=True)
def update_k_star(
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


# =============================================================================
# GAUSSIAN PROCESS PREDICTIONS
# =============================================================================


@njit
def update_mean(
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
def update_variance(
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
            variance, MIN_VARIANCE
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
