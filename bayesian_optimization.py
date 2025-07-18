"""
Multi-Objective Bayesian optimization optimized class.
"""

import os
import time
import numpy as np

# Debug flag - setup from launch configuration or environment variable
DEBUG_MODE = os.environ.get("BAYESIAN_DEBUG", "False").lower() in ("true", "1", "yes")

# Conditional imports and setup
if DEBUG_MODE:
    print("🐛 DEBUG MODE - Numba disabled")

    # Define dummy decorators that do nothing
    def njit(*args, **kwargs):  # pylint: disable=unused-argument
        """
        Dummy njit decorator for debugging purposes.
        """

        def decorator(func):
            """
            Dummy decorator function that returns the function as is.
            """
            return func

        if len(args) == 1 and callable(args[0]):
            # Called as @njit without parentheses
            return args[0]
        # Called as @njit()
        return decorator

    def prange(n):
        """
        Dummy prange function for debugging purposes.
        """
        return range(n)

else:
    print("🚀 PRODUCTION MODE - Numba enabled")
    from numba import njit, prange

X_MAX = 20
Y_MAX = 20
Z_MAX = 20


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
    g_x = -((x[1] - 1) ** 4) + 20
    h_x = -((x[2] - 5) ** 2) + 120

    return np.array([f_x, g_x, h_x])


@njit
def initialize_samples(x_vector, y_vector, bounds, function, n_samples=8):
    """
    Initialize sample points using uniform grid sampling for maximum coverage.

    Args:
        x_vector (np.ndarray): Array to store evaluated points.
        y_vector (np.ndarray): Array to store objective values at evaluated points.
        bounds (np.ndarray): Array of (min, max) bounds for each dimension.
        function (callable): The function to optimize.
        n_samples (int): Number of initial samples to generate.

    Returns:
        int: Number of evaluations performed.
    """

    dim = len(bounds)

    # Calculate grid size for uniform distribution
    # For n_samples points in dim dimensions: grid_size = ceil(n_samples^(1/dim))
    grid_size = int(np.ceil(n_samples ** (1.0 / dim)))

    # Generate uniform grid points
    initial_guesses = generate_uniform_grid(bounds, dim, grid_size, n_samples)

    # Evaluate all initial samples
    n_evaluations = 0
    for i in range(len(initial_guesses)):  # pylint: disable=consider-using-enumerate
        x_vector[i] = initial_guesses[i]
        y_vector[i] = function(x_vector[i])
        n_evaluations += 1

        if DEBUG_MODE:
            print(
                f"➡️  Debug: Initial sample {i+1}: x = {x_vector[i]}, y = {y_vector[i]}"
            )

    if DEBUG_MODE:
        print(f"🎯 Debug: Initialized {n_evaluations} samples uniformly.\n")

    return n_evaluations


@njit
def generate_uniform_grid(bounds, dim, grid_size, max_samples):
    """
    Generate uniformly distributed grid points within bounds.

    Args:
        bounds (np.ndarray): Array of (min, max) bounds for each dimension.
        dim (int): Number of dimensions.
        grid_size (int): Number of points per dimension.
        max_samples (int): Maximum number of samples to return.

    Returns:
        list: List of uniformly distributed sample points.
    """

    # Create grid coordinates for each dimension
    grid_coords = []
    for d in range(dim):
        min_val, max_val = bounds[d][0], bounds[d][1]

        if grid_size == 1:
            # If only one point per dimension, use midpoint
            coords = np.array([(min_val + max_val) // 2], dtype=np.int32)
        else:
            # Create evenly spaced points including endpoints
            coords = np.linspace(min_val, max_val, grid_size).astype(np.int32)

        grid_coords.append(coords)

    # Generate all combinations of grid coordinates
    samples = []
    total_combinations = grid_size**dim

    for i in range(min(total_combinations, max_samples)):
        sample = np.zeros(dim, dtype=np.int32)

        # Convert linear index to multi-dimensional coordinates
        temp_i = i
        for d in range(dim):
            coord_idx = temp_i % grid_size
            sample[d] = grid_coords[d][coord_idx]
            temp_i //= grid_size

        samples.append(sample)

        if len(samples) >= max_samples:
            break

    return samples


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

    # Compute the inverse once
    base_matrix_inv = np.ascontiguousarray(np.linalg.inv(base_matrix))

    for obj_idx in range(n_objectives):
        # Scale with prior variance
        kernel_matrix_inv[obj_idx] = np.ascontiguousarray(
            base_matrix_inv / prior_variance[obj_idx]
        )

    return kernel_matrix_inv


@njit
def rbf_kernel(x1, x2, var, length_scale=1.0):
    """
    Radial basis function (RBF) kernel for multi-dimensional inputs.

    Args:
        x1 (np.ndarray): First input point in the hyperplane.
        x2 (np.ndarray): Second input point in the hyperplane.
        var (float): Variance parameter for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        float: The value of the RBF kernel between x1 and x2.
    """

    euclidean_distance_2 = np.sum((x1 - x2) ** 2)
    return var * np.exp(-0.5 * euclidean_distance_2 / (length_scale**2))


@njit
def update_k(
    kernel_matrix,
    x_vector,
    current_eval,
    var,
    length_scale=1.0,
):
    """
    Compute the kernel matrix for the training points.

    Args:
        kernel_matrix (np.ndarray): Preallocated kernel matrix to fill.
        x_vector (np.ndarray): Training points.
        current_eval (int): Current number of evaluations.
        var (float): Variance parameter for the kernel.
        length_scale (float): Length scale parameter for the kernel.
    """

    n_objectives = kernel_matrix.shape[0]

    for i in prange(current_eval):  # pylint: disable=not-an-iterable
        # Compute only the upper triangle (kernel is symmetric)
        for j in range(i, current_eval):
            kernel_matrix[:, i, j] = rbf_kernel(
                x_vector[i], x_vector[j], 1.0, length_scale
            )
            # Symmetric entry
            kernel_matrix[:, j, i] = kernel_matrix[:, i, j]

    # The only difference in the matrices for different objectives is the prior variance
    for obj_idx in range(n_objectives):
        kernel_matrix[obj_idx] *= var[obj_idx]


@njit
def update_k_star(
    k_star,
    x_vector,
    input_space,
    current_eval,
    var,
    length_scale=1.0,
):
    """
    Update the kernel vector for a new point based on the training points.

    Args:
        k_star (np.ndarray): Preallocated kernel vector to fill up to the current_eval.
        x_vector (np.ndarray): Training points.
        input_space (np.ndarray): Discretized input space.
        current_eval (int): Current number of evaluations.
        var (np.ndarray): Variance parameter for the kernel.
        length_scale (float): Length scale parameter for the kernel.
    """

    n_objectives = k_star.shape[0]
    n = len(input_space)

    # Compute the same rbf kernel for all objectives, as the only difference is the variance
    for e in range(current_eval):
        eval_x = x_vector[e]
        for i in range(n):
            x_star = input_space[i]
            k_star[:, e, i] = rbf_kernel(eval_x, x_star, 1.0, length_scale)

    # Modify the  k_star based on the prior variance for each objective
    for obj_idx in range(n_objectives):
        k_star[obj_idx] *= var[obj_idx]


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
        mu_objectives (np.ndarray): Preallocated mean predictions for each objective.
        k_star (np.ndarray): Kernel vector for the new point.
        inverted_kernel_matrix (np.ndarray): Inverted kernel matrix for the training points.
        y_vector (np.ndarray): Objective values at evaluated points.
        prior_mean (np.ndarray): Prior mean for each objective.
        current_eval (int): Current number of evaluations.
    """

    n_objectives = mu_objectives.shape[0]

    for obj_idx in range(n_objectives):
        # Access the inverted kernel matrix for this objective
        kernel_matrix_inv = inverted_kernel_matrix[obj_idx]

        # Precompute delta_y
        delta_y = np.ascontiguousarray(
            y_vector[:current_eval, obj_idx] - prior_mean[obj_idx]
        )

        # Extract k_star for this objective and ensure contiguous
        k_star_obj = np.ascontiguousarray(k_star[obj_idx, :current_eval, :])

        # Vectorized computation: k_star.T @ (K_inv @ delta_y)
        partial_result = kernel_matrix_inv @ delta_y

        # Transpose k_star for proper matrix multiplication
        k_star_t = np.ascontiguousarray(k_star_obj.T)  # (n_points, current_eval)

        # Compute all means at once
        mu_objectives[obj_idx, :] = prior_mean[obj_idx] + k_star_t @ partial_result


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
        variance_objectives (np.ndarray): Preallocated variance predictions for each objective.
        k_star (np.ndarray): Kernel vector for the new point.
        inverted_kernel_matrix (np.ndarray): Inverted kernel matrix for the training points.
        prior_variance (np.ndarray): Prior variance for each objective.
        current_eval (int): Current number of evaluations.
    """

    n_objectives = variance_objectives.shape[0]

    for obj_idx in range(n_objectives):
        # Access the inverted kernel matrix for this objective
        kernel_matrix_inv = inverted_kernel_matrix[obj_idx]

        # Extract k_star for this objective and ensure contiguous
        k_star_obj = np.ascontiguousarray(k_star[obj_idx, :current_eval, :])

        # Transpose k_star for proper matrix operations
        k_star_t = np.ascontiguousarray(k_star_obj.T)  # (n_points, current_eval)

        # Vectorized computation: k_star.T @ K_inv @ k_star
        intermediate = kernel_matrix_inv @ k_star_obj  # (current_eval, n_points)

        # Second: k_star.T @ intermediate -> (n_points,)
        quadratic_form = np.sum(k_star_t * intermediate.T, axis=1)

        # Compute all variances at once
        variance_objectives[obj_idx, :] = prior_variance[obj_idx] - quadratic_form


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

        # Standardize variance: var / prior_variance
        std_variance_objectives[obj_idx] = (
            variance_objectives[obj_idx] / prior_variance[obj_idx]
        )


@njit
def upper_confidence_bound(mu, variance, beta=2.0):
    """
    Compute the upper confidence bound for a single objective, vectorized.

    Args:
        mu (np.ndarray): Mean predictions for the objective.
        variance (np.ndarray): Variance predictions for the objective.
        beta (float): Exploration-exploitation trade-off parameter.

    Returns:
        np.ndarray: Upper confidence bound values for the objective.
    """

    return mu + beta * np.sqrt(np.abs(variance))


@njit
def update_ucb(
    ucb,
    mu_objectives,
    variance_objectives,
    beta=2.0,
):
    """
    Update the upper confidence bound acquisition function values.

    Args:
        ucb (np.ndarray): Preallocated upper confidence bound values.
        mu_objectives (np.ndarray): Mean predictions for each objective.
        variance_objectives (np.ndarray): Variance predictions for each objective.
        beta (float): Exploration-exploitation trade-off parameter.
    """

    n_objectives = mu_objectives.shape[0]

    # Compute the UCB for each point vectorized
    for obj_idx in range(n_objectives):
        ucb[obj_idx] = upper_confidence_bound(
            mu_objectives[obj_idx],
            variance_objectives[obj_idx],
            beta,
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


@njit
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
    n_iterations,
    n_objectives,  # pylint: disable=unused-argument
    function,
    beta=2.0,
    length_scale=3.0,
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
        n_iterations (int): Total number of iterations.
        n_objectives (int): Number of objectives.
        function (callable): The function to optimize.
        beta (float): Exploration-exploitation trade-off parameter.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        tuple: Updated x_vector, y_vector, and number of evaluations.
    """

    # Total number of evaluations
    last_eval = 0

    for current_eval in range(n_evaluations, n_iterations):
        if DEBUG_MODE:
            print(
                f"🔄 Debug: Starting iteration {current_eval}, n_evaluations={n_evaluations}"
            )

        # Update kernel matrices for each objective
        update_k(
            kernel_matrix=kernel_matrices,
            x_vector=x_vector,
            current_eval=current_eval,
            var=prior_variance,
            length_scale=length_scale,
        )

        # Invert matrix
        kernel_matrix_inv = invert_k(
            current_eval=current_eval,
            kernel_matrix=kernel_matrices,
            prior_variance=prior_variance,
        )

        # Update k star for each objective
        update_k_star(
            k_star=k_star,
            x_vector=x_vector,
            input_space=input_space,
            current_eval=current_eval,
            var=prior_variance,
            length_scale=length_scale,
        )

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
            beta=beta,
        )

        # Update hypervolume improvement acquisition function
        update_hypervolume_improvement(
            acquisition_values=acquisition_values,
            ucb=ucb,
        )

        # Select the next point to evaluate
        x_next = input_space[np.argmax(acquisition_values)]

        if DEBUG_MODE:
            print(
                f"""🔍 Debug: Selected next point: {x_next} """
                f"""with hypervolume improvement {acquisition_values.max()}"""
            )

        # Check if x_next is already evaluated
        already_evaluated = False
        for j in range(current_eval):
            if np.all(x_vector[j] == x_next):
                already_evaluated = True
                break

        # Update the total number of evaluations
        last_eval = current_eval

        if not already_evaluated:
            # Evaluate the function at the new point
            x_vector[current_eval] = x_next
            y_vector[current_eval] = function(x_next)

            if DEBUG_MODE:
                print(f"✅ Debug: Objective values: {y_vector[current_eval]}\n")
        else:
            if DEBUG_MODE:
                print("🎯 Debug: Point already evaluated, stopping optimization\n")
            break

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
        """

        self.function = function
        self.bounds = bounds
        self.n_objectives = n_objectives
        self.n_iterations = n_iterations

        # If prior mean and variance are not provided, calculate them later from initial samples
        self.prior_mean = np.array(kwargs.get("prior_mean", [0.0] * n_objectives))
        self.prior_variance = np.array(
            kwargs.get("prior_variance", [1.0] * n_objectives)
        )

        # Initial number of samples to evaluate
        self.initial_samples = kwargs.get("initial_samples", 3)

        # Dimensionality of the input space
        self.dim = len(bounds)

        # Preallocate the input space as a unique array of cartesian points
        ranges = [np.arange(b[0], b[1]) for b in bounds]
        mesh = np.meshgrid(*ranges, indexing="ij")
        self.input_space = np.stack([m.ravel() for m in mesh], axis=-1)

        # Preallocate the function evaluations
        self.x_vector = np.zeros((n_iterations, self.dim))
        self.y_vector = np.zeros((n_iterations, n_objectives))  # Multiple objectives

        # Preallocate the kernel matrices for each objective
        self.kernel_matrices = np.zeros(
            (self.n_objectives, self.n_iterations, self.n_iterations), dtype=np.float64
        )

        # Preallocate the kernel vector kstar for each objective
        self.k_star = np.zeros(
            (self.n_objectives, self.n_iterations, len(self.input_space)),
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
        self.n_evaluations = initialize_samples(
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

    def optimize(self, beta=2.0, length_scale=3.0):
        """
        Perform the Multi-Objective Bayesian optimization.

        Args:
            beta (float): Exploration-exploitation trade-off parameter.
            length_scale (float): Length scale parameter for the kernel.
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
            n_iterations=self.n_iterations,
            n_objectives=self.n_objectives,
            function=self.function,
            beta=beta,
            length_scale=length_scale,
        )

    def pareto_analysis(self):
        """
        Perform Pareto analysis on the results of the optimization.
        """

        is_efficient = is_pareto_efficient(self.y_vector[: self.n_evaluations])

        # Extract Pareto-efficient points
        pareto_points = self.y_vector[is_efficient]

        # Extract point of the input space corresponding to Pareto-efficient points
        pareto_indices = np.where(is_efficient)[0]
        pareto_input_points = self.x_vector[pareto_indices]

        print("📊 Pareto Analysis Results:")

        for i, point in enumerate(pareto_points):
            print(f"Input: {pareto_input_points[i]}, Pareto Point {i + 1}: {point}")

        return pareto_points


if __name__ == "__main__":
    # Example usage
    _bounds = [
        (0, X_MAX),
        (0, Y_MAX),
        (0, Z_MAX),
    ]

    start_time = time.time()
    print("\n⚡ Starting optimization...\n")

    optimizer = BayesianOptimization(
        toy_function,
        _bounds,
        n_objectives=len(_bounds),
        n_iterations=40,
        # prior_mean=[50] * len(_bounds),  # Initial prior mean
        # prior_variance=[400] * len(_bounds),  # Initial prior variance
        initial_samples=2 ** len(_bounds),  # 8 initial samples (2^3 for 3D space)
    )

    optimizer.optimize(
        beta=1.0,
        length_scale=2.0,
    )

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    optimizer.pareto_analysis()
