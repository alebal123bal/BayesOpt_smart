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
        # Called as @njit() or @njit(parallel=True)
        return decorator

    def prange(n):
        """
        Dummy prange function for debugging purposes.
        """
        return range(n)

else:
    print("🚀 PRODUCTION MODE - Numba enabled")
    from numba import njit, prange

X_MAX = 30
Y_MAX = 30


@njit
def toy_function(x):
    """
    A multi-objective toy function to optimize.

    Args:
        x (np.ndarray): Input array (e.g., [x1, x2, ..., xd]).

    Returns:
        np.ndarray: Output array containing [f(x), g(x), h(x)].
    """
    f_x = -np.sum((x - 12) ** 2) + 100
    g_x = -np.sum((x - 12) ** 2) + 80
    h_x = -np.sum((x - 12) ** 2) + 120

    return np.array([f_x, g_x, h_x])


@njit
def initialize_samples(x_vector, y_vector, dim, function, n_samples=3):
    """
    Initialize the first few sample points for the optimization.

    Args:
        x_vector (np.ndarray): Array to store evaluated points.
        y_vector (np.ndarray): Array to store objective values at evaluated points.
        dim (int): Dimensionality of the input space.
        function (callable): The function to optimize.
        n_samples (int): Number of initial samples to generate.

    Returns:
        int: Number of evaluations performed.
    """
    # Initial guesses (keep integers for simplicity)
    initial_guesses = np.random.randint(low=0, high=X_MAX, size=(n_samples, dim))
    n_evaluations = 0

    for i in range(n_samples):
        x_vector[i] = initial_guesses[i]
        y_vector[i] = function(x_vector[i])
        n_evaluations += 1
        if DEBUG_MODE:
            print(
                f"➡️  Debug: Initial sample {i+1}: x = {x_vector[i]}, y = {y_vector[i]}"
            )

    if DEBUG_MODE:
        print(f"🎯 Debug: Initialized {n_evaluations} samples.\n")

    return n_evaluations


@njit
def rbf_kernel(x1, x2, sigma, length_scale=1.0):
    """
    Radial basis function (RBF) kernel for multi-dimensional inputs.

    Args:
        x1 (np.ndarray): First input array.
        x2 (np.ndarray): Second input array.
        sigma (float): Standard deviation for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        float: The value of the RBF kernel between x1 and x2.
    """
    distance_squared = 0.0
    for i in range(len(x1)):  # pylint: disable=consider-using-enumerate
        distance_squared += (x1[i] - x2[i]) ** 2
    return sigma**2 * np.exp(-0.5 * distance_squared / (length_scale**2))


@njit
def compute_k_star(x_vector, x_star, sigma, length_scale=1.0):
    """
    Compute the kernel vector between the training points and a new point.

    Args:
        x_vector (np.ndarray): Training points.
        x_star (np.ndarray): New point.
        sigma (float): Standard deviation for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        np.ndarray: Kernel vector between training points and the new point.
    """
    n = len(x_vector)
    k_star = np.empty(n, dtype=np.float64)  # Preallocate array for the kernel vector

    for i in range(n):
        k_star[i] = rbf_kernel(x_vector[i], x_star, sigma, length_scale)

    return k_star


@njit(parallel=True)
def compute_k(x_vector, sigma, length_scale=1.0):
    """
    Compute the kernel matrix for the training points.

    Args:
        x_vector (np.ndarray): Training points.
        sigma (float): Standard deviation for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        np.ndarray: Kernel matrix for the training points.
    """
    n = len(x_vector)
    kernel_matrix = np.empty((n, n), dtype=np.float64)  # Preallocate the matrix

    for i in prange(n):  # pylint: disable=not-an-iterable
        for j in range(i, n):  # Compute only the upper triangle (kernel is symmetric)
            value = rbf_kernel(x_vector[i], x_vector[j], sigma, length_scale)
            kernel_matrix[i, j] = value
            kernel_matrix[j, i] = value  # Fill the symmetric element

    return kernel_matrix


@njit
def compute_delta_mu(k_star, kernel_matrix, y_vector, prior_mean):
    """
    Update the mean of the Gaussian process at a new point.

    Args:
        k_star (np.ndarray): Kernel vector for the new point.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.
        y_vector (np.ndarray): Function values at the training points.
        prior_mean (float): Prior mean of the Gaussian process.

    Returns:
        float: Delta mean at the new point to be added to precomputed vector.
    """
    kernel_matrix_inv = np.linalg.inv(kernel_matrix)
    mu = k_star.T @ kernel_matrix_inv @ (y_vector - prior_mean)
    return mu


@njit
def compute_delta_variance(k_star, kernel_matrix):
    """
    Update the variance of the Gaussian process at a new point.

    Args:
        k_star (np.ndarray): Kernel vector for the new point.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.

    Returns:
        float: Delta variance at the new point to be added to precomputed vector.
    """
    kernel_matrix_inv = np.linalg.inv(kernel_matrix)
    variance = -k_star.T @ kernel_matrix_inv @ k_star
    return variance


@njit
def upper_confidence_bound(mu, variance, beta=2.0):
    """
    Compute the upper confidence bound for a Gaussian process.

    Args:
        mu (float): Mean of the Gaussian process at a point.
        variance (float): Variance of the Gaussian process at a point.
        beta (float): Exploration-exploitation trade-off parameter.

    Returns:
        float: Upper confidence bound value.
    """
    return mu + beta * np.sqrt(np.abs(variance))


@njit
def hypervolume_improvement(
    mu_objectives,
    variance_objectives,
    reference_point,  # pylint: disable=unused-argument
    beta=2.0,
):
    """
    Compute a multi-objective acquisition function based on hypervolume improvement.
    This is a simplified version - you may want to use more sophisticated methods.

    Args:
        mu_objectives (np.ndarray): Mean predictions for each objective.
        variance_objectives (np.ndarray): Variance predictions for each objective.
        reference_point (np.ndarray): Reference point for hypervolume calculation.
        beta (float): Exploration parameter.

    Returns:
        float: Acquisition function value.
    """
    n_objectives = len(mu_objectives)

    # Preallocate ucb_values array
    ucb_values = np.empty(n_objectives, dtype=np.float64)

    # Compute UCB values for each objective
    for i in range(n_objectives):
        ucb_values[i] = upper_confidence_bound(
            mu_objectives[i], variance_objectives[i], beta
        )

    # Preallocate weights array (equal weights for all objectives)
    weights = np.ones(n_objectives, dtype=np.float64)

    # Compute weighted sum of UCB values
    acquisition_value = np.dot(weights, ucb_values)

    return acquisition_value


@njit
def optimize(
    x_vector,
    y_vector,
    kernel_matrices,
    mu_objectives,
    variance_objectives,
    acquisition_values,
    input_space,
    prior_mean,
    prior_variance,
    reference_point,
    n_evaluations,
    n_iterations,
    n_objectives,
    function,
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

    Returns:
        Updated `x_vector` and `y_vector` after optimization.
    """
    for f in range(3, n_iterations):  # pylint: disable=unused-variable
        if DEBUG_MODE:
            print(f"🔄 Debug: Starting iteration {f}, n_evaluations={n_evaluations}")

        # Compute kernel matrices for each objective
        for obj_idx in range(n_objectives):
            kernel_matrices[obj_idx, :n_evaluations, :n_evaluations] = compute_k(
                x_vector[:n_evaluations],
                sigma=np.sqrt(prior_variance[obj_idx]),
                length_scale=3.0,
            )

        for i in range(len(input_space)):  # pylint: disable=consider-using-enumerate
            x_star = input_space[i]

            # Compute the kernel vector for the new point
            k_star = compute_k_star(
                x_vector[:n_evaluations],
                x_star,
                sigma=np.sqrt(np.abs(prior_variance[obj_idx])),
                length_scale=3.0,
            )

            # Initialize mean and variance predictions for each objective
            mu_pred = np.zeros(n_objectives, dtype=np.float64)
            var_pred = np.zeros(n_objectives, dtype=np.float64)

            # Update mean and variance for each objective
            for obj_idx in range(n_objectives):
                # Update mean
                delta_mu = compute_delta_mu(
                    k_star=k_star,
                    kernel_matrix=kernel_matrices[
                        obj_idx, :n_evaluations, :n_evaluations
                    ],
                    y_vector=y_vector[:n_evaluations, obj_idx],
                    prior_mean=prior_mean[obj_idx],
                )

                mu_objectives[obj_idx, i] = prior_mean[obj_idx] + delta_mu
                mu_pred[obj_idx] = mu_objectives[obj_idx, i]

                # Update variance
                delta_variance = compute_delta_variance(
                    k_star=k_star,
                    kernel_matrix=kernel_matrices[
                        obj_idx, :n_evaluations, :n_evaluations
                    ],
                )

                variance_objectives[obj_idx, i] = (
                    prior_variance[obj_idx] + delta_variance
                )
                var_pred[obj_idx] = variance_objectives[obj_idx, i]

            # Compute multi-objective acquisition function
            acquisition_values[i] = hypervolume_improvement(
                mu_pred,
                var_pred,
                reference_point,
                beta=2.0,
            )

        # Select the next point to evaluate
        x_next = input_space[np.argmax(acquisition_values)]

        if DEBUG_MODE:
            print(f"  Debug: Selected next point: {x_next}")

        # Check if x_next is already evaluated
        already_evaluated = False
        for j in range(n_evaluations):
            if np.all(x_vector[j] == x_next):
                already_evaluated = True
                break

        if not already_evaluated:
            # Evaluate the function at the new point
            x_vector[n_evaluations] = x_next
            y_vector[n_evaluations] = function(x_next)

            if DEBUG_MODE:
                print(f"  Debug: Function value: {y_vector[n_evaluations]}")

            # Increment the number of evaluations
            n_evaluations += 1
        else:
            if DEBUG_MODE:
                print("🎯 Debug: Point already evaluated, stopping optimization")
            # Stop if the point is already evaluated
            break

    return x_vector, y_vector, n_evaluations


def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.

    Args:
        costs (np.ndarray): An (n_points, n_objectives) array

    Returns:
        np.ndarray: A boolean array indicating which points are Pareto efficient
    """

    # Efficient set is defined as the set of points that are not dominated by any other point
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0

    # Iterate through each point and check if it is dominated by any other point
    while next_point_index < len(costs):
        # Check if the current point is dominated by any other point
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # If the current point is dominated by any other point, remove it from the efficient set
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]

        # Move to the next point
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


class MultiObjectiveBayesianOptimization:
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
        self.prior_mean = np.array(kwargs.get("prior_mean", [0.0] * n_objectives))
        self.prior_variance = np.array(
            kwargs.get("prior_variance", [1.0] * n_objectives)
        )
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

        # Mean for each objective's Gaussian process
        self.mu_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Variance for each objective's Gaussian process
        self.variance_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate acquisition function values for each point
        self.acquisition_values = np.zeros(len(self.input_space), dtype=np.float64)

        # Initial guesses
        self.n_evaluations = initialize_samples(
            self.x_vector, self.y_vector, self.dim, self.function
        )

        # Reference point for hypervolume (should be worse than any expected objective value)
        self.reference_point = np.array([0.0] * n_objectives)

    def optimize(self):
        """
        Perform the Multi-Objective Bayesian optimization.
        """
        # Optimize with numba
        self.x_vector, self.y_vector, self.n_evaluations = optimize(
            self.x_vector,
            self.y_vector,
            self.kernel_matrices,
            self.mu_objectives,
            self.variance_objectives,
            self.acquisition_values,
            self.input_space,
            self.prior_mean,
            self.prior_variance,
            self.reference_point,
            self.n_evaluations,
            self.n_iterations,
            self.n_objectives,
            self.function,
        )

    def pareto_analysis(self):
        """
        Perform Pareto analysis on the results of the optimization.
        """

        # Find Pareto frontier
        pareto_mask = is_pareto_efficient(
            -self.y_vector[: self.n_evaluations]
        )  # Negative for maximization
        pareto_solutions = self.x_vector[: self.n_evaluations][pareto_mask]
        pareto_objectives = self.y_vector[: self.n_evaluations][pareto_mask]

        # Print results
        print("Final results:")
        for i in range(self.n_evaluations):
            print(f"x = {self.x_vector[i]}, objectives = {self.y_vector[i]}")

        print(f"\nPareto optimal solutions found: {len(pareto_solutions)}")
        for i, (x_pareto, obj_pareto) in enumerate(
            zip(pareto_solutions, pareto_objectives)
        ):
            print(f"Pareto {i+1}: x = {x_pareto}, objectives = {obj_pareto}")


if __name__ == "__main__":
    # Example usage
    _bounds = [
        (0, X_MAX),
        (0, Y_MAX),
    ]

    start_time = time.time()
    print("\nStarting optimization...\n")

    optimizer = MultiObjectiveBayesianOptimization(
        toy_function,
        _bounds,
        n_objectives=3,
        n_iterations=20,
        prior_mean=[50] * 3,
        prior_variance=[400.0] * 3,
    )

    optimizer.optimize()
    end_time = time.time()
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds.\n")

    optimizer.pareto_analysis()
