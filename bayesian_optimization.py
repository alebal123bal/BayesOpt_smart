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
    initial_guesses = [
        np.array([5, 1, 1], dtype=np.int32),
        np.array([10, 2, 2], dtype=np.int32),
        np.array([15, 3, 3], dtype=np.int32),
    ]
    n_evaluations = 0

    for i in range(len(initial_guesses)):
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
def normalize_mean(mu, norm_mu):
    """
    Scale the mean predictions to a range of [0 + eps, 1].

    Args:
        mu (np.ndarray): Mean predictions for each objective.
        norm_mu (np.ndarray): Normalized mean predictions for each objective.
    """

    for obj_idx in range(mu.shape[0]):
        max_value = np.max(mu[obj_idx])
        min_value = np.min(mu[obj_idx])
        eps = 1e-15  # Small epsilon to avoid division by zero
        norm_mu[obj_idx] = (mu[obj_idx] - min_value + eps) / (
            max_value - min_value + eps
        )

    return norm_mu


@njit
def normalize_variance(var, norm_var):
    """
    Scale in-place the variance predictions to a range of [eps, 1].

    Args:
        var (np.ndarray): Variance predictions for each objective.
        norm_var (np.ndarray): Normalized variance predictions for each objective.
    """

    for obj_idx in range(var.shape[0]):
        max_value = np.max(var[obj_idx])
        norm_var[obj_idx] = var[obj_idx] / max_value

    return norm_var


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
def compute_k_star(x_vector, x_star, var, length_scale=1.0):
    """
    Compute the kernel vector between the training points and a new point.

    Args:
        x_vector (np.ndarray): Training points.
        x_star (np.ndarray): New point.
        var (float): Variance parameter for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        np.ndarray: Kernel vector between training points and the new point.
    """
    n = len(x_vector)
    k_star = np.empty(n, dtype=np.float64)  # Preallocate array for the kernel vector

    for i in range(n):
        k_star[i] = rbf_kernel(x_vector[i], x_star, var, length_scale)

    return k_star


@njit(parallel=True)
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

    # TODO : avoid recalculating the kernel matrix for already evaluated points (make it grow)
    for i in prange(current_eval):  # pylint: disable=not-an-iterable
        # Compute only the upper triangle (kernel is symmetric)
        for j in range(i, current_eval):
            kernel_matrix[:, i, j] = rbf_kernel(
                x_vector[i, 0], x_vector[j, 0], 1.0, length_scale
            )
            # Symmetric entry
            kernel_matrix[:, j, i] = kernel_matrix[:, i, j]

    # The only difference between the kernel matrices for different objectives
    # is the variance parameter
    for obj_idx in range(n_objectives):
        kernel_matrix[obj_idx] *= var[obj_idx]


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
    n_total = 0

    # Preallocate normalized mean and variance arrays
    norm_mu_objectives = np.zeros((n_objectives, len(input_space)), dtype=np.float64)
    norm_variance_objectives = np.zeros(
        (n_objectives, len(input_space)), dtype=np.float64
    )

    for current_eval in range(n_evaluations, n_iterations):
        if DEBUG_MODE:
            print(
                f"🔄 Debug: Starting iteration {current_eval}, n_evaluations={n_evaluations}"
            )

        # Compute kernel matrices for each objective
        update_k(
            kernel_matrix=kernel_matrices,
            x_vector=x_vector,
            current_eval=current_eval,
            var=prior_variance,
            length_scale=length_scale,
        )

        # Loop through each objective to compute mean and variance predictions
        for obj_idx in range(n_objectives):
            # Loop through the input space to update mean and variance predictions
            for i in range(
                len(input_space)
            ):  # pylint: disable=consider-using-enumerate
                x_star = input_space[i]

                # Compute the kernel vector for the new point
                k_star = compute_k_star(
                    x_vector[:current_eval],
                    x_star,
                    var=prior_variance[obj_idx],
                    length_scale=length_scale,
                )

                # Update mean and variance for each objective
                delta_mu = compute_delta_mu(
                    k_star=k_star,
                    kernel_matrix=kernel_matrices[
                        obj_idx, :current_eval, :current_eval
                    ],
                    y_vector=y_vector[:current_eval, obj_idx],
                    prior_mean=prior_mean[obj_idx],
                )

                mu_objectives[obj_idx, i] = prior_mean[obj_idx] + delta_mu

                # Update variance
                delta_variance = compute_delta_variance(
                    k_star=k_star,
                    kernel_matrix=kernel_matrices[
                        obj_idx, :current_eval, :current_eval
                    ],
                )

                variance_objectives[obj_idx, i] = (
                    prior_variance[obj_idx] + delta_variance
                )

        # Normalize the mean and variance predictions for the hypervolume improvement
        normalize_mean(mu=mu_objectives, norm_mu=norm_mu_objectives)
        normalize_variance(var=variance_objectives, norm_var=norm_variance_objectives)

        # Loop through the input space to compute acquisition function values
        for obj_idx in range(n_objectives):
            for i in range(len(input_space)):
                # Compute multi-objective acquisition function
                acquisition_values[i] = hypervolume_improvement(
                    norm_mu_objectives[:, i],
                    norm_variance_objectives[:, i],
                    reference_point,
                    beta=beta,
                )

        # Select the next point to evaluate
        x_next = input_space[np.argmax(acquisition_values)]

        if DEBUG_MODE:
            print(
                f"🔍 Debug: Selected next point: {x_next} with hypervolume improvement {acquisition_values.max()}"
            )

        # Check if x_next is already evaluated
        already_evaluated = False
        for j in range(current_eval):
            if np.all(x_vector[j] == x_next):
                already_evaluated = True
                break

        if not already_evaluated:
            # Evaluate the function at the new point
            x_vector[current_eval] = x_next
            y_vector[current_eval] = function(x_next)

            if DEBUG_MODE:
                print(f"✅ Debug: Objective values: {y_vector[current_eval]}\n")
        else:
            if DEBUG_MODE:
                print("🎯 Debug: Point already evaluated, stopping optimization\n")
                n_total = current_eval
            break

    return x_vector, y_vector, n_total


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

        # Preallocate the mean for each objective's Gaussian process
        self.mu_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the variance for each objective's Gaussian process
        self.variance_objectives = np.zeros(
            (n_objectives, len(self.input_space)), dtype=np.float64
        )

        # Preallocate the acquisition function values for each point
        self.acquisition_values = np.zeros(len(self.input_space), dtype=np.float64)

        # Initial guesses
        self.n_evaluations = initialize_samples(
            self.x_vector,
            self.y_vector,
            self.dim,
            self.function,
            self.initial_samples,
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
            beta=beta,
            length_scale=length_scale,
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

        if DEBUG_MODE:
            # Print results
            print("\n📊 Final results:")
            for i in range(self.n_evaluations):
                print(f"x = {self.x_vector[i]}, objectives = {self.y_vector[i]}")

        print(f"\n⚖️  Pareto optimal solutions found: {len(pareto_solutions)}")
        for i, (x_pareto, obj_pareto) in enumerate(
            zip(pareto_solutions, pareto_objectives)
        ):
            print(f"Pareto {i+1}: x = {x_pareto}, objectives = {obj_pareto}")


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
        n_objectives=len(
            _bounds
        ),  # Number of objectives is equal to the number of bounds
        n_iterations=30,
        initial_samples=2 ** len(_bounds),  # 8 initial samples (2^3 for 3D space)
    )

    optimizer.optimize(
        beta=1.5,
        length_scale=3.0,
    )

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    optimizer.pareto_analysis()
