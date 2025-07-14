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
    g_x = -((x[1] - 1) ** 2) + 20
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
        np.array([17, 17, 17], dtype=np.int32),
        np.array([20, 4, 4], dtype=np.int32),
    ]
    # TODO: very important to have an even initial space for u and sigma eval

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

    # TODO : avoid recalculating the k star for already evaluated points (make it grow)
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
    kernel_matrix,
    y_vector,
    prior_mean,
    current_eval,
):
    """
    Update the mean predictions for each objective based on the kernel vector and training points.

    Args:
        mu_objectives (np.ndarray): Preallocated mean predictions for each objective.
        k_star (np.ndarray): Kernel vector for the new point.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.
        y_vector (np.ndarray): Function values at the training points.
        prior_mean (np.ndarray): Prior mean for each objective.
        current_eval (int): Current number of evaluations.
    """

    n_objectives = mu_objectives.shape[0]
    n = len(k_star[0, 0])  # Number of points in the input space

    for obj_idx in range(n_objectives):
        # Precompute the inverse of the kernel matrix for the current objective
        _kernel_matrix_inv = np.linalg.inv(
            kernel_matrix[obj_idx, :current_eval, :current_eval]
        )

        # Precompute the difference between the current evaluations and the prior mean
        _delta_y = y_vector[:current_eval, obj_idx] - prior_mean[obj_idx]

        # Precompute their dot product with the kernel vector
        _partial_mu = _kernel_matrix_inv @ _delta_y

        # Cycle the input space to compute the mean for each point
        for i in range(n):
            # Compute the mean prediction for the current point
            mu_objectives[obj_idx, i] = (
                prior_mean[obj_idx] + k_star[obj_idx, :current_eval, i].T @ _partial_mu
            )


@njit
def update_variance(
    variance_objectives,
    k_star,
    kernel_matrix,
    prior_variance,
    current_eval,
):
    """
    Update the variance predictions for each objective based on the kernel vector
    and training points.

    Args:
        variance_objectives (np.ndarray): Preallocated variance predictions for each objective.
        k_star (np.ndarray): Kernel vector for the new point.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.
        prior_variance (np.ndarray): Prior variance for each objective.
        current_eval (int): Current number of evaluations.
    """

    n_objectives = variance_objectives.shape[0]
    n = len(k_star[0, 0])  # Number of points in the input space

    for obj_idx in range(n_objectives):
        # Precompute the inverse of the kernel matrix for the current objective
        _kernel_matrix_inv = np.linalg.inv(
            kernel_matrix[obj_idx, :current_eval, :current_eval]
        )

        # Cycle through the input space to compute the variance for each point
        for i in range(n):
            # Compute the variance prediction for the current point
            variance_objectives[obj_idx, i] = (
                prior_variance[obj_idx]
                - k_star[obj_idx, :current_eval, i].T
                @ _kernel_matrix_inv
                @ k_star[obj_idx, :current_eval, i]
            )


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
    n_points = len(ucb[0])

    # Compute the UCB for each point
    for obj_idx in range(n_objectives):
        for i in range(n_points):
            ucb[obj_idx, i] = upper_confidence_bound(
                mu_objectives[obj_idx, i],
                variance_objectives[obj_idx, i],
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
    ucb,
    acquisition_values,
    input_space,
    prior_mean,
    prior_variance,
    reference_point,  # pylint: disable=unused-argument
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
    norm_mu_objectives = np.zeros(
        (n_objectives, len(input_space)), dtype=np.float64
    )  # pylint: disable=unused-variable
    norm_variance_objectives = np.zeros(
        (n_objectives, len(input_space)), dtype=np.float64
    )  # pylint: disable=unused-variable

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
            kernel_matrix=kernel_matrices,
            y_vector=y_vector,
            prior_mean=prior_mean,
            current_eval=current_eval,
        )

        # Update variance predictions for each objective
        update_variance(
            variance_objectives=variance_objectives,
            k_star=k_star,
            kernel_matrix=kernel_matrices,
            prior_variance=prior_variance,
            current_eval=current_eval,
        )

        # Normalize the mean and variance predictions for the hypervolume improvement

        # Update Upper Confidence Bound (UCB)
        update_ucb(
            ucb=ucb,
            mu_objectives=mu_objectives,
            variance_objectives=variance_objectives,
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
                f"🔍 Debug: Selected next point: {x_next} with hypervolume improvement {acquisition_values.max()}"
            )

        # Check if x_next is already evaluated
        already_evaluated = False
        for j in range(current_eval):
            if np.all(x_vector[j] == x_next):
                already_evaluated = True
                break

        # Update the total number of evaluations
        n_total = current_eval

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

    return x_vector, y_vector, n_total + 1


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
            elif np.all(y_vector_neg[i] <= y_vector_neg[j]) and np.any(
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

        # Preallocate the upper confidence bound acquisition function values for each objective
        self.ucb = np.zeros((n_objectives, len(self.input_space)), dtype=np.float64)

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
            x_vector=self.x_vector,
            y_vector=self.y_vector,
            kernel_matrices=self.kernel_matrices,
            k_star=self.k_star,
            mu_objectives=self.mu_objectives,
            variance_objectives=self.variance_objectives,
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

        if DEBUG_MODE:
            print("📊 Pareto Analysis Results:")

            for i, point in enumerate(pareto_points):
                print(
                    f"  Pareto Point {i + 1}: {point}, Input: {pareto_input_points[i]}"
                )

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
        n_iterations=10,
        # prior_mean=[50] * len(_bounds),  # Initial prior mean
        # prior_variance=[400] * len(_bounds),  # Initial prior variance
        initial_samples=2 ** len(_bounds),  # 8 initial samples (2^3 for 3D space)
    )

    optimizer.optimize(
        beta=2.0,
        length_scale=3.0,
    )

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    optimizer.pareto_analysis()
