"""
Multi-Objective Bayesian optimization optimized class.
"""

import time
import numpy as np
from numba import njit

X_MAX = 200
Y_MAX = 200


def toy_function(x):
    """
    A multi-objective toy function to optimize.

    Args:
        x (np.ndarray): Input array (e.g., [x1, x2, ..., xd]).

    Returns:
        np.ndarray: Output array containing [f(x), g(x), h(x)].
    """
    f_x = -np.sum(((x - 12) ** 2)) + 100
    g_x = -np.sum(((x - 12) ** 2)) + 80
    h_x = -np.sum(((x - 12) ** 2)) + 120

    return np.array([f_x, g_x, h_x])


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
    for i in range(len(x1)):
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
    kernel_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel_matrix[i, j] = rbf_kernel(
                x_vector[i], x_vector[j], sigma, length_scale
            )
    return kernel_matrix


def compute_delta_mu(k_star, kernel_matrix, y_vector, prior_mean):
    """
    Update the mean of the Gaussian process at a new point.

    Args:
        k_star (np.ndarray): Kernel vector for the new point.
        y_vector (np.ndarray): Function values at the training points.
        kernel_matrix (np.ndarray): Kernel matrix for the training points.
        prior_mean (float): Prior mean of the Gaussian process.

    Returns:
        float: Delta mean at the new point to be added to precomputed vector.
    """
    kernel_matrix_inv = np.linalg.inv(kernel_matrix)
    mu = k_star.T @ kernel_matrix_inv @ (y_vector - prior_mean)
    return mu


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
    return np.array(variance)


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
    return mu + beta * np.sqrt(variance)


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

    # Simple weighted sum of UCB values for each objective
    ucb_values = np.array(
        [
            upper_confidence_bound(mu_objectives[i], variance_objectives[i], beta)
            for i in range(len(mu_objectives))
        ]
    )

    # You can adjust weights based on objective importance
    weights = np.array([1.0] * len(mu_objectives))  # Equal weights
    return np.sum(weights * ucb_values)


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
        self.prior_mean = kwargs.get("prior_mean", [0.0] * n_objectives)
        self.prior_variance = kwargs.get("prior_variance", [1.0] * n_objectives)

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
        self.kernel_matrices = [
            np.zeros((n_iterations, n_iterations)) for _ in range(n_objectives)
        ]

        # Mean for each objective's Gaussian process
        self.mu_objectives = [
            np.array([self.prior_mean[obj_idx]] * self.input_space.shape[0]).reshape(
                (self.input_space.shape[0], 1)
            )
            for obj_idx in range(n_objectives)
        ]

        # Variance for each objective's Gaussian process
        self.variance_objectives = [
            np.array(
                [self.prior_variance[obj_idx]] * self.input_space.shape[0]
            ).reshape((self.input_space.shape[0], 1))
            for obj_idx in range(n_objectives)
        ]

        # Preallocate acquisition function values for each point
        self.acquisition_values = np.array([0.0] * self.input_space.shape[0]).reshape(
            (self.input_space.shape[0], 1)
        )

        # Initial guesses
        self.x_vector[0] = np.array([5.0] * self.dim)
        self.y_vector[0] = self.function(self.x_vector[0])

        self.x_vector[1] = np.array([10.0] * self.dim)
        self.y_vector[1] = self.function(self.x_vector[1])

        self.x_vector[2] = np.array([15.0] * self.dim)
        self.y_vector[2] = self.function(self.x_vector[2])

        # Keep track of the number of evaluations
        self.n_evaluations = 3

        # Reference point for hypervolume (should be worse than any expected objective value)
        self.reference_point = np.array([0.0, 0.0, 0.0])

    def optimize(self):
        """
        Perform the Multi-Objective Bayesian optimization.
        """
        for f in range(3, self.n_iterations):
            # Compute kernel matrices for each objective
            for obj_idx in range(self.n_objectives):
                self.kernel_matrices[obj_idx][
                    : self.n_evaluations, : self.n_evaluations
                ] = compute_k(
                    self.x_vector[: self.n_evaluations],
                    sigma=np.sqrt(self.prior_variance[obj_idx]),
                    length_scale=3.0,
                )

            for i, x_star in enumerate(self.input_space):
                # Compute the kernel vector for the new point
                k_star = compute_k_star(
                    x_vector=self.x_vector[: self.n_evaluations],
                    x_star=x_star,
                    sigma=np.sqrt(self.prior_variance[obj_idx]),
                    length_scale=3.0,
                )

                # Initialize mean and variance predictions for each objective
                mu_pred = np.zeros(self.n_objectives, dtype=np.float64)
                var_pred = np.zeros(self.n_objectives, dtype=np.float64)

                # Update mean and variance for each objective
                for obj_idx in range(self.n_objectives):
                    # Update mean
                    delta_mu = compute_delta_mu(
                        k_star=k_star,
                        kernel_matrix=self.kernel_matrices[obj_idx][
                            : self.n_evaluations, : self.n_evaluations
                        ],
                        y_vector=self.y_vector[: self.n_evaluations, obj_idx],
                        prior_mean=self.prior_mean[obj_idx],
                    )

                    self.mu_objectives[obj_idx][i] = (
                        self.prior_mean[obj_idx] + delta_mu.item()
                    )
                    mu_pred[obj_idx] = self.mu_objectives[obj_idx][i].item()

                    # Update variance
                    delta_variance = compute_delta_variance(
                        k_star=k_star,
                        kernel_matrix=self.kernel_matrices[obj_idx][
                            : self.n_evaluations, : self.n_evaluations
                        ],
                    )

                    self.variance_objectives[obj_idx][i] = (
                        self.prior_variance[obj_idx] + delta_variance.item()
                    )
                    var_pred[obj_idx] = self.variance_objectives[obj_idx][i].item()

                # Compute multi-objective acquisition function
                self.acquisition_values[i] = hypervolume_improvement(
                    mu_pred,
                    var_pred,
                    self.reference_point,
                    beta=2.0,
                )

            # Select the next point to evaluate
            x_next = self.input_space[np.argmax(self.acquisition_values)]

            # Check if x_next is already evaluated
            if not np.any(
                np.all(self.x_vector[: self.n_evaluations] == x_next, axis=1)
            ):
                # Evaluate the function at the new point
                self.x_vector[f] = x_next
                self.y_vector[f] = self.function(x_next)

                # Increment the number of evaluations
                self.n_evaluations += 1
            else:
                print(f"Point {x_next} already evaluated, breaking.")
                break

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

    optimizer = MultiObjectiveBayesianOptimization(
        toy_function,
        _bounds,
        n_objectives=3,
        n_iterations=20,
        prior_mean=[50] * 3,
        prior_variance=[400.0] * 3,
    )

    start_time = time.time()
    print("Starting optimization...")
    optimizer.optimize()
    optimizer.pareto_analysis()
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
