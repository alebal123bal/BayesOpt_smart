"""
Bayesian optimization optimized class.
"""

import numpy as np

X_MAX = 30
Y_MAX = 30


def toy_function(x):
    """
    A simple toy function to optimize.

    Args:
        x (np.ndarray): Input array (e.g., [x1, x2, ..., xd]).

    Returns:
        float: Output of the toy function.
    """

    return -np.sum(((x - 12) ** 2)) + 100


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

    distance_squared = np.sum((x1 - x2) ** 2)
    return sigma**2 * np.exp(-0.5 * distance_squared / length_scale**2)


def compute_k_star(x_vector, x_star, sigma, length_scale=1.0):
    """
    Compute the kernel vector between the training points and a new point.

    Args:
        x_vector (np.ndarray): Training points.
        x_star (float): New point.
        sigma (float): Standard deviation for the kernel.
        length_scale (float): Length scale parameter for the kernel.

    Returns:
        np.ndarray: Kernel vector between training points and the new point.
    """

    return np.array([rbf_kernel(x, x_star, sigma, length_scale) for x in x_vector])


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


class BayesianOptimization:
    """
    A class for Bayesian optimization.
    """

    def __init__(
        self, function, bounds, n_iterations=10, prior_mean=0.0, prior_variance=1.0
    ):
        """
        Initialize the Bayesian optimization class.

        Args:
            function (callable): The function to optimize.
            bounds (tuple): The bounds for the input variable (min, max).
            n_iterations (int): The number of iterations for the optimization.
            prior_mean (float): The prior mean for the Gaussian process.
            prior_variance (float): The prior variance for the Gaussian process.
        """

        self.function = function
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        # Dimensionality of the input space
        self.dim = len(bounds)

        # Preallocate the input space as a unique array of cartesian points
        ranges = [np.arange(b[0], b[1]) for b in bounds]
        mesh = np.meshgrid(*ranges, indexing="ij")
        self.input_space = np.stack([m.ravel() for m in mesh], axis=-1)

        # Preallocate the function evaluations
        self.x_vector = np.zeros((n_iterations, self.dim))
        self.y_vector = np.zeros((n_iterations, 1))  # Assuming single objective

        # Preallocate the kernel matrix
        self.kernel_matrix = np.empty((n_iterations, n_iterations))

        # Mean for the Gaussian process
        self.mu = np.array([self.prior_mean] * self.input_space.shape[0]).reshape(
            (self.input_space.shape[0], 1)
        )

        # Variance for the Gaussian process
        self.variance = np.array(
            [self.prior_variance] * self.input_space.shape[0]
        ).reshape((self.input_space.shape[0], 1))

        # Preallocate UCB values for each point
        self.ucb = np.array([0.0] * self.input_space.shape[0]).reshape(
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

    def optimize(self):
        """
        Perform the Bayesian optimization.
        """

        for f in range(3, self.n_iterations):
            # Compute the kernel matrix for the current evaluations
            self.kernel_matrix[: self.n_evaluations, : self.n_evaluations] = compute_k(
                self.x_vector[: self.n_evaluations],
                sigma=np.sqrt(self.prior_variance),
                length_scale=3.0,
            )

            for i, x_star in enumerate(self.input_space):
                # Compute the kernel vector for the new point
                k_star = compute_k_star(
                    x_vector=self.x_vector[: self.n_evaluations],
                    x_star=x_star,
                    sigma=np.sqrt(self.prior_variance),
                    length_scale=3.0,
                )

                # Update mean
                delta_mu = compute_delta_mu(
                    k_star=k_star,
                    kernel_matrix=self.kernel_matrix[
                        : self.n_evaluations, : self.n_evaluations
                    ],
                    y_vector=self.y_vector[: self.n_evaluations],
                    prior_mean=self.prior_mean,
                )

                # Sum the calculated mean
                self.mu[i] = self.prior_mean + delta_mu.item()

                # Update variance
                delta_variance = compute_delta_variance(
                    k_star=k_star,
                    kernel_matrix=self.kernel_matrix[
                        : self.n_evaluations, : self.n_evaluations
                    ],
                )

                # Sum the calculated variance
                self.variance[i] = self.prior_variance + delta_variance.item()

                # Compute the upper confidence bound
                self.ucb[i] = upper_confidence_bound(
                    mu=self.mu[i],
                    variance=self.variance[i],
                    beta=2.0,
                )

            # Select the next point to evaluate if not already evaluated
            x_next = self.input_space[np.argmax(self.ucb)]

            # Check if x_next is in self.X[:self.n_evaluations]
            if not np.any(
                np.all(self.x_vector[: self.n_evaluations] == x_next, axis=1)
            ):
                # Evaluate the function at the new point
                self.x_vector[f] = x_next
                self.y_vector[f] = self.function(x_next)

                # Increment the number of evaluations
                self.n_evaluations += 1

            else:
                # If already evaluated, skip this iteration
                print(f"Point {x_next} already evaluated, breaking.")
                break

        # Print the final results
        print("Final results:")

        for i in range(self.n_iterations):
            print(f"x = {self.x_vector[i]}, f(x) = {self.y_vector[i]}")

        # Max value found
        max_index = np.argmax(self.y_vector[: self.n_iterations])
        max_x = self.x_vector[max_index]
        max_y = self.y_vector[max_index]

        print(f"Maximum value found: f({max_x}) = {max_y}")


if __name__ == "__main__":
    # Example usage
    _bounds = [
        (0, X_MAX),
        (0, Y_MAX),
    ]

    optimizer = BayesianOptimization(
        toy_function,
        _bounds,
        n_iterations=20,
        prior_mean=50.0,
        prior_variance=400.0,
    )

    optimizer.optimize()
