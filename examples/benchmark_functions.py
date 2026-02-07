"""
Benchmark functions for testing Bayesian Optimization.

This module contains example objective functions that can be used
to test and demonstrate the optimization algorithms.
"""

import numpy as np
from bayesopt.config import DEBUG_MODE

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

else:
    from numba import njit


# =============================================================================
# 2D TOY FUNCTION
# =============================================================================


@njit
def toy_function(x: np.ndarray) -> np.ndarray:
    """
    A multi-objective toy function to optimize (2 objectives).

    This is a simple benchmark with two quadratic objectives centered
    at different points, useful for visualizing 2D optimization.

    Args:
        x: Input array (e.g., [x1, x2]).

    Returns:
        Output array containing [f(x), g(x)].
    """
    f_x = -((x[0] - 150) ** 2) / 100 + 100
    g_x = -((x[1] - 150) ** 2) / 100 + 20

    return np.array([f_x, g_x])


# =============================================================================
# 3D TOY FUNCTION
# =============================================================================


@njit
def toy_function_3d(x: np.ndarray) -> np.ndarray:
    """
    A multi-objective toy function to optimize (3 objectives).

    Args:
        x: Input array (e.g., [x1, x2, x3]).

    Returns:
        Output array containing [f(x), g(x), h(x)].
    """
    f_x = -((x[0] - 150) ** 2) / 100 + 100
    g_x = -((x[1] - 150) ** 2) / 100 + 20
    h_x = -((x[2] - 5)) + 120

    return np.array([f_x, g_x, h_x])


# =============================================================================
# CLASSIC BENCHMARK FUNCTIONS
# =============================================================================


@njit
def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere function (single objective).

    Global minimum at x = [0, 0, ...] with f(x) = 0.

    Args:
        x: Input array of arbitrary dimension.

    Returns:
        Single-objective value (negated for maximization).
    """
    # Negate for maximization (original is minimization)
    return np.array([-1.0 * np.sum(x**2)])
