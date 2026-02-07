"""
Acquisition functions for Bayesian Optimization.

This module contains numba-accelerated acquisition functions
that guide the exploration-exploitation trade-off during optimization.
"""

import numpy as np
from config import DEBUG_MODE

# Conditional imports and setup
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
# UPPER CONFIDENCE BOUND (UCB) ACQUISITION
# =============================================================================


@njit
def upper_confidence_bound(
    mu: np.ndarray,
    variance: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute the upper confidence bound for a single objective, vectorized.

    UCB balances exploitation (mean) and exploration (uncertainty).

    Args:
        mu: Mean predictions for the objective (n_candidates,).
        variance: Variance predictions for the objective (n_candidates,).
        beta: Exploration-exploitation trade-off parameter.

    Returns:
        Upper confidence bound values for the objective (n_candidates,).
    """
    return mu + beta * np.sqrt(np.abs(variance))


@njit
def update_ucb(
    ucb: np.ndarray,
    mu_objectives: np.ndarray,
    variance_objectives: np.ndarray,
    betas: np.ndarray,
) -> None:
    """
    Update the upper confidence bound acquisition function values.

    Modifies the ucb array in-place.

    Args:
        ucb: Preallocated upper confidence bound values (n_objectives, n_candidates).
        mu_objectives: Mean predictions for each objective (n_objectives, n_candidates).
        variance_objectives: Variance predictions (n_objectives, n_candidates).
        betas: Exploration-exploitation trade-off parameters (n_objectives,).
    """
    n_objectives = mu_objectives.shape[0]

    # Compute the UCB for each point vectorized
    for obj_idx in range(n_objectives):
        ucb[obj_idx] = upper_confidence_bound(
            mu_objectives[obj_idx],
            variance_objectives[obj_idx],
            betas[obj_idx],
        )


# =============================================================================
# HYPERVOLUME IMPROVEMENT ACQUISITION
# =============================================================================


@njit
def update_hypervolume_improvement(
    acquisition_values: np.ndarray,
    ucb: np.ndarray,
) -> None:
    """
    Update the hypervolume improvement acquisition function values.

    This is a simple scalarization that sums UCB across all objectives.
    Modifies the acquisition_values array in-place.

    Args:
        acquisition_values: Preallocated acquisition values (n_candidates,).
        ucb: Upper confidence bound values for each point (n_objectives, n_candidates).
    """
    n_points = len(acquisition_values)

    # Compute the hypervolume improvement acquisition function values
    for i in range(n_points):
        acquisition_values[i] = np.sum(ucb[:, i])
