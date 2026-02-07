"""
Pareto analysis utilities for Bayesian Optimization.

This module contains functions for computing and analyzing
Pareto-efficient solutions in multi-objective optimization.
"""

from typing import Tuple
import numpy as np


def is_pareto_efficient(y_vector: np.ndarray) -> np.ndarray:
    """
    Find the Pareto-efficient points from objective evaluations.

    A point is Pareto-efficient if no other point dominates it
    (i.e., is better in all objectives).

    Args:
        y_vector: Objective values at evaluated points (n_points, n_objectives).

    Returns:
        Boolean array indicating which points are Pareto-efficient (n_points,).
    """
    # Invert the y_vector to find the Pareto-efficient points
    # (we maximize, so we need to minimize the negation)
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


def compute_pareto_front(
    x_vector: np.ndarray,
    y_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pareto front from evaluated points.

    Args:
        x_vector: Input points (n_points, n_dimensions).
        y_vector: Objective values (n_points, n_objectives).

    Returns:
        Tuple of (pareto_inputs, pareto_objectives) containing only
        the Pareto-efficient points and their objectives.
    """
    is_efficient = is_pareto_efficient(y_vector)
    return x_vector[is_efficient], y_vector[is_efficient]


def print_pareto_analysis(
    pareto_inputs: np.ndarray,
    pareto_objectives: np.ndarray,
) -> None:
    """
    Print formatted Pareto analysis results.

    Args:
        pareto_inputs: Input points on the Pareto front (n_pareto, n_dimensions).
        pareto_objectives: Objective values on the Pareto front (n_pareto, n_objectives).
    """
    print("📊 Pareto Analysis Results:")
    for i, (input_point, obj_values) in enumerate(zip(pareto_inputs, pareto_objectives)):
        print(f"Input: {input_point}, Pareto Point {i + 1}: {obj_values}")
