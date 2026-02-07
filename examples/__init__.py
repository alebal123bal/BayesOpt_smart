"""
Examples package for Bayesian Optimization.

This package contains benchmark functions and demonstration scripts.
"""

from .benchmark_functions import (
    toy_function,
    toy_function_3d,
    sphere,
)

__all__ = [
    "toy_function",
    "toy_function_3d",
    "sphere",
]
