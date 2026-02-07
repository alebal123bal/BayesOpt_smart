"""
BayesOpt_smart - Multi-Objective Bayesian Optimization

A high-performance Bayesian optimization library for multi-objective problems,
leveraging Numba-accelerated Gaussian Processes and efficient acquisition functions.

Main Components:
    - BayesianOptimization: Main optimizer class
    - Kernel functions: GP operations (numba_kernels module)
    - Acquisition functions: UCB and hypervolume improvement
    - Utilities: Pareto analysis, batch selection
    - Configuration: Centralized settings

Example:
    >>> from bayesian_optimization import BayesianOptimization
    >>> from examples import toy_function
    >>> 
    >>> optimizer = BayesianOptimization(
    ...     function=toy_function,
    ...     bounds=[(0, 300), (0, 300)],
    ...     n_objectives=2,
    ...     n_iterations=10,
    ... )
    >>> optimizer.optimize()
    >>> pareto_front = optimizer.pareto_analysis()
"""

__version__ = "2.0.0"
__author__ = "BayesOpt_smart Team"

# Main optimizer class
from .bayesian_optimization import BayesianOptimization

# Utility functions
from .pareto import (
    is_pareto_efficient,
    compute_pareto_front,
    print_pareto_analysis,
)
from .acquisition import (
    select_next_batch,
)

# Configuration
from .config import (
    DEBUG_MODE,
    RANDOM_SEED,
    DEFAULT_PRIOR_MEAN,
    DEFAULT_PRIOR_VARIANCE,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_BETA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INITIAL_SAMPLES,
)

# Optional: Export plotting classes if available
try:
    from plotting import (
        Plotter, 
        PlotterStatic,
        PyQtPlotter, 
        PyQtPlotterStatic,
        HeatmapPlotterDaemon, 
        HeatmapPlotterStatic
    )
    __all__ = [
        # Main optimizer
        "BayesianOptimization",
        # Utilities
        "select_next_batch",
        "is_pareto_efficient",
        "compute_pareto_front",
        "print_pareto_analysis",
        # Plotting (new fast PyQtGraph)
        "Plotter",
        "PlotterStatic",
        "PyQtPlotter",
        "PyQtPlotterStatic",
        # Plotting (legacy Matplotlib)
        "HeatmapPlotterDaemon",
        "HeatmapPlotterStatic",
        # Version
        "__version__",
    ]
except ImportError:
    __all__ = [
        # Main optimizer
        "BayesianOptimization",
        # Utilities
        "select_next_batch",
        "is_pareto_efficient",
        "compute_pareto_front",
        "print_pareto_analysis",
        # Version
        "__version__",
    ]
