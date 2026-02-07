"""
2D Bayesian Optimization Demo.

This example demonstrates multi-objective Bayesian optimization
on a 2D toy function with visualization.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from bayesopt import BayesianOptimization
from examples.benchmark_functions import sphere, toy_function
from plotting import PyQtPlotter


def main():
    """Run 2D optimization demo."""
    X_MAX = 300
    Y_MAX = 300

    # Define bounds for 2D optimization
    bounds = [
        (0, X_MAX),
        (0, Y_MAX),
    ]

    # Initialize PyQtGraph Plotter (fast, OpenGL-accelerated)
    plotter = PyQtPlotter(
        bounds=bounds,
        n_objectives=len(bounds),
    )

    start_time = time.time()
    print("\n⚡ Starting 2D optimization demo...\n")

    # Create optimizer
    optimizer = BayesianOptimization(
        toy_function,
        bounds,
        n_objectives=len(bounds),
        initial_samples=(X_MAX + Y_MAX) // 100,  # 1% of grid size
        n_iterations=15,
        batch_size=X_MAX // 100,  # 1% of grid size
        betas=np.array([2.0] * len(bounds)),
        plot=True,
        plotter=plotter,
    )

    # Run optimization
    optimizer.optimize()

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    # Analyze results
    optimizer.pareto_analysis()

    # Keep the plot window open
    plotter.show()


if __name__ == "__main__":
    main()
