"""
2D Bayesian Optimization Demo.

This example demonstrates multi-objective Bayesian optimization
on a 2D toy function with visualization using the callback architecture.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from bayesopt import BayesianOptimization
from bayesopt.callbacks import PlotterCallback, ProgressLogger, GraphSaverCallback
from examples.benchmark_functions import toy_function
from plotting import PyQtPlotter, PyQtPlotterStatic


def main():
    """Run 2D optimization demo with callback architecture."""
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

    # Setup callbacks
    plotter_callback = PlotterCallback(plotter)
    progress_logger = ProgressLogger(
        log_file="outputs/logs/optimization.log", verbose=True
    )
    graph_saver = GraphSaverCallback(
        plotter_class=PyQtPlotterStatic,
        bounds=bounds,
        n_objectives=len(bounds),
        save_every=1,
        save_format="png",
    )

    start_time = time.time()
    print("\n⚡ Starting 2D optimization demo with callback architecture...\n")

    # Create optimizer with callbacks
    optimizer = BayesianOptimization(
        toy_function,
        bounds,
        n_objectives=len(bounds),
        initial_samples=(X_MAX + Y_MAX) // 100,  # 1% of grid size
        n_iterations=15,
        batch_size=X_MAX // 100,  # 1% of grid size
        betas=np.array([2.0] * len(bounds)),
        callbacks=[
            # plotter_callback,
            progress_logger,
            graph_saver,
        ],
    )

    # Run optimization (callbacks will be called automatically)
    optimizer.optimize()

    end_time = time.time()
    print(f"\n🎉 Optimization completed in {end_time - start_time:.2f} seconds.")

    # Analyze results
    optimizer.pareto_analysis()

    # Keep the plot window open
    plotter.show()


if __name__ == "__main__":
    main()
