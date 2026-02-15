# BayesOpt Smart

A high-performance multi-objective Bayesian optimization library with real-time visualization and advanced Gaussian process modeling.

![Optimization Demo](resources/GIFs/optimization.gif)

## Overview

BayesOpt Smart is a Python library for multi-objective Bayesian optimization that leverages Numba-accelerated computation and Gaussian processes to efficiently optimize expensive-to-evaluate functions. The library features a flexible callback architecture for monitoring optimization progress, extensible acquisition functions, and rich visualization capabilities.

## Key Features

- **High-Performance Computing**: Numba-accelerated kernels for fast GP operations
- **Multi-Objective Optimization**: Simultaneous optimization of multiple competing objectives
- **Batch Evaluation**: Evaluate multiple candidate points per iteration
- **Real-Time Visualization**: Interactive heatmaps, acquisition function plots, and Pareto frontier analysis
- **Flexible Callback System**: Monitor progress, log metrics, and save visualizations during optimization
- **Gaussian Process Modeling**: RBF kernel-based surrogate models with hyperparameter optimization
- **Acquisition Functions**: Upper Confidence Bound (UCB) and Hypervolume Improvement
- **Pareto Front Analysis**: Automatic detection and analysis of Pareto-optimal solutions

## Installation

### Requirements

- Python 3.8+
- NumPy
- Numba
- PyQt5 (for visualization)
- Matplotlib
- SciPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alebal123bal/BayesOpt_smart
cd BayesOpt_smart
```

2. Install dependencies:
```bash
pip install numpy numba pyqt5 matplotlib scipy
```

## Quick Start

```python
from bayesopt import BayesianOptimization
from bayesopt.callbacks import (
    ProgressLogger,
    PerformanceMonitor,
    OptimizationLogger,
    GraphSaverCallback,
)
from plotting import StaticPlotter

# Setup callbacks
callbacks = [
    ProgressLogger(log_file="outputs/logs/optimization.log", verbose=True),
    PerformanceMonitor(),
    OptimizationLogger(),
    GraphSaverCallback(
        plotter_class=StaticPlotter,
        bounds=bounds,
        n_objectives=2,
        save_every=1,
        save_format="png"
    )
]

# Create optimizer with callbacks
optimizer = BayesianOptimization(
    objective_function,
    bounds,
    n_objectives=2,
    callbacks=callbacks,
    initial_samples=10,
    n_iterations=20
)

optimizer.optimize()
```

### Real-Time Visualization

For interactive real-time plotting during optimization:

```python
from bayesopt.callbacks import PlotterCallback
from plotting import PyQtPlotter

# Initialize plotter
plotter = PyQtPlotter(bounds=bounds, n_objectives=2)
plotter_callback = PlotterCallback(plotter)

# Add to optimizer callbacks
optimizer = BayesianOptimization(
    objective_function,
    bounds,
    callbacks=[plotter_callback],
    ...
)
```


## Project Structure

```
BayesOpt_smart/
├── bayesopt/                  # Core optimization library
│   ├── __init__.py
│   ├── bayesian_optimization.py  # Main optimizer class
│   ├── acquisition.py         # Acquisition functions (UCB, HVI)
│   ├── numba_kernels.py       # Numba-accelerated GP operations
│   ├── pareto.py              # Pareto frontier analysis
│   ├── callbacks.py           # Monitoring callbacks
│   └── config.py              # Configuration and defaults
├── plotting/                  # Visualization modules
│   ├── __init__.py
│   └── pyqt_plotter.py        # Real-time plotting with PyQt
├── examples/                  # Example scripts and functions
│   ├── demo_2d.py             # 2D optimization demo
│   └── benchmark_functions.py # Test functions
├── outputs/                   # Output directory
│   ├── figures/               # Saved visualizations
│   └── logs/                  # Optimization logs
├── resources/                 # Documentation and media
│   ├── GIFs/
│   └── theory/
├── BayesianOptimization_Tutorial.ipynb  # Comprehensive tutorial
└── README.md
```

## Tutorial

A comprehensive Jupyter notebook tutorial is included that covers:

- Theoretical foundations of Bayesian optimization
- Gaussian process regression and RBF kernels
- Multi-objective acquisition functions
- Implementation details and performance tips
- Advanced configuration and customization

To explore the tutorial:
```bash
jupyter notebook BayesianOptimization_Tutorial.ipynb
```

## Performance Optimization

The library is optimized for performance through:

- **Numba JIT compilation**: Critical computational kernels are compiled to native code
- **Vectorized operations**: Efficient NumPy array operations throughout
- **Batch evaluation**: Amortize overhead by evaluating multiple points
- **Cached computations**: Reuse kernel matrices and decompositions where possible
- **Optional debug mode**: Disable Numba compilation for debugging (set `DEBUG_MODE=True` in config.py)

## Examples

### Running the 2D Demo

Press F5.

This demonstrates optimization on a simple 2D toy function with real-time visualization and comprehensive logging.

### Custom Objective Functions

Define your own objective function following this signature:

```python
def my_objective(x: np.ndarray) -> np.ndarray:
    """
    Compute objective values.
    
    Args:
        x: Input array of shape (n_dimensions,)
    
    Returns:
        Array of shape (n_objectives,) containing objective values
    """
    obj1 = ...  # First objective
    obj2 = ...  # Second objective
    return np.array([obj1, obj2])
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

## Acknowledgments

This implementation is inspired by modern Bayesian optimization research and leverages:
- Gaussian processes for surrogate modeling
- UCB and hypervolume improvement acquisition functions
- Numba for high-performance computing
- PyQt for interactive visualization
