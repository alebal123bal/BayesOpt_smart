# Multi-Objective Bayesian Optimization

A high-performance Multi-Objective Bayesian Optimization implementation with optional Numba acceleration for computationally intensive optimization problems.

## Features

- **Multi-objective optimization** using Gaussian Process regression
- **Numba acceleration** for production use with debug mode fallback
- **Pareto frontier analysis** for identifying optimal trade-offs
- **Hypervolume-based acquisition function** for multi-objective exploration
- **Configurable priors** and optimization parameters
- **Debug mode** for development and troubleshooting

## Installation

```bash
# Required dependencies
pip install numpy numba

# Optional for visualization
pip install matplotlib
```

## Quick Start

```python
from bayesian_moo import BayesianOptimization
import numpy as np

# Define your multi-objective function
def my_function(x):
    """Returns array of objectives to maximize"""
    f1 = -(x[0] - 5)**2 + 25
    f2 = -(x[1] - 3)**2 + 15
    return np.array([f1, f2])

# Set bounds for each dimension
bounds = [(0, 10), (0, 8)]

# Create and run optimizer
optimizer = BayesianOptimization(
    function=my_function,
    bounds=bounds,
    n_objectives=2,
    n_iterations=20
)

optimizer.optimize()
optimizer.pareto_analysis()
```

## Configuration

### Debug Mode

Toggle between production (Numba-accelerated) and debug modes:

```bash
# Enable debug mode
export BAYESIAN_DEBUG=true
python your_script.py

# Production mode (default)
python your_script.py
```

Or set in VS Code launch configuration:
```json
{
    "env": {
        "BAYESIAN_DEBUG": "true"
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | callable | - | Multi-objective function returning np.array |
| `bounds` | list of tuples | - | [(min, max), ...] for each dimension |
| `n_objectives` | int | 3 | Number of objective functions |
| `n_iterations` | int | 10 | Total optimization iterations |
| `prior_mean` | list/array | [0.0, ...] | Prior mean for each objective |
| `prior_variance` | list/array | [1.0, ...] | Prior variance for each objective |
| `initial_samples` | int | 3 | Number of initial random samples |

## Example: 3D Optimization

```python
import numpy as np
from bayesian_moo import BayesianOptimization

# 3D multi-objective function
def toy_function(x):
    f1 = -((x[0] - 12) ** 2) + 100
    f2 = -((x[1] - 1) ** 4) + 20  
    f3 = -((x[2] - 5) ** 2) + 120
    return np.array([f1, f2, f3])

# Optimize over 3D space
bounds = [(0, 30), (0, 30), (0, 30)]

optimizer = BayesianOptimization(
    function=toy_function,
    bounds=bounds,
    n_objectives=3,
    n_iterations=30,
    prior_mean=[50, 10, 60],
    prior_variance=[400.0, 100.0, 500.0],
    initial_samples=8
)

optimizer.optimize()
optimizer.pareto_analysis()
```

## Output

The optimizer provides:

1. **Progress tracking** (in debug mode)
2. **Final results** showing all evaluated points
3. **Pareto frontier** with optimal trade-off solutions

```
🚀 PRODUCTION MODE - Numba enabled

Starting optimization...

Optimization completed in 2.34 seconds.

Final results:
x = [12.  1.  5.], objectives = [100.  19.  120.]
x = [11.  2.  4.], objectives = [ 99.  4. 119.]
...

Pareto optimal solutions found: 5
Pareto 1: x = [12.  1.  5.], objectives = [100.  19.  120.]
Pareto 2: x = [11.  0.  5.], objectives = [ 99.  19.  120.]
...
```

## Performance

- **Production mode**: Numba-compiled for maximum speed
- **Debug mode**: Pure Python for easier debugging and development
- **Memory efficient**: Pre-allocated arrays for large-scale problems
- **Parallel computation**: Kernel matrix computation uses parallel processing

## Architecture

The implementation uses:

- **Gaussian Process** regression for each objective
- **RBF kernel** with configurable length scales
- **Upper Confidence Bound** acquisition strategy
- **Hypervolume improvement** for multi-objective selection
- **Pareto efficiency** analysis for result interpretation

## Customization

### Custom Acquisition Functions

```python
@njit
def my_acquisition_function(mu_objectives, variance_objectives, reference_point, beta=2.0):
    # Your custom multi-objective acquisition logic
    return acquisition_value
```

### Custom Kernels

```python
@njit
def my_kernel(x1, x2, sigma, length_scale=1.0):
    # Your custom kernel implementation
    return kernel_value
```

## Troubleshooting

### Common Issues

1. **Numba compilation errors**: Enable debug mode to isolate issues
2. **Memory errors**: Reduce `n_iterations` or input space size
3. **Slow convergence**: Adjust prior parameters or acquisition function

### Debug Mode Benefits

- Detailed iteration logging
- No Numba compilation overhead
- Standard Python debugging tools work
- Easier development and testing

## License

This implementation is provided as-is for research and educational purposes.