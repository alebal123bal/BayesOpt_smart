# 🚀 Multi-Objective Bayesian Optimization

A **high-performance**, **production-ready** Multi-Objective Bayesian Optimization implementation with advanced features and comprehensive visualization capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Numba Accelerated](https://img.shields.io/badge/numba-accelerated-green.svg)](https://numba.pydata.org/)
[![Multi-Objective](https://img.shields.io/badge/optimization-multi--objective-orange.svg)]()

## 🎬 Live Visualization Demo

![Bayesian Optimization Heatmap Demo](videos/heatmap_demo.gif)

*Real-time heatmap updates during optimization, showing mean predictions, uncertainty, and acquisition function.*

## ✨ Key Features

- 🎯 **Multi-Objective Optimization**: Optimize multiple competing objectives simultaneously
- ⚡ **High Performance**: Numba JIT compilation for 10x+ speed improvements
- 📊 **Rich Visualization**: Interactive heatmaps and Pareto front analysis
- 🔧 **Production Ready**: Robust error handling and comprehensive testing
- 📈 **Batch Optimization**: Evaluate multiple points per iteration for efficiency
- 🔍 **Hyperparameter Optimization**: Automatic tuning via marginal likelihood maximization
- 🌟 **Pareto Analysis**: Intelligent trade-off identification and ranking

## 📦 Installation

### Requirements
- Python 3.8+
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- Numba >= 0.53.0 (for performance acceleration)
- PyQtGraph >= 0.12.0 (for visualization)

### Quick Install
```bash
pip install numpy scipy numba pyqtgraph
```

## 🚀 Quick Start

See the comprehensive [Jupyter Tutorial](BayesianOptimization_Tutorial.ipynb) for detailed examples and explanations.

### Basic Usage
```python
from bayesian_optimization import BayesianOptimization, toy_function

# Create optimizer
optimizer = BayesianOptimization(
    function=toy_function,
    bounds=[(0, 30), (0, 30)],
    n_objectives=2,
    initial_samples=8,
    n_iterations=5
)

# Run optimization
optimizer.optimize()

# Analyze results
pareto_points = optimizer.pareto_analysis()
```

## 📊 Configuration

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | callable | **Required** | Multi-objective function returning `np.array` |
| `bounds` | list of tuples | **Required** | `[(min, max), ...]` for each dimension |
| `n_objectives` | int | `3` | Number of objective functions |
| `n_iterations` | int | `10` | Number of optimization iterations |
| `initial_samples` | int | `3` | Initial Latin Hypercube samples |
| `batch_size` | int | `1` | Points to evaluate per iteration |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_mean` | list/array | `[0.0, ...]` | Prior mean for each objective |
| `prior_variance` | list/array | `[1.0, ...]` | Prior variance for each objective |
| `length_scales` | list/array | `[1.0, ...]` | RBF kernel length scales |
| `betas` | list/array | `[2.0, ...]` | Exploration-exploitation trade-off |

## 📚 Tutorial & Documentation

### Interactive Jupyter Tutorial

A comprehensive tutorial is included: **[BayesianOptimization_Tutorial.ipynb](BayesianOptimization_Tutorial.ipynb)**

**Tutorial Contents:**
- ✅ Theory: Gaussian Processes and multi-objective optimization
- ✅ Implementation: Understanding the codebase architecture  
- ✅ Usage: From basic to advanced configurations
- ✅ Visualization: Heatmap interpretation and Pareto analysis
- ✅ Performance: Parameter tuning and optimization tips
- ✅ Examples: Real-world optimization scenarios

**Launch Tutorial:**
```bash
# Open in VS Code with Jupyter extension
code BayesianOptimization_Tutorial.ipynb

# Or use Jupyter Lab/Notebook
jupyter lab BayesianOptimization_Tutorial.ipynb
```

## ⚡ Performance

### Speed
- **10-50x faster** than pure Python with Numba JIT compilation
- **Memory-efficient** with O(n² × m) complexity (n=evaluations, m=objectives)
- **Production mode** (default): Full acceleration
- **Debug mode**: Set `BAYESIAN_DEBUG=true` for debugging

### Performance Scaling

| Problem Size | Evaluations | Time (Production) |
|--------------|-------------|-------------------|
| 2D, 2 obj    | 50 evals    | ~2-5 seconds     |
| 2D, 3 obj    | 100 evals   | ~5-15 seconds    |
| 3D, 2 obj    | 200 evals   | ~15-45 seconds   |

## 🏗️ Technical Overview

### Core Components
- **Gaussian Process Regression**: Individual GP for each objective
- **RBF Kernel**: Configurable length scales and prior variance
- **Latin Hypercube Sampling**: Efficient initial exploration
- **Upper Confidence Bound**: Balanced exploration-exploitation
- **Pareto Analysis**: Multi-objective solution ranking

### Mathematical Foundation
- **GP Prior**: f(x) ~ GP(μ(x), k(x,x'))
- **RBF Kernel**: k(x,x') = σ² exp(-||x-x'||²/(2ℓ²))
- **UCB Acquisition**: UCB(x) = μ(x) + β×σ(x)

## 📖 References

1. Rasmussen & Williams, "Gaussian Processes for Machine Learning"
2. Knowles, "ParEGO: A Hybrid Algorithm"  
3. Srinivas et al., "Gaussian Process Optimization in the Bandit Setting"

## 📄 License

MIT License - Copyright (c) 2025

---

**Happy Optimizing!** 🚀🎯📊