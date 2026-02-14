"""
Configuration and constants for Bayesian Optimization.

This module centralizes all configuration settings, making them
easy to modify and maintain in a single location.
"""

import os
import numpy as np

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

# Debug flag - can be set via environment variable
DEBUG_MODE = os.environ.get("BAYESIAN_DEBUG", "False").lower() in ("true", "1", "yes")

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42

# Set numpy random seed
np.random.seed(RANDOM_SEED)

# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================

# Default prior mean for objectives (0.0 means calculate from initial samples)
DEFAULT_PRIOR_MEAN = 0.0

# Default prior variance for objectives (1.0 means calculate from initial samples)
DEFAULT_PRIOR_VARIANCE = 1.0

# Default length scale for kernel
DEFAULT_LENGTH_SCALE = 1.0

# Default beta (exploration-exploitation trade-off)
DEFAULT_BETA = 1.0

# Default batch size for batch acquisition
DEFAULT_BATCH_SIZE = 3

# Default number of initial samples (Latin Hypercube Sampling)
DEFAULT_INITIAL_SAMPLES = 3

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Float type for Numba-optimized functions
NUMBA_FLOAT_TYPE = np.float64

# Adjust jitter values based on precision
if NUMBA_FLOAT_TYPE == np.float32:
    # Float32 needs much larger jitter due to ~7 decimal digit precision
    KERNEL_JITTER = 1e-3
    CHOLESKY_JITTER = 1e-4
    MIN_VARIANCE = 1e-6
else:
    # Float64 has ~15 decimal digit precision
    KERNEL_JITTER = 1e-6
    CHOLESKY_JITTER = 1e-8
    MIN_VARIANCE = 1e-10

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================

# Hyperparameter optimization method
HYPERPARAM_METHOD = "Powell"

# Hyperparameter optimization tolerance
HYPERPARAM_XTOL = 1e-3
HYPERPARAM_FTOL = 1e-4

# Maximum iterations for hyperparameter optimization
HYPERPARAM_MAXITER = 1000

# Minimum bound for hyperparameters (length scales, variances)
HYPERPARAM_MIN_BOUND = 1e-5

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Default plotting flag
DEFAULT_PLOT_ENABLED = True

# =============================================================================
# NUMBA CONFIGURATION
# =============================================================================

# Print mode message
if DEBUG_MODE:
    print("🐛 DEBUG MODE - Numba disabled (config.py)")
else:
    print("🚀 PRODUCTION MODE - Numba enabled (config.py)")
