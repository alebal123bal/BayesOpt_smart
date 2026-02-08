"""
Plotting utilities for Bayesian Optimization visualization.

This module provides real-time visualization using PyQtGraph and
static image exports using Matplotlib.
"""

# Import plotting utilities
try:
    from .pyqt_plotter import PyQtPlotter, StaticPlotter, create_optimization_gif

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    PyQtPlotter = None
    StaticPlotter = None
    create_optimization_gif = None

# Export based on availability
if PYQTGRAPH_AVAILABLE:
    # Use PyQtGraph for live plotting, Matplotlib for static exports
    Plotter = PyQtPlotter
    PlotterStatic = StaticPlotter

    __all__ = [
        "Plotter",
        "PlotterStatic",
        "PyQtPlotter",
        "StaticPlotter",
        "create_optimization_gif",
    ]
else:
    # No plotting available
    Plotter = None
    PlotterStatic = None
    __all__ = []
