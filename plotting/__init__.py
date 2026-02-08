"""
Plotting utilities for Bayesian Optimization visualization.

This module provides real-time and static visualization using PyQtGraph
for high-performance OpenGL-accelerated rendering.
"""

# Import PyQtGraph plotting
try:
    from .pyqt_plotter import PyQtPlotter, PyQtPlotterStatic, create_optimization_gif

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    PyQtPlotter = None
    PyQtPlotterStatic = None
    create_optimization_gif = None

# Export based on availability
if PYQTGRAPH_AVAILABLE:
    # Use PyQtGraph as default
    Plotter = PyQtPlotter
    PlotterStatic = PyQtPlotterStatic

    __all__ = [
        "Plotter",
        "PlotterStatic",
        "PyQtPlotter",
        "PyQtPlotterStatic",
        "create_optimization_gif",
    ]
else:
    # No plotting available
    Plotter = None
    PlotterStatic = None
    __all__ = []
