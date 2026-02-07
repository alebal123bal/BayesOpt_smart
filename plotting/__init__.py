"""
Plotting utilities for Bayesian Optimization visualization.

This module provides real-time and static visualization using PyQtGraph
for high-performance OpenGL-accelerated rendering.
"""

# Try to import PyQtGraph (new default)
try:
    from .pyqt_plotter import PyQtPlotter, PyQtPlotterStatic
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    PyQtPlotter = None
    PyQtPlotterStatic = None

# Also import Matplotlib (legacy/fallback)
try:
    from .heatmap_plotter import HeatmapPlotterDaemon, HeatmapPlotterStatic
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    HeatmapPlotterDaemon = None
    HeatmapPlotterStatic = None

# Export based on availability (prefer PyQtGraph)
if PYQTGRAPH_AVAILABLE:
    # Use PyQtGraph as default (much faster)
    Plotter = PyQtPlotter
    PlotterStatic = PyQtPlotterStatic
    
    if MATPLOTLIB_AVAILABLE:
        # Both available
        __all__ = [
            "Plotter",
            "PlotterStatic",
            "PyQtPlotter",
            "PyQtPlotterStatic",
            "HeatmapPlotterDaemon",
            "HeatmapPlotterStatic",
        ]
    else:
        # Only PyQtGraph
        __all__ = [
            "Plotter",
            "PlotterStatic",
            "PyQtPlotter",
            "PyQtPlotterStatic",
        ]
elif MATPLOTLIB_AVAILABLE:
    # Fallback to matplotlib
    Plotter = HeatmapPlotterDaemon
    PlotterStatic = HeatmapPlotterStatic
    __all__ = [
        "Plotter",
        "PlotterStatic",
        "HeatmapPlotterDaemon",
        "HeatmapPlotterStatic",
    ]
else:
    # No plotting available
    Plotter = None
    PlotterStatic = None
    __all__ = []

