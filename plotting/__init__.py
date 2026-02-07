"""
Plotting utilities for Bayesian Optimization visualization.

This module provides real-time and static heatmap visualization
for monitoring optimization progress.
"""

from .heatmap_plotter import HeatmapPlotterDaemon, HeatmapPlotterStatic

__all__ = [
    "HeatmapPlotterDaemon",
    "HeatmapPlotterStatic",
]
