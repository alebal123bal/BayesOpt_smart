"""
Fast real-time plotting using PyQtGraph.

High-performance OpenGL-accelerated visualization for Bayesian Optimization.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


class PyQtPlotter:
    """
    High-performance interactive plotter using PyQtGraph with OpenGL acceleration.
    
    Significantly faster than Matplotlib for real-time updates.
    """

    def __init__(self, bounds, n_objectives):
        """
        Initialize the PyQtGraph plotter.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max)]
            n_objectives (int): Number of objectives
        """
        self.bounds = bounds
        self.n_objectives = n_objectives

        # Grid parameters
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.x_grid = np.arange(x_min, x_max)
        self.y_grid = np.arange(y_min, y_max)
        self.nx, self.ny = len(self.x_grid), len(self.y_grid)

        # Initialize Qt Application (only once)
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Delay window creation until first plot call
        self.win = None
        self.plots = []
        self.image_items = []
        self.scatter_items = []
        self._initialized = False

    def _create_plots(self):
        """Create the plot layout."""
        for obj_idx in range(self.n_objectives):
            row_plots = []
            row_images = []
            row_scatters = []

            # Mean plot
            p1 = self.win.addPlot(title=f"Objective {obj_idx}: Mean (μ)")
            p1.setAspectLocked(False)
            p1.showGrid(x=True, y=True, alpha=0.3)
            img1 = pg.ImageItem()
            p1.addItem(img1)
            scatter1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
            p1.addItem(scatter1)
            
            # Add colorbar for mean
            colorbar1 = pg.ColorBarItem(values=(0, 100), colorMap='viridis')
            colorbar1.setImageItem(img1)
            
            row_plots.append(p1)
            row_images.append(img1)
            row_scatters.append(scatter1)

            # Uncertainty plot
            p2 = self.win.addPlot(title=f"Objective {obj_idx}: Uncertainty (σ)")
            p2.setAspectLocked(False)
            p2.showGrid(x=True, y=True, alpha=0.3)
            img2 = pg.ImageItem()
            p2.addItem(img2)
            scatter2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
            p2.addItem(scatter2)
            
            # Add colorbar for uncertainty
            colorbar2 = pg.ColorBarItem(values=(0, 10), colorMap='plasma')
            colorbar2.setImageItem(img2)
            
            row_plots.append(p2)
            row_images.append(img2)
            row_scatters.append(scatter2)

            # Acquisition plot
            p3 = self.win.addPlot(title=f"Objective {obj_idx}: Acquisition")
            p3.setAspectLocked(False)
            p3.showGrid(x=True, y=True, alpha=0.3)
            img3 = pg.ImageItem()
            p3.addItem(img3)
            scatter3 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
            p3.addItem(scatter3)
            
            # Scatter for next points (stars)
            scatter_next = pg.ScatterPlotItem(
                size=15, 
                pen=pg.mkPen('r', width=2), 
                brush=pg.mkBrush(255, 0, 0, 200),
                symbol='star'
            )
            p3.addItem(scatter_next)
            
            # Add colorbar for acquisition
            colorbar3 = pg.ColorBarItem(values=(0, 100), colorMap='inferno')
            colorbar3.setImageItem(img3)
            
            row_plots.append(p3)
            row_images.append(img3)
            row_scatters.append([scatter3, scatter_next])

            self.plots.append(row_plots)
            self.image_items.append(row_images)
            self.scatter_items.append(row_scatters)

            # Move to next row
            self.win.nextRow()

    def plot(
        self,
        x_vector,
        y_vector,  # Not used in current implementation
        mu_objectives,
        variance_objectives,
        acquisition_values,
        x_next=None,
    ):
        """
        Update the plot with new data (non-blocking, very fast).

        Args:
            x_vector (np.ndarray): Evaluated points, shape (n_eval, dim)
            y_vector (np.ndarray): Objective values (not used in current implementation)
            mu_objectives (np.ndarray): Mean predictions, shape (n_objectives, n_grid_points)
            variance_objectives (np.ndarray): Variance predictions, shape (n_objectives, n_grid_points)
            acquisition_values (np.ndarray): Acquisition values per grid point, shape (n_grid_points,)
            x_next (np.ndarray): Next candidate points to evaluate, shape (n_next, dim)
        """
        if x_vector.shape[1] != 2:
            print("⚠️ PyQtPlotter only supports 2D input space.")
            return

        # Create window on first call when we have data
        if not self._initialized:
            print("📊 Opening plot window with initial data...")
            self.win = pg.GraphicsLayoutWidget(show=True, title="Bayesian Optimization")
            self.win.resize(1800, 600 * self.n_objectives)
            self.win.setWindowTitle(f"Bayesian Optimization - {self.nx}×{self.ny} Grid")
            self._create_plots()
            self._initialized = True

        # Reshape to 2D grids (flip for correct orientation)
        mu_grids = [mu.reshape(self.nx, self.ny) for mu in mu_objectives]
        sigma_grids = [np.sqrt(var.reshape(self.nx, self.ny)) for var in variance_objectives]
        acquisition_grid = acquisition_values.reshape(self.nx, self.ny)

        for obj_idx in range(self.n_objectives):
            # Update mean image
            self.image_items[obj_idx][0].setImage(mu_grids[obj_idx].T, autoLevels=True)
            self.image_items[obj_idx][0].setRect(QtCore.QRectF(self.x_min, self.y_min, 
                                                                self.x_max - self.x_min, 
                                                                self.y_max - self.y_min))
            
            # Update uncertainty image
            self.image_items[obj_idx][1].setImage(sigma_grids[obj_idx].T, autoLevels=True)
            self.image_items[obj_idx][1].setRect(QtCore.QRectF(self.x_min, self.y_min,
                                                                self.x_max - self.x_min,
                                                                self.y_max - self.y_min))
            
            # Update acquisition image
            self.image_items[obj_idx][2].setImage(acquisition_grid.T, autoLevels=True)
            self.image_items[obj_idx][2].setRect(QtCore.QRectF(self.x_min, self.y_min,
                                                                self.x_max - self.x_min,
                                                                self.y_max - self.y_min))

            # Update scatter points (evaluated points)
            spots = [{'pos': (x[0], x[1]), 'brush': pg.mkBrush(255, 255, 0, 200)} 
                     for x in x_vector]
            self.scatter_items[obj_idx][0].setData(spots=spots)
            self.scatter_items[obj_idx][1].setData(spots=spots)
            self.scatter_items[obj_idx][2][0].setData(spots=spots)

            # Update next points on acquisition plot
            if x_next is not None and len(x_next) > 0:
                next_spots = [{'pos': (x[0], x[1])} for x in x_next]
                self.scatter_items[obj_idx][2][1].setData(spots=next_spots)

        # Process events to update display
        self.app.processEvents()

    def show(self):
        """Keep the window open (blocking until closed)."""
        if self._initialized:
            print("\n📊 Optimization complete. Close the plot window to exit.")
            self.app.exec()
        else:
            print("⚠️ No plot window created (no data was plotted).")

    def close(self):
        """Close the plot window."""
        if self.win is not None:
            self.win.close()
        self.win.close()


class PyQtPlotterStatic:
    """
    Static plotter using PyQtGraph (blocks until window closed).
    
    For single snapshot visualizations.
    """

    def __init__(self, bounds, n_objectives):
        """
        Initialize the static PyQtGraph plotter.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max)]
            n_objectives (int): Number of objectives
        """
        self.bounds = bounds
        self.n_objectives = n_objectives

        # Grid parameters
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.x_grid = np.arange(x_min, x_max)
        self.y_grid = np.arange(y_min, y_max)
        self.nx, self.ny = len(self.x_grid), len(self.y_grid)

    def plot(
        self,
        x_vector,
        _y_vector,  # Not used in current implementation
        mu_objectives,
        variance_objectives,
        acquisition_values,
        x_next=None,
    ):
        """
        Create and show a static plot (blocks until closed).

        Args:
            x_vector (np.ndarray): Evaluated points, shape (n_eval, dim)
            _y_vector (np.ndarray): Objective values (not used in current implementation)
            mu_objectives (np.ndarray): Mean predictions, shape (n_objectives, n_grid_points)
            variance_objectives (np.ndarray): Variance predictions, shape (n_objectives, n_grid_points)
            acquisition_values (np.ndarray): Acquisition values per grid point, shape (n_grid_points,)
            x_next (np.ndarray): Next candidate points to evaluate, shape (n_next, dim)
        """
        if x_vector.shape[1] != 2:
            print("⚠️ PyQtPlotterStatic only supports 2D input space.")
            return

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(show=True, title="Bayesian Optimization")
        win.resize(1800, 600 * self.n_objectives)
        win.setWindowTitle(f"Bayesian Optimization - {self.nx}×{self.ny} Grid")

        # Reshape to 2D grids
        mu_grids = [mu.reshape(self.nx, self.ny) for mu in mu_objectives]
        sigma_grids = [np.sqrt(var.reshape(self.nx, self.ny)) for var in variance_objectives]
        acquisition_grid = acquisition_values.reshape(self.nx, self.ny)

        for obj_idx in range(self.n_objectives):
            # Mean plot
            p1 = win.addPlot(title=f"Objective {obj_idx}: Mean (μ)")
            p1.setAspectLocked(False)
            p1.showGrid(x=True, y=True, alpha=0.3)
            img1 = pg.ImageItem()
            img1.setImage(mu_grids[obj_idx].T, autoLevels=True)
            img1.setRect(QtCore.QRectF(self.x_min, self.y_min, 
                                       self.x_max - self.x_min, 
                                       self.y_max - self.y_min))
            p1.addItem(img1)
            scatter1 = pg.ScatterPlotItem(
                x=x_vector[:, 0], 
                y=x_vector[:, 1],
                size=10, 
                pen=pg.mkPen(None), 
                brush=pg.mkBrush(255, 255, 0, 200)
            )
            p1.addItem(scatter1)

            # Uncertainty plot
            p2 = win.addPlot(title=f"Objective {obj_idx}: Uncertainty (σ)")
            p2.setAspectLocked(False)
            p2.showGrid(x=True, y=True, alpha=0.3)
            img2 = pg.ImageItem()
            img2.setImage(sigma_grids[obj_idx].T, autoLevels=True)
            img2.setRect(QtCore.QRectF(self.x_min, self.y_min,
                                       self.x_max - self.x_min,
                                       self.y_max - self.y_min))
            p2.addItem(img2)
            scatter2 = pg.ScatterPlotItem(
                x=x_vector[:, 0], 
                y=x_vector[:, 1],
                size=10, 
                pen=pg.mkPen(None), 
                brush=pg.mkBrush(255, 255, 0, 200)
            )
            p2.addItem(scatter2)

            # Acquisition plot
            p3 = win.addPlot(title=f"Objective {obj_idx}: Acquisition")
            p3.setAspectLocked(False)
            p3.showGrid(x=True, y=True, alpha=0.3)
            img3 = pg.ImageItem()
            img3.setImage(acquisition_grid.T, autoLevels=True)
            img3.setRect(QtCore.QRectF(self.x_min, self.y_min,
                                       self.x_max - self.x_min,
                                       self.y_max - self.y_min))
            p3.addItem(img3)
            scatter3 = pg.ScatterPlotItem(
                x=x_vector[:, 0], 
                y=x_vector[:, 1],
                size=10, 
                pen=pg.mkPen(None), 
                brush=pg.mkBrush(255, 255, 0, 200)
            )
            p3.addItem(scatter3)

            if x_next is not None and len(x_next) > 0:
                scatter_next = pg.ScatterPlotItem(
                    x=x_next[:, 0],
                    y=x_next[:, 1],
                    size=15,
                    pen=pg.mkPen('r', width=2),
                    brush=pg.mkBrush(255, 0, 0, 200),
                    symbol='star'
                )
                p3.addItem(scatter_next)

            win.nextRow()

        app.exec()

    def close(self):
        """Close any open windows."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.closeAllWindows()
