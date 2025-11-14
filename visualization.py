"""Enhanced visualization components for TSP GA.
This module provides advanced real-time plotting utilities with 2x2 layout
for visualizing multiple algorithms' progress in solving TSP, including
route evolution, fitness convergence, and pheromone matrix visualization.

Author: gimocimo
Date: 14/11/2025
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class TSPPlotter:
    """Enhanced plotter for visualizing multiple algorithms on TSP."""

    def __init__(self, cities):
        """Initialize the plotter with city data.

        Args:
            cities: NumPy array of city coordinates
        """
        self.cities = cities
        plt.ion() # Enable interactive mode

        # Create 2x2 subplot layout
        self.fig, self.ax_array = plt.subplots(2, 2, figsize=(14, 10))
        self.sga_route_ax = self.ax_array[0, 0]
        self.hga_route_ax = self.ax_array[1, 0]
        self.convergence_ax = self.ax_array[0, 1]
        self.pheromone_ax = self.ax_array[1, 1]

        # Adjust spacing
        self.fig.subplots_adjust(hspace=0.3, wspace=0.25)

        # Setup individual plots
        self._setup_sga_route_plot()
        self._setup_hga_route_plot()
        self._setup_convergence_plot()
        self._setup_pheromone_plot()

        # Track plot objects
        self.convergence_lines = {}
        self.route_lines = {"SGA": None, "HGA-ACO": None}
        self.pheromone_image = None
        self.pheromone_colorbar = None

    def _setup_sga_route_plot(self):
        """Setup the SGA route visualization plot."""
        self.sga_route_ax.set_title("SGA Best Route Evolution")
        self.sga_route_ax.set_xlabel("X-coordinate")
        self.sga_route_ax.set_ylabel("Y-coordinate")

        # Plot cities
        self.sga_route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )

        # Add city labels
        for i, city_coord in enumerate(self.cities):
            self.sga_route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        self.sga_route_ax.legend(loc='upper right')

    def _setup_hga_route_plot(self):
        """Setup the HGA-ACO route visualization plot."""
        self.hga_route_ax.set_title("HGA-ACO Best Route Evolution")
        self.hga_route_ax.set_xlabel("X-coordinate")
        self.hga_route_ax.set_ylabel("Y-coordinate")

        # Plot cities
        self.hga_route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )

        # Add city labels
        for i, city_coord in enumerate(self.cities):
            self.hga_route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        self.hga_route_ax.legend(loc='upper right')

    def _setup_convergence_plot(self):
        """Setup the convergence visualization plot."""
        self.convergence_ax.set_title("Fitness Convergence Comparison")
        self.convergence_ax.set_xlabel("Generation")
        self.convergence_ax.set_ylabel("Best Cost (Distance)")

    def _setup_pheromone_plot(self):
        """Setup the pheromone heatmap plot."""
        self.pheromone_ax.set_title("HGA-ACO Pheromone Matrix")
        self.pheromone_ax.set_xlabel("City Index")
        self.pheromone_ax.set_ylabel("City Index")

    def _get_route_ax(self, algo_name):
        """Get the appropriate route axis for an algorithm.

        Args:
            algo_name: name of the algorithm

        Returns:
            matplotlib axis object
        """
        if "SGA" in algo_name:
            return self.sga_route_ax
        elif "HGA-ACO" in algo_name:
            return self.hga_route_ax
        return None

    def update_live_route_plot(self, best_tour_indices, algo_name,
                               generation, best_cost, update_freq):
        """Update the route plot with current best tour.

        Args:
            best_tour_indices: list of city indices in best tour
            algo_name: name of the algorithm
            generation: current generation number (-1 for final)
            best_cost: cost of the best tour
            update_freq: update frequency setting
        """
        target_ax = self._get_route_ax(algo_name)
        if not target_ax:
            return

        # Check if update is needed
        is_update_time = (
                update_freq > 0 and (
                generation % update_freq == 0 or
                generation == -1 or
                generation == 0
        )
        )

        if not is_update_time and generation > 0:
            return

        # Remove old route line
        if self.route_lines[algo_name]:
            try:
                self.route_lines[algo_name].pop(0).remove()
            except (AttributeError, IndexError, ValueError):
                self.route_lines[algo_name] = None

        # Create tour coordinates
        tour_coords = np.array([
            self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]
        ])

        # Choose color based on algorithm
        line_color = 'blue' if "SGA" in algo_name else 'green'

        # Plot new route
        line = target_ax.plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            color=line_color,
            linestyle='-',
            marker='.',
            label="Current Best"
        )
        self.route_lines[algo_name] = line

        # Update title
        gen_display = "Final" if generation == -1 else str(generation)
        target_ax.set_title(
            f"{algo_name} Route - Gen: {gen_display}, Cost: {best_cost:.2f}"
        )

        # Update legend
        handles, labels = target_ax.get_legend_handles_labels()
        by_label = {"Cities": handles[labels.index("Cities")]}
        if self.route_lines[algo_name]:
            by_label["Current Best"] = self.route_lines[algo_name][0]
        target_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.pause(0.01)

    def update_convergence_plot(self, history, algo_name, color_val):
        """Update the convergence plot with cost history.

        Args:
            history: list of best costs per generation
            algo_name: name of the algorithm
            color_val: color for the plot line
        """
        generations_axis = list(range(len(history)))
        label_prefix = f"{algo_name} Best Cost"
        current_label = label_prefix

        # Check if time was already added to label
        existing_legend = self.convergence_ax.get_legend()
        if existing_legend:
            for text in existing_legend.get_texts():
                if text.get_text().startswith(label_prefix) and "(Time:" in text.get_text():
                    current_label = text.get_text()
                    break

        if algo_name in self.convergence_lines:
            self.convergence_lines[algo_name].set_data(generations_axis, history)
            self.convergence_lines[algo_name].set_label(current_label)
        else:
            line, = self.convergence_ax.plot(
                generations_axis,
                history,
                label=current_label,
                color=color_val
            )
            self.convergence_lines[algo_name] = line

        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def update_pheromone_heatmap(self, pheromone_matrix, generation, update_freq):
        """Update the pheromone heatmap visualization.

        Args:
            pheromone_matrix: current pheromone matrix
            generation: current generation number
            update_freq: update frequency setting
        """
        # Check if update is needed
        if not (update_freq > 0 and generation % update_freq == 0):
            if generation != -1: # Allow final update
                return

        if self.pheromone_image is None:
            # Create initial heatmap
            self.pheromone_image = self.pheromone_ax.imshow(
                pheromone_matrix,
                cmap='viridis',
                aspect='auto',
                interpolation='nearest'
            )
            self.pheromone_colorbar = self.fig.colorbar(
                self.pheromone_image,
                ax=self.pheromone_ax,
                orientation='vertical'
            )
            self.pheromone_ax.set_title("HGA-ACO Pheromone Matrix")
        else:
            # Update existing heatmap
            self.pheromone_image.set_data(pheromone_matrix)
            # Update color limits based on data range
            self.pheromone_image.set_clim(
                vmin=np.min(pheromone_matrix),
                vmax=np.max(pheromone_matrix)
            )

        # Update title with generation
        gen_display = "Final" if generation == -1 else str(generation)
        self.pheromone_ax.set_title(f"HGA Pheromones - Gen: {gen_display}")
        plt.pause(0.01)

    def display_execution_times(self, sga_time, hga_time):
        """Update convergence plot with execution times.

        Args:
            sga_time: SGA execution time
            hga_time: HGA-ACO execution time
        """
        if "SGA" in self.convergence_lines:
            label = f"SGA Best Cost (Time: {sga_time:.2f}s)"
            self.convergence_lines["SGA"].set_label(label)
        if "HGA-ACO" in self.convergence_lines:
            label = f"HGA-ACO Best Cost (Time: {hga_time:.2f}s)"
            self.convergence_lines["HGA-ACO"].set_label(label)

        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def show_final_routes(self, sga_best_ind, hga_best_ind):
        """Display the final best routes for both algorithms.

        Args:
            sga_best_ind: best Individual from SGA
            hga_best_ind: best Individual from HGA-ACO
        """
        # Clear and redraw SGA final route
        self.sga_route_ax.cla()
        self.sga_route_ax.set_title(f"SGA Final Route - Cost: {sga_best_ind.cost:.2f}")
        self.sga_route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )
        for i, city_coord in enumerate(self.cities):
            self.sga_route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        sga_tour_coords = np.array([
            self.cities[i] for i in sga_best_ind.tour + [sga_best_ind.tour[0]]
        ])
        self.sga_route_ax.plot(
            sga_tour_coords[:, 0],
            sga_tour_coords[:, 1],
            'b-',
            label="SGA Final Path"
        )
        self.sga_route_ax.legend(loc='upper right')

        # Clear and redraw HGA-ACO final route
        self.hga_route_ax.cla()
        self.hga_route_ax.set_title(f"HGA-ACO Final Route - Cost: {hga_best_ind.cost:.2f}")
        self.hga_route_ax.scatter(
            self.cities[:, 0],
            self.cities[:, 1],
            c='red',
            marker='o',
            label='Cities',
            zorder=5
        )
        for i, city_coord in enumerate(self.cities):
            self.hga_route_ax.text(
                city_coord[0] + 0.5,
                city_coord[1] + 0.5,
                str(i),
                fontsize=9
            )

        hga_tour_coords = np.array([
            self.cities[i] for i in hga_best_ind.tour + [hga_best_ind.tour[0]]
        ])
        self.hga_route_ax.plot(
            hga_tour_coords[:, 0],
            hga_tour_coords[:, 1],
            'g-',
            label="HGA-ACO Final Path"
        )
        self.hga_route_ax.legend(loc='upper right')

        plt.pause(0.1)

    def keep_plot_open(self):
        """Keep the plot window open after execution."""
        self.fig.suptitle(
            "TSP Solver Comparison: SGA vs. HGA-ACO",
            fontsize=16,
            y=0.99
        )
        plt.ioff()
        plt.show()
