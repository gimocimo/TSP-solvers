"""Core components for TSP GA.
This module contains all core components including city management,
distance calculations, and individual representation.

Author: gimocimo
Date: 14/11/2025
"""

import random
import numpy as np

# ============= City Management =============
def get_fixed_cities():
    """Return a fixed set of cities for testing.

    Returns:
        np.ndarray: array of (x, y) coordinates for 10 cities
    """
    cities = [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120)
    ]
    return np.array(cities)


def generate_cities(num_cities, width=100, height=100, seed=None):
    """Generate random cities in a given space.

    Args:
        num_cities: number of cities to generate
        width: width of the grid (default: 100)
        height: height of the grid (default: 100)
        seed: random seed for reproducibility (default: None)

    Returns:
        np.ndarray: array of (x, y) coordinates
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cities = []
    for _ in range(num_cities):
        x = random.randint(0, width)
        y = random.randint(0, height)
        cities.append((x, y))

    return np.array(cities)


# ============= Distance Calculations =============
def euclidean_distance(city1, city2):
    """Compute Euclidean distance between two cities.

    Args:
        city1: array of (x, y) coordinates
        city2: array of (x, y) coordinates

    Returns:
        float: Euclidean distance
    """
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def calculate_distance_matrix(cities):
    """Calculate distance matrix for all city pairs.

    Args:
        cities: NumPy array of (x, y) coordinates

    Returns:
        dist_matrix: 2D array of distances between cities
    """
    num_cities = len(cities)
    # Initialize distance matrix
    dist_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = euclidean_distance(cities[i], cities[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return dist_matrix


def calculate_tour_cost(tour, distance_matrix):
    """Calculate total cost (distance) for a complete tour.

    Args:
        tour: list of city indices representing the tour
        distance_matrix: NumPy 2D array of distances between cities

    Returns:
        cost: total tour distance
    """
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return cost


# ============= Individual Representation =============
class Individual:
    """Represents a solution (tour) in the GA population."""

    def __init__(self, tour):
        """Initialize an individual with a tour.

        Args:
            tour: list of city indices representing the tour
        """
        self.tour = list(tour)
        self.cost = float('inf')

    def calculate_cost(self, distance_matrix):
        """Calculate and store the tour cost.

        Args:
            distance_matrix: NumPy 2D array of distances between cities

        Returns:
            cost: calculated cost
        """
        self.cost = calculate_tour_cost(self.tour, distance_matrix)
        return self.cost

    def __lt__(self, other):
        """Enable sorting by cost."""
        return self.cost < other.cost

    def __repr__(self):
        """String representation for console output."""
        # Shorten tour representation if too long
        if len(self.tour) < 15:
            tour_str = str(self.tour)
        else:
            tour_str = str(self.tour[:7] + ["..."] + self.tour[-7:])
        return f"Tour: {tour_str} Cost: {self.cost:.2f}"
