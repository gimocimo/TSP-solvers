"""Base class for TSP algorithms.

Author: gimocimo
Date: 14/11/2025
"""

from abc import ABC, abstractmethod

class TSPAlgorithm(ABC):
    """Abstract base class for TSP solving algorithms."""

    def __init__(self, cities, distance_matrix):
        """Initialize algorithm with problem data.

        Args:
            cities: NumPy array of city coordinates
            distance_matrix: pre-calculated distance matrix
        """
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.num_cities = len(cities)
        self.best_individual = None
        self.cost_history = [] # Track cost over generations

    @abstractmethod
    def solve(self, **kwargs):
        """Solve the TSP problem.

        Returns:
            tuple: (best_individual, cost_history)
        """
        pass
