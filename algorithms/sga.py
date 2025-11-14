"""Standard Genetic Algorithm implementation.

Author: gimocimo
Date: 14/11/2025
"""

import random
import copy
from algorithms.base import TSPAlgorithm
from core import Individual

class StandardGA(TSPAlgorithm):
    """Standard Genetic Algorithm for TSP."""

    def __init__(self, cities, distance_matrix):
        """Initialize SGA.

        Args:
            cities: NumPy array of city coordinates
            distance_matrix: pre-calculated distance matrix
        """
        super().__init__(cities, distance_matrix)

    def initialize_population(self, population_size):
        """Create initial random population.

        Args:
            population_size: number of individuals in population

        Returns:
            population: list of Individual objects
        """
        population = []
        base_tour = list(range(self.num_cities))

        for _ in range(population_size):
            tour = random.sample(base_tour, self.num_cities)
            population.append(Individual(tour))

        return population

    def selection_tournament(self, population, tournament_size):
        """Tournament selection operator.

        Args:
            population: current population
            tournament_size: number of individuals in each tournament

        Returns:
            selected_parents: list of selected parents (mating pool)
        """
        selected_parents = []

        for _ in range(len(population)):
            aspirants = random.sample(population, tournament_size)
            winner = min(aspirants, key=lambda ind: ind.cost)
            selected_parents.append(winner)

        return selected_parents

    def crossover_ordered(self, parent1_ind, parent2_ind):
        """Ordered crossover (OX) operator.

        Args:
            parent1_ind: first parent Individual
            parent2_ind: second parent Individual

        Returns:
            Individual: offspring individual
        """
        parent1_tour = parent1_ind.tour
        parent2_tour = parent2_ind.tour
        size = len(parent1_tour)

        # Initialize child tour
        child_tour = [-1] * size

        # Select random segment from parent1
        start, end = sorted(random.sample(range(size), 2))
        child_tour[start:end + 1] = parent1_tour[start:end + 1]

        # Fill remaining from parent2
        p2_idx = 0
        for i in range(size):
            if child_tour[i] == -1:
                while parent2_tour[p2_idx] in child_tour[start:end + 1]:
                    p2_idx += 1
                child_tour[i] = parent2_tour[p2_idx]
                p2_idx += 1

        return Individual(child_tour)

    def mutate_swap(self, individual, mutation_prob):
        """Swap mutation operator.

        Args:
            individual: individual to mutate
            mutation_prob: probability of mutation
        """
        if random.random() < mutation_prob:
            tour = individual.tour
            idx1, idx2 = random.sample(range(len(tour)), 2)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

    def solve(self, population_size, generations, crossover_rate,
              mutation_rate, tournament_size, elitism_size=0,
              plotter=None, plot_freq=1):
        """Run the SGA to solve TSP.

        Args:
            population_size: size of population
            generations: number of generations to run
            crossover_rate: probability of crossover
            mutation_rate: probability of mutation
            tournament_size: size of tournament for selection
            elitism_size: number of best individuals to preserve
            plotter: optional plotter instance for visualization
            plot_freq: update frequency for live plotting

        Returns:
            tuple: (best_individual, cost_history)
        """
        # Initialize population
        population = self.initialize_population(population_size)

        # Evaluate initial population
        for ind in population:
            ind.calculate_cost(self.distance_matrix)

        # Sort and track best
        population.sort()
        self.best_individual = copy.deepcopy(population[0])
        self.cost_history = [self.best_individual.cost]

        algo_name = "SGA"
        print(f"\n--- Running {algo_name} for {self.num_cities} cities ---")
        print(f"Initial best cost: {self.best_individual.cost:.2f}")

        # Initial plot update if plotter provided
        if plotter:
            plotter.update_live_route_plot(
                self.best_individual.tour, algo_name, 0,
                self.best_individual.cost, plot_freq
            )

        # Evolution loop
        for gen in range(1, generations + 1):
            new_population = []

            # Apply elitism
            if elitism_size > 0:
                elites = copy.deepcopy(population[:elitism_size])
                new_population.extend(elites)

            # Create mating pool
            mating_pool = self.selection_tournament(population, tournament_size)

            # Generate offspring
            offspring_idx = 0
            while len(new_population) < population_size:
                # Select parents
                parent1 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1
                parent2 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1

                # Apply crossover
                if random.random() < crossover_rate:
                    child = self.crossover_ordered(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))

                # Apply mutation
                self.mutate_swap(child, mutation_rate)

                # Evaluate offspring
                child.calculate_cost(self.distance_matrix)
                new_population.append(child)

            # Replace population
            population = new_population
            population.sort()

            # Update best if improved
            if population[0].cost < self.best_individual.cost:
                self.best_individual = copy.deepcopy(population[0])

            self.cost_history.append(self.best_individual.cost)

            # Progress output
            if gen % 10 == 0 or gen == generations:
                print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {self.best_individual.cost:.2f}")

            # Update plots if plotter provided
            if plotter:
                if gen % plot_freq == 0 or gen == generations:
                    plotter.update_live_route_plot(
                        self.best_individual.tour, algo_name, gen,
                        self.best_individual.cost, plot_freq
                    )
                plotter.update_convergence_plot(self.cost_history, algo_name, "blue")

        print(f"{algo_name} Final Best Tour: {self.best_individual.tour} with Cost: {self.best_individual.cost:.2f}")

        # Final plot update
        if plotter:
            plotter.update_live_route_plot(
                self.best_individual.tour, algo_name, generations,
                self.best_individual.cost, plot_freq
            )

        return self.best_individual, self.cost_history
