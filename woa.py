"""
WOA implementation based on Mirjalili & Lewis 2016 paper.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WhaleOptimization:
    """
    WOA for hyperparam optimization
    
    Implements: 
        encircling prey (exploitation), spiral updating position, search for prey (exploration)
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        bounds: Dict[str, Tuple[float, float]],
        population_size: int = 30,
        max_iterations: int = 50,
        a_max: float = 2.0,
        spiral_constant: float = 1.0
    ):
        """
        Initializiation
        Args:
            fitness_function: function to minimize (returns RMSE)
            bounds: dictionary of param bounds
            population_size: num of whales in population
            max_iterations: max num of iterations
            a_max: maximum val for 'a' param
            spiral_constant: spiral shape constant
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.a_max = a_max
        self.spiral_constant = spiral_constant
        
        # Extract param names and bounds
        self.param_names = list(bounds.keys())
        self.dim = len(self.param_names)
        self.lower_bounds = np.array([bounds[k][0] for k in self.param_names])
        self.upper_bounds = np.array([bounds[k][1] for k in self.param_names])
        
        # Initialize population
        self.population = None
        self.fitness_values = None
        self.best_position = None
        self.best_fitness = float('inf')
        
        # Tracking
        self.fitness_history = []
        
    def initialize_population(self, initial_solutions: np.ndarray | None):
        """
        Args:
            initial_solutions: optional initial solutions - in this case, from GA
        """
        if initial_solutions is not None:
            # Use provided solutions
            self.population = initial_solutions.copy()
            self.best_position = self.population[0].copy()
            
            # Fill remaining with random if needed
            if len(self.population) < self.population_size:
                n_random = self.population_size - len(self.population)
                random_pop = np.random.uniform(
                    self.lower_bounds,
                    self.upper_bounds,
                    size=(n_random, self.dim)
                )
                self.population = np.vstack([self.population, random_pop])
        else:
            # Random initialization
            self.population = np.random.uniform(
                self.lower_bounds,
                self.upper_bounds,
                size=(self.population_size, self.dim)
            )
        
        self.fitness_values = np.full(self.population_size, float('inf'))
        
    def apply_bounds(self, position: np.ndarray) -> np.ndarray:
        """Apply boundary constraints to position"""
        return np.clip(position, self.lower_bounds, self.upper_bounds)
    
    def evaluate_population(self):
        """Evaluate fitness for all whales in population"""
        for i in range(self.population_size):
            if self.population is None or self.fitness_values is None:
                raise RuntimeError("Call initialize_population() first")
            # Convert position to param dictionary
            params = {
                name: self.population[i, j]
                for j, name in enumerate(self.param_names)
            }
            
            # Evaluate fitness
            fitness = self.fitness_function(params)
            
            # Handle NaN or invalid fitness
            if np.isnan(fitness) or np.isinf(fitness):
                fitness = 1e10  # Large penalty
            
            self.fitness_values[i] = fitness
            
            # Update best solution
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.population[i].copy()
    
    def update_position(self, whale_idx: int, iteration: int):
        """
        Update whale position using WOA equations.
        
        Args:
            whale_idx: Index of whale to update
            iteration: Current iteration number
        """
        # Calculate 'a' param (linearly decreases from 2 to 0)
        a = self.a_max - iteration * (self.a_max / self.max_iterations)
        
        # Random params
        r = np.random.random()  # Random [0,1]
        A = 2 * a * r - a  # Step size coefficient
        C = 2 * r  # Random weight for distance
        p = np.random.random()  # Probability to choose behavior
        l = np.random.uniform(-1, 1)  # Random for spiral
        
        if self.population is None:
            raise RuntimeError("Call initialize_population() first")
        current_position = self.population[whale_idx]
        
        if p < 0.5:
            # Encircling prey or search for prey
            if np.abs(A) < 1:
                # Encircling prey (exploitation)
                # D = |C * W_best - W|
                D = np.abs(C * self.best_position - current_position)
                # W(t+1) = W_best - A * D
                new_position = self.best_position - A * D
            else:
                # Search for prey (exploration)
                # Select random whale
                random_idx = np.random.randint(0, self.population_size)
                random_whale = self.population[random_idx]
                # D = |C * W_rand - W|
                D = np.abs(C * random_whale - current_position)
                # W(t+1) = W_rand - A * D
                new_position = random_whale - A * D
        else:
            # Spiral updating position
            # D' = |W_best - W|
            D_prime = np.abs(self.best_position - current_position)
            # W(t+1) = D' * e^(b*l) * cos(2*pi*l) + W_best
            # where b is spiral constant, l is random [-1,1]
            b = self.spiral_constant
            new_position = (
                D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) +
                self.best_position
            )
        
        # Apply boundary constraints
        new_position = self.apply_bounds(new_position)
        
        return new_position
    
    def optimize(self, initial_solutions: np.ndarray | None) -> Tuple[Dict, float, List]:
        """
        Run WOA optimization.
        
        Args:
            initial_solutions: Optional initial solutions
            
        Returns:
            tuple of (best_params, best_fitness, fitness_history)
        """
        logger.info("Starting Whale Optimization Algorithm...")
        
        # Initialize population
        self.initialize_population(initial_solutions)
        
        # Initial evaluation
        self.evaluate_population()
        self.fitness_history.append(self.best_fitness)
        
        logger.info(f"Initial best fitness: {self.best_fitness:.6f}")

        # It's assumed there's a population initialized, otherwise values are None
        if self.population is None or self.fitness_values is None or self.best_position is None:
            raise RuntimeError("Call initialize_population() first")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            
            # Update each whale's position
            for i in range(self.population_size):
                new_position = self.update_position(i, iteration)
                
                # Evaluate new position
                params = {
                    name: new_position[j]
                    for j, name in enumerate(self.param_names)
                }
                new_fitness = self.fitness_function(params)
                
                # Handle invalid fitness
                if np.isnan(new_fitness) or np.isinf(new_fitness):
                    new_fitness = 1e10
                
                # Update if improved
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_position
                    self.fitness_values[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_position.copy()
            
            # Track fitness history
            self.fitness_history.append(self.best_fitness)
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"WOA Iteration {iteration+1}/{self.max_iterations}, "
                          f"Best Fitness: {self.best_fitness:.6f}")
        
        # Convert best position to param dictionary
        best_params = {
            name: self.best_position[j]
            for j, name in enumerate(self.param_names)
        }
        
        logger.info(f"WOA completed. Final best fitness: {self.best_fitness:.6f}")
        
        return best_params, self.best_fitness, self.fitness_history
    
    def get_population_solutions(self) -> np.ndarray:
        """Get current population solutions"""

        if self.population is None:
            raise RuntimeError("Call initialize_population() first")

        return self.population.copy()

    # def __initialization_check__(self, *attrs) -> None:
    #     missing = [name for name in attrs if getattr(self, name, None) is None]

    #     if missing:
    #         raise RuntimeError(
    #             f"Call initialize_population() first. Missing: {', '.join(missing)}"
    #         )