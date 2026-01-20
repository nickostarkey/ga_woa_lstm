"""
Uses real-valued encoding with SBX crossover and polynomial mutation
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm for hyperparam optimization
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        bounds: Dict[str, Tuple[float, float]],
        population_size: int = 50,
        max_iterations: int = 20,
        elite_ratio: float = 0.2,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 3
    ):
        """
        Args:
            fitness_function: objective function to minimize (returns RMSE)
            bounds: dictionary of parame bounds
            population_size: num of individuals
            max_iterations: num of generations
            elite_ratio: proportion of elite individuals to keep
            crossover_rate: probability of crossover
            mutation_rate: probability of mutation
            tournament_size: size of tournament for selection
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Extract param info
        self.param_names = list(bounds.keys())
        self.dim = len(self.param_names)
        self.lower_bounds = np.array([bounds[k][0] for k in self.param_names])
        self.upper_bounds = np.array([bounds[k][1] for k in self.param_names])
        
        # Calculate elite size
        self.elite_size = int(population_size * elite_ratio)
        
        # Population and fitness
        self.population = None
        self.fitness_values = None
        self.best_individual = None
        self.best_fitness = float('inf')
        
        # Tracking
        self.fitness_history = []
        
    def initialize_population(self):
        """Initialize population with random individuals"""
        self.population = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.population_size, self.dim)
        )
        self.fitness_values = np.full(self.population_size, float('inf'))
        
    def evaluate_population(self):
        """Evaluate fitness for all individuals"""
        
        for i in range(self.population_size):
            # Resolve Pyright's error for population and fitness_values None type 
            if self.population is None or self.fitness_values is None:
                raise RuntimeError("Call initialize_population() first")
            params = {
                name: self.population[i, j]
                for j, name in enumerate(self.param_names)
            }
            
            fitness = self.fitness_function(params)
            
            # Handle invalid fitness
            if np.isnan(fitness) or np.isinf(fitness):
                fitness = 1e10
            
            self.fitness_values[i] = fitness
            
            # Update best
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = self.population[i].copy()
    
    def tournament_selection(self) -> np.ndarray:
        """
        Select individual using tournament selection
        
        Returns: selected individual
        """
        # Randomly select tournament_size individuals
        tournament_indices = np.random.choice(
            self.population_size,
            size=self.tournament_size,
            replace=False
        )
        
        if self.population is None or self.fitness_values is None:
            raise RuntimeError("Call initialize_population() first")
        # Get their fitness values
        tournament_fitness = self.fitness_values[tournament_indices]
        
        # Select best from tournament
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return self.population[winner_idx].copy()
    
    def simulated_binary_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            parent1: 1st parent
            parent2: 2nd parent
            eta: distribution index
            
        Returns: two offspring
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for i in range(self.dim):
            if np.random.random() < 0.5:

                # Perform crossover for this gene
                if np.abs(parent1[i] - parent2[i]) > 1e-10:
                    y1 = min(parent1[i], parent2[i])
                    y2 = max(parent1[i], parent2[i])
                    
                    # Calculate beta
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                    
                    # Create offspring
                    offspring1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    offspring2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Apply bounds
                    offspring1[i] = np.clip(
                        offspring1[i],
                        self.lower_bounds[i],
                        self.upper_bounds[i]
                    )
                    offspring2[i] = np.clip(
                        offspring2[i],
                        self.lower_bounds[i],
                        self.upper_bounds[i]
                    )
        
        return offspring1, offspring2
    
    def polynomial_mutation(
        self,
        individual: np.ndarray,
        eta: float = 20.0
    ) -> np.ndarray:
        """
        Args:
            individual: individual to mutate
            eta: distribution index
            
        Returns: mutated individual
        """
        mutated = individual.copy()
        
        for i in range(self.dim):
            if np.random.random() < self.mutation_rate:
                y = mutated[i]
                yl = self.lower_bounds[i]
                yu = self.upper_bounds[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                mutated[i] = y + deltaq * (yu - yl)
                mutated[i] = np.clip(mutated[i], yl, yu)
        
        return mutated
    
    def evolve_population(self) -> np.ndarray:
        """
        Create next gen through selection, crossover, and mutation
        
        Returns: new population
        """
        if self.population is None or self.fitness_values is None:
            raise RuntimeError("Call initialize_population() first")
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_values)
        
        # Keep elite individuals
        elite = self.population[sorted_indices[:self.elite_size]].copy()
        
        # Generate offspring
        offspring = []
        
        while len(offspring) < self.population_size - self.elite_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self.simulated_binary_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self.polynomial_mutation(child1)
            child2 = self.polynomial_mutation(child2)
            
            offspring.append(child1)
            if len(offspring) < self.population_size - self.elite_size:
                offspring.append(child2)
        
        # Combine elite and offspring
        offspring = np.array(offspring[:self.population_size - self.elite_size])
        new_population = np.vstack([elite, offspring])
        
        return new_population
    
    def get_elite_individuals(self) -> np.ndarray:
        """Get elite individuals from current population"""

        if self.population is None or self.fitness_values is None:
            raise RuntimeError("Call initialize_population() first")

        sorted_indices = np.argsort(self.fitness_values)
        return self.population[sorted_indices[:self.elite_size]].copy()
    
    def optimize(self) -> Tuple[Dict, float, List]:
        """
        Run GA optimization
        
        Returns:
            tuple of (best_params, best_fitness, fitness_history)
        """
        logger.info("Starting GA...")
        
        # Initialize
        self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population()
        self.fitness_history.append(self.best_fitness)
        
        logger.info(f"Gen 0: best fitness = {self.best_fitness:.6f}")
        
        # Evolution loop
        for generation in range(self.max_iterations):
            # Create next generation
            self.population = self.evolve_population()
            
            # Evaluate new population
            self.evaluate_population()
            
            # Track progress
            self.fitness_history.append(self.best_fitness)
            
            logger.info(f"Gen {generation+1}/{self.max_iterations}: "
                       f"best fitness = {self.best_fitness:.6f}")
        
        # Convert best to params
        if self.best_individual is None:
            raise RuntimeError("Call initialize_population() first")
        best_params = {
            name: self.best_individual[j]
            for j, name in enumerate(self.param_names)
        }
        
        logger.info(f"Status: GA COMPLETED. Final best fitness: {self.best_fitness:.6f}")
        
        return best_params, self.best_fitness, self.fitness_history