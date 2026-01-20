"""
Combines global search (via GA) with local refinement (via WOA).
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
import logging
from ga import GeneticAlgorithm
from woa import WhaleOptimization

logger = logging.getLogger(__name__)


class HybridGAWOA:
    """
    Here's how it works:
        1. GA performs global search
        2. After each GA generation, elite solutions are refined using WOA
        3. Refined solutions are merged back into GA population
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        bounds: Dict[str, Tuple[float, float]],
        ga_config: Dict,
        woa_config: Dict
    ):
        """
        Args:
            fitness_function: objective function
            bounds: param bounds
            ga_config: GA config
            woa_config: WOA config
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.ga_config = ga_config
        self.woa_config = woa_config
        
        # Tracking
        self.best_params = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.ga_history = []
        self.woa_history = []

        # To track hyperparams 
        self.hyperparam_history = []  
        
    def optimize(self) -> Tuple[Dict, float, Dict]:
        """
        Returns:
            tuple of (best_params, best_fitness, history_dict)
        """
        logger.info("="*60)
        logger.info("Starting hybrid GA-WOA optimization...")
        logger.info("="*60)
        
        # Initialize GA
        ga = GeneticAlgorithm(
            fitness_function=self.fitness_function,
            bounds=self.bounds,
            population_size=self.ga_config['population_size'],
            max_iterations=self.ga_config['max_iterations'],
            elite_ratio=self.ga_config['elite_ratio'],
            crossover_rate=self.ga_config['crossover_rate'],
            mutation_rate=self.ga_config['mutation_rate'],
            tournament_size=self.ga_config['tournament_size']
        )
        
        # Initialize GA population
        ga.initialize_population()
        ga.evaluate_population()
        
        self.best_fitness = ga.best_fitness

        if ga.best_individual is None or ga.fitness_values is None:
            raise RuntimeError("Call GeneticAlgorithm.initialize_population() first")
        self.best_params = {
            name: ga.best_individual[i]
            for i, name in enumerate(ga.param_names)
        }
        
        logger.info(f"Initial GA best fitness: {self.best_fitness:.6f}")
        
        # Main hybrid loop
        for ga_iter in range(self.ga_config['max_iterations']):
            logger.info(f"\n{'='*60}")
            logger.info(f"GA iteration {ga_iter + 1}/{self.ga_config['max_iterations']}")
            logger.info(f"{'='*60}")
            
            # Get elite individuals from GA
            elite_solutions = ga.get_elite_individuals()
            
            logger.info(f"Elite solutions extracted: {len(elite_solutions)}")
            logger.info(f"Current GA best fitness: {ga.best_fitness:.6f}")
            
            # Initialize WOA with elite solutions for local refinement
            woa = WhaleOptimization(
                fitness_function=self.fitness_function,
                bounds=self.bounds,
                population_size=self.woa_config['population_size'],
                max_iterations=self.woa_config['max_iterations'],
                a_max=self.woa_config['a_max'],
                spiral_constant=self.woa_config.get('spiral_constant', 1.0)
            )
            
            # Run WOA local search
            logger.info(f"\nRunning WOA local refinement...")
            woa_best_params, woa_best_fitness, woa_history = woa.optimize(
                initial_solutions=elite_solutions
            )
            
            # Get refined WOA population
            woa_population = woa.get_population_solutions()
            
            # Update global best in case WOA found better solution
            if woa_best_fitness < self.best_fitness:
                self.best_fitness = woa_best_fitness
                self.best_params = woa_best_params
                logger.info(f"✓ New global best found by WOA: {self.best_fitness:.6f}")
            
            # Merge WOA-refined solutions with GA offspring + create next GA generation
            ga_offspring = ga.evolve_population()
            
            # Replace worst GA individuals with best WOA solutions + sort WOA population by fitness
            woa_fitness = np.array([
                self.fitness_function({
                    name: woa_population[i, j]
                    for j, name in enumerate(ga.param_names)
                })
                for i in range(len(woa_population))
            ])
            
            woa_sorted_idx = np.argsort(woa_fitness)
            best_woa = woa_population[woa_sorted_idx[:len(elite_solutions)]]
            
            # Replace worst in GA offspring with best from WOA
            ga.population = ga_offspring
            ga.evaluate_population()
            
            ga_sorted_idx = np.argsort(ga.fitness_values)
            worst_indices = ga_sorted_idx[-len(best_woa):]
            
            ga.population[worst_indices] = best_woa
            
            # Re-evaluate after merging
            ga.evaluate_population()
            
            # Track history
            self.ga_history.append(ga.best_fitness)
            self.woa_history.append(woa_best_fitness)
            self.fitness_history.append(self.best_fitness)
            self.hyperparam_history.append(self.best_params.copy())
            
            logger.info(f"\nITERATION {ga_iter + 1} SUMMARY:")
            logger.info(f"  GA best fitness: {ga.best_fitness:.6f}")
            logger.info(f"  WOA best fitness: {woa_best_fitness:.6f}")
            logger.info(f"  Global best fitness: {self.best_fitness:.6f}")
        
        logger.info("\n" + "="*60)
        logger.info("Status: Hybrid GA-WOA Optimization COMPLETED")
        logger.info(f"Final Best Fitness: {self.best_fitness:.6f}")
        logger.info("="*60)
        
        # Prepare history dictionary
        history = {
            'fitness_history': self.fitness_history,
            'ga_history': self.ga_history,
            'woa_history': self.woa_history,
            'hyperparam_history': self.hyperparam_history,
            'best_params': self.best_params,
            'best_fitness': self.best_fitness
        }
        
        return self.best_params, self.best_fitness, history


def create_fitness_function(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    training_config: Dict
) -> Callable:
    """
    Create fitness function for optimization
        
    Returns:
        fitness function that takes params and returns RMSE
    """
    from lstm_model import train_lstm_model
    
    def fitness_function(params: Dict) -> float:
        """
        Train LSTM with given params and return validation RMSE.
        
        Args:
            params: dictionary of hyperparams
            
        Returns:
            validation RMSE (fitness value to minimize)
        """
        try:
            # Extract hyperparams
            hidden_units = int(params['hidden_units'])
            learning_rate = float(params['learning_rate'])
            dropout = float(params['dropout'])
            batch_size = int(params['batch_size'])
            
            # Train model
            _, val_rmse = train_lstm_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                hidden_units=hidden_units,
                learning_rate=learning_rate,
                dropout_rate=dropout,
                batch_size=batch_size,
                epochs=training_config['epochs'],
                patience=training_config['patience'],
                min_delta=training_config['min_delta'],
                verbose=training_config.get('verbose', 0)
            )
            
            return val_rmse
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")

            # Return large penalty for failed evals
            return 1e10  
    
    return fitness_function