"""
Contains all hyperparam bounds and algo params
"""

# Stock data config
STOCK_CONFIG = {
    'ticker': '^GSPC',
    'period': '5y',
    'lookback': 60, # days
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

RANDOM_SEED = 42

# Output directories
OUTPUT_CONFIG = {
    'plots_dir': 'plots',
    'models_dir': 'models',
    'results_dir': 'results'
}

# LSTM hyperparam bounds
LSTM_BOUNDS = {
    'hidden_units': (32, 256),
    'learning_rate': (0.0001, 0.01),
    'dropout': (0.1, 0.5),
    'batch_size': (16, 128)
}

# GA params
GA_CONFIG = {
    'population_size': 50,
    'max_iterations': 20,
    'elite_ratio': 0.2,
    'crossover_rate': 0.8,
    'mutation_rate': 0.2,
    'tournament_size': 3
}

# WOA params
WOA_CONFIG = {
    'population_size': 30,
    'max_iterations': 50,
    'a_max': 2.0,  # Starting val for 'a' param
    'spiral_constant': 1.0  # 'b' constant for spiral shape
}

# LSTM training params
TRAINING_CONFIG = {
    'epochs': 100,
    'patience': 10,
    'min_delta': 1e-4,
    'verbose': 0
}