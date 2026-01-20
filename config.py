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