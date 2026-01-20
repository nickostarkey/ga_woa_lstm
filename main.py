import os
import numpy as np
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Project modules
from config import *
from data_loader import StockDataLoader
from lstm_model import train_lstm_model, evaluate_model
from ga_woa_hybrid import HybridGAWOA, create_fitness_function
from utilities import (
    plot_convergence, 
    plot_predictions, 
    plot_residuals, 
    plot_hyperparam_evolution,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def create_output_directories():
    """Create output directories if they don't exist"""
    for dir_path in OUTPUT_CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Output directories created")

def main():
    logger.info("="*70)
    logger.info("GA-WOA-LSTM Stock Price Prediction")
    logger.info("="*70)

    # Create output directories
    create_output_directories() 

    logger.info("\n" + "="*70)
    logger.info("1/6 - Data Loading and Preprocessing")
    logger.info("="*70)

    # Load and prepare data
    data_loader = StockDataLoader(
        ticker=STOCK_CONFIG['ticker'],
        period=STOCK_CONFIG['period'],
        lookback=STOCK_CONFIG['lookback'],
        train_split=STOCK_CONFIG['train_split'],
        val_split=STOCK_CONFIG['val_split'],
        test_split=STOCK_CONFIG['test_split']
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = data_loader.prepare_data(
        use_technical_indicators=True
    )


    # Run hybrid GA-WOA optimization
    logger.info("\n" + "="*70)
    logger.info("*** 2/6 - Hybrid GA-WOA Optimization ***")
    logger.info("="*70)
    
    # Create fitness function
    fitness_func = create_fitness_function(
        X_train, y_train, X_val, y_val, TRAINING_CONFIG
    )
    
    # Initialize hybrid optimizer
    hybrid_optimizer = HybridGAWOA(
        fitness_function=fitness_func,
        bounds=LSTM_BOUNDS,
        ga_config=GA_CONFIG,
        woa_config=WOA_CONFIG
    )
    
    # Run optimization
    best_params, best_fitness, history = hybrid_optimizer.optimize()
    
    logger.info("\n" + "="*70)
    logger.info("Optimization results:")
    logger.info(f"  Best validation RMSE: {best_fitness:.6f}")
    logger.info("  Optimal hyperparams:")
    for param, value in best_params.items():
        logger.info(f"    {param}: {value:.6f}")
    logger.info("="*70)
    
    # Train final model with optimal hyperparams
    logger.info("\n" + "="*70)
    logger.info("*** 3/6 - Training Final Model ***")
    logger.info("="*70)
    
    final_model, _ = train_lstm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hidden_units=int(best_params['hidden_units']),
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout'],
        batch_size=int(best_params['batch_size']),
        epochs=TRAINING_CONFIG['epochs'],
        patience=TRAINING_CONFIG['patience'],
        verbose=1
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("*** 4/6 - Model Evaluation ***")
    logger.info("="*70)
    
    test_metrics = evaluate_model(final_model, X_test, y_test)
    
    logger.info("Test Set Performance:")
    logger.info(f"  RMSE: {test_metrics['RMSE']:.6f}")
    logger.info(f"  MAE:  {test_metrics['MAE']:.6f}")
    logger.info(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    logger.info(f"  R2:   {test_metrics['R2']:.6f}")
    
    # Inverse transform predictions
    y_test_original = data_loader.inverse_transform_predictions(y_test)
    y_pred_original = data_loader.inverse_transform_predictions(test_metrics['predictions'])
    
    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("*** 5/6 - Generating Visualizations ***")
    logger.info("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convergence plot
    plot_convergence(
        history,
        os.path.join(OUTPUT_CONFIG['plots_dir'], f'convergence_{timestamp}.png')
    )
    
    # Predictions plot
    plot_predictions(
        y_test_original,
        y_pred_original,
        'Test Set',
        os.path.join(OUTPUT_CONFIG['plots_dir'], f'predictions_{timestamp}.png')
    )
    
    # Residuals plot
    plot_residuals(
        y_test_original,
        y_pred_original,
        os.path.join(OUTPUT_CONFIG['plots_dir'], f'residuals_{timestamp}.png')
    )
    
    # Hyperparams plot
    plot_hyperparam_evolution(
        history,
        os.path.join(OUTPUT_CONFIG['plots_dir'], f'hyperparams_{timestamp}.png')
    )
    
    # Save results
    logger.info("\n" + "="*70)
    logger.info("*** 6/6 - Saving results ***")
    logger.info("="*70)
    
    # Save model
    model_path = os.path.join(OUTPUT_CONFIG['models_dir'], f'best_model_{timestamp}.pth')
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save results
    results_path = os.path.join(OUTPUT_CONFIG['results_dir'], f'results_{timestamp}.csv')
    save_results(history, test_metrics, results_path)
    
    # Save configuration
    config_dict = {
        'stock_config': STOCK_CONFIG,
        'lstm_bounds': LSTM_BOUNDS,
        'ga_config': GA_CONFIG,
        'woa_config': WOA_CONFIG,
        'training_config': TRAINING_CONFIG,
        'best_params': best_params,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(OUTPUT_CONFIG['results_dir'], f'config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Config saved to {config_path}")
    
    logger.info("\n" + "="*70)
    logger.info("GA-WOA-LSTM OPTIMIZATION COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"\nFinal Test Performance:")
    logger.info(f"  RMSE: {test_metrics['RMSE']:.6f}")
    logger.info(f"  MAE:  {test_metrics['MAE']:.6f}")
    logger.info(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    logger.info(f"  R2:   {test_metrics['R2']:.6f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()