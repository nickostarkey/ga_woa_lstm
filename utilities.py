import matplotlib.pyplot as plt
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def plot_convergence(history: dict, save_path: str):
    """Plot optimization convergence curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot overall convergence
    ax1 = axes[0]
    iterations = range(1, len(history['fitness_history']) + 1)
    ax1.plot(iterations, history['fitness_history'], 'b-', linewidth=2, label='Global Best')
    ax1.plot(iterations, history['ga_history'], 'g--', alpha=0.7, label='GA Best')
    ax1.plot(iterations, history['woa_history'], 'r--', alpha=0.7, label='WOA Best')
    ax1.set_xlabel('GA Iteration', fontsize=12)
    ax1.set_ylabel('Fitness (RMSE)', fontsize=12)
    ax1.set_title('Hybrid GA-WOA Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot improvement percentage
    ax2 = axes[1]
    initial_fitness = history['fitness_history'][0]
    improvement = [(initial_fitness - f) / initial_fitness * 100 
                   for f in history['fitness_history']]
    ax2.plot(iterations, improvement, 'purple', linewidth=2)
    ax2.set_xlabel('GA iteration', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Optimization improvement over time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Convergence plot saved to {save_path}")


def plot_predictions(y_true, y_pred, title, save_path):
    """Plot actual vs. predicted values"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot time series
    ax1 = axes[0]
    indices = range(len(y_true))
    ax1.plot(indices, y_true, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(indices, y_pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Stock Price', fontsize=12)
    ax1.set_title(f'{title} - Time series comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual price', fontsize=12)
    ax2.set_ylabel('Predicted price', fontsize=12)
    ax2.set_title('Actual vs predicted scatter plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Predictions plot saved to {save_path}")


def plot_residuals(y_true, y_pred, save_path):
    """Plot residual analysis"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    ax1 = axes[0, 0]
    ax1.plot(residuals, 'b-', alpha=0.6, linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Residual', fontsize=11)
    ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residual', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Residuals vs predicted
    ax4 = axes[1, 1]
    ax4.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Value', fontsize=11)
    ax4.set_ylabel('Residual', fontsize=11)
    ax4.set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Residuals plot saved to {save_path}")


def plot_hyperparameter_evolution(history: dict, save_path: str):
    """Plot evolution of best hyperparams over iterations"""
    # When tracking hyperparams at each iteration
    # Let's test creating a summary plot first
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_params = history['best_params']
    param_names = list(best_params.keys())
    param_values = list(best_params.values())
    
    # Create bar plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.barh(param_names, param_values, color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, param_values)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=10)
    
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_title('Optimal Hyperparams', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Hyperparams plot saved to {save_path}")


def save_results(history, metrics, save_path):
    """Save optimization results to csv"""
    
    results = {
        'Metric': ['Best Fitness', 'RMSE', 'MAE', 'MAPE', 'R2'],
        'Value': [
            history['best_fitness'],
            metrics['RMSE'],
            metrics['MAE'],
            metrics['MAPE'],
            metrics['R2']
        ]
    }
    
    df_metrics = pd.DataFrame(results)
    
    # Save hyperparams
    params_df = pd.DataFrame([history['best_params']])
    
    # Combine and save
    with pd.ExcelWriter(save_path.replace('.csv', '.xlsx'), engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
        params_df.to_excel(writer, sheet_name='Hyperparams', index=False)
        
        # Save iteration history
        hist_df = pd.DataFrame({
            'Iteration': range(1, len(history['fitness_history']) + 1),
            'Global_Best': history['fitness_history'],
            'GA_Best': history['ga_history'],
            'WOA_Best': history['woa_history']
        })
        hist_df.to_excel(writer, sheet_name='Convergence', index=False)
    
    logger.info(f"Results saved to {save_path.replace('.csv', '.xlsx')}")