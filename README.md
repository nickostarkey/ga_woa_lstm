# GA-WOA-LSTM Hybrid Optimization System

A complete implementation of a hybrid genetic algorithm (GA) and whale optimization algorithm (WOA) for optimizing LSTM hyperparams in stock market closing price prediction.


**Note**: This is a research implementation. Stock market prediction is inherently uncertain. This tool should not be used for actual trading decisions without proper validation and risk assessment.

## Overview

This system combines:
- **Genetic Algorithm (GA)**: global search across the hyperparam space
- **Whale Optimization Algorithm (WOA)**: local refinement of elite solutions
- **LSTM Neural Network**: time series prediction model with optimized hyperparams


## Data

At the time this repo is last tested (22 Jan, 2026):
- 502 data points

Therefore: 
- Training samples: 278
- Validation samples: 72
- Test samples: 73

## Features

- Two-stage hybrid optimization (GA → WOA)
- Real-time data fetching from Yahoo Finance
- Technical indicators (moving averages, RSI)
- Proper data preprocessing with no leakage: 
   - split before normalization
   - validation set extraction -> sequences after training period
   - test set extraction -> sequences after validation period
- Early stopping to prevent overfitting
- Visualization methods
- Detailed logging and progress tracking
- Comprehensive error handling
- Reproducible results with random seeds

## Project Structure

```
ga-woa-lstm/
├── config.py               # Configuration and hyperparam bounds
├── data_loader.py          # Data pipeline and preprocessing
├── lstm_model.py          
├── ga.py                   # GA
├── woa.py                  # WOA
├── ga_woa_hybrid.py        # Hybrid optimization framework
├── main.py                 
├── utilities.py            # Plotting and results utilities
├── requirements.txt        # Dependencies
├── README.md             
├── pyrightconfig.json     # Pyright config 
├── plots/                 # Generated visualization plots 
├── models/                # Saved trained models
└── results/               # Optimization results and metrics
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nickostarkey/ga_woa_lstm.git
cd ga-woa-lstm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Config
Modify `config.py` to customize parameters and hyperparams.

### Basic Usage

Run the complete optimization pipeline:

```bash
python3 main.py
```

This will:
1. Download 2 years of S&P 500 data
2. Preprocess and create sequences
3. Run GA-WOA hybrid optimization
4. Train final model with optimal hyperparams
5. Evaluate on test set
6. Generate visualizations
7. Save results


### Advanced Usage

#### Using Custom Stock Data

```python
from data_loader import StockDataLoader

data_loader = StockDataLoader(
    ticker='AAPL',  # Apple stock
    period='5y',
    lookback=60,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)

X_train, y_train, X_val, y_val, X_test, y_test, scaler = data_loader.prepare_data()
```

#### Running Only GA

```python
from ga import GeneticAlgorithm
from ga_woa_hybrid import create_fitness_function

fitness_func = create_fitness_function(X_train, y_train, X_val, y_val, TRAINING_CONFIG)

ga = GeneticAlgorithm(
    fitness_function=fitness_func,
    bounds=LSTM_BOUNDS,
    population_size=50,
    max_iterations=30
)

best_params, best_fitness, history = ga.optimize()
```

#### Running Only WOA

```python
from woa import WhaleOptimization

woa = WhaleOptimization(
    fitness_function=fitness_func,
    bounds=LSTM_BOUNDS,
    population_size=30,
    max_iterations=100
)

best_params, best_fitness, history = woa.optimize()
```

#### Run with real time logs

```bash
nohup python main.py > run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Monitor progress:
```bash
tail -f run_*.log
```

## Output

### Console Logs
Real-time progress with:
- Current iteration
- Best fitness values
- Hyperparam values
- Training metrics

### Visualizations

1. **Convergence Plot**: fitness vs iteration for GA, WOA, and global best
2. **Predictions Plot**: actual vs predicted stock prices
3. **Residuals Plot**: error analysis with histogram, Q-Q plot, and scatter
4. **Hyperparams Plot**: final optimal hyperparam values

### Saved Files

- `models/best_model_<timestamp>.pth`: Trained PyTorch model
- `results/results_<timestamp>.xlsx`: Metrics, hyperparams, and convergence history
- `results/config_<timestamp>.json`: Complete configuration
- `plots/*.png`: All visualization plots

## Performance Metrics

The system evaluates models using RMSE, MAE, MAPE, and R2

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in LSTM_BOUNDS
   - Reduce population sizes in GA_CONFIG and WOA_CONFIG

2. **Slow Training**
   - Reduce max_iterations in GA_CONFIG and WOA_CONFIG
   - Reduce LSTM epochs in TRAINING_CONFIG

4. **Data Download Fails**
   - Check internet connection
   - Try different period

## References

1. Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. *Advances in Engineering Software*, 95, 51-67.

2. Deb, K., & Agrawal, R. B. (1995). Simulated Binary Crossover for Continuous Search Space. *Complex Systems*, 9(2), 115-148.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

4. Huiyong, W., & Wang., Z. (2025). Stock market forecasting research based on GA-WOA-LSTM. *PLoS One*.

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ga_woa_lstm,
  title={GA-WOA-LSTM: Bio-Inspired Hybrid Optimization for Stock Price Prediction},
  author={Nicko Starkey},
  year={2026},
  url={https://github.com/nickostarkey/ga_woa_lstm.git}
}
```

The corresponding manuscript can be accessed on Research Square via (this link)[].

## Contributions

Contributions are welcome. Please:

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

