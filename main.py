import numpy as np
import torch
import logging

# Project modules
from config import *
from data_loader import StockDataLoader

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


def main():
    logger.info("="*70)
    logger.info("GA-WOA-LSTM Stock Price Prediction")
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

    print(f"X_train\n{X_train}\n")

if __name__ == "__main__":
    main()