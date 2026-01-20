import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """Download, preprocess, create sequences"""
    
    def __init__(self, ticker: str, period: str, lookback: int,
                 train_split: float, val_split: float, test_split: float):
        """
        Args:
            ticker: stock ticker symbol
            period: period to download (e.g.,'5y')
            lookback: num of days to look back for prediction
            train_split, val_split, test_split: proportions for training, validation, and test sets
        """
        self.ticker = ticker
        self.period = period
        self.lookback = lookback
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        
    def download_data(self) -> pd.DataFrame:
        """Download stock data from Yahoo Finance"""
        logger.info(f"Downloading {self.ticker} data for period {self.period}...")
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period)
        
        if df.empty:
            raise ValueError(f"No data downloaded for {self.ticker}")
        
        logger.info(f"Downloaded {len(df)} data points")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features"""
        df = df.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Relative strength index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop NaN created by indicators
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            data: normalized data
            
        Returns:
            X: input sequences (samples, lookback, features)
            y: target values (samples, )
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            
            # Predict closing price (first column)
            y.append(data[i, 0])  
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, use_technical_indicators: bool = True) -> Tuple:
        """
        Args:
            use_technical_indicators: boolean flag - whether to include technical indicators
            
        Returns:
            tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        """
        # Download data
        df = self.download_data()
        
        if use_technical_indicators:
            df = self.add_technical_indicators(df)
            features = ['Close', 'MA_5', 'MA_20', 'RSI']
        else:
            features = ['Close']
        
        # Extract features
        data = df[features].values
        
        # Calculate split indices
        n = len(data)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        
        # Split data before normalization to prevent data leakage
        train_data = data[:train_end]
        
        # Include training data for sequence creation
        val_data = data[:val_end]  

        # Include all data for sequence creation
        test_data = data[:n] 
        
        # Fit scaler only on training data
        self.scaler.fit(train_data)
        
        # Transform all data
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        X_val_full, y_val_full = self.create_sequences(val_scaled)
        X_test_full, y_test_full = self.create_sequences(test_scaled)
        
        # Extract val set - after training period
        val_start_idx = len(y_train)
        val_end_idx = val_start_idx + int(n * self.val_split)
        X_val = X_val_full[val_start_idx:val_end_idx]
        y_val = y_val_full[val_start_idx:val_end_idx]
        
        # Extract test set - after val period
        test_start_idx = len(y_train) + len(y_val)
        X_test = X_test_full[test_start_idx:]
        y_test = y_test_full[test_start_idx:]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Feature shape: {X_train.shape}")
        
        self.data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        return X_train, y_train, X_val, y_val, X_test, y_test, self.scaler
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale"""
        
        # Create dummy array with same shape as orig features
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        
        # Inverse transform
        original_scale = self.scaler.inverse_transform(dummy)
        
        return original_scale[:, 0]