import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)

class EvalMetrics(TypedDict):
    RMSE: float
    MAE: float
    MAPE: float
    R2: float
    predictions: np.ndarray


class LSTMModel(nn.Module):
    """LSTM for time series prediction with early stopping"""
    
    def __init__(self, input_size: int, hidden_units: int, 
                 dropout_rate: float, num_layers: int = 2):
        """
        Args:
            input_size: num of input features
            hidden_units: num of hidden units in LSTM
            dropout_rate: dropout rate for regularization
            num_layers: num of LSTM layers
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        """Forward pass through the network"""
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        dropped = self.dropout(last_output)
        
        # Final prediction
        output = self.fc(dropped)
        
        return output


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Initialize early stopping
        
        Args:
            patience: num of epochs to wait for improvement
            min_delta: min change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: current validation loss
            
        Returns: True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_units: int,
    learning_rate: float,
    dropout_rate: float,
    batch_size: int,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    verbose: int = 0
) -> Tuple[LSTMModel, float]:
    """
    Train LSTM model with given hyperparams
    
    Args:
        X_train: training input sequences
        y_train: training targets
        X_val: val input sequences
        y_val: val targets
        hidden_units: num of LSTM hidden units
        learning_rate: learning rate for optimizer
        dropout_rate: dropout rate
        batch_size: batch size for training
        epochs: max num of epochs
        patience: early stopping patience
        min_delta: min delta for early stopping
        verbose: verbosity level
        
    Returns: trained model and validation RMSE
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, int(hidden_units), dropout_rate).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # Training loop
    best_model_state = None
    best_val_rmse = float('inf')
    
    batch_size = int(batch_size)
    num_batches = len(X_train_t) // batch_size
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_X = X_train_t[batch_indices]
            batch_y = y_train_t[batch_indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_rmse = torch.sqrt(val_loss).item()
        
        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
        
        if verbose > 0 and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, "
                       f"Train Loss: {np.mean(train_losses):.6f}, "
                       f"Val RMSE: {val_rmse:.6f}")
        
        # Early stopping check
        if early_stopping(val_rmse):
            if verbose > 0:
                logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_rmse


def evaluate_model(model: LSTMModel, X: np.ndarray, y: np.ndarray) -> EvalMetrics:
    """
    Args:
        model: trained LSTM model
        X: input sequences
        y: true targets
        
    Returns: dictionary of evaluation metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(device)
        
        predictions = model(X_t).cpu().numpy().flatten()
        y_true = y.flatten()
        
        # Calculate metrics
        mse = np.mean((y_true - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - predictions))
        mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100
        
        # R^2
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'predictions': predictions
    }