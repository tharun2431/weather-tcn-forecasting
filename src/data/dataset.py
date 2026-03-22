"""
Weather Dataset Loading and Preprocessing

This module handles loading weather data and creating sequences for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    """
    PyTorch Dataset for weather time series data.
    
    Args:
        data: Numpy array of shape (n_samples, n_features)
        sequence_length: Number of time steps in input sequence
        forecast_horizon: Number of time steps to predict
        target_col: Index of the target column (default: 0, typically temperature)
    """
    
    def __init__(
        self, 
        data: np.ndarray,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        target_col: int = 0
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence: all features for sequence_length time steps
        x = self.data[idx:idx + self.sequence_length]
        
        # Target: target column for forecast_horizon time steps
        y = self.data[
            idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon,
            self.target_col
        ]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)
        )


def load_weather_data(
    filepath: str,
    date_col: str = 'DATE',
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load weather data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        date_col: Name of the date column
        feature_cols: List of feature columns to use (None = use all numeric)
        
    Returns:
        DataFrame with datetime index and selected features
    """
    df = pd.read_csv(filepath)
    
    # Parse date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    
    # Select features
    if feature_cols:
        df = df[feature_cols]
    else:
        # Use all numeric columns
        df = df.select_dtypes(include=[np.number])
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    handle_missing: str = 'interpolate'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess weather data: handle missing values, normalize, and split.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        handle_missing: Method to handle missing values ('interpolate', 'drop', 'ffill')
        
    Returns:
        Tuple of (train_data, val_data, test_data, scaler)
    """
    # Handle missing values
    if handle_missing == 'interpolate':
        df = df.interpolate(method='linear', limit_direction='both')
    elif handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'ffill':
        df = df.ffill().bfill()
    
    data = df.values
    n = len(data)
    
    # Split data (maintaining temporal order)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Normalize using training data statistics
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    return train_data, val_data, test_data, scaler


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    sequence_length: int = 30,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    target_col: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train_data: Training data array
        val_data: Validation data array
        test_data: Test data array
        sequence_length: Input sequence length
        forecast_horizon: Number of steps to forecast
        batch_size: Batch size for DataLoader
        target_col: Target column index
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = WeatherDataset(train_data, sequence_length, forecast_horizon, target_col)
    val_dataset = WeatherDataset(val_data, sequence_length, forecast_horizon, target_col)
    test_dataset = WeatherDataset(test_data, sequence_length, forecast_horizon, target_col)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
