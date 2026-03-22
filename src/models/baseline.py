"""
Baseline Models for Comparison

This module implements LSTM and GRU baseline models for weather forecasting.
These serve as comparison benchmarks for the TCN model.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMBaseline(nn.Module):
    """
    LSTM-based baseline model for time series forecasting.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        output_size: Number of output features
        dropout: Dropout probability (applied between LSTM layers)
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take output from last time step
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out


class GRUBaseline(nn.Module):
    """
    GRU-based baseline model for time series forecasting.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of GRU hidden units
        num_layers: Number of GRU layers
        output_size: Number of output features
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional GRU
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Take output from last time step
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline that flattens the input sequence.
    
    Args:
        input_size: Number of input features
        sequence_length: Length of input sequence
        hidden_size: Hidden layer size
        output_size: Number of output features
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        hidden_size: int = 128,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        flat_size = input_size * sequence_length
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        Returns:
            Output tensor of shape (batch, output_size)
        """
        return self.network(x)


def get_baseline_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    sequence_length: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: One of 'lstm', 'gru', 'mlp'
        input_size: Number of input features
        output_size: Number of output features
        sequence_length: Required for MLP model
        **kwargs: Additional arguments for the model
        
    Returns:
        Instantiated model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMBaseline(input_size, output_size=output_size, **kwargs)
    elif model_type == 'gru':
        return GRUBaseline(input_size, output_size=output_size, **kwargs)
    elif model_type == 'mlp':
        if sequence_length is None:
            raise ValueError("sequence_length required for MLP model")
        return SimpleMLP(input_size, sequence_length, output_size=output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
