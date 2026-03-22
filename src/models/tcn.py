"""
Temporal Convolutional Network (TCN) Implementation

This module implements the TCN architecture for time series forecasting.
Based on: Bai et al. (2018) - "An Empirical Evaluation of Generic Convolutional 
and Recurrent Networks for Sequence Modeling"
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Optional


class CausalConv1d(nn.Module):
    """
    Causal Convolution Layer - ensures output at time t only depends on inputs up to time t.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        dilation: Spacing between kernel elements
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        ))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
        Returns:
            Output tensor of shape (batch, channels, sequence_length)
        """
        out = self.conv(x)
        # Remove extra padding from the end to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    TCN Residual Block with two causal convolutions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        dilation: Spacing between kernel elements
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # First causal convolution
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second causal convolution
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channel mismatch)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
        Returns:
            Output tensor of shape (batch, out_channels, sequence_length)
        """
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        residual = x if self.downsample is None else self.downsample(x)
        
        return self.relu_out(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting.
    
    The network uses dilated causal convolutions with exponentially increasing
    dilation factors to achieve a large receptive field while maintaining causality.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features (forecast horizon)
        num_channels: List of channel sizes for each TCN layer
        kernel_size: Kernel size for all layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        num_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # TCN expects (batch, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        out = self.network(x)
        
        # Take the last time step and apply fully connected layer
        out = out[:, :, -1]
        out = self.fc(out)
        
        return out
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN."""
        num_levels = len(self.network)
        kernel_size = 3  # Assuming default
        return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)


class TCNConfig:
    """Configuration class for TCN hyperparameters."""
    
    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 1,
        num_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        sequence_length: int = 30,
        epochs: int = 100
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.epochs = epochs
        
    def to_dict(self) -> dict:
        return self.__dict__


def build_tcn(config: TCNConfig) -> TCN:
    """Build a TCN model from config."""
    return TCN(
        input_size=config.input_size,
        output_size=config.output_size,
        num_channels=config.num_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout
    )
