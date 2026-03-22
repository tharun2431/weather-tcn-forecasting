"""
Visualization Utilities

This module contains plotting functions for training curves, predictions, and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = 'Training and Validation Loss',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots()
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = 'Actual vs Predicted',
    n_samples: int = 200,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        n_samples: Number of samples to plot
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots()
    
    n = min(n_samples, len(actual))
    x = range(n)
    
    ax.plot(x, actual[:n], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax.plot(x, predicted[:n], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = 'Prediction Scatter Plot',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot scatter of actual vs predicted with regression line.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots()
    
    ax.scatter(actual, predicted, alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
    title: str = 'Model Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing different models.
    
    Args:
        results: Dictionary with model names as keys and metrics dict as values
        metric: Metric to compare
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots()
    
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    
    colors = sns.color_palette('viridis', len(models))
    bars = ax.bar(models, values, color=colors)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = 'Feature Importance',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Horizontal bar chart of feature importance.
    
    Args:
        feature_names: Names of features
        importances: Importance values
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    
    colors = sns.color_palette('viridis', len(feature_names))
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color=colors)
    
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = 'Residual Analysis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residual distribution and over time.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = actual - predicted
    
    # Residuals over time
    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
