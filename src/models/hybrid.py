"""
Hybrid and Ensemble Deep Learning Models for Weather Prediction.

This module implements three hybrid architectures:
1. TCN-LSTM Hybrid: TCN encoder → LSTM decoder
2. CNN-Attention Hybrid: TCN backbone + Multi-Head Self-Attention
3. Stacking Ensemble: Meta-learner on base model predictions
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


# ── Shared building blocks ──────────────────────────────────────

class CausalConv1d(nn.Module):
    """1D causal convolution with padding to preserve sequence length."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.padding, dilation=dilation))

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Residual TCN block with two causal convolutions."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        res = self.downsample(x) if self.downsample else x
        return self.relu(out + res)


# ═══════════════════════════════════════════════════════════════
# 1. TCN-LSTM Hybrid
# ═══════════════════════════════════════════════════════════════

class TCN_LSTM(nn.Module):
    """
    Hybrid TCN-LSTM: TCN encoder extracts local temporal features,
    LSTM decoder captures long-range sequential dependencies.

    Architecture: Input → TCN(3 blocks) → LSTM(128) → FC → Output
    """
    def __init__(self, input_size, tcn_channels=[64, 64, 64],
                 lstm_hidden=128, lstm_layers=1, kernel_size=3, dropout=0.2):
        super().__init__()

        # TCN Encoder
        tcn_layers = []
        for i, out_ch in enumerate(tcn_channels):
            in_ch = input_size if i == 0 else tcn_channels[i - 1]
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size,
                                       dilation=2**i, dropout=dropout))
        self.tcn_encoder = nn.Sequential(*tcn_layers)

        # LSTM Decoder
        self.lstm = nn.LSTM(
            tcn_channels[-1], lstm_hidden, lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        # TCN expects (batch, channels, seq_len)
        tcn_out = self.tcn_encoder(x.transpose(1, 2))  # → (B, C, T)
        # LSTM expects (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(tcn_out.transpose(1, 2))
        # Use last hidden state
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 2. CNN-Attention Hybrid
# ═══════════════════════════════════════════════════════════════

class CNN_Attention(nn.Module):
    """
    Hybrid CNN-Attention: TCN backbone extracts temporal features,
    Multi-Head Self-Attention learns which time steps are most important.

    Architecture: Input → TCN(3 blocks) → MultiHeadAttention → FC → Output
    """
    def __init__(self, input_size, tcn_channels=[64, 64, 64],
                 n_heads=4, kernel_size=3, dropout=0.2):
        super().__init__()

        # TCN backbone
        tcn_layers = []
        for i, out_ch in enumerate(tcn_channels):
            in_ch = input_size if i == 0 else tcn_channels[i - 1]
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size,
                                       dilation=2**i, dropout=dropout))
        self.tcn_backbone = nn.Sequential(*tcn_layers)

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=tcn_channels[-1],
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True)

        self.layer_norm = nn.LayerNorm(tcn_channels[-1])
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Store attention weights for visualisation
        self.attn_weights = None

    def forward(self, x):
        # x: (batch, seq_len, features)
        tcn_out = self.tcn_backbone(x.transpose(1, 2))  # → (B, C, T)
        tcn_out = tcn_out.transpose(1, 2)  # → (B, T, C)

        # Self-attention with residual connection
        attn_out, self.attn_weights = self.attention(
            tcn_out, tcn_out, tcn_out)
        attn_out = self.layer_norm(tcn_out + self.dropout(attn_out))

        # Use last time step (or could pool)
        out = attn_out[:, -1, :]
        return self.fc(out).squeeze(-1)

    def get_attention_weights(self):
        """Return last computed attention weights for visualisation."""
        return self.attn_weights


# ═══════════════════════════════════════════════════════════════
# 3. Stacking Ensemble
# ═══════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Stacking Ensemble: trains a meta-learner (Ridge regression) on
    the predictions of multiple base models.

    This is not a nn.Module — it uses sklearn for the meta-learner
    and wraps trained PyTorch base models.
    """
    def __init__(self, base_models, device='cpu'):
        """
        Args:
            base_models: dict of {name: trained_model}
            device: torch device
        """
        self.base_models = base_models
        self.device = device
        self.meta_learner = None
        self.model_names = list(base_models.keys())

    def _get_base_predictions(self, loader):
        """Get predictions from all base models on a data loader."""
        all_preds = {name: [] for name in self.model_names}
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                for name, model in self.base_models.items():
                    model.eval()
                    pred = model(X_batch).cpu().numpy()
                    all_preds[name].extend(pred)
                all_targets.extend(y_batch.numpy())

        # Stack into (n_samples, n_models) matrix
        meta_features = np.column_stack(
            [np.array(all_preds[name]) for name in self.model_names])
        targets = np.array(all_targets)
        return meta_features, targets

    def fit(self, val_loader):
        """Fit the meta-learner on validation set predictions."""
        from sklearn.linear_model import RidgeCV
        meta_features, targets = self._get_base_predictions(val_loader)
        self.meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        self.meta_learner.fit(meta_features, targets)
        return self

    def predict(self, loader):
        """Generate ensemble predictions using fitted meta-learner."""
        meta_features, targets = self._get_base_predictions(loader)
        preds = self.meta_learner.predict(meta_features)
        return preds, targets

    def get_weights(self):
        """Return the learned model combination weights."""
        if self.meta_learner is None:
            return None
        weights = self.meta_learner.coef_
        return dict(zip(self.model_names, weights))


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def get_hybrid_model(name, input_size, **kwargs):
    """Factory function to create hybrid models by name."""
    models = {
        'tcn-lstm': TCN_LSTM,
        'cnn-attention': CNN_Attention,
    }
    name_lower = name.lower()
    if name_lower not in models:
        raise ValueError(f"Unknown hybrid model: {name}. "
                         f"Choose from: {list(models.keys())}")
    return models[name_lower](input_size, **kwargs)
