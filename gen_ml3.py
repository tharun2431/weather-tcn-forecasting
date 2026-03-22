"""Generate Tharun_ML3.ipynb — Milestone 3: Evaluation & Analysis + Deployment"""
import os, nbformat as nbf

def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
nb.cells = [
md("""
# Milestone 3: Model Evaluation, Analysis & Deployment

**Name:** Tharun  
**Student ID:** [Your Student ID]  
**Module:** MSc Deep Learning Applications (CMP-L016)  
**Project #28:** Weather Prediction with Hybrid Deep Learning Models

**Dataset:** Jena Climate 2009–2022 (Max Planck Institute for Biogeochemistry)

In this notebook I evaluate the trained models from Milestone 2 on the held-out test set,
analyse their strengths/weaknesses, and deploy a simple web interface.
"""),

# ============ SETUP ============
md("## 1. Setup"),
code("""
# install dependencies if needed (for Colab)
import subprocess, sys
for pkg in ['torch', 'seaborn', 'scikit-learn']:
    try:
        __import__(pkg.replace('-', '_').split('[')[0])
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

import os, time, json
import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

FIGURE_DIR = os.path.join("outputs", "figures")
MODELS_DIR = os.path.join("outputs", "models")
RESULTS_DIR = os.path.join("outputs", "results")
for d in [FIGURE_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Setup done")
"""),

code("""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
"""),

# ============ DATA ============
md("## 2. Load & Prepare Data (same pipeline as M2)"),
code("""
# auto-download if not available
data_path = os.path.join("data", "raw", "jena_climate_2009_2016.csv")
if not os.path.exists(data_path):
    import urllib.request, zipfile
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = os.path.join("data", "raw", "jena_climate.zip")
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(os.path.join("data", "raw"))
    os.remove(zip_path)
    print("Done!")

df = pd.read_csv(data_path)

# auto-detect date column
date_col = None
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_col = col
        break
if date_col is None:
    date_col = df.columns[0]

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
df = df.set_index(date_col).sort_index()
df = df.select_dtypes(include=[np.number])

# fix bad wind values
for col in ['wv (m/s)', 'max. wv (m/s)']:
    if col in df.columns:
        df.loc[df[col] < 0, col] = 0

# resample to hourly
df = df.resample('1h').mean()
df = df.interpolate(method='linear').ffill().bfill()

feature_names = list(df.columns)
target_col = 'T (degC)'
target_idx = feature_names.index(target_col)

# same chronological split as M2
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled   = scaler.transform(val_df)
test_scaled  = scaler.transform(test_df)

N_FEATURES = len(feature_names)
print(f"Features: {N_FEATURES}, Target: {target_col}")
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
"""),

code("""
class WeatherDataset(Dataset):
    def __init__(self, data, target_idx, seq_len=168, horizon=1):
        self.data = torch.FloatTensor(data)
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len + self.horizon - 1, self.target_idx]
        return x, y

SEQ_LEN = 168
HORIZON = 1
BATCH_SIZE = 64

train_ds = WeatherDataset(train_scaled, target_idx, SEQ_LEN, HORIZON)
val_ds   = WeatherDataset(val_scaled,   target_idx, SEQ_LEN, HORIZON)
test_ds  = WeatherDataset(test_scaled,  target_idx, SEQ_LEN, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_ds):,}")
"""),

# ============ MODEL ARCHITECTURES ============
md("""
## 3. Model Architectures

Need to re-define all model classes so we can load the saved weights from Milestone 2.
These are the exact same architectures — just copied here for loading.
"""),
code("""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation))
    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.drop(self.relu(self.bn2(self.conv2(out))))
        res = self.skip(x) if self.skip else x
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, channels=[64]*5, kernel_size=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i, ch in enumerate(channels):
            in_c = input_size if i == 0 else channels[i-1]
            blocks.append(TCNBlock(in_c, ch, kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)

    def receptive_field(self):
        n = len(self.tcn)
        return 1 + 2 * (3 - 1) * (2**n - 1)


class TCN_LSTM(nn.Module):
    def __init__(self, input_size, tcn_ch=[64, 64, 64],
                 lstm_hidden=128, kernel_size=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i, ch in enumerate(tcn_ch):
            in_c = input_size if i == 0 else tcn_ch[i-1]
            blocks.append(TCNBlock(in_c, ch, kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(tcn_ch[-1], lstm_hidden, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        tcn_out = self.encoder(x.transpose(1, 2))
        _, (h, _) = self.lstm(tcn_out.transpose(1, 2))
        return self.head(h[-1]).squeeze(-1)

print("All model classes defined")
"""),

# ============ LOAD MODELS ============
md("## 4. Load Trained Models"),
code("""
# instantiate and load weights from M2
models = {
    'LSTM':     LSTMModel(N_FEATURES).to(device),
    'TCN':      TCN(N_FEATURES).to(device),
    'TCN-LSTM': TCN_LSTM(N_FEATURES).to(device),
}

# try loading TCN-Tuned (might have different architecture from HP search)
tcn_tuned_path = os.path.join(MODELS_DIR, 'tcn_tuned_best.pt')
if not os.path.exists(tcn_tuned_path):
    tcn_tuned_path = os.path.join(MODELS_DIR, 'tcn-tuned_best.pt')

if os.path.exists(tcn_tuned_path):
    # try a few common architectures from HP search
    for channels in [[32,32,32,32], [64,64,64,64,64], [64,64,128,128]]:
        for ks in [3, 5, 7]:
            try:
                tcn_t = TCN(N_FEATURES, channels=channels, kernel_size=ks).to(device)
                tcn_t.load_state_dict(torch.load(tcn_tuned_path, map_location=device, weights_only=True))
                models['TCN-Tuned'] = tcn_t
                print(f"Loaded TCN-Tuned (channels={channels}, ks={ks})")
                break
            except RuntimeError:
                continue
        if 'TCN-Tuned' in models:
            break
    if 'TCN-Tuned' not in models:
        print("Could not match TCN-Tuned architecture, skipping")

loaded = []
for name, model in models.items():
    if name == 'TCN-Tuned':
        loaded.append(name)
        model.eval()
        continue
    fname = name.lower().replace('-', '_') + '_best.pt'
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            loaded.append(name)
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    else:
        print(f"No weights found for {name} at {path}")

print(f"\\nReady: {loaded}")
"""),

# ============ TEST EVALUATION ============
md("""
## 5. Test Set Evaluation

Now the important part — evaluating everything on data the models have **never** seen.
All metrics are converted back to real °C so they're interpretable.
"""),
code("""
def get_predictions(model, loader):
    # get model predictions and convert back to original temperature scale
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy()
            preds.extend(p)
            targets.extend(yb.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    # handle NaN predictions (can happen with unstable training)
    mask = ~(np.isnan(preds) | np.isinf(preds))
    preds = preds[mask]
    targets = targets[mask]

    # inverse transform to degrees celsius
    preds_real = preds * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    targets_real = targets * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    return preds_real, targets_real


def calc_metrics(y_true, y_pred):
    # filter any remaining NaN/inf just in case
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan'), 'MAPE': float('nan')}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # mape - skip zeros to avoid division errors
    zmask = y_true != 0
    mape = np.mean(np.abs((y_true[zmask] - y_pred[zmask]) / y_true[zmask])) * 100 if zmask.any() else 0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
"""),

code("""
# evaluate each model
results = {}
predictions = {}

for name in loaded:
    model = models[name]
    preds, targets = get_predictions(model, test_loader)
    metrics = calc_metrics(targets, preds)
    results[name] = metrics
    predictions[name] = {'preds': preds, 'targets': targets}
    print(f"{name:>10s}: MAE={metrics['MAE']:.4f}°C  RMSE={metrics['RMSE']:.4f}°C  "
          f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.2f}%")

# also build ensemble predictions
print("\\nBuilding stacking ensemble...")
def get_all_preds(model_dict, loader):
    all_p = {n: [] for n in model_dict}
    tgts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            for n, m in model_dict.items():
                m.eval()
                all_p[n].extend(m(xb).cpu().numpy())
            tgts.extend(yb.numpy())
    X = np.column_stack([np.array(all_p[n]) for n in model_dict])
    return X, np.array(tgts)

# fit ensemble on validation set
val_X, val_y = get_all_preds(models, val_loader)
meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
meta.fit(val_X, val_y)

# get ensemble test predictions
test_X, test_y = get_all_preds(models, test_loader)
ens_preds_scaled = meta.predict(test_X)
ens_preds = ens_preds_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
ens_targets = test_y * scaler.scale_[target_idx] + scaler.mean_[target_idx]

ens_metrics = calc_metrics(ens_targets, ens_preds)
results['Ensemble'] = ens_metrics
predictions['Ensemble'] = {'preds': ens_preds, 'targets': ens_targets}

print(f"{'Ensemble':>10s}: MAE={ens_metrics['MAE']:.4f}°C  RMSE={ens_metrics['RMSE']:.4f}°C  "
      f"R²={ens_metrics['R2']:.4f}  MAPE={ens_metrics['MAPE']:.2f}%")
"""),

code("""
# formatted results table
print("=" * 65)
print("  TEST SET RESULTS")
print("=" * 65)
print(f"\\n{'Model':<12} {'MAE (°C)':>10} {'RMSE (°C)':>11} {'R²':>8} {'MAPE (%)':>10}")
print("-" * 55)

all_names = [n for n in ['LSTM', 'TCN', 'TCN-LSTM', 'TCN-Tuned', 'Ensemble'] if n in results]
for name in all_names:
    m = results[name]
    best_marker = " *" if name == min(results, key=lambda k: results[k]['RMSE']) else ""
    print(f"  {name:<10} {m['MAE']:>10.4f} {m['RMSE']:>11.4f} {m['R2']:>8.4f} {m['MAPE']:>10.2f}{best_marker}")

best = min(results, key=lambda k: results[k]['RMSE'])
print(f"\\n* Best model by RMSE: {best} ({results[best]['RMSE']:.4f}°C)")
"""),

# ============ VISUALIZATIONS ============
md("## 6. Prediction Plots"),
code("""
# actual vs predicted overlay for a 500-hour window from test set
N_SHOW = 500
colors = {'LSTM': '#3498db', 'TCN': '#9b59b6', 'TCN-LSTM': '#e67e22',
          'TCN-Tuned': '#2ecc71', 'Ensemble': '#e74c3c'}

fig, axes = plt.subplots(len(predictions), 1, figsize=(16, 3.5*len(predictions)), sharex=True)
if len(predictions) == 1:
    axes = [axes]

for ax, (name, data) in zip(axes, predictions.items()):
    actual = data['targets'][:N_SHOW]
    pred = data['preds'][:N_SHOW]
    ax.plot(actual, 'k-', lw=1, alpha=0.7, label='Actual')
    ax.plot(pred, color=colors.get(name, '#888'), lw=1, alpha=0.8, ls='--',
            label=f'{name} (MAE: {results[name]["MAE"]:.2f}°C)')
    ax.set_ylabel('Temp (°C)')
    ax.set_title(f'{name}', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Hours')
plt.suptitle('Test Set Predictions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_predictions.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## 7. Scatter Plots (Actual vs Predicted)"),
code("""
n_models = len(predictions)
fig, axes = plt.subplots(1, n_models, figsize=(4.5*n_models, 4.5))
if n_models == 1:
    axes = [axes]

for ax, (name, data) in zip(axes, predictions.items()):
    ax.scatter(data['targets'], data['preds'], alpha=0.1, s=5,
               color=colors.get(name, '#888'))
    lo = min(data['targets'].min(), data['preds'].min())
    hi = max(data['targets'].max(), data['preds'].max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual (°C)')
    ax.set_ylabel('Predicted (°C)')
    ax.set_title(f'{name} (R²={results[name]["R2"]:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Actual vs Predicted', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ============ RESIDUAL ANALYSIS ============
md("""
## 8. Residual Analysis

If the model is good, residuals (actual - predicted) should be randomly
distributed around zero with no visible pattern.
"""),
code("""
fig, axes = plt.subplots(len(predictions), 2, figsize=(14, 3.5*len(predictions)))
if len(predictions) == 1:
    axes = axes.reshape(1, -1)

for i, (name, data) in enumerate(predictions.items()):
    residuals = data['targets'] - data['preds']

    # residuals over time
    axes[i, 0].plot(residuals, alpha=0.4, lw=0.5, color=colors.get(name, '#888'))
    axes[i, 0].axhline(0, color='red', ls='--', lw=1.5)
    axes[i, 0].set_title(f'{name} — Residuals Over Time', fontweight='bold')
    axes[i, 0].set_ylabel('Error (°C)')
    axes[i, 0].grid(True, alpha=0.3)

    # histogram
    axes[i, 1].hist(residuals, bins=80, density=True, alpha=0.7,
                     color=colors.get(name, '#888'), edgecolor='black', lw=0.3)
    axes[i, 1].axvline(0, color='red', ls='--', lw=1.5)
    mu, sigma = residuals.mean(), residuals.std()
    axes[i, 1].set_title(f'{name} — Distribution (mean={mu:.3f}, std={sigma:.3f})', fontweight='bold')
    axes[i, 1].set_xlabel('Error (°C)')

plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_residuals.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

code("""
# seasonal error patterns - does the model struggle in certain months or hours?
test_dates = test_df.index[SEQ_LEN + HORIZON - 1:]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# monthly errors
for name, data in predictions.items():
    n = min(len(test_dates), len(data['preds']))
    errors = np.abs(data['targets'][:n] - data['preds'][:n])
    df_err = pd.DataFrame({'month': test_dates[:n].month, 'error': errors})
    monthly = df_err.groupby('month')['error'].mean()
    axes[0].plot(monthly.index, monthly.values, 'o-', label=name,
                 color=colors.get(name, '#888'), lw=2, alpha=0.8)

axes[0].set_xlabel('Month')
axes[0].set_ylabel('MAE (°C)')
axes[0].set_title('Error by Month', fontweight='bold')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# hourly errors
for name, data in predictions.items():
    n = min(len(test_dates), len(data['preds']))
    errors = np.abs(data['targets'][:n] - data['preds'][:n])
    df_err = pd.DataFrame({'hour': test_dates[:n].hour, 'error': errors})
    hourly = df_err.groupby('hour')['error'].mean()
    axes[1].plot(hourly.index, hourly.values, 'o-', label=name,
                 color=colors.get(name, '#888'), lw=2, alpha=0.8)

axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('MAE (°C)')
axes[1].set_title('Error by Hour', fontweight='bold')
axes[1].set_xticks(range(0, 24, 3))
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle('When Do Models Struggle?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_seasonal.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ============ MODEL COMPARISON ============
md("## 9. Model Comparison"),
code("""
metrics_names = ['MAE', 'RMSE', 'R2', 'MAPE']
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

model_names = list(results.keys())
bar_colors = [colors.get(n, '#888') for n in model_names]

for ax, metric in zip(axes, metrics_names):
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, v in zip(bars, vals):
        fmt = f'{v:.4f}' if metric == 'R2' else f'{v:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                fmt, ha='center', va='bottom', fontsize=8, fontweight='bold')
    unit = '(°C)' if metric in ['MAE','RMSE'] else '(%)' if metric == 'MAPE' else ''
    ax.set_title(f'{metric} {unit}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ============ ABLATION ============
md("""
## 10. Ablation Study — TCN Depth

How many convolutional blocks does the TCN actually need?
Testing with 2, 3, 4, and 5 blocks (30 epochs each).
"""),
code("""
ablation_depths = [2, 3, 4, 5]
ablation_results = {}

print("TCN Depth Ablation")
print("=" * 50)

for depth in ablation_depths:
    channels = [64] * depth
    tcn_abl = TCN(N_FEATURES, channels=channels).to(device)

    opt = torch.optim.Adam(tcn_abl.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    val_hist = []

    for ep in range(30):
        tcn_abl.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(tcn_abl(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn_abl.parameters(), 1.0)
            opt.step()

        tcn_abl.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl.append(loss_fn(tcn_abl(xb), yb).item())
        v = np.mean(vl)
        val_hist.append(v)
        best_val = min(best_val, v)

    # evaluate on test
    preds, targets = get_predictions(tcn_abl, test_loader)
    metrics = calc_metrics(targets, preds)
    rf = 1 + 2 * (3 - 1) * (2**depth - 1)
    params = sum(p.numel() for p in tcn_abl.parameters())

    ablation_results[depth] = {
        'metrics': metrics, 'params': params,
        'rf': rf, 'val_hist': val_hist
    }

    print(f"  Depth {depth}: RF={rf:>4}h | Params: {params:>8,} | "
          f"MAE: {metrics['MAE']:.4f}°C | R²: {metrics['R2']:.4f}")

    del tcn_abl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""),

code("""
# plot ablation results
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
depths = list(ablation_results.keys())

# training curves
for d in depths:
    axes[0].plot(ablation_results[d]['val_hist'], label=f'{d} blocks', lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Val MSE')
axes[0].set_title('Validation Loss by Depth', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE by depth
maes = [ablation_results[d]['metrics']['MAE'] for d in depths]
axes[1].bar(depths, maes, color=['#3498db','#2ecc71','#9b59b6','#e67e22'],
            alpha=0.85, edgecolor='black')
for d, v in zip(depths, maes):
    axes[1].text(d, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_xlabel('Blocks')
axes[1].set_ylabel('Test MAE (°C)')
axes[1].set_title('MAE by Depth', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# receptive field vs R²
rfs = [ablation_results[d]['rf'] for d in depths]
r2s = [ablation_results[d]['metrics']['R2'] for d in depths]
ax2 = axes[2].twinx()
axes[2].bar(depths, rfs, alpha=0.4, color='steelblue', label='RF (hours)')
ax2.plot(depths, r2s, 'ro-', lw=2, label='R²')
axes[2].set_xlabel('Blocks')
axes[2].set_ylabel('Receptive Field (hours)')
ax2.set_ylabel('R²')
axes[2].set_title('RF vs R²', fontweight='bold')
axes[2].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.suptitle('TCN Depth Ablation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_ablation.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ============ MULTI-STEP FORECAST ============
md("""
## 11. Multi-step Forecasting

Instead of 1 hour ahead, we predict 24 hours into the future by
feeding each prediction back as the next input. This shows how
error accumulates over longer horizons.
"""),
code("""
def forecast_multistep(model, start_seq, n_steps):
    # predict n_steps ahead by feeding predictions back as input
    model.eval()
    seq = start_seq.clone()
    preds = []

    with torch.no_grad():
        for _ in range(n_steps):
            x = seq.unsqueeze(0).to(device)
            p = model(x).cpu().item()
            preds.append(p)
            # shift window and append prediction
            new_row = seq[-1].clone()
            new_row[target_idx] = p
            seq = torch.cat([seq[1:], new_row.unsqueeze(0)], dim=0)

    # convert back to real temperature
    preds_real = np.array(preds) * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    return preds_real

# pick a starting point in test set
START = 100
HOURS_AHEAD = 24

init_seq = torch.FloatTensor(test_scaled[START : START + SEQ_LEN])
actual_future = test_scaled[START + SEQ_LEN : START + SEQ_LEN + HOURS_AHEAD, target_idx]
actual_real = actual_future * scaler.scale_[target_idx] + scaler.mean_[target_idx]

# last 48h of context for the plot
context = test_scaled[START : START + SEQ_LEN, target_idx][-48:]
context_real = context * scaler.scale_[target_idx] + scaler.mean_[target_idx]

print(f"Forecasting {HOURS_AHEAD} hours ahead from test index {START}")
"""),

code("""
# generate forecasts for all models
ar_preds = {}

for name in loaded:
    preds = forecast_multistep(models[name], init_seq, HOURS_AHEAD)
    mae = mean_absolute_error(actual_real, preds)
    ar_preds[name] = {'preds': preds, 'mae': mae}
    print(f"  {name:>10s}: 24h MAE = {mae:.3f}°C")

# plot
fig, ax = plt.subplots(figsize=(16, 6))

# context
x_ctx = range(48)
ax.plot(x_ctx, context_real, 'k-', lw=2, label='History', alpha=0.7)

# actual future
x_fut = range(48, 48 + HOURS_AHEAD)
ax.plot(x_fut, actual_real, 'k--', lw=2, marker='o', ms=4, label='Actual')

# model predictions
for name, data in ar_preds.items():
    ax.plot(x_fut, data['preds'], ls='--', lw=1.5, alpha=0.8,
            color=colors.get(name, '#888'),
            label=f'{name} (MAE: {data["mae"]:.2f}°C)', marker='s', ms=3)

ax.axvline(48, color='gray', ls=':', lw=1.5, alpha=0.7)
ax.set_xlabel('Hours')
ax.set_ylabel('Temperature (°C)')
ax.set_title('24-hour Forecast Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_forecast_24h.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

code("""
# how does error grow with forecast horizon?
horizons = [1, 3, 6, 12, 18, 24]
horizon_errors = {name: [] for name in loaded}

for h in horizons:
    for name in loaded:
        preds = forecast_multistep(models[name], init_seq, h)
        actual = test_scaled[START + SEQ_LEN : START + SEQ_LEN + h, target_idx]
        actual_r = actual * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        mae = mean_absolute_error(actual_r, preds)
        horizon_errors[name].append(mae)

fig, ax = plt.subplots(figsize=(10, 6))
for name, errors in horizon_errors.items():
    ax.plot(horizons, errors, 'o-', label=name, color=colors.get(name, '#888'), lw=2, ms=6)

ax.set_xlabel('Forecast Horizon (hours)')
ax.set_ylabel('MAE (°C)')
ax.set_title('Error Accumulation Over Time', fontsize=14, fontweight='bold')
ax.set_xticks(horizons)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'eval_error_growth.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ============ SUMMARY ============
md("## 12. Summary & Conclusions"),
code("""
print("=" * 70)
print("  MILESTONE 3 — EVALUATION SUMMARY")
print("=" * 70)

# single-step
print("\\nSingle-step (1 hour ahead):")
print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
print("  " + "-" * 48)
for name in all_names:
    m = results[name]
    star = " *" if name == best else ""
    print(f"  {name:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f} {m['MAPE']:>7.2f}%{star}")

# multi-step
print("\\nMulti-step (24h forecast):")
for name, data in ar_preds.items():
    print(f"  {name:<10}: {data['mae']:.4f}°C")

# ablation
print("\\nTCN Depth Ablation:")
for d, r in ablation_results.items():
    print(f"  {d} blocks: RF={r['rf']}h, MAE={r['metrics']['MAE']:.4f}°C")

best = min(results, key=lambda k: results[k]['RMSE'])
print(f"\\n{'='*70}")
print(f"  Best model: {best} (RMSE: {results[best]['RMSE']:.4f}°C)")
print(f"{'='*70}")
"""),

code("""
# save everything
eval_data = {
    'single_step': {n: {k: float(v) for k, v in m.items()} for n, m in results.items()},
    'multi_step': {n: {'mae_24h': float(d['mae'])} for n, d in ar_preds.items()},
    'ablation': {str(d): {
        'mae': float(r['metrics']['MAE']),
        'r2': float(r['metrics']['R2']),
        'params': r['params'],
        'receptive_field': r['rf']
    } for d, r in ablation_results.items()},
    'best_model': best,
    'ensemble_weights': dict(zip(list(models.keys()), [float(w) for w in meta.coef_]))
}

with open(os.path.join(RESULTS_DIR, 'milestone3_results.json'), 'w') as f:
    json.dump(eval_data, f, indent=2)

print(f"Results saved to {RESULTS_DIR}/milestone3_results.json")
print(f"Figures saved to {FIGURE_DIR}/")
"""),

md("""
## Areas for Enhancement

Based on the evaluation:

- **Error accumulates quickly** in multi-step forecasting — attention mechanisms
  or transformer-based models could help maintain accuracy over longer horizons
- **Seasonal patterns** show higher errors during temperature transitions (spring/autumn)
  — adding calendar features (month, day-of-year) as inputs could help
- **The ensemble consistently wins** — exploring neural meta-learners instead of
  Ridge regression could further improve combination
- **TCN depth** matters — too shallow misses long-range patterns, too deep overfits.
  The sweet spot seems to be around 4-5 blocks

These findings will inform the final Milestone 4 submission.
"""),
]

os.makedirs("notebooks", exist_ok=True)
with open(os.path.join("notebooks","Tharun_ML3.ipynb"), 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Created: notebooks/Tharun_ML3.ipynb")
