"""
Weather Prediction Dashboard
Tharun — MSc Deep Learning Applications (CMP-L016)
Project #28: Weather Prediction with Hybrid Deep Learning Models

Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import os, json
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Weather Prediction — Hybrid DL Models",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* headers */
    h1, h2, h3 {
        color: #e0e7ff !important;
    }
    
    /* custom card */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .info-card h4 {
        color: #00d4ff !important;
        margin-bottom: 0.5rem;
    }
    
    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #a0aec0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 212, 255, 0.15);
        color: #00d4ff !important;
    }
    
    /* divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL DEFINITIONS ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation))
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out

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
            blocks.append(TCNBlock(in_c, ch, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        return self.head(out[:, :, -1]).squeeze(-1)

class TCN_LSTM(nn.Module):
    def __init__(self, input_size, tcn_ch=[64,64,64], lstm_hidden=128,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i, ch in enumerate(tcn_ch):
            in_c = input_size if i == 0 else tcn_ch[i-1]
            blocks.append(TCNBlock(in_c, ch, kernel_size, dilation=2**i, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(tcn_ch[-1], lstm_hidden, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, x):
        tcn_out = self.encoder(x.transpose(1, 2))
        _, (h, _) = self.lstm(tcn_out.transpose(1, 2))
        return self.head(h[-1]).squeeze(-1)


# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    data_path = os.path.join("data", "raw", "jena_climate_2009_2016.csv")
    if not os.path.exists(data_path):
        return None, None, None, None

    df = pd.read_csv(data_path)
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

    for col in ['wv (m/s)', 'max. wv (m/s)']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = 0

    df = df.resample('1h').mean().interpolate('linear').ffill().bfill()

    n = len(df)
    train_end = int(n * 0.7)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end])
    scaled = scaler.transform(df)

    return df, scaled, scaler, list(df.columns)


@st.cache_resource
def load_models(n_features):
    device = torch.device('cpu')
    models_dict = {}
    model_classes = {
        'LSTM': lambda: LSTMModel(n_features),
        'TCN': lambda: TCN(n_features),
        'TCN-LSTM': lambda: TCN_LSTM(n_features),
    }
    for name, create_fn in model_classes.items():
        fname = name.lower().replace('-', '_') + '_best.pt'
        path = os.path.join("outputs", "models", fname)
        if os.path.exists(path):
            try:
                model = create_fn().to(device)
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model.eval()
                models_dict[name] = model
            except Exception:
                pass
    return models_dict


def predict_future(model, data_scaled, start_idx, seq_len, n_steps, target_idx, scaler):
    model.eval()
    seq = torch.FloatTensor(data_scaled[start_idx : start_idx + seq_len])
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = seq.unsqueeze(0)
            p = model(x).item()
            preds.append(p)
            new_row = seq[-1].clone()
            new_row[target_idx] = p
            seq = torch.cat([seq[1:], new_row.unsqueeze(0)], dim=0)
    preds_real = np.array(preds) * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    return preds_real


# ===================== PAGES =====================
def page_home():
    # hero section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 2.5rem; background: linear-gradient(90deg, #00d4ff, #7b68ee);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-weight: 800;'>
                🌤️ Weather Prediction Dashboard
            </h1>
            <p style='color: #a0aec0; font-size: 1.1rem; max-width: 600px; margin: auto;'>
                Hybrid Deep Learning Models for Temperature Forecasting<br>
                MSc Deep Learning Applications — Project #28
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # load data
    df, scaled, scaler, features = load_data()
    if df is None:
        st.error("Dataset not found. Place the CSV in `data/raw/`")
        return

    target_col = 'T (degC)'
    target_idx = features.index(target_col)
    models = load_models(len(features))

    if not models:
        st.warning("No trained models found in `outputs/models/`. Run the M2 notebook first.")
        return

    # sidebar controls
    st.sidebar.header("⚙️ Controls")
    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()), index=0)
    forecast_hours = st.sidebar.slider("Forecast Horizon", 1, 48, 24, help="Hours to predict ahead")

    n = len(df)
    test_start = int(n * 0.85)
    max_idx = n - 168 - forecast_hours
    start_point = st.sidebar.slider("Start Point", test_start, max_idx, test_start + 200)

    # current conditions
    current = df.iloc[start_point + 167]
    st.subheader("📊 Current Conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌡️ Temperature", f"{current[target_col]:.1f}°C")
    with c2:
        if 'rh (%)' in features:
            st.metric("💧 Humidity", f"{current['rh (%)']:.1f}%")
    with c3:
        if 'p (mbar)' in features:
            st.metric("📊 Pressure", f"{current['p (mbar)']:.1f} mbar")
    with c4:
        if 'wv (m/s)' in features:
            st.metric("💨 Wind", f"{current['wv (m/s)']:.1f} m/s")

    st.markdown("---")

    # forecast
    st.subheader(f"📈 {forecast_hours}h Forecast — {selected_model}")

    model = models[selected_model]
    preds = predict_future(model, scaled, start_point, 168, forecast_hours, target_idx, scaler)

    actual_slice = df.iloc[start_point + 168 : start_point + 168 + forecast_hours]
    actual_temps = actual_slice[target_col].values

    history = df.iloc[start_point + 120 : start_point + 168]
    pred_dates = pd.date_range(start=df.index[start_point + 168], periods=forecast_hours, freq='h')

    # plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history[target_col],
                             mode='lines', name='History',
                             line=dict(color='#a0aec0', width=2)))
    fig.add_trace(go.Scatter(x=pred_dates, y=actual_temps[:forecast_hours],
                             mode='lines+markers', name='Actual',
                             line=dict(color='#00d4ff', width=2),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=pred_dates, y=preds,
                             mode='lines+markers', name=f'{selected_model} Prediction',
                             line=dict(color='#7b68ee', width=2, dash='dash'),
                             marker=dict(size=4)))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation='h', y=-0.15),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Temperature (°C)', gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # error metrics
    if len(actual_temps) >= forecast_hours:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(actual_temps[:forecast_hours], preds)
        rmse = np.sqrt(mean_squared_error(actual_temps[:forecast_hours], preds))
        r2 = r2_score(actual_temps[:forecast_hours], preds)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.3f}°C")
        m2.metric("RMSE", f"{rmse:.3f}°C")
        m3.metric("R²", f"{r2:.4f}")

    st.markdown("---")

    # compare all models
    st.subheader("🔄 Compare All Models")

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Scatter(x=pred_dates, y=actual_temps[:forecast_hours],
                                     mode='lines+markers', name='Actual',
                                     line=dict(color='#ffffff', width=2.5),
                                     marker=dict(size=4)))

    model_colors = {'LSTM': '#3498db', 'TCN': '#9b59b6', 'TCN-LSTM': '#e67e22'}
    for name, m in models.items():
        p = predict_future(m, scaled, start_point, 168, forecast_hours, target_idx, scaler)
        compare_fig.add_trace(go.Scatter(
            x=pred_dates, y=p, mode='lines', name=name,
            line=dict(color=model_colors.get(name, '#888'), width=2, dash='dash')))

    compare_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation='h', y=-0.15),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Temperature (°C)', gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    # performance table
    if len(actual_temps) >= forecast_hours:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        perf = []
        for name, m in models.items():
            p = predict_future(m, scaled, start_point, 168, forecast_hours, target_idx, scaler)
            a = actual_temps[:forecast_hours]
            perf.append({
                'Model': name,
                'MAE (°C)': round(mean_absolute_error(a, p), 4),
                'RMSE (°C)': round(np.sqrt(mean_squared_error(a, p)), 4),
                'R²': round(r2_score(a, p), 4),
            })
        st.dataframe(pd.DataFrame(perf).set_index('Model'),
                      use_container_width=True)


def page_about():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #e0e7ff;'>About This Project</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class='info-card'>
    <h4>📋 Project Overview</h4>
    <p style='color: #cbd5e0;'>
    This dashboard is part of <strong>Project #28: Weather Prediction with Hybrid Deep Learning Models</strong>,
    developed for the MSc Deep Learning Applications module (CMP-L016).
    </p>
    <p style='color: #cbd5e0;'>
    The project investigates whether combining Temporal Convolutional Networks (TCN) with
    Long Short-Term Memory (LSTM) networks produces more accurate weather forecasts than
    either architecture alone.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # model architecture cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='info-card'>
        <h4>🔄 LSTM</h4>
        <p style='color: #cbd5e0; font-size: 0.9rem;'>
        <strong>Type:</strong> Recurrent Baseline<br>
        <strong>Params:</strong> 214,145<br>
        <strong>Layers:</strong> 2-layer LSTM + FC head<br>
        <strong>Role:</strong> Captures sequential dependencies through gating mechanisms
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
        <h4>⚡ TCN</h4>
        <p style='color: #cbd5e0; font-size: 0.9rem;'>
        <strong>Type:</strong> Convolutional Baseline<br>
        <strong>Params:</strong> 121,025<br>
        <strong>Layers:</strong> 5 dilated causal conv blocks<br>
        <strong>Role:</strong> Parallel processing with large receptive field (125h)
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-card'>
        <h4>🧬 TCN-LSTM Hybrid</h4>
        <p style='color: #cbd5e0; font-size: 0.9rem;'>
        <strong>Type:</strong> Hybrid (Innovation)<br>
        <strong>Params:</strong> 174,273<br>
        <strong>Layers:</strong> 3 TCN blocks → LSTM decoder<br>
        <strong>Role:</strong> TCN extracts local features, LSTM models long-range patterns
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class='info-card'>
    <h4>📊 Dataset</h4>
    <p style='color: #cbd5e0;'>
    <strong>Jena Climate Dataset</strong> — Max Planck Institute for Biogeochemistry<br>
    • 14 meteorological features recorded every 10 minutes<br>
    • Covers 2009–2022 (resampled to hourly)<br>
    • ~70,000+ samples after preprocessing<br>
    • 70/15/15 chronological train/val/test split
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class='info-card'>
    <h4>🛠️ Tech Stack</h4>
    <p style='color: #cbd5e0;'>
    Python 3.10+ • PyTorch • Streamlit • Plotly • scikit-learn • Pandas • NumPy
    </p>
    </div>
    """, unsafe_allow_html=True)


def page_results():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #e0e7ff;'>📊 Training Results</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # try loading saved results
    m2_path = os.path.join("outputs", "results", "milestone2_results.json")
    m3_path = os.path.join("outputs", "results", "milestone3_results.json")

    if os.path.exists(m2_path):
        with open(m2_path) as f:
            m2_results = json.load(f)

        st.subheader("Milestone 2 — Training Summary")
        if 'models' in m2_results:
            rows = []
            for name, info in m2_results['models'].items():
                rows.append({
                    'Model': name,
                    'Type': info.get('type', ''),
                    'Parameters': f"{info.get('params', 0):,}",
                    'Epochs': info.get('epochs', ''),
                    'Val MSE': f"{info.get('val_mse', 0):.6f}",
                    'Time': f"{info.get('time', 0):.0f}s"
                })
            st.dataframe(pd.DataFrame(rows).set_index('Model'),
                          use_container_width=True)

    if os.path.exists(m3_path):
        with open(m3_path) as f:
            m3_results = json.load(f)

        st.subheader("Milestone 3 — Test Set Evaluation")
        if 'single_step' in m3_results:
            rows = []
            for name, metrics in m3_results['single_step'].items():
                rows.append({
                    'Model': name,
                    'MAE (°C)': f"{metrics['MAE']:.4f}",
                    'RMSE (°C)': f"{metrics['RMSE']:.4f}",
                    'R²': f"{metrics['R2']:.4f}",
                    'MAPE (%)': f"{metrics['MAPE']:.2f}"
                })
            st.dataframe(pd.DataFrame(rows).set_index('Model'),
                          use_container_width=True)

            best = m3_results.get('best_model', 'Unknown')
            st.success(f"🏆 Best model: **{best}**")

        if 'ensemble_weights' in m3_results:
            st.subheader("Ensemble Weights")
            weights = m3_results['ensemble_weights']
            fig = go.Figure(data=[go.Bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                marker_color=['#3498db', '#9b59b6', '#e67e22', '#2ecc71'][:len(weights)]
            )])
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                yaxis_title='Weight',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

    if not os.path.exists(m2_path) and not os.path.exists(m3_path):
        st.info("No saved results found. Run the M2 and M3 notebooks first to generate results.")


# ==================== MAIN ====================
def main():
    # sidebar navigation
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #00d4ff; font-size: 1.3rem;'>🌤️ Navigation</h2>
        </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("", ["🏠 Dashboard", "📊 Results", "ℹ️ About"],
                            label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center; color: #718096; font-size: 0.8rem; padding-top: 1rem;'>
            <p>Tharun — CMP-L016<br>
            Project #28<br>
            MSc Deep Learning Applications</p>
        </div>
    """, unsafe_allow_html=True)

    if page == "🏠 Dashboard":
        page_home()
    elif page == "📊 Results":
        page_results()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
