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
import requests
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Jena Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Protect material icons from being overridden by custom fonts */
    .material-symbols-rounded, .stIcon, [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Google Weather dark theme override */
    .stApp {
        background-color: #202124;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* headers */
    h1, h2, h3 {
        color: #e8eaed !important;
        font-weight: 400 !important;
    }

    /* tabs (Google style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 0 !important;
        color: #9aa0a6 !important;
        padding: 10px 16px;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        outline: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #e8eaed !important;
        border-bottom: 2px solid #8ab4f8 !important;
    }
    
    /* Forecast cards */
    .forecast-card {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        background: transparent;
        transition: background 0.2s ease;
    }
    .forecast-card:hover {
        background: rgba(255,255,255,0.05);
    }
    .forecast-day {
        color: #e8eaed;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .forecast-icon {
        font-size: 1.8rem;
        margin: 5px 0;
    }
    .forecast-temp {
        color: #9aa0a6;
        font-size: 0.9rem;
    }
    .forecast-temp span {
        color: #e8eaed;
        font-weight: 600;
        margin-right: 4px;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.05);
        margin: 2rem 0;
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
JENA_CSV_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

@st.cache_data
def load_data():
    """Load and preprocess the Jena Climate dataset.

    Reads the raw CSV, resamples to hourly frequency, handles missing
    values via interpolation, and fits a StandardScaler on the training
    split (first 70%).  If the CSV is not present locally it is
    automatically downloaded from the TensorFlow public mirror.

    Returns:
        tuple: (df, scaled_array, scaler, feature_names) or four Nones
               if the dataset cannot be obtained.
    """
    data_path = os.path.join("data", "raw", "jena_climate_2009_2016.csv")
    if not os.path.exists(data_path):
        # Auto-download for Streamlit Cloud deployment
        try:
            import zipfile, io, urllib.request
            st.info("⬇️ Downloading Jena Climate dataset (first run only)...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            resp = urllib.request.urlopen(JENA_CSV_URL)
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                # Extract the CSV inside the zip
                csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
                with zf.open(csv_name) as src, open(data_path, 'wb') as dst:
                    dst.write(src.read())
            st.success("✅ Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Could not download dataset: {e}")
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


@st.cache_data(ttl=3600)
def fetch_live_jena_data(features, _scaler):
    """Fetch live data from Open-Meteo API and construct the 14 features."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 50.9272,
        "longitude": 11.5861,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                   "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Europe/Berlin"
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    
    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "T (degC)": hourly["temperature_2m"],
        "rh (%)": hourly["relative_humidity_2m"],
        "Tdew (degC)": hourly["dew_point_2m"],
        "p (mbar)": hourly["surface_pressure"],
        "wv (m/s)": np.array(hourly["wind_speed_10m"]) / 3.6,
        "max. wv (m/s)": np.array(hourly["wind_gusts_10m"]) / 3.6,
        "wd (deg)": hourly["wind_direction_10m"]
    })
    
    df = df.dropna().reset_index(drop=True)
    
    T = df["T (degC)"]
    p = df["p (mbar)"]
    rh = df["rh (%)"]
    
    T_K = T + 273.15
    df["Tpot (K)"] = T_K * (1000 / p) ** 0.286
    df["VPmax (mbar)"] = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    df["VPact (mbar)"] = df["VPmax (mbar)"] * (rh / 100)
    df["VPdef (mbar)"] = df["VPmax (mbar)"] - df["VPact (mbar)"]
    df["sh (g/kg)"] = 622 * df["VPact (mbar)"] / (p - 0.378 * df["VPact (mbar)"])
    df["H2OC (mmol/mol)"] = (df["VPact (mbar)"] / p) * 1000
    Tv = T_K * (1 + (df["sh (g/kg)"] / 1000) * 0.61)
    df["rho (g/m**3)"] = (p * 100 / (287.05 * Tv)) * 1000
    
    df = df.set_index("time")
    
    current_hour = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).floor('h')
    
    # get the nearest index that isn't later than the current_hour
    try:
        idx = df.index.get_indexer([current_hour], method='pad')[0]
    except Exception:
        idx = len(df) - 1
        
    if idx < 167:
        df_168 = df.iloc[:168].copy()
    else:
        df_168 = df.iloc[idx-167:idx+1].copy()
        
    df_168 = df_168[features]
    scaled = _scaler.transform(df_168)
    return df_168, scaled



@st.cache_resource
def load_models(n_features):
    """Load trained model weights from outputs/models/.

    Attempts to load LSTM, TCN, and TCN-LSTM models. Models whose
    weight files (.pt or .pth) are not found are silently skipped.

    Args:
        n_features (int): Number of input features (must match training).

    Returns:
        dict: Mapping of model name to loaded nn.Module in eval mode.
    """
    device = torch.device('cpu')
    models_dict = {}
    model_classes = {
        'LSTM': lambda: LSTMModel(n_features),
        'TCN': lambda: TCN(n_features),
        'TCN-LSTM': lambda: TCN_LSTM(n_features),
    }
    for name, create_fn in model_classes.items():
        # try both .pt and .pth extensions
        base = name.lower().replace('-', '_') + '_best'
        for ext in ['.pt', '.pth']:
            path = os.path.join("outputs", "models", base + ext)
            if os.path.exists(path):
                try:
                    model = create_fn().to(device)
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                    model.eval()
                    models_dict[name] = model
                except Exception:
                    pass
                break
    return models_dict


def predict_future(model, data_scaled, start_idx, seq_len, n_steps, target_idx, scaler):
    """Generate an autoregressive multi-step temperature forecast.

    Feeds each predicted value back as input for the next step.
    Results are inverse-transformed to degrees Celsius.

    Args:
        model (nn.Module): Trained model in eval mode.
        data_scaled (np.ndarray): Full scaled dataset.
        start_idx (int): Index where the input window begins.
        seq_len (int): Length of the input window (hours).
        n_steps (int): Number of hours to forecast ahead.
        target_idx (int): Column index of the target variable.
        scaler (StandardScaler): Fitted scaler for inverse transform.

    Returns:
        np.ndarray: Predicted temperatures in °C, shape (n_steps,).
    """
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
    """Render the main dashboard page styled like Google Weather."""
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
    model_options = list(models.keys()) + ["Ensemble (Best)"]
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=len(model_options)-1)
    forecast_hours = st.sidebar.slider("Forecast Horizon", 1, 48, 24, help="Hours to predict ahead")

    data_mode = st.sidebar.radio("Data Source", ["Historical (Jena Dataset)", "Live Forecast (Open-Meteo API)"])

    if data_mode == "Historical (Jena Dataset)":
        n = len(df)
        test_start = int(n * 0.85)
        max_idx = n - 168 - forecast_hours
        start_point = st.sidebar.slider("Start Point", test_start, max_idx, test_start + 200)
    else:
        st.sidebar.info("Fetching real-time weather data for Jena, Germany...")
        try:
            df, scaled = fetch_live_jena_data(features, scaler)
            start_point = 0
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return

    # current conditions
    current = df.iloc[start_point + 167]
    dt = current.name
    
    # Map basic conditions based on temp/humidity
    temp = current[target_col]
    rh = current['rh (%)'] if 'rh (%)' in features else 50
    if rh > 85:
        icon = '🌧️'
        cond_text = 'Rainy'
    elif temp < 0:
        icon = '❄️'
        cond_text = 'Snow'
    elif rh > 60:
        icon = '☁️'
        cond_text = 'Cloudy'
    else:
        icon = '☀️'
        cond_text = 'Clear'

    day_str = dt.strftime("%A %H:%M")

    # ====== HEADER UI ======
    st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;'>
<!-- Left Side: Weather Icon, Temp, Extra -->
<div style='display: flex; align-items: center;'>
<div style='font-size: 4.5rem; line-height: 1; margin-right: 20px;'>{icon}</div>
<div style='display: flex; align-items: flex-start;'>
<div style='font-size: 4rem; font-weight: 300; color: #e8eaed; line-height: 1;'>{int(temp)}</div>
<div style='font-size: 1.5rem; color: #e8eaed; padding-top: 5px; margin-right: 20px;'>°C</div>
</div>
<div style='display: flex; flex-direction: column; justify-content: center; color: #9aa0a6; font-size: 0.85rem; padding-top: 10px;'>
<div>Pressure: {current['p (mbar)'] if 'p (mbar)' in features else 0:.1f} mbar</div>
<div>Humidity: {rh:.0f}%</div>
<div>Wind: {current['wv (m/s)'] if 'wv (m/s)' in features else 0:.1f} m/s</div>
</div>
</div>

<!-- Right Side: Weather, Time, Condition -->
<div style='text-align: right;'>
<div style='font-size: 1.4rem; color: #e8eaed; font-weight: 400;'>Weather</div>
<div style='color: #9aa0a6; font-size: 1rem;'>{day_str}</div>
<div style='color: #9aa0a6; font-size: 1rem;'>{cond_text}</div>
<div style='margin-top: 10px; opacity: 0.5; font-size: 0.8rem; color: #9aa0a6;'>Jena, Germany</div>
</div>
</div>
""", unsafe_allow_html=True)

    # forecast
    if selected_model == "Ensemble (Best)":
        # Run all base models and combine with learned stacking weights
        ensemble_weights = {'LSTM': 0.45, 'TCN': 0.15, 'TCN-LSTM': 0.10}
        # Add TCN-Tuned if available
        if 'TCN-Tuned' in models:
            ensemble_weights['TCN-Tuned'] = 0.30
        # Normalize weights to available models
        available = {k: v for k, v in ensemble_weights.items() if k in models}
        total_w = sum(available.values())
        available = {k: v / total_w for k, v in available.items()}
        
        preds = np.zeros(forecast_hours)
        for mname, weight in available.items():
            p = predict_future(models[mname], scaled, start_point, 168, forecast_hours, target_idx, scaler)
            preds += weight * p
    else:
        model = models[selected_model]
        preds = predict_future(model, scaled, start_point, 168, forecast_hours, target_idx, scaler)
    actual_slice = df.iloc[start_point + 168 : start_point + 168 + forecast_hours]
    actual_temps = actual_slice[target_col].values if not actual_slice.empty else None
    
    if start_point + 168 < len(df):
        pred_start = df.index[start_point + 168]
    else:
        pred_start = df.index[-1] + pd.Timedelta(hours=1)
    
    pred_dates = pd.date_range(start=pred_start, periods=forecast_hours, freq='h')

    # ====== TABS FOR CHARTS ======
    tabs = st.tabs(["Temperature", "Compare Data"])

    def create_minimal_chart(x_data, y_data, color_fill, color_line, y_actual=None):
        fig = go.Figure()
        
        # Calculate padding to prevent text cutoff
        all_y = list(y_data)
        if y_actual is not None:
            all_y.extend(list(y_actual))
        y_min, y_max = min(all_y), max(all_y)
        y_pad_top = max(4.0, (y_max - y_min) * 0.4)
        y_pad_bot = max(1.0, (y_max - y_min) * 0.1)

        # Glow / shadow effect beneath the line
        fig.add_trace(go.Scatter(x=x_data, y=y_data,
                                 mode='lines', showlegend=False,
                                 line=dict(color=color_line, width=8, shape='spline'),
                                 opacity=0.15, hoverinfo='skip'))
        
        # Main forecast line
        fig.add_trace(go.Scatter(x=x_data, y=y_data,
                                 mode='lines+markers+text', name='Forecast',
                                 line=dict(color=color_line, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor=color_fill,
                                 marker=dict(size=8, color='#202124', line=dict(color=color_line, width=2)),
                                 text=[f"<b>{int(y)}°</b>" for y in y_data], textposition="top center",
                                 textfont=dict(color='#ffffff', size=14)))
                                 
        if y_actual is not None:
             fig.add_trace(go.Scatter(x=x_data, y=y_actual,
                                 mode='lines+markers', name='Actual',
                                 line=dict(color='#ffab40', width=2, dash='dot', shape='spline'),
                                 marker=dict(size=6, color='#ffab40')))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=280, margin=dict(l=0, r=0, t=50, b=0),
            showlegend=y_actual is not None, legend=dict(orientation="h", y=1.2, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        
        # Minimalist axes with dynamic range
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=True, tickfont=dict(color='#9aa0a6', size=11))
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False, range=[y_min - y_pad_bot, y_max + y_pad_top])
        return fig

    with tabs[0]:
        st.plotly_chart(create_minimal_chart(
            pred_dates, preds, 
            'rgba(251, 188, 5, 0.2)', '#fbbc05',  # Google Gold
            actual_temps
        ), use_container_width=True)

    with tabs[1]:
        # Compare all models minimalist view
        fig_cmp = go.Figure()
        model_colors = {'LSTM': '#4285f4', 'TCN': '#ea4335', 'TCN-LSTM': '#34a853', 'TCN-Tuned': '#fbbc05'}
        all_model_preds = {}
        for name, m in models.items():
            p = predict_future(m, scaled, start_point, 168, forecast_hours, target_idx, scaler)
            all_model_preds[name] = p
            fig_cmp.add_trace(go.Scatter(x=pred_dates, y=p, mode='lines', name=name,
                line=dict(color=model_colors.get(name, '#fff'), width=2, shape='spline')))

        # Add Ensemble line (weighted combination of all base models)
        ens_w = {'LSTM': 0.45, 'TCN': 0.15, 'TCN-LSTM': 0.10, 'TCN-Tuned': 0.30}
        avail_w = {k: v for k, v in ens_w.items() if k in all_model_preds}
        tw = sum(avail_w.values())
        ens_pred = np.zeros(forecast_hours)
        for mname, w in avail_w.items():
            ens_pred += (w / tw) * all_model_preds[mname]
        fig_cmp.add_trace(go.Scatter(x=pred_dates, y=ens_pred, mode='lines', name='Ensemble (Best)',
            line=dict(color='#f59e0b', width=3, dash='dash', shape='spline')))

        if actual_temps is not None:
            fig_cmp.add_trace(go.Scatter(x=pred_dates, y=actual_temps, mode='lines', name='Actual',
                    line=dict(color='#ffffff', width=2, dash='dot')))
        fig_cmp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            hovermode="x unified")
        fig_cmp.update_xaxes(showgrid=False, zeroline=False)
        fig_cmp.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ====== HORIZONTAL FORECAST ROW ======
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    n_cols = 7
    cols = st.columns(n_cols)
    
    # We step through the forecast hours to show 7 distinct points
    step = max(1, forecast_hours // n_cols)
    
    for i in range(n_cols):
        idx = min(i * step, len(preds) - 1)
        p_val = preds[idx]
        p_dt = pred_dates[idx]
        
        # simplistic icon selection based on temp value (to mimic variation)
        if p_val > 25: c_icon = '☀️'
        elif p_val > 15: c_icon = '⛅'
        elif p_val > 5: c_icon = '☁️'
        elif p_val < 0: c_icon = '❄️'
        else: c_icon = '🌧️'
        
        with cols[i]:
            st.markdown(f"""
            <div class='forecast-card'>
                <div class='forecast-day'>{p_dt.strftime('%a %H:%M')}</div>
                <div class='forecast-icon'>{c_icon}</div>
                <div class='forecast-temp'><span>{int(p_val)}°</span></div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # error metrics (kept simple at the bottom)
    if actual_temps is not None and len(actual_temps) >= forecast_hours:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(actual_temps[:forecast_hours], preds)
        rmse = np.sqrt(mean_squared_error(actual_temps[:forecast_hours], preds))
        r2 = r2_score(actual_temps[:forecast_hours], preds)

        st.markdown(f"<div style='color:#9aa0a6; font-size:0.85rem;'>Metrics for {selected_model}: MAE: {mae:.2f}°C | RMSE: {rmse:.2f}°C | R²: {r2:.3f}</div>", unsafe_allow_html=True)
    elif actual_temps is None:
        st.markdown(f"<div style='color:#9aa0a6; font-size:0.85rem;'>Live forecast for {selected_model} (Metrics unavailable until future occurs)</div>", unsafe_allow_html=True)


def page_about():
    """Render the About page with project overview and model architecture details."""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <div style='display: inline-block; padding: 4px 12px; background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 20px; color: #00e5ff; font-weight: 600; font-size: 0.85rem; margin-bottom: 1rem; letter-spacing: 1px;'>
                📍 ABOUT
            </div>
            <h1 style='color: #ffffff; font-weight: 800; margin-bottom: 0;'>About This Project</h1>
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
    """Render the Results page showing training and evaluation summaries."""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <div style='display: inline-block; padding: 4px 12px; background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 20px; color: #a855f7; font-weight: 600; font-size: 0.85rem; margin-bottom: 1rem; letter-spacing: 1px;'>
                📈 PERFORMANCE
            </div>
            <h1 style='color: #ffffff; font-weight: 800; margin-bottom: 0;'>Training Results</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    m2_path = os.path.join("outputs", "results", "milestone2_results.json")
    m3_path = os.path.join("outputs", "results", "milestone3_results.json")

    # ── Milestone 2 ──
    if os.path.exists(m2_path):
        with open(m2_path) as f:
            m2_results = json.load(f)

        st.subheader("Milestone 2 — Training Summary")

        # M2 data may be flat (model-name keys at top level) or nested under 'models'
        model_data = m2_results.get('models', None)
        if model_data is None:
            # flat structure: every key except meta keys is a model
            model_data = {k: v for k, v in m2_results.items() if isinstance(v, dict)}

        if model_data:
            rows = []
            for name, info in model_data.items():
                rows.append({
                    'Model': name,
                    'Type': info.get('type', '—'),
                    'Parameters': f"{info.get('params', 0):,}" if info.get('params') else '—',
                    'Epochs': info.get('epochs', '—'),
                    'Best Val Loss': f"{info.get('best_val_loss', info.get('val_mse', 0)):.6f}",
                    'Time': f"{info.get('time_seconds', info.get('time', 0)):.0f}s" if info.get('time_seconds', info.get('time')) else '—'
                })
            st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

            # Interactive Plotly chart — Val Loss comparison
            names = [r['Model'] for r in rows]
            losses = [float(r['Best Val Loss']) for r in rows]
            colors_m2 = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71', '#f39c12']
            fig_m2 = go.Figure(data=[go.Bar(
                x=names, y=losses,
                marker_color=colors_m2[:len(names)],
                text=[f"{v:.4f}" for v in losses],
                textposition='outside',
                textfont=dict(color='white', size=12)
            )])
            fig_m2.update_layout(
                title=dict(text='Validation Loss Comparison', font=dict(color='white', size=16)),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                yaxis_title='Best Val MSE',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
            )
            st.plotly_chart(fig_m2, use_container_width=True)

    st.markdown("---")

    # ── Milestone 3 ──
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
            st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

            # ── Dynamically find the best model by lowest MAE ──
            best_name = min(m3_results['single_step'].items(), key=lambda x: x[1]['MAE'])[0]
            best_mae = m3_results['single_step'][best_name]['MAE']
            best_r2 = m3_results['single_step'][best_name]['R2']

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
                        border: 1px solid rgba(16,185,129,0.4); border-radius: 12px;
                        padding: 1.2rem 1.5rem; margin: 1rem 0;'>
                <div style='display: flex; align-items: center; gap: 12px;'>
                    <span style='font-size: 2rem;'>🏆</span>
                    <div>
                        <div style='color: #10b981; font-weight: 700; font-size: 1.1rem;'>Best Model: {best_name}</div>
                        <div style='color: #6ee7b7; font-size: 0.9rem;'>MAE: {best_mae:.2f}°C  •  R²: {best_r2:.3f}  •  Lowest error across all architectures</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Interactive MAE / RMSE grouped bar chart ──
            model_names = list(m3_results['single_step'].keys())
            mae_vals = [m3_results['single_step'][n]['MAE'] for n in model_names]
            rmse_vals = [m3_results['single_step'][n]['RMSE'] for n in model_names]

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='MAE (°C)', x=model_names, y=mae_vals,
                                       marker_color='#f59e0b',
                                       text=[f"{v:.2f}" for v in mae_vals],
                                       textposition='outside', textfont=dict(color='#fbbf24', size=11)))
            fig_comp.add_trace(go.Bar(name='RMSE (°C)', x=model_names, y=rmse_vals,
                                       marker_color='#ef4444',
                                       text=[f"{v:.2f}" for v in rmse_vals],
                                       textposition='outside', textfont=dict(color='#fca5a5', size=11)))
            fig_comp.update_layout(
                barmode='group',
                title=dict(text='MAE vs RMSE — All Models', font=dict(color='white', size=16)),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                legend=dict(font=dict(color='white')),
                yaxis_title='Error (°C)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # ── R² comparison (horizontal bar) ──
            r2_vals = [m3_results['single_step'][n]['R2'] for n in model_names]
            colors_r2 = ['#3b82f6', '#a855f7', '#f97316', '#10b981', '#eab308']
            fig_r2 = go.Figure(data=[go.Bar(
                y=model_names, x=r2_vals, orientation='h',
                marker_color=colors_r2[:len(model_names)],
                text=[f"{v:.4f}" for v in r2_vals],
                textposition='inside', textfont=dict(color='white', size=12)
            )])
            fig_r2.update_layout(
                title=dict(text='R² Score Comparison (closer to 1.0 = better)', font=dict(color='white', size=16)),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                xaxis=dict(range=[0.98, 1.0], gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        # ── Ablation Depth Results ──
        if 'ablation_depth' in m3_results:
            st.markdown("---")
            st.subheader("TCN Depth Ablation Study")
            abl = m3_results['ablation_depth']
            depths = sorted(abl.keys(), key=int)
            abl_mae = [abl[d]['MAE'] for d in depths]
            abl_rf = [abl[d]['rf'] for d in depths]

            fig_abl = go.Figure()
            fig_abl.add_trace(go.Scatter(
                x=[f"{d} blocks" for d in depths], y=abl_mae,
                mode='lines+markers+text', name='MAE',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=12, color='#f59e0b', line=dict(color='white', width=2)),
                text=[f"{v:.2f}°C" for v in abl_mae],
                textposition='top center', textfont=dict(color='#fbbf24', size=11)
            ))
            fig_abl.update_layout(
                title=dict(text='MAE by TCN Depth (Receptive Field Expansion)', font=dict(color='white', size=16)),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                yaxis_title='Test MAE (°C)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
            )
            st.plotly_chart(fig_abl, use_container_width=True)

            # RF info cards
            rf_cols = st.columns(len(depths))
            for i, d in enumerate(depths):
                with rf_cols[i]:
                    st.markdown(f"""
                    <div style='background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.25);
                                border-radius: 10px; padding: 0.8rem; text-align: center;'>
                        <div style='color: #a855f7; font-weight: 700; font-size: 1.3rem;'>{d}</div>
                        <div style='color: #94a3b8; font-size: 0.75rem;'>blocks</div>
                        <div style='color: #e2e8f0; font-weight: 600; margin-top: 4px;'>RF: {abl_rf[i]}h</div>
                    </div>
                    """, unsafe_allow_html=True)

    if not os.path.exists(m2_path) and not os.path.exists(m3_path):
        st.info("No saved results found. Run the M2 and M3 notebooks first to generate results.")


# ==================== MAIN ====================
def main():
    """Entry point: configure sidebar navigation and route to pages."""
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
