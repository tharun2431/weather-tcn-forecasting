"""
Weather Prediction Dashboard - TCN vs LSTM
Flask web application with live weather prediction
"""
from flask import Flask, render_template, jsonify, request
import numpy as np
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

app = Flask(__name__)

# ============================================
# CITIES DATABASE
# ============================================
CITIES = {
    "jena": {"name": "Jena, Germany", "lat": 50.93, "lon": 11.59, "emoji": "🇩🇪"},
    "london": {"name": "London, UK", "lat": 51.51, "lon": -0.13, "emoji": "🇬🇧"},
    "paris": {"name": "Paris, France", "lat": 48.86, "lon": 2.35, "emoji": "🇫🇷"},
    "berlin": {"name": "Berlin, Germany", "lat": 52.52, "lon": 13.41, "emoji": "🇩🇪"},
    "new_york": {"name": "New York, USA", "lat": 40.71, "lon": -74.01, "emoji": "🇺🇸"},
    "tokyo": {"name": "Tokyo, Japan", "lat": 35.69, "lon": 139.69, "emoji": "🇯🇵"},
    "sydney": {"name": "Sydney, Australia", "lat": -33.87, "lon": 151.21, "emoji": "🇦🇺"},
    "mumbai": {"name": "Mumbai, India", "lat": 19.08, "lon": 72.88, "emoji": "🇮🇳"},
    "dubai": {"name": "Dubai, UAE", "lat": 25.20, "lon": 55.27, "emoji": "🇦🇪"},
    "toronto": {"name": "Toronto, Canada", "lat": 43.65, "lon": -79.38, "emoji": "🇨🇦"},
}

# ============================================
# PRE-COMPUTED MODEL RESULTS
# ============================================
RESULTS = {
    "short_term": {
        "description": "Short-term forecasting (1 hour ahead)",
        "horizon": "1 hour",
        "input": "72 hours (3 days)",
        "lstm": {"R2": 99.52, "RMSE": 0.62, "MAE": 0.41, "MAPE": 3.21},
        "tcn":  {"R2": 99.37, "RMSE": 0.71, "MAE": 0.48, "MAPE": 3.85}
    },
    "long_term": {
        "24h": {
            "lstm": {"R2": 90.96, "RMSE": 2.35, "MAE": 1.80, "MAPE": 82.60},
            "tcn":  {"R2": 87.47, "RMSE": 2.77, "MAE": 2.19, "MAPE": 104.78}
        },
        "48h": {
            "lstm": {"R2": 84.45, "RMSE": 3.08, "MAE": 2.39, "MAPE": 128.19},
            "tcn":  {"R2": 80.01, "RMSE": 3.49, "MAE": 2.73, "MAPE": 124.11}
        },
        "72h": {
            "lstm": {"R2": 80.80, "RMSE": 3.42, "MAE": 2.66, "MAPE": 119.21},
            "tcn":  {"R2": 74.86, "RMSE": 3.92, "MAE": 3.07, "MAPE": 127.05}
        }
    }
}

# Generate sample prediction data
np.random.seed(42)
hours = 200
t = np.linspace(0, 4 * np.pi, hours)
actual_temps = 10 + 8 * np.sin(t) + 3 * np.sin(2.5 * t) + np.random.normal(0, 0.5, hours)
lstm_preds = actual_temps + np.random.normal(0, 0.7, hours)
tcn_preds = actual_temps + np.random.normal(0, 0.8, hours)

PREDICTION_DATA = {
    "actual": actual_temps.tolist(),
    "lstm": lstm_preds.tolist(),
    "tcn": tcn_preds.tolist(),
    "hours": list(range(hours))
}


# ============================================
# LIVE PREDICTION FUNCTIONS
# ============================================
def fetch_weather_data(lat, lon):
    """Fetch recent weather from Open-Meteo API (free, no key needed)"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
        f"wind_speed_10m,dew_point_2m"
        f"&past_days=7&forecast_days=3"
        f"&timezone=auto"
    )
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'WeatherTCN/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None


def simulate_predictions(actual_temps, model_type="tcn"):
    """
    Simulate model predictions based on actual temperature data.
    In production, this would load actual PyTorch models and run inference.
    Here we simulate realistic predictions based on model accuracy metrics.
    """
    np.random.seed(hash(model_type) % 2**31)
    temps = np.array(actual_temps, dtype=float)
    
    if model_type == "tcn":
        # TCN: slightly more noise but better at trends
        noise_std = 0.8
        trend_weight = 0.95
    else:
        # LSTM: lower noise, good sequential prediction
        noise_std = 0.6
        trend_weight = 0.97
    
    predictions = []
    for i in range(24):
        if i < len(temps):
            base = temps[-(24-i)] if (24-i) <= len(temps) else temps[-1]
        else:
            base = predictions[-1] if predictions else temps[-1]
        
        # Add realistic noise that increases with horizon
        noise = np.random.normal(0, noise_std * (1 + i * 0.05))
        pred = base * trend_weight + temps[-1] * (1 - trend_weight) + noise
        predictions.append(round(pred, 2))
    
    # Generate confidence intervals (wider for longer horizons)
    upper = [round(p + 1.5 * (1 + i * 0.08), 2) for i, p in enumerate(predictions)]
    lower = [round(p - 1.5 * (1 + i * 0.08), 2) for i, p in enumerate(predictions)]
    
    return predictions, upper, lower


# ============================================
# ROUTES
# ============================================
@app.route('/')
def dashboard():
    return render_template('index.html', cities=CITIES)


@app.route('/api/results')
def get_results():
    return jsonify(RESULTS)


@app.route('/api/predictions')
def get_predictions():
    return jsonify(PREDICTION_DATA)


@app.route('/api/cities')
def get_cities():
    return jsonify(CITIES)


@app.route('/api/live_predict')
def live_predict():
    """Live prediction endpoint"""
    city_key = request.args.get('city', 'jena')
    
    if city_key not in CITIES:
        return jsonify({"error": "City not found"}), 404
    
    city = CITIES[city_key]
    weather_data = fetch_weather_data(city['lat'], city['lon'])
    
    if not weather_data or 'hourly' not in weather_data:
        return jsonify({"error": "Could not fetch weather data"}), 500
    
    hourly = weather_data['hourly']
    temps = hourly['temperature_2m']
    times = hourly['time']
    
    # Split into past (7 days) and actual future (3 days)
    now_idx = min(168, len(temps))  # 7 days * 24 hours
    past_temps = temps[:now_idx]
    future_actual = temps[now_idx:now_idx+24]
    past_times = times[:now_idx]
    future_times = times[now_idx:now_idx+24]
    
    # Generate predictions from both models
    tcn_preds, tcn_upper, tcn_lower = simulate_predictions(past_temps, "tcn")
    lstm_preds, lstm_upper, lstm_lower = simulate_predictions(past_temps, "lstm")
    
    # Current conditions
    current_temp = past_temps[-1] if past_temps else 0
    current_humidity = hourly['relative_humidity_2m'][now_idx-1] if now_idx > 0 else 0
    current_pressure = hourly['pressure_msl'][now_idx-1] if now_idx > 0 else 0
    current_wind = hourly['wind_speed_10m'][now_idx-1] if now_idx > 0 else 0
    
    return jsonify({
        "city": city,
        "current": {
            "temperature": current_temp,
            "humidity": current_humidity,
            "pressure": current_pressure,
            "wind_speed": current_wind,
            "time": past_times[-1] if past_times else ""
        },
        "past": {
            "temps": past_temps[-48:],  # Last 48 hours
            "times": past_times[-48:]
        },
        "predictions": {
            "times": future_times[:24],
            "tcn": {
                "values": tcn_preds,
                "upper": tcn_upper,
                "lower": tcn_lower
            },
            "lstm": {
                "values": lstm_preds,
                "upper": lstm_upper,
                "lower": lstm_lower
            },
            "actual": future_actual[:24]
        }
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Weather Prediction Dashboard")
    print("  Open: http://localhost:8080")
    print("="*50 + "\n")
    app.run(debug=True, port=8080)
