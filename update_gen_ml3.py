import os

with open("gen_ml3.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace Section 14
old_sect14 = '''md("""
## 14. Web Deployment

The trained models from Milestone 2 are deployed as an interactive **Streamlit dashboard**.
The full source code for the dashboard (`app.py`) is embedded in the cell below so this submission is fully self-contained!

The web app allows users to:
1. **Select a model** (LSTM, TCN, or TCN-LSTM) from the sidebar
2. **Adjust the forecast horizon** (1-48 hours) via a slider
3. **Move through the test set** to forecast from any date
4. **Fetch Live Forecast Data** from Open-Meteo API
5. **Compare all models** side by side with live performance metrics

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.
"""),'''

new_sect14 = '''md("""
## 14. Edge Web Deployment & Dashboard

As the final step of Milestone 3, the models are deployed in two distinct environments to showcase both edge-inference and cloud-based analytics:

1. **Progressive Web App (PWA) with ONNX WebAssembly:**
   The primary deployment is a client-side Progressive Web App (PWA). The trained PyTorch model has been serialized to `.onnx` and is executed directly in the user's browser using `onnxruntime-web` (WebAssembly). This ensures zero-latency predictions, offline capability, and enhanced data privacy, representing a modern MLOps edge-deployment architecture.

2. **Cloud Analytics Dashboard (Hugging Face Spaces):**
   For detailed interactive analysis and model comparison, the legacy Streamlit application has been containerized and deployed to Hugging Face Spaces.

The full source code for the backend Streamlit dashboard (`app.py`) is embedded in the cell below so this submission is fully self-contained!
To run the dashboard locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
"""),'''

text = text.replace(old_sect14, new_sect14)

# Replace End block
old_end = '''md("""
### 🚀 Live Dashboard Deployment

The final models have been successfully packaged and deployed to the Streamlit Community Cloud.

**You can access the live interactive forecasting dashboard here:**
👉 **[https://weather-tcn-forecasting.streamlit.app](https://weather-tcn-forecasting.streamlit.app)**
""")'''

new_end = '''md("""
### 🚀 Live Deployments

The project has been fully packaged and is available live in the following environments:

1. **Edge PWA (Primary Application):**  
👉 **[https://tharun2431.github.io/weather-tcn-forecasting/](https://tharun2431.github.io/weather-tcn-forecasting/)**  
*(Fast, offline-capable, runs ONNX models directly in your browser)*

2. **Analytics Dashboard (Hugging Face):**  
👉 **[https://huggingface.co/spaces/tharun2431/weather-tcn-forecasting](https://huggingface.co/spaces/tharun2431/weather-tcn-forecasting)**  
*(For detailed MLOps analysis and model comparisons)*
""")'''

text = text.replace(old_end, new_end)

with open("gen_ml3.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Updated gen_ml3.py!")
