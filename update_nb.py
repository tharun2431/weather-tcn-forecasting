import nbformat
import sys

nb_path = "notebooks/Tharun_ML3.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        src = "".join(cell.source)
        if "## 14. Web Deployment" in src:
            cell.source = """## 14. Edge Web Deployment & Dashboard

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
```"""
        
        elif "### 🚀 Live Dashboard Deployment" in src:
            cell.source = """### 🚀 Live Deployments

The project has been fully packaged and is available live in the following environments:

1. **Edge PWA (Primary Application):**  
👉 **[https://tharun2431.github.io/weather-tcn-forecasting/](https://tharun2431.github.io/weather-tcn-forecasting/)**  
*(Fast, offline-capable, runs ONNX models directly in your browser)*

2. **Analytics Dashboard (Hugging Face):**  
👉 **[https://huggingface.co/spaces/tharun2431/weather-tcn-forecasting](https://huggingface.co/spaces/tharun2431/weather-tcn-forecasting)**  
*(For detailed MLOps analysis and model comparisons)*"""

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Updated notebooks/Tharun_ML3.ipynb successfully!")
