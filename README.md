# 🌤️ Weather Prediction with Hybrid Deep Learning Models

**MSc Deep Learning Applications (CMP-L016) — Project #28**  
**Author:** Tharun Bisai

---

## Overview

This project investigates hybrid deep learning architectures for short-term temperature forecasting using the Jena Climate dataset (2009–2022). We compare LSTM, TCN, TCN-LSTM hybrid, and stacking ensemble approaches.

## Models

| Model | Type | Parameters |
|-------|------|-----------|
| LSTM | Recurrent Baseline | 214,145 |
| TCN | Convolutional Baseline | 121,025 |
| TCN-LSTM | Hybrid (Innovation) | 174,273 |
| Ensemble | Stacking Meta-learner | — |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training (Colab)
Upload `notebooks/Tharun_ML2.ipynb` to Google Colab and run all cells.

### 3. Run Evaluation
Upload `notebooks/Tharun_ML3.ipynb` to Colab (after M2 finishes).

### 4. Launch Dashboard

**🌍 Live Cloud Deployment:**  
👉 **[https://huggingface.co/spaces/Tharunp2431/weather-forecasting)**

*To run legally/locally:*
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                     # Streamlit web dashboard
├── requirements.txt           # Python dependencies
├── notebooks/
│   ├── Tharun_ML2.ipynb       # Model training
│   └── Tharun_ML3.ipynb       # Evaluation & analysis
├── docs/
│   └── Tharunbisai_FinalReport.md  # IEEE report
├── data/raw/                  # Dataset (not tracked)
├── outputs/
│   ├── figures/               # Generated plots
│   ├── models/                # Saved weights (.pt)
│   └── results/               # JSON results
└── src/models/hybrid.py       # Model class definitions
```

## Dataset

**Jena Climate Dataset** — Max Planck Institute for Biogeochemistry  
14 meteorological features, hourly resolution, 2009–2016

## Results

- **Best model:** Stacking Ensemble (MSE ~0.0043)
- **Best standalone:** LSTM (MSE ~0.0044)
- **Fastest training:** TCN (24 epochs vs 84 for LSTM)

## Tech Stack

Python 3.10+ • PyTorch • Streamlit • Plotly • scikit-learn

## License

Academic use only — MSc coursework submission.
