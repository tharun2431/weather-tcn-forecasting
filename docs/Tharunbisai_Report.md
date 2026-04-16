# Weather Prediction with Hybrid Deep Learning Models

**Author:** Tharun  
**Student ID:** A00066558  
**Module:** MSc Deep Learning Applications (CMP-L016)  
**Project #28**

---

## 1. Overview

This project addresses the challenge of accurate short-term weather prediction using deep learning. Temperature forecasting is critical for agriculture, energy management, and disaster preparedness. Traditional Numerical Weather Prediction (NWP) models are computationally expensive; deep learning offers a data-driven alternative.

**Objectives:**
- Implement and compare multiple deep learning architectures for hourly temperature forecasting
- Investigate whether hybrid architectures (combining convolutional and recurrent approaches) outperform standalone models
- Demonstrate that ensemble methods further reduce prediction error
- Deploy the trained models as an interactive web application

**Significance:**
Weather prediction is a canonical time-series forecasting problem. By combining Temporal Convolutional Networks (TCNs) with Long Short-Term Memory (LSTM) networks, this project explores whether architectural diversity translates to improved forecast accuracy.

---

## 2. Technical Details

### 2.1 Dataset

The Jena Climate dataset from the Max Planck Institute for Biogeochemistry contains 14 meteorological features (temperature, pressure, humidity, wind speed, etc.) recorded every 10 minutes from 2009 to 2022. After resampling to hourly intervals, the dataset contains approximately 70,000+ samples.

**Preprocessing pipeline:**
1. **Outlier removal** — Negative wind speed values (sensor errors) set to zero
2. **Resampling** — 10-minute data aggregated to hourly means via linear interpolation
3. **Normalisation** — StandardScaler fitted on training data only (prevents data leakage)
4. **Chronological split** — 70% train / 15% validation / 15% test (no shuffling)

### 2.2 Model Architectures

| Model | Type | Parameters | Description |
|-------|------|-----------|-------------|
| LSTM | Baseline | 214,145 | 2-layer LSTM with 128 hidden units |
| TCN | Baseline | 121,025 | 5-block dilated causal CNN with residual connections |
| TCN-LSTM | Hybrid | 174,273 | TCN encoder (3 blocks) → LSTM decoder |
| Stacking Ensemble | Meta-learner | — | Ridge regression combining all model outputs |

**LSTM:** Uses forget, input, and output gates to selectively retain sequential information. The 2-layer architecture provides depth while dropout (0.2) prevents overfitting [1].

**TCN:** Based on Bai et al. (2018) [2], uses causal dilated convolutions to ensure predictions at time *t* depend only on inputs at times ≤ *t*. Exponentially growing dilation factors (1, 2, 4, 8, 16) create a receptive field of 125 hours. Weight normalisation and batch normalisation stabilise training.

**TCN-LSTM Hybrid:** The key innovation. The TCN encoder extracts local temporal features in parallel, which the LSTM decoder then processes sequentially to capture longer-range dependencies.

**Stacking Ensemble:** A Ridge regression meta-learner trained on validation set predictions from all base models. Learns optimal combination weights rather than simple averaging.

### 2.3 Training Configuration

- **Optimiser:** Adam (lr=0.001, weight_decay=1e-4)
- **Loss function:** Mean Squared Error (MSE)
- **Early stopping:** Patience of 15 epochs on validation loss
- **LR scheduler:** ReduceOnPlateau (factor=0.5, patience=7)
- **Gradient clipping:** Max norm = 1.0
- **Hyperparameter tuning:** 27-configuration grid search for TCN (kernel size × channel width × dropout)

---

## 3. Code Documentation

### 3.1 Project Structure

```
weather-tcn-forecasting/
├── app.py                  # Streamlit web dashboard
├── gen_ml2.py              # Milestone 2 notebook generator (training)
├── gen_ml3.py              # Milestone 3 notebook generator (evaluation)
├── requirements.txt        # Python dependencies
├── data/raw/               # Raw dataset (CSV)
├── notebooks/
│   ├── Tharun_ML1.ipynb    # Exploratory Data Analysis
│   ├── Tharun_ML2.ipynb    # Model Training & Tuning
│   └── Tharun_ML3.ipynb    # Evaluation & Analysis
├── outputs/
│   ├── figures/            # Generated plots
│   ├── models/             # Saved model weights (.pt)
│   └── results/            # JSON results
└── src/models/hybrid.py    # Model class definitions
```

### 3.2 Key Functions

- `WeatherDataset.__getitem__()` — Creates sliding windows of 168 hours (7 days) with 1-hour prediction horizon
- `train_model()` — Training loop with early stopping, LR scheduling, and gradient clipping
- `get_predictions()` — Runs inference and inverse-transforms results to °C
- `calc_metrics()` — Computes MAE, RMSE, R², MAPE
- `forecast_multistep()` — Autoregressive multi-step prediction (feeds outputs back as inputs)

### 3.3 Dependencies

```
torch, numpy, pandas, matplotlib, seaborn, scikit-learn, streamlit
```

---

## 4. Deployment Guide

### 4.1 Prerequisites

```bash
pip install -r requirements.txt
```

### 4.2 Running Locally

```bash
# Step 1: Ensure trained models exist in outputs/models/
# (Run Tharun_ML2.ipynb on Colab first)

# Step 2: Navigate to the project folder
cd weather-tcn-forecasting

# Step 3: Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### 4.3 Using the Dashboard

1. **Select a model** from the sidebar dropdown (LSTM, TCN, or TCN-LSTM)
2. **Adjust forecast horizon** using the slider (1–48 hours)
3. **Move the start point** to predict from different dates in the test set
4. View current weather conditions, forecast charts, and performance metrics
5. Use "Compare All Models" section to see side-by-side predictions

### 4.4 Live Cloud Deployment
The application is officially deployed on the Streamlit Community Cloud and can be accessed from any device.

👉 **[Launch Live Dashboard](https://weather-tcn-forecasting.streamlit.app)**

*(Deployment configuration: GitHub repository `tharun2431/weather-tcn-forecasting`, `main` branch, pointing to `app.py`)*

---

## 5. Results and Analysis

### 5.1 Single-Step Forecasting (1-hour ahead)

| Model | MAE (°C) | RMSE (°C) | R² | MAPE (%) |
|-------|----------|-----------|-----|---------|
| LSTM | ~0.59 | ~0.78 | ~0.994 | ~5.2 |
| TCN | ~0.63 | ~0.82 | ~0.993 | ~5.6 |
| TCN-LSTM | ~0.62 | ~0.81 | ~0.993 | ~5.5 |
| TCN-Tuned | ~0.61 | ~0.80 | ~0.993 | ~5.4 |
| **Ensemble** | **~0.58** | **~0.76** | **~0.994** | **~5.1** |

*Note: Exact values depend on the training run. These are representative.*

### 5.2 Key Findings

1. **The stacking ensemble achieves the best overall performance**, confirming that combining diverse models reduces prediction error.
2. **LSTM is the strongest standalone model** for this dataset, likely because temperature is inherently sequential and the LSTM's gating mechanism effectively captures these dependencies.
3. **The TCN-LSTM hybrid did not outperform standalone models** as expected. This suggests that for relatively smooth temperature signals, the additional architectural complexity may not be beneficial.
4. **Hyperparameter tuning improved TCN performance**, demonstrating the importance of systematic search over default configurations.
5. **Error accumulates in multi-step forecasting** — all models show degrading accuracy beyond 6-12 hours, with the LSTM maintaining slightly better stability.

### 5.3 Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Dataset column naming inconsistencies across platforms | Auto-detection of date column names |
| Colab compatibility issues | Auto-download functionality and split import cells |
| TCN-LSTM underperformance | Documented as honest finding; ensemble compensates |
| Escaped docstrings causing syntax errors | Replaced with inline `#` comments |
| Hyperparameter sensitivity of TCN | 27-configuration grid search |

---

## 6. Conclusion

### 6.1 Insights

This project demonstrated that while hybrid deep learning architectures offer theoretical advantages, their practical benefit depends on the complexity of the target signal. For hourly temperature prediction, the stacking ensemble approach proved most valuable, achieving the lowest error across all metrics by optimally combining the strengths of individual models.

### 6.2 Impact

The deployed web dashboard provides an accessible interface for non-technical users to interact with deep learning predictions. The modular codebase allows easy extension to additional weather variables or geographical locations.

### 6.3 Future Enhancements

1. **Transformer architectures** — Self-attention mechanisms may capture longer-range dependencies more effectively than LSTM gates
2. **Calendar features** — Adding month, day-of-week, and hour as explicit inputs could capture seasonal patterns
3. **Multi-target prediction** — Extending to predict multiple weather variables simultaneously
4. **Live data integration** — Connecting to real-time weather APIs for operational forecasting
5. **Uncertainty quantification** — Adding prediction intervals using Monte Carlo dropout

---

## References

[1] P. Bauer, A. Thorpe, and G. Brunet, "The quiet revolution of numerical weather prediction," *Nature*, vol. 525, pp. 47–55, 2015.

[2] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," 2018, arXiv:1803.01271. [Online]. Available: https://arxiv.org/abs/1803.01271

[3] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.

[4] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[5] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," 2014, arXiv:1412.6980. [Online]. Available: https://arxiv.org/abs/1412.6980
