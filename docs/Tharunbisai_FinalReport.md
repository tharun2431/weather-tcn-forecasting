# Weather Prediction Using Hybrid Deep Learning Models

**Author:** Tharun Bisai  
**Student ID:** [Your Student ID]  
**Module:** MSc Deep Learning Applications (CMP-L016)  
**Project #28:** Weather Prediction with Temporal Convolutional Networks

---

## Abstract

This paper presents a hybrid deep learning approach for short-term weather prediction using the Jena Climate dataset. We compare three neural architectures — LSTM, Temporal Convolutional Network (TCN), and a novel TCN-LSTM hybrid — alongside a stacking ensemble that combines their predictions. Experimental results on 13 years of hourly weather data demonstrate that the ensemble approach achieves the lowest prediction error (MSE ~0.0043), confirming that architectural diversity improves forecasting accuracy. The trained models are deployed as an interactive web dashboard using Streamlit.

---

## 1. Background and Problem Definition

Weather prediction is a fundamental challenge with direct impact on agriculture, energy management, transportation, and disaster preparedness. Traditional approaches rely on Numerical Weather Prediction (NWP) models, which solve complex atmospheric differential equations and require massive computational resources [1].

Deep learning offers a data-driven alternative: given sufficient historical data, neural networks can learn temporal patterns without explicit physical modelling. This is particularly appealing for localised, short-term forecasting where station-level data is abundant but NWP resolution is insufficient.

**The Problem:** Can a hybrid deep learning architecture that combines convolutional and recurrent components produce more accurate short-term temperature forecasts than either approach alone?

**Hypothesis:** A TCN-LSTM hybrid should capture both local temporal patterns (via dilated convolutions) and longer-range sequential dependencies (via LSTM gates), outperforming standalone architectures.

---

## 2. Proposed Deep Learning Solution

We investigate four complementary approaches:

| # | Model | Architecture | Role |
|---|-------|-------------|------|
| 1 | **LSTM** | 2-layer, 128 hidden units | Recurrent baseline |
| 2 | **TCN** | 5-block dilated causal CNN | Convolutional baseline |
| 3 | **TCN-LSTM** | TCN encoder → LSTM decoder | Hybrid (key innovation) |
| 4 | **Stacking Ensemble** | Ridge regression meta-learner | Optimal model combination |

**Why these models?**

- **LSTM** [2] is the established standard for sequence modelling. Its forget, input, and output gates selectively retain information across time steps, making it naturally suited to weather's sequential nature.
- **TCN** [3] uses causal dilated convolutions to process entire sequences in parallel. Exponentially growing dilation factors create a receptive field of 125 hours (5+ days) while maintaining computational efficiency.
- **TCN-LSTM Hybrid** combines both: the TCN encoder extracts local temporal features, which the LSTM decoder then processes to capture longer-range dependencies. This is our primary contribution.
- **Stacking Ensemble** [4] trains a meta-learner (Ridge regression) on the validation-set predictions of all base models, learning optimal combination weights rather than simple averaging.

---

## 3. Data Selection

We selected the **Jena Climate Dataset** from the Max Planck Institute for Biogeochemistry for several reasons:

- **Size:** ~420,000 records (10-minute intervals from 2009–2022), providing sufficient data for deep learning
- **Features:** 14 meteorological variables (temperature, pressure, humidity, wind speed/direction, etc.) — multivariate input supports richer models
- **Quality:** Collected from a professional weather station with consistent sensor calibration
- **Relevance:** Temperature prediction is a canonical time-series forecasting problem with clear evaluation metrics
- **Accessibility:** Openly available, enabling reproducibility

**Source:** [Max Planck Institute — Jena Weather Station](https://www.bgc-jena.mpg.de/wetter/)

---

## 4. Preparing the Data

### 4.1 Missing Values
The dataset was checked for null values. A small number of gaps (caused by sensor downtime) were identified and filled using linear interpolation, followed by forward-fill and back-fill for edge cases.

### 4.2 Outlier Handling
- **Wind speed:** Negative values (physically impossible) were detected and replaced with zero — these represent sensor errors
- **All features:** An IQR (Interquartile Range) analysis was conducted across all 14 features. While statistical outliers were identified (particularly in wind gusts and dewpoint), these were deliberately retained as they represent genuine weather extremes (storms, cold snaps, heatwaves) that the model should learn to predict

### 4.3 Resampling
The original 10-minute resolution was resampled to hourly means. This reduces noise, computational cost, and memory requirements while preserving the temporal patterns relevant for forecasting (hourly resolution is standard in operational weather prediction).

### 4.4 Normalisation
StandardScaler (zero mean, unit variance) was applied, fitted exclusively on the training set to prevent data leakage. StandardScaler was chosen over MinMaxScaler because weather variables are unbounded — a heatwave could produce values outside any historical min/max range.

### 4.5 Data Splitting
A chronological 70/15/15 split was used (train/validation/test). No shuffling was applied because shuffling time-series data causes information leakage — future values would inform past predictions, producing unrealistically optimistic results.

### 4.6 Windowing
A sliding window of 168 hours (7 days) was used as input, predicting 1 hour ahead. The 7-day window captures weekly weather patterns (e.g., weather fronts typically take 3–7 days to pass through mid-latitude locations like Jena).

---

## 5. Defining the Deep-Learning Model

### 5.1 LSTM (214,145 parameters)
```
Input (168, 14) → LSTM(128, 2 layers, dropout=0.2) → FC(128→64) → ReLU → Dropout → FC(64→1)
```
The 2-layer LSTM processes the 7-day input sequence step by step, maintaining a hidden state that encodes the relevant temporal context. The final hidden state is passed through a fully-connected head for prediction.

### 5.2 TCN (121,025 parameters)
```
Input (168, 14) → [CausalConv → BatchNorm → ReLU → Dropout] × 5 blocks → FC(64→1)
```
Each block uses dilated causal convolutions with dilation factors [1, 2, 4, 8, 16], creating a receptive field of 125 hours. Weight normalisation stabilises training. Residual connections with 1×1 convolutions handle channel dimension changes.

### 5.3 TCN-LSTM Hybrid (174,273 parameters)
```
Input (168, 14) → TCN Encoder (3 blocks) → LSTM(128, 1 layer) → FC(64→1)
```
The TCN encoder extracts local patterns (dilated conv features), and the LSTM decoder processes these features sequentially to capture longer-range dependencies. This architecture is the project's primary innovation.

### 5.4 Stacking Ensemble
```
LSTM Predictions ⊕ TCN Predictions ⊕ TCN-LSTM Predictions → RidgeCV → Final Prediction
```
Ridge regression with cross-validated regularisation (α ∈ {0.01, 0.1, 1.0, 10.0}) learns the optimal linear combination of base model predictions from the validation set.

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden size | 128 | Balance between capacity and overfitting |
| Dropout | 0.2 | Standard regularisation for medium-sized models |
| Kernel size (TCN) | 3 | Captures local 3-hour patterns per layer |
| Seq length | 168 | 7 days covers weekly weather cycles |
| Batch size | 64 | Fits GPU memory while providing gradient stability |

---

## 6. Training and Fine-Tuning

### 6.1 Training Configuration
- **Optimiser:** Adam (lr=0.001, weight_decay=1e-4) — adapts per-parameter learning rates
- **Loss function:** MSE — penalises large errors more than MAE, critical for weather extremes. Huber Loss was considered but MSE produced more stable convergence
- **Early stopping:** Patience of 15 epochs on validation loss
- **LR scheduler:** ReduceLROnPlateau (factor=0.5, patience=7)
- **Gradient clipping:** Max norm = 1.0 (prevents exploding gradients in LSTM)

### 6.2 Hyperparameter Tuning
A 27-configuration grid search was conducted for the TCN:
- **Kernel sizes:** {3, 5, 7}
- **Channel widths:** {[32]×4, [64]×5, [64,64,128,128]}
- **Dropout rates:** {0.1, 0.2, 0.3}

### 6.3 Training Results

| Model | Params | Epochs | Best Val MSE | Time |
|-------|--------|--------|-------------|------|
| LSTM | 214,145 | ~84 | ~0.0044 | ~767s |
| TCN | 121,025 | ~24 | ~0.0051 | ~222s |
| TCN-LSTM | 174,273 | ~24 | ~0.0049 | ~200s |
| TCN-Tuned | 26,529 | ~27 | ~0.0048 | ~217s |
| **Ensemble** | — | — | **~0.0043** | — |

### 6.4 Challenges
- **GPU memory:** The TCN-LSTM hybrid required careful batch sizing to avoid OOM errors
- **Convergence speed:** TCN converged much faster than LSTM (24 vs 84 epochs) but to a slightly worse optimum
- **Hyperparameter sensitivity:** TCN performance varied significantly across configurations, justifying the grid search

---

## 7. Testing with New Data

The models were evaluated on the held-out **test set** (15% of data, ~10,000+ hourly samples) that was never seen during training or hyperparameter tuning.

### 7.1 Single-Step Metrics (1-hour ahead)

| Model | MAE (°C) | RMSE (°C) | R² | MAPE (%) |
|-------|----------|-----------|-----|---------|
| LSTM | ~0.59 | ~0.78 | ~0.994 | ~5.2 |
| TCN | ~0.63 | ~0.82 | ~0.993 | ~5.6 |
| TCN-LSTM | ~0.62 | ~0.81 | ~0.993 | ~5.5 |
| **Ensemble** | **~0.58** | **~0.76** | **~0.994** | **~5.1** |

### 7.2 Multi-Step Forecasting
Autoregressive 24-hour forecasts showed that error accumulates with horizon length, with all models maintaining reasonable accuracy up to ~6 hours but degrading beyond 12 hours.

### 7.3 Residual Analysis
Residual distributions were approximately normal and centred near zero for all models, with no systematic bias. Seasonal analysis revealed higher errors during spring/autumn temperature transitions.

---

## 8. Deploying the Model

The trained models are deployed as an interactive web dashboard using **Streamlit**.

### 8.1 Features
- **Model selection** — choose between LSTM, TCN, or TCN-LSTM
- **Adjustable forecast horizon** — 1 to 48 hours ahead
- **Interactive time slider** — select any point in the test set
- **Real-time comparison** — all models plotted simultaneously
- **Performance metrics** — MAE, RMSE, R² displayed live
- **About page** — model architecture descriptions and dataset information

### 8.2 Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The dashboard opens at `http://localhost:8501`.

### 8.3 Technology Stack
Python 3.10+ • PyTorch • Streamlit • Plotly • scikit-learn

---

## 9. Results and Analysis

### 9.1 Key Findings
1. **The stacking ensemble achieves the best overall performance**, confirming that combining diverse model predictions reduces error
2. **LSTM is the strongest standalone model** — its gating mechanism effectively captures temperature's sequential nature
3. **The TCN-LSTM hybrid did not surpass the standalone LSTM**, suggesting that for relatively smooth temperature signals, the additional architectural complexity introduces more parameters without proportional benefit
4. **TCN trains 3× faster** than LSTM (24 vs 84 epochs) while achieving competitive accuracy (within 0.04°C MAE)
5. **Hyperparameter tuning reduced TCN error by ~8%**, demonstrating the importance of systematic search

### 9.2 Ablation Study
TCN depth analysis (2–5 blocks) showed that 4–5 blocks provides the best balance between receptive field size and overfitting risk.

### 9.3 Limitations
- Single-point prediction (no uncertainty quantification)
- Single target variable (temperature only)
- Autoregressive multi-step forecast degrades beyond 12 hours
- No external features (satellite imagery, NWP forecasts)

---

## 10. Conclusion

### 10.1 Insights
This project demonstrated that ensemble methods consistently outperform individual architectures for weather time-series forecasting. While the TCN-LSTM hybrid did not achieve the expected improvement over standalone models, the stacking ensemble validated the hypothesis that architectural diversity reduces prediction error.

### 10.2 Impact
The deployed Streamlit dashboard provides an accessible, interactive interface for exploring model predictions, making deep learning weather forecasting tangible for non-technical users.

### 10.3 Future Enhancements
1. **Transformer architectures** — self-attention may capture multi-scale temporal dependencies more effectively
2. **Probabilistic forecasting** — adding prediction intervals via Monte Carlo dropout or quantile regression
3. **Multi-variate prediction** — forecasting temperature, humidity, and wind simultaneously
4. **Real-time data integration** — connecting to live weather APIs for operational forecasting
5. **Larger ensemble diversity** — adding GRU, attention-based models, and simple baselines

---

## 11. References

[1] P. Bauer, A. Thorpe, and G. Brunet, "The quiet revolution of numerical weather prediction," *Nature*, vol. 525, pp. 47–55, 2015.

[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[3] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.

[4] D. H. Wolpert, "Stacked Generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.

[5] A. van den Oord et al., "WaveNet: A Generative Model for Raw Audio," *arXiv:1609.03499*, 2016.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proc. IEEE CVPR*, pp. 770–778, 2016.

[7] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *Proc. ICLR*, 2015.

[8] K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," *arXiv:1406.1078*, 2014.

[9] Y. Wu and H. He, "A Hybrid Deep Learning Model for Temperature Prediction," *IEEE Access*, vol. 8, pp. 207886–207896, 2020.

[10] A. Graves, "Generating Sequences with Recurrent Neural Networks," *arXiv:1308.0850*, 2013.
