# Weather Prediction with Hybrid Deep Learning Models: Model Evaluation, Deployment, and Documentation

**Tharun Bisai (A00066558)**  
**MSc Data Science**  
**Deep Learning Applications (CMP-L016)**  
**University of Roehampton, London, United Kingdom**  
**Project #28: Weather Prediction with Temporal Convolutional Networks**

---

## Abstract

This report presents the final milestone of a weather prediction project that investigates hybrid deep learning architectures for short-term hourly temperature forecasting. Four model architectures — a Long Short-Term Memory (LSTM) baseline, a Temporal Convolutional Network (TCN), a novel TCN-LSTM hybrid combining convolutional and recurrent paradigms, and a stacking ensemble meta-learner — were trained on the Jena Climate dataset (2009–2016) and rigorously evaluated on held-out test data. To assess real-world generalisation, the models were additionally tested against live weather data from Berlin, Germany, representing a completely unseen geographic domain. The stacking ensemble achieved the best single-step performance on the Jena test set (MAE ≈ 0.37°C, R² ≈ 0.99), while the Berlin transfer test revealed expected domain-shift degradation (MAE ≈ 6.7°C) but confirmed that the networks had learned genuine diurnal temperature physics rather than memorising the training distribution. The trained models were deployed as a production-ready, Google Weather-inspired interactive Streamlit dashboard, accessible live at **https://weather-tcn-forecasting.streamlit.app**, capable of real-time inference via the Open-Meteo API. This report documents the full evaluation pipeline, deployment architecture, critical analysis of results, and directions for future work.

**Keywords**: weather prediction, temporal convolutional network, LSTM, hybrid architecture, ensemble learning, time-series forecasting, web deployment, generalisation

---

## I. INTRODUCTION

### A. Problem Statement and Motivation

Accurate short-term weather prediction is fundamental to modern society, underpinning decision-making in agriculture, energy grid management, transportation logistics, and disaster preparedness [1]. Traditional Numerical Weather Prediction (NWP) models solve complex systems of partial differential equations governing atmospheric dynamics, requiring enormous computational resources and extensive domain expertise to configure [1]. While NWP remains the gold standard for medium-range forecasting (3–10 days), its computational cost and latency make it impractical for rapid, localised short-term predictions at the hourly scale.

Deep learning offers a compelling data-driven alternative. Neural networks can learn complex non-linear mappings directly from historical sensor data without explicit physical modelling, enabling fast inference suitable for real-time applications [2]. Recent advances in sequence modelling — particularly recurrent neural networks and temporal convolutional architectures — have demonstrated competitive performance on time-series forecasting tasks across multiple domains [3][4].

### B. Project Objectives

This project pursued four specific objectives:

1. **Implement and compare** multiple deep learning architectures (LSTM, TCN, and a novel TCN-LSTM hybrid) for hourly temperature forecasting using the Jena Climate dataset.
2. **Investigate** whether hybrid architectures that combine convolutional feature extraction with recurrent sequence modelling outperform standalone models on smooth meteorological signals.
3. **Demonstrate** that ensemble methods (stacking meta-learners) can further reduce prediction error by optimally combining diverse model outputs.
4. **Deploy** the trained models as an interactive, publicly accessible web application capable of live real-time forecasting.

### C. Why Deep Learning is Appropriate

Temperature forecasting from multivariate sensor data is inherently a sequence-to-sequence regression problem with strong temporal autocorrelation. Deep learning is particularly suited because: (a) LSTMs can selectively retain information across hundreds of time steps via gating mechanisms [4]; (b) TCNs can extract hierarchical temporal features through dilated causal convolutions with parallelised training [3]; and (c) modern frameworks (PyTorch, Streamlit) enable seamless transition from research prototyping to production deployment.

---

## II. BACKGROUND AND LITERATURE REVIEW

### A. Classical Weather Forecasting

Numerical Weather Prediction, pioneered by Richardson (1922) and operationalised from the 1950s onward, remains the dominant approach for weather forecasting globally. Bauer et al. [1] describe this as a "quiet revolution" in which NWP accuracy has improved steadily through finer grid resolution and better data assimilation. However, NWP models such as ECMWF's IFS require supercomputing infrastructure and hours of computation per forecast cycle, making them inaccessible for localised, rapid predictions.

### B. Recurrent Neural Networks for Time-Series

Hochreiter and Schmidhuber [4] introduced the LSTM architecture in 1997 to address the vanishing gradient problem in standard RNNs. The LSTM's forget, input, and output gates enable selective information retention across long sequences. Subsequent work demonstrated LSTMs' effectiveness on diverse time-series tasks including natural language processing, speech recognition, and financial forecasting. For weather prediction specifically, Salman et al. [7] showed that LSTM networks could predict temperature with MAE values below 1°C on the Jena Climate dataset, establishing a strong baseline for data-driven weather forecasting.

### C. Temporal Convolutional Networks

Bai et al. [3] conducted a landmark empirical comparison between recurrent and convolutional architectures for sequence modelling, concluding that TCNs — networks built on causal dilated convolutions — outperformed LSTMs on a majority of benchmark tasks while offering significant advantages in training parallelism and memory efficiency. The key innovation of TCNs is the use of exponentially increasing dilation factors (1, 2, 4, 8, ...), which expand the receptive field without increasing computational cost. Weight normalisation and residual connections further stabilise training in deep TCN architectures [3].

### D. Hybrid and Ensemble Approaches

The idea of combining convolutional feature extractors with recurrent decoders has gained traction in recent literature. Hewage et al. [8] demonstrated that CNN-LSTM hybrids can capture both local patterns (via convolutions) and long-range dependencies (via recurrent processing) in environmental time-series data. Separately, Wolpert's stacked generalisation framework [5] provides a principled method for combining multiple models: a meta-learner (e.g., Ridge regression) is trained on the validation-set predictions of diverse base models, learning optimal combination weights that typically outperform any single model or simple averaging.

### E. The Jena Climate Dataset

The Jena Climate dataset, collected by the Max Planck Institute for Biogeochemistry, has become a standard benchmark for time-series forecasting in the deep learning community. It contains 14 meteorological variables (temperature, atmospheric pressure, humidity, wind speed and direction, etc.) recorded at 10-minute intervals from 2009 to 2016, comprising over 420,000 data points. Its widespread use in tutorials, textbooks, and research papers [7][9] makes it an ideal benchmark for comparing novel architectures against established baselines.

---

## III. METHODOLOGY

### A. Data Preprocessing

The raw Jena Climate dataset required several preprocessing steps before model training:

1. **Outlier Removal:** Negative wind speed values (physically impossible; caused by sensor malfunction) were clipped to zero. This affected approximately 0.3% of records in the `wv (m/s)` and `max. wv (m/s)` columns.

2. **Temporal Resampling:** The original 10-minute resolution was resampled to hourly means via `pandas.DataFrame.resample('1h').mean()`, followed by linear interpolation and forward/backward filling to handle any residual NaN values. This reduced the dataset from ~420,000 to ~70,000 samples while preserving the dominant temporal patterns.

3. **Normalisation:** A `StandardScaler` (zero mean, unit variance) was fitted exclusively on the training partition (first 70% of data). This scaler was then applied to the validation and test sets, strictly preventing data leakage — a critical methodological consideration often overlooked in time-series work.

4. **Chronological Split:** The dataset was divided chronologically (no shuffling) into 70% training, 15% validation, and 15% test partitions. This preserves the temporal ordering essential for time-series evaluation and prevents future information from leaking into training.

5. **Sliding Window Construction:** Input sequences of 168 hours (7 days) across all 14 features were constructed as model inputs, with the target being the temperature value 1 hour ahead. This window length was chosen to capture weekly periodicities in weather patterns.

### B. Model Architectures

**LSTM Baseline (214,145 parameters):** A 2-layer LSTM with 128 hidden units per layer, followed by a fully connected head (128 → 64 → 1) with ReLU activations and dropout (p=0.2). The LSTM processes the 168-step input window sequentially, with the final hidden state passed to the prediction head. This architecture provides a strong recurrent baseline [4].

**TCN Baseline (121,025 parameters):** A 5-block Temporal Convolutional Network based on the architecture proposed by Bai et al. [3]. Each block contains two layers of dilated causal convolutions with weight normalisation, ReLU activation, and dropout (p=0.2). Dilation factors grow exponentially as (1, 2, 4, 8, 16), creating a receptive field of 125 time steps. Residual skip connections bypass each block to enable gradient flow in deeper configurations.

**TCN-LSTM Hybrid (174,273 parameters):** The primary architectural contribution of this project. A 3-block TCN encoder extracts local temporal features in parallel, producing a sequence of convolutional feature maps. These are then fed into a 1-layer LSTM decoder (64 hidden units) that processes the TCN outputs sequentially, combining the parallel efficiency of convolutions with the selective memory of recurrent gating. The final LSTM hidden state is passed through a linear prediction head (64 → 1).

**Stacking Ensemble (Meta-learner):** A Ridge regression model (α=1.0) trained on the validation-set predictions of all three base models. Rather than simple averaging, the meta-learner discovers the optimal linear combination weights for each base model's output, typically assigning higher weight to the strongest performer while still benefiting from the diversity of weaker models [5].

### C. Training Configuration

All base models were trained with identical hyperparameters to ensure fair comparison:

- **Optimiser:** Adam (lr=0.001, weight_decay=1e-4) [6]
- **Loss Function:** Mean Squared Error (MSE)
- **Early Stopping:** Patience of 15 epochs on validation loss
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=7)
- **Gradient Clipping:** Maximum norm of 1.0 to prevent exploding gradients
- **Batch Size:** 64
- **Maximum Epochs:** 100

Additionally, a 27-configuration grid search was conducted for the TCN architecture, varying kernel size ∈ {3, 5, 7}, channel width ∈ {32, 64, 128}, and dropout rate ∈ {0.1, 0.2, 0.3}. The optimal configuration (kernel=5, channels=64, dropout=0.2) was designated as **TCN-Tuned**.

---

## IV. EXPERIMENTS AND RESULTS

### A. Single-Step Performance on Jena Test Set

The held-out test set comprises the final 15% of the chronologically ordered Jena dataset, representing entirely unseen future data. All four architectures were evaluated on identical test samples.

| Model | MAE (°C) | RMSE (°C) | R² | MAPE (%) |
|-------|----------|-----------|-----|---------|
| LSTM Baseline | 0.39 | 0.52 | 0.99 | 5.2 |
| TCN Baseline | 0.42 | 0.56 | 0.99 | 5.6 |
| TCN-LSTM Hybrid | 0.40 | 0.54 | 0.99 | 5.5 |
| TCN-Tuned | 0.39 | 0.53 | 0.99 | 5.4 |
| **Stacking Ensemble** | **0.37** | **0.49** | **0.99** | **5.1** |

*Table 1: Single-step (1-hour ahead) test set performance. The stacking ensemble achieves the lowest error across all metrics.*

### B. Generalisation Testing on New Geography (Berlin)

To rigorously test the models' real-world robustness beyond the training distribution, a transfer-learning inference experiment was conducted against live weather data from **Berlin, Germany** — an entirely unseen geographic location with differing urban topology and micro-climate characteristics.

Live data was retrieved via the Open-Meteo API. The same preprocessing pipeline and trained scaler were applied to the Berlin data. As expected, applying models trained exclusively on Jena's local conditions to Berlin's differing environment resulted in significantly higher error (MAE ≈ 6.7°C). However, critically, the models retained the structural shape of diurnal temperature curves (day/night cycles), demonstrating that the neural networks had learned genuine physical patterns of temperature variation rather than memorising the Jena training distribution.

### C. Autoregressive Multi-Step Forecasting

For predictions beyond 1 hour, autoregressive looping was employed: the model's output at time (t+1) is appended to the input feature array to predict (t+2), and so forth. All models maintained high accuracy up to approximately 6 hours, with errors naturally compounding as the horizon expanded:

| Horizon | LSTM MAE (°C) | TCN MAE (°C) | Hybrid MAE (°C) |
|---------|---------------|--------------|-----------------|
| 1 hour | 0.39 | 0.42 | 0.40 |
| 6 hours | 1.2 | 1.4 | 1.3 |
| 12 hours | 2.1 | 2.5 | 2.3 |
| 24 hours | 3.2 | 3.6 | 3.4 |

*Table 2: Autoregressive multi-step MAE. Error grows approximately linearly, with the LSTM maintaining slightly better stability at longer horizons.*

---

## V. DISCUSSION

### A. Why the Ensemble Outperformed Individual Models

The stacking ensemble's superiority (MAE 0.37°C vs. the next-best 0.39°C) confirms Wolpert's theoretical framework [5]: by training a meta-learner on diverse base model predictions, the ensemble exploits the complementary strengths of each architecture. The LSTM captures smooth sequential trends, the TCN detects short-term local patterns, and the hybrid provides an intermediate representation. The Ridge regression meta-learner assigns optimal weights to each, effectively reducing the variance component of the prediction error.

### B. Why the Hybrid Did Not Surpass Standalone Models

A key finding of this project is that the TCN-LSTM hybrid architecture, despite its theoretical appeal, did not significantly outperform the standalone LSTM. This outcome, while initially surprising, has a clear explanation: the Jena temperature signal is relatively smooth and strongly autocorrelated at the hourly scale. The LSTM's gating mechanism is already highly effective at capturing this type of gradual sequential variation. Adding a convolutional encoder introduces additional parameters and architectural complexity without providing access to fundamentally different information — the local patterns extracted by the TCN are redundant when the recurrent decoder already captures the dominant temporal structure.

This finding aligns with Bai et al.'s [3] observation that the relative advantage of convolutional vs. recurrent architectures is task-dependent. For signals with sharp discontinuities or multi-scale patterns (e.g., audio waveforms), TCNs demonstrate clear advantages. For slowly varying physical processes like temperature, recurrent models remain competitive.

### C. Domain Shift in Berlin Transfer Test

The degradation in Berlin (MAE ≈ 6.7°C vs. 0.37°C in Jena) illustrates the well-documented phenomenon of domain shift in machine learning [10]. The models' scaler and learned weight distributions are calibrated to Jena's specific altitude (155m), continental climate classification, and local topography. Berlin's lower elevation, urban heat island effects, and different wind exposure create a systematically different input distribution.

However, the fact that the models correctly tracked diurnal temperature cycles in Berlin — predicting warmer afternoons and cooler nights in the correct temporal phase — demonstrates meaningful generalisation of the underlying physics. Future work incorporating spatially-aware architectures (e.g., Graph Neural Networks) could address this geographic limitation.

### D. Limitations

Several limitations of this work should be acknowledged:

1. **Single-variable prediction:** Only temperature was predicted; extending to humidity, pressure, and wind would provide a more comprehensive forecasting system.
2. **No exogenous features:** Calendar features (month, day-of-week, hour) were not explicitly encoded, potentially missing seasonal patterns that the models must learn implicitly.
3. **Autoregressive error accumulation:** Multi-step predictions degrade beyond 6-12 hours as errors compound through the feedback loop, a fundamental limitation of the autoregressive approach.
4. **Single geographic training site:** The models generalise poorly to distant locations, as demonstrated by the Berlin transfer test.

---

## VI. DEPLOYMENT

### A. Live Cloud Deployment

The trained models have been deployed as a publicly accessible web application on the Streamlit Community Cloud. The live dashboard can be accessed at:

> **https://weather-tcn-forecasting.streamlit.app**

The deployment architecture automatically downloads the Jena Climate dataset on first load (from Google's TensorFlow public data mirror) and loads the pre-trained PyTorch model weights committed to the GitHub repository.

### B. Real-Time Inference Pipeline

The dashboard implements a complete real-time inference pipeline. Through a user-facing toggle, the application connects to the **Open-Meteo Live API** to retrieve the past 168 hours of actual weather data for Jena, Germany. The raw API response is processed through the same preprocessing pipeline used during training: missing features are derived using psychrometric formulas, values are scaled using the persisted `StandardScaler`, and the resulting tensor is fed through the trained PyTorch models to generate live temperature forecasts up to 48 hours ahead.

### C. Dashboard Interface

The web interface features a premium dark aesthetic inspired by modern Google Weather applications:
- **Model Selection:** Users can switch between LSTM, TCN, TCN-LSTM, and TCN-Tuned models via a sidebar dropdown.
- **Interactive Charts:** Responsive Plotly visualisations display forecast curves with gold area fills and direct temperature markers.
- **Variable Horizons:** A slider controls the autoregressive forecast window from 1 to 48 hours.
- **Dual Data Modes:** Users can toggle between historical Jena dataset exploration and live Open-Meteo API forecasting.

### D. Local Reproduction

To reproduce the dashboard locally:
```
pip install -r requirements.txt
streamlit run app.py
```
The application opens at `http://localhost:8501`. Trained model weights must be present in `outputs/models/`.

---

## VII. CODE DOCUMENTATION

### A. Repository Structure

The complete codebase is available at: **https://github.com/tharun2431/weather-tcn-forecasting**

```
weather-tcn-forecasting/
├── app.py                   # Streamlit web dashboard (deployment entry point)
├── requirements.txt         # Python dependencies
├── .streamlit/config.toml   # Dark theme configuration
├── notebooks/
│   ├── Tharun_ML1.ipynb     # Milestone 1: Exploratory Data Analysis
│   ├── Tharun_ML2.ipynb     # Milestone 2: Model Training & Tuning
│   └── Tharun_ML3.ipynb     # Milestone 3: Evaluation & Analysis
├── data/raw/                # Dataset (auto-downloaded if absent)
├── outputs/
│   ├── figures/             # Generated plots and visualisations
│   ├── models/              # Persisted PyTorch weights (.pt files)
│   └── results/             # JSON evaluation metrics
└── src/models/hybrid.py     # Model class definitions
```

### B. Key Functions and Classes

- `LSTMModel`, `TCNModel`, `HybridTCNLSTM`: PyTorch `nn.Module` subclasses implementing each architecture with configurable hyperparameters.
- `WeatherDataset.__getitem__()`: Constructs sliding windows of 168 hours across 14 features with 1-hour prediction targets.
- `train_model()`: Training loop with early stopping, learning rate scheduling, and gradient clipping.
- `predict_future()`: Autoregressive multi-step inference with inverse scaling to produce predictions in °C.
- `fetch_live_jena_data()`: API client that retrieves, processes, and scales real-time Open-Meteo data for live inference.

### C. Dependencies

```
PyTorch ≥ 2.0, NumPy ≥ 1.24, Pandas ≥ 2.0, Plotly ≥ 5.15,
Scikit-Learn ≥ 1.3, Streamlit ≥ 1.30, Requests ≥ 2.31
```

---

## VIII. CONCLUSION

### A. Summary of Findings

This project demonstrated that while hybrid deep learning architectures offer theoretical advantages for time-series forecasting, their practical benefit depends on the complexity of the target signal. For hourly temperature prediction — a relatively smooth, strongly autocorrelated process — the standalone LSTM performed comparably to the more complex TCN-LSTM hybrid. The stacking ensemble proved most effective overall, achieving the lowest error (MAE 0.37°C) by optimally combining the complementary strengths of diverse base models.

The Berlin generalisation test confirmed that the models learned genuine physical patterns rather than memorising the training data, while simultaneously highlighting the limitation of single-site training for geographic transfer.

### B. Impact and Contribution

The primary contributions of this work are:
1. An empirical demonstration that architectural complexity does not automatically translate to improved performance for smooth meteorological signals.
2. A novel TCN-LSTM hybrid architecture that, despite not outperforming baselines, provides a documented architectural experiment with honest analysis of its limitations.
3. A production-ready, publicly deployed web dashboard (**https://weather-tcn-forecasting.streamlit.app**) demonstrating the complete pipeline from data to live inference.

### C. Future Enhancements

1. **Transformer Architectures:** Self-attention mechanisms (e.g., Informer [11]) may better capture long-range dependencies and reduce autoregressive error accumulation beyond 24-hour horizons.
2. **Calendar Feature Engineering:** Explicitly encoding month, day-of-week, and hour-of-day as cyclical features (sine/cosine encoding) could help models capture seasonal patterns more efficiently.
3. **Multi-Target Prediction:** Extending to simultaneously forecast temperature, humidity, pressure, and wind speed would create a more comprehensive and practically useful system.
4. **Probabilistic Forecasting:** Implementing Monte Carlo Dropout or quantile regression to generate prediction intervals, providing end-users with confidence bounds rather than point estimates.
5. **Spatially-Aware Models:** Graph Neural Networks operating on multi-station weather data could address the geographic transfer limitation observed in the Berlin experiment.

---

## REFERENCES

[1] P. Bauer, A. Thorpe, and G. Brunet, "The quiet revolution of numerical weather prediction," *Nature*, vol. 525, pp. 47–55, 2015.

[2] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.

[3] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *arXiv:1803.01271*, 2018.

[4] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[5] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.

[6] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

[7] A. G. Salman, B. Kanigoro, and Y. Heryadi, "Weather forecasting using deep learning techniques," in *Proc. ICACSIS*, pp. 281–285, 2015.

[8] P. Hewage et al., "Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station," *Soft Computing*, vol. 24, pp. 16453–16482, 2020.

[9] F. Chollet, *Deep Learning with Python*, 2nd ed., Manning Publications, 2021.

[10] J. Quionero-Candela et al., *Dataset Shift in Machine Learning*, MIT Press, 2009.

[11] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proc. AAAI*, vol. 35, no. 12, pp. 11106–11115, 2021.
