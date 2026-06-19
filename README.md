# 📈 Stock Market Prediction Using Deep Learning — Final Year Project

A comprehensive deep learning pipeline for **intraday stock market direction prediction** on S&P 500 equities. The system collects 1-minute OHLCV data and news sentiment from the Polygon.io API, engineers technical and Kalman-filtered features, and benchmarks **five distinct neural architectures** in both a feature-enriched ("Normal") and raw-data ("Baseline") configuration — exposing whether hand-crafted features genuinely improve predictive power over letting the model learn its own representations.

---

## 🏗️ Architecture Overview

```
├── data_collection.py          # Automated data pipeline (Polygon.io API → cleaned CSVs)
├── data_visualizations.py      # Technical indicator & equity curve charting
├── engine.py                   # Orchestration engine for the Flask web app
├── app.py                      # Flask web dashboard (collection, training, visualization)
│
├── lstm.py / lstm_baseline.py              # LSTM model (Normal / Baseline)
├── gru.py / gru_baseline.py               # GRU model (Normal / Baseline)
├── TFT.py / TFT_baseline.py              # Temporal Fusion Transformer (Normal / Baseline)
├── tabnet.py / tabnet_baseline.py         # TabNet model (Normal / Baseline)
├── 1DCapsnetLstmHybrid.py / *_baseline.py # 1D-CapsNet + LSTM Hybrid (Normal / Baseline)
│
├── data/                       # Processed CSV datasets per ticker (~860 MB)
├── saved_models/               # Trained model checkpoints per ticker (~170 MB)
├── visualizations/             # Confusion matrices, equity curves, reports
├── lightning_logs/             # PyTorch Lightning training logs
├── static/                    # Static assets for the web dashboard
└── templates/                 # Jinja2 HTML templates for the Flask UI
```

---

## 🔬 Models Implemented

| Model | Type | Framework |
|-------|------|-----------|
| **LSTM** | Recurrent | Keras / PyTorch |
| **GRU** | Recurrent | Keras |
| **Temporal Fusion Transformer (TFT)** | Attention-based | PyTorch Lightning |
| **TabNet** | Attention-based Tabular | PyTorch TabNet |
| **1D-CapsNet + LSTM Hybrid** | Capsule Network | PyTorch |

Each model is tested in two modes:
- **Normal**: Uses hand-crafted features (SMA, EMA, MACD, RSI, Kalman Filter, rolling highs/lows, news sentiment)
- **Baseline**: Uses only raw OHLCV data, letting the network learn its own representations

---

## 📊 Feature Engineering Pipeline

The `data_collection.py` script builds a rich feature set for each ticker:

- **Price Data**: 1-minute OHLCV bars from Polygon.io (market hours only)
- **Technical Indicators**: SMA(10, 50), EMA(10, 50), MACD, RSI(14)
- **Rolling Features**: 20-bar rolling high/low, 1-minute returns
- **Kalman Filter**: A **causal** (forward-pass only) Kalman filter
- **News Sentiment**: Backward-merged Polygon.io news articles with sentiment labels and staleness tracking

---

## 🚀 Quick Start

### Prerequisites

```
Python 3.10+
pip install requests pandas pandas_ta pytz pykalman
pip install torch torchvision pytorch-lightning pytorch-tabnet
pip install keras tensorflow plotly flask
```

### 1. Collect Data

```bash
python data_collection.py
```

This fetches 1-minute stock data + news for 13 S&P 500 tickers and saves processed CSVs into `data/`.

### 2. Train Models

Use the Jupyter notebooks for interactive training and evaluation:

```
lstm.ipynb          # LSTM Normal
lstm_baseline.ipynb # LSTM Baseline
GRU.ipynb           # GRU Normal
gru_baseline.ipynb  # GRU Baseline
TFT.ipynb           # TFT Normal
TFT_baseline.ipynb  # TFT Baseline
tabnet.ipynb        # TabNet Normal
tabnet_baseline.ipynb # TabNet Baseline
1Dcapsnetlstmhybrid.ipynb          # CapsNet+LSTM Normal
1Dcapsnetlstmhybrid_baseline.ipynb # CapsNet+LSTM Baseline
```

Or use the standalone `.py` scripts directly:

```bash
python lstm.py
python TFT.py
```

### 3. Launch Web Dashboard

```bash
python app.py
```

Navigate to `http://localhost:5000` to access the interactive dashboard for data collection, model training, and visualization.

---

## 📁 S&P 500 Tickers Covered

```
MSFT, COST, HD, JNJ, JPM, LLY, MA, META, PG, UNH, V, WMT, XOM

```

## 📄 License

This project was developed as a Final Year Project for academic purposes.
