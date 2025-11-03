# 🪙 Ethereum Next-Day High Prediction API

FastAPI service that predicts **tomorrow’s HIGH** for **Ethereum (ETH-USD)** using today’s UTC hourly market data from **Kraken** plus an **sklearn Ridge pipeline**.

**Version:** 2.2  
**Author:** Monika Shakya (25548660)  
**Project:** AT3 – Data Product with ML (Crypto Prediction)

---

## ✨ Features

- Live hourly ETH data via **Kraken OHLC** (with retry + backoff).  
- Daily feature engineering (lags, rolling stats, ATR, cyclical time).  
- **GET** prediction endpoint: `/predict/ethereum`.  
- Production-friendly: **Docker** image, healthcheck, clean logs.

---

## 📦 Repository Layout
.
├── app/
│   ├── main.py                 # FastAPI application (your code here)
│   └── __init__.py             # (optional)
├── models/
│   ├── eth_ridge_pipeline.pkl  # Trained sklearn pipeline (required)
│   └── feature_names.json      # Feature order (optional; has fallback)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md


✅ The API boots even if `feature_names.json` is missing (falls back to a default order).  
❗ `models/eth_ridge_pipeline.pkl` must exist and be loadable.

---

## 🧰 Tech Stack

- **FastAPI** + **Uvicorn**  
- **pandas / numpy / scikit-learn / joblib**  
- **requests** with **urllib3 Retry**  
- **Python 3.11**

---

## 🚀 Quickstart

### 🧪 Option A — Local (Python)

**Install dependencies:**
```bash
pip install --no-cache-dir -r requirements.txt
