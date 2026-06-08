# 🪙 NextDayCrypto — Ethereum Price Prediction API

A production-ready REST API that predicts **tomorrow's HIGH price for Ethereum (ETH-USD)** using live market data and a trained machine learning pipeline. Built with FastAPI and deployed via Docker and Render.

**Author:** Monika Shakya · Master of Data Science and Innovation, UTS

---

## 🔍 Project Overview

Cryptocurrency markets are highly volatile and data-rich — making them an ideal domain for applied machine learning. This project demonstrates an end-to-end ML data product:

- Pulls **live hourly OHLC data** from the Kraken exchange API
- Engineers daily features including **lag variables, rolling statistics, ATR, and cyclical time encodings**
- Serves next-day HIGH predictions through a clean **REST API endpoint**
- Packaged as a **Docker image** and deployed to the cloud via **Render**

The goal was to build something production-grade: resilient data ingestion, a clean API contract, and a deployment pipeline — not just a notebook.

---

## ✨ Features

- 📡 **Live data** — fetches today's UTC hourly ETH/USD OHLC from Kraken with automatic retry and exponential backoff (`tenacity`)
- 🧠 **ML pipeline** — pre-trained sklearn Ridge regression pipeline with feature scaling and engineering
- 🔮 **Prediction endpoint** — single GET request returns tomorrow's predicted ETH HIGH price
- 🩺 **Health check** — `/health/` endpoint for uptime monitoring
- 🐳 **Dockerised** — fully containerised for reproducible local and cloud deployment
- ☁️ **Render-ready** — includes `render.yaml` for one-click cloud deployment

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** + **Uvicorn** | REST API framework and ASGI server |
| **scikit-learn** | Ridge regression pipeline + feature transformation |
| **pandas** + **numpy** | Feature engineering and data manipulation |
| **joblib** | Model serialisation / deserialisation |
| **requests** + **tenacity** | Kraken API calls with retry and backoff |
| **Docker** | Containerisation |
| **Render** | Cloud deployment |
| **Python 3.11** | Runtime |

---

## 🗂️ Project Structure

```
NextDayCrypto_app/
│
├── app/
│   └── main.py               # FastAPI app — data fetch, feature engineering, prediction
│
├── models/
│   ├── eth_ridge_pipeline.pkl # Trained sklearn pipeline (required at boot)
│   └── feature_names.json    # Feature order (optional — has hardcoded fallback)
│
├── Dockerfile                # Container build for local and cloud deployment
├── render.yaml               # Render cloud deployment config
├── requirements.txt          # Pinned runtime dependencies
├── pyproject.toml            # Project metadata and dependency spec
├── .python-version           # Python 3.11 pin (for pyenv)
└── README.md
```

---

## 🔄 How It Works

```
GET /predict/ethereum
        ↓
  Fetch live hourly ETH/USD OHLC from Kraken (last ~24h)
        ↓
  Aggregate to daily OHLC features
        ↓
  Engineer features: lags, rolling mean/std, ATR, cyclical hour/day encodings
        ↓
  Load eth_ridge_pipeline.pkl (Ridge + StandardScaler)
        ↓
  Return predicted next-day HIGH price (USD)
```

---

## 🚀 Quickstart

### Option A — Local (Python)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option B — Docker

```bash
docker build -t nextdaycrypto .
docker run -p 8000:8000 nextdaycrypto
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /predict/ethereum` | Returns predicted next-day ETH HIGH (USD) |
| `GET /health/` | Health check |
| `GET /docs` | Auto-generated Swagger UI |

---

## ☁️ Cloud Deployment (Render)

The included `render.yaml` configures automatic deployment to [Render](https://render.com):

```yaml
services:
  - type: web
    name: eth-high-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Push to `main` and the service redeploys automatically.

> **Note:** `models/eth_ridge_pipeline.pkl` must be present in the repo for the API to boot. The `feature_names.json` is optional — the app falls back to a default feature order if it's missing.

---

## 📦 Dependencies

```
fastapi==0.111.0
uvicorn==0.30.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
requests==2.31.0
tenacity==8.2.2
```

---

## 👩‍💻 Author

**Monika Shakya**
Master of Data Science and Innovation
University of Technology Sydney
