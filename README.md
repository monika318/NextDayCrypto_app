# 🪙 NextDayCrypto — Ethereum Next-Day HIGH Prediction API

A production-ready REST API that predicts **tomorrow's HIGH price for Ethereum (ETH-USD)** using today's live UTC market data and a trained scikit-learn Ridge pipeline. Built with FastAPI, containerised with Docker, and deployed on Render.

**Author:** Monika Shakya · Master of Data Science and Innovation, UTS

---

## 🔍 Project Overview

Cryptocurrency markets are highly volatile and data-rich — making them an ideal domain for applied machine learning. This project demonstrates a complete end-to-end ML data product:

- Pulls **live hourly OHLC data** from the Kraken public API (no API key required)
- Engineers **15 daily features** from today's intraday window — including price lags, rolling statistics, ATR, body size, and cyclical time encodings
- Serves next-day HIGH predictions via a single **GET endpoint**
- Packaged as a **Docker image** and deployed to the cloud via **Render**

The focus was on building something genuinely production-grade: resilient data ingestion with retry logic, robust custom transformers, clean API responses, and a real deployment pipeline.

---

## ✨ Features

- 📡 **Live data** — fetches today's UTC hourly ETH/USD OHLC from Kraken with automatic retry and exponential backoff (`urllib3 Retry`)
- 🧠 **Custom sklearn pipeline** — includes `FFillBFillImputer`, `WinsorizeCapper`, and `Log1pNamedColumns` transformers, with a Ridge regression estimator
- 🔮 **Tomorrow-forecast mode** — builds features from today's partial UTC window and predicts the next day's HIGH (D → D+1)
- 📦 **Rich API response** — returns predicted price, feature day, prediction day, timestamp, and all 15 features used
- 🩺 **Health check** — `/health/` reports model load status
- 🐳 **Dockerised** — fully containerised for reproducible local and cloud deployment
- ☁️ **Render-ready** — `render.yaml` for one-click cloud deployment

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** + **Uvicorn** | REST API framework and ASGI server |
| **scikit-learn** | Ridge regression + custom transformers pipeline |
| **pandas** + **numpy** | Feature engineering and data manipulation |
| **joblib** | Model serialisation / deserialisation |
| **requests** + **urllib3 Retry** | Kraken API calls with retry and backoff |
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
│   └── feature_names.json    # Feature order (optional — falls back to hardcoded default)
│
├── Dockerfile                # Container build
├── render.yaml               # Render cloud deployment config
├── requirements.txt          # Pinned runtime dependencies
├── pyproject.toml            # Project metadata
├── .python-version           # Python 3.11 pin
└── README.md
```

---

## 🔄 How It Works

```
GET /predict/ethereum
        ↓
  Fetch today's UTC hourly ETH/USD OHLC from Kraken
  (retries on 429/5xx with exponential backoff)
        ↓
  Engineer 15 features from today's intraday window:
    close, low, volume, close_lag_1, volume_lag_1,
    roll_mean_close_7, roll_std_close_7, return_1d,
    price_range, body_size, month_sin, month_cos,
    hour_high_sin, hour_high_cos, atr_14
        ↓
  Load eth_ridge_pipeline.pkl
  (FFillBFillImputer → WinsorizeCapper → Log1pNamedColumns → Ridge)
        ↓
  Return predicted next-day HIGH (USD) + full feature snapshot
```

**Forecast logic:** called on day D, features are built from D's intraday data, prediction is for day D+1's HIGH.

---

## 📤 Example Response

```json
{
  "token": "ethereum",
  "predicted_next_day_high": 3842.17,
  "units": "USD",
  "feature_day_utc": "2024-11-03",
  "prediction_for_day_utc": "2024-11-04",
  "timestamp_utc": "2024-11-03T14:22:10+00:00",
  "features_used": {
    "close": 3801.45,
    "low": 3765.20,
    "volume": 124500.0,
    "close_lag_1": 3789.10,
    ...
  }
}
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
| `GET /health/` | Health check — confirms model is loaded |
| `GET /docs` | Auto-generated Swagger UI |

> **Note:** `models/eth_ridge_pipeline.pkl` must be present for the API to boot. `feature_names.json` is optional — the app falls back to a hardcoded default feature order if missing.

---

## ☁️ Cloud Deployment (Render)

The included `render.yaml` configures deployment to [Render](https://render.com):

```yaml
services:
  - type: web
    name: eth-high-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

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
