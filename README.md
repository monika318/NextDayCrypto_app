Ethereum Next-Day High Prediction API
FastAPI service that predicts tomorrow’s HIGH for Ethereum (ETH-USD) using today’s UTC hourly market data from Kraken plus an sklearn Ridge pipeline.
Version: 2.2
Author: Monika Shakya (25548660)
Project: AT3 – Data Product with ML (Crypto Prediction)

✨ Features


Live hourly ETH data via Kraken OHLC (with retry + backoff).


Daily feature engineering (lags, rolling stats, ATR, cyclical time).


GET prediction endpoint: /predict/ethereum.


Production-friendly: Docker image, healthcheck, clean logs.



📦 Repository Layout
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


✅ The API boots even if feature_names.json is missing (falls back to a default order).
❗ models/eth_ridge_pipeline.pkl must exist and be loadable.


🧰 Tech Stack


FastAPI + Uvicorn


pandas / numpy / scikit-learn / joblib


requests with urllib3 Retry


Python 3.11



🚀 Quickstart
Option A — Local (Python)


Install deps:


pip install --no-cache-dir -r requirements.txt



Run the server:


uvicorn app.main:app --host 0.0.0.0 --port 8000



Open:




Docs (Swagger): http://127.0.0.1:8000/docs


Health: http://127.0.0.1:8000/health/


Predict: http://127.0.0.1:8000/predict/ethereum



Option B — Docker
Build:
docker build -t eth-nextday-api .

Run:
docker run -d -p 8000:8000 --name eth-api eth-nextday-api

Test:
curl http://localhost:8000/health/
curl http://localhost:8000/predict/ethereum


🔌 Endpoints
MethodPathDescriptionGET/health/Service status; confirms model availabilityGET/predict/ethereumPredicts next-day HIGH (USD)
Example: GET /predict/ethereum
200 OK
{
  "token": "ethereum",
  "predicted_next_day_high": 3215.82,
  "units": "USD",
  "feature_day_utc": "2025-11-03",
  "prediction_for_day_utc": "2025-11-04",
  "timestamp_utc": "2025-11-03T09:00:22Z",
  "features_used": {
    "close": 3175.23,
    "low": 3150.12,
    "volume": 12345.67,
    "close_lag_1": 3168.10,
    "volume_lag_1": 980.0,
    "roll_mean_close_7": 3180.55,
    "roll_std_close_7": 12.44,
    "return_1d": 0.004,
    "price_range": 48.7,
    "body_size": 10.1,
    "month_sin": -0.5,
    "month_cos": 0.87,
    "hour_high_sin": 0.26,
    "hour_high_cos": 0.96,
    "atr_14": 25.3
  }
}

Possible errors


400 — unsupported token (only ethereum).


502 — upstream data missing from Kraken for the window.


503 — model not loaded.


500 — unexpected error during feature build/prediction.



🧠 How the Prediction Window Works


API uses today’s UTC hourly OHLCV (partial day until UTC midnight).


It builds features for D (today) and predicts D+1 (tomorrow) high.


Example: On Nov 3 (UTC) → builds features from Nov 3 → predicts Nov 4 HIGH.



🧪 Model & Pipeline


Model: Ridge Regression (scikit-learn)


Custom transformers registered for safe unpickling:


FFillBFillImputer – forward/backward fill


WinsorizeCapper – robust outlier capping


Log1pNamedColumns – safe log1p on selected columns




Engineered features:


Lags: close_lag_1, volume_lag_1


Rolling: roll_mean_close_7, roll_std_close_7


Volatility/momentum: return_1d, atr_14, price_range, body_size


Time (cyclical): month_sin/cos, hour_high_sin/cos




If models/feature_names.json is present, it dictates column order; otherwise the app uses:
["close","low","volume","close_lag_1","volume_lag_1",
 "roll_mean_close_7","roll_std_close_7","return_1d",
 "price_range","body_size","month_sin","month_cos",
 "hour_high_sin","hour_high_cos","atr_14"]


⚙️ Configuration
Environment variables (all optional):
NamePurposeDefaultPORTUvicorn port8000MODEL_PATHPath to model picklemodels/eth_ridge_pipeline.pklFEATURES_PATHPath to feature names JSONmodels/feature_names.json

Note: Your current main.py reads model/feature paths from the repo layout above; override via code or env if you refactor.


🐳 Dockerfile (for reference)
FROM python:3.11.4-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


📚 Dependencies
Pinned versions (from requirements.txt / pyproject.toml):
fastapi==0.111.0
uvicorn==0.30.1
pandas==2.2.2
scikit-learn==1.5.1
numpy==1.26.4
joblib==1.4.2
requests==2.31.0
tenacity==8.2.2


🧯 Troubleshooting


model_not_loaded in /health/: Ensure models/eth_ridge_pipeline.pkl exists and matches the custom transformers in main.py.


Kraken returns empty data (502): Upstream may be temporarily unavailable, or the time window has no bars yet; try again later.


NaN/inf feature error (500): Usually due to insufficient intraday data; call again closer to UTC midnight or next hour.


CORS (if calling from browser): Add a CORS middleware to FastAPI if you serve from a different origin.


Timezone confusion: The service operates in UTC for data windows and timestamps.



📜 License / Academic Use
Developed for UTS 36120 Advanced Machine Learning & Agents – AT3: Data Product with ML.
For academic demonstration and coursework submission.

✅ Submission Checklist


 app/main.py present and imports succeed


 models/eth_ridge_pipeline.pkl included


 (Optional) models/feature_names.json included


 requirements.txt / pyproject.toml pinned


 Docker image builds and runs locally


 /docs, /health/, and /predict/ethereum all tested



Need a deployment section (Render or Cloud Run) added as well? I can append step-by-step deploy instructions with health checks and example test commands.
