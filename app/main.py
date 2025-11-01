# # app/main.py
# from fastapi import FastAPI, HTTPException, Path
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Optional
# import numpy as np
# import pandas as pd
# import joblib
# import json
# from datetime import datetime, timezone

# # -------------------------------------------------------------------
# # 🎯 Configuration
# # -------------------------------------------------------------------
# ALLOWED_TOKENS = {"ethereum"}  # only ETH supported for this student API

# MODEL_PATH = "models/ridge_final_on_scaled.pkl"   # <-- update if your file name differs
# FEATURES_PATH = "models/feature_names.json"       # optional; list of feature names in training order

# # If feature_names.json is missing, fall back to this exact order:
# FEATURE_ORDER_DEFAULT = ["year", "month", "hour_high", "hour_low", "log_close", "log_volume"]

# # -------------------------------------------------------------------
# # 🧠 Load model and (optional) feature order
# # -------------------------------------------------------------------
# try:
#     model = joblib.load(MODEL_PATH)
#     print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     model = None
#     print(f"[WARN] Could not load model at startup: {e}")

# try:
#     with open(FEATURES_PATH, "r") as f:
#         FEATURE_ORDER = json.load(f)  # must be a list[str] in the exact order used at training
#         print(f"[INFO] Feature order loaded from {FEATURES_PATH}: {len(FEATURE_ORDER)} features")
# except Exception:
#     FEATURE_ORDER = None
#     print("[WARN] No feature order JSON found. Will use default order.")

# # -------------------------------------------------------------------
# # 📦 Pydantic schema (users send raw close/volume; API computes logs)
# # -------------------------------------------------------------------
# class EthInput(BaseModel):
#     year: int = Field(..., ge=2000, description="Calendar year (UTC).")
#     month: int = Field(..., ge=1, le=12, description="Month [1..12].")
#     hour_high: int = Field(..., ge=0, le=23, description="Hour of day when HIGH occurred.")
#     hour_low: int = Field(..., ge=0, le=23, description="Hour of day when LOW occurred.")
#     close: float = Field(..., ge=0, description="Raw close price (USD).")
#     volume: float = Field(..., ge=0, description="Raw traded volume (non-negative).")

# # -------------------------------------------------------------------
# # 🔧 Helpers
# # -------------------------------------------------------------------
# def build_feature_row(payload: EthInput) -> pd.DataFrame:
#     """
#     Convert raw inputs into the model's features:
#       - log_close = log1p(close)
#       - log_volume = log1p(volume)
#     Arrange columns in FEATURE_ORDER (from file) or FEATURE_ORDER_DEFAULT.
#     """
#     log_close = float(np.log1p(payload.close))
#     log_volume = float(np.log1p(payload.volume))

#     row = {
#         "year": payload.year,
#         "month": payload.month,
#         "hour_high": payload.hour_high,
#         "hour_low": payload.hour_low,
#         "log_close": log_close,
#         "log_volume": log_volume,
#     }

#     cols = FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT
#     missing = [c for c in cols if c not in row]
#     if missing:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Server misconfiguration: missing features {missing} in request construction."
#         )

#     return pd.DataFrame([row], columns=cols)

# # -------------------------------------------------------------------
# # 🚀 FastAPI application
# # -------------------------------------------------------------------
# app = FastAPI(
#     title="AT3 – Next-Day HIGH Predictor (Ethereum API)",
#     description=(
#         "Predicts next-day HIGH price (t+1) for Ethereum (ETH-USD). "
#         "Send raw features (year, month, hour_high, hour_low, close, volume); "
#         "the API computes log-transforms and returns the prediction."
#     ),
#     version="1.0.0",
# )

# # ---------------- Root & health ----------------
# @app.get("/", tags=["info"])
# def root():
#     return {
#         "project": "AT3 – Crypto Next-Day HIGH Prediction (Student API)",
#         "student": "Monika Shakya",
#         "token_supported": list(ALLOWED_TOKENS),
#         "endpoints": {
#             "/": "GET – project info",
#             "/health/": "GET – service healthcheck",
#             "/predict/{token}": "POST – next-day HIGH prediction for the token",
#         },
#         "expected_input_example": {
#             "year": 2025,
#             "month": 11,
#             "hour_high": 21,
#             "hour_low": 3,
#             "close": 3445.72,
#             "volume": 1283450.0
#         },
#         "model_artifacts": {
#             "model_path": MODEL_PATH,
#             "feature_order_file": FEATURES_PATH,
#             "feature_order_present": FEATURE_ORDER is not None,
#             "feature_order_used_if_missing": FEATURE_ORDER_DEFAULT,
#         },
#         "repo": "https://github.com/monika318/NextDayCrypto_app.git",
#     }

# @app.get("/health/", tags=["info"])
# def health():
#     status = model is not None
#     return JSONResponse(
#         status_code=200 if status else 503,
#         content={"status": "ok" if status else "model_not_loaded"},
#     )

# # ---------------- Prediction ----------------
# @app.post("/predict/{token}", tags=["prediction"])
# def predict(
#     token: str = Path(..., description="Must be 'ethereum'"),
#     payload: EthInput = ...
# ):
#     if token.lower() not in ALLOWED_TOKENS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported token '{token}'. Supported: {list(ALLOWED_TOKENS)}",
#         )

#     if model is None:
#         raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

#     # Build feature vector in the exact order expected by the model
#     X = build_feature_row(payload)

#     # Predict
#     try:
#         yhat = float(model.predict(X)[0])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

#     return {
#         "token": token.lower(),
#         "predicted_next_day_high": yhat,
#         "units": "USD",
#         "timestamp_utc": datetime.now(timezone.utc).isoformat(),
#         "n_features": int(X.shape[1]),
#         "feature_order_used": list(X.columns),
#     }

# app/main.py
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import requests
import joblib
import json
from datetime import datetime, timezone

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
ALLOWED_TOKENS = {"ethereum"}  # your token for AT3
MODEL_PATH = "models/ridge_final_on_scaled.pkl"         # your saved model/pipeline
FEATURES_PATH = "models/feature_names.json"             # optional: exact training feature order
FEATURE_ORDER_DEFAULT = ["year", "month", "hour_high", "hour_low", "log_close", "log_volume"]

# -------------------------------------------------------------
# Load model + feature order
# -------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)   # ideally this is a Pipeline(scaler + ridge)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

try:
    with open(FEATURES_PATH, "r") as f:
        FEATURE_ORDER = json.load(f)  # list[str] in the training order
        print(f"[INFO] Feature order loaded: {FEATURES_PATH}")
except Exception:
    FEATURE_ORDER = None
    print("[WARN] No feature_names.json found; using default order.")

# -------------------------------------------------------------
# Data fetchers (free APIs)
# -------------------------------------------------------------
def kraken_eth_hourly_today() -> pd.DataFrame:
    """
    Pull today's (UTC) ETH/USD 60-min candles from Kraken.
    Computes intraday hours for high/low and fresh close/volume.
    """
    # Midnight UTC today
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    since_unix = int(today_utc.timestamp())

    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "ETHUSD", "interval": 60, "since": since_unix}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(f"Kraken error: {js['error']}")

    # Result key is the pair, e.g. "ETHUSD"
    key = next((k for k in js["result"].keys() if k != "last"), None)
    rows = js["result"].get(key, [])
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    cols = ["ts", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(rows, columns=cols)
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]

def build_fresh_eth_features_for_today() -> dict:
    """
    Build the freshest feature payload for the model:
      - year/month: now (UTC)
      - hour_high/hour_low: hours of max high / min low from today's intraday candles
      - close: latest hourly close
      - volume: sum of today's hourly volumes
    """
    hourly = kraken_eth_hourly_today()
    if hourly.empty:
        # Fallback defaults (should be rare)
        now = datetime.now(timezone.utc)
        return {
            "year": now.year,
            "month": now.month,
            "hour_high": 15,
            "hour_low": 3,
            "close": 3000.0,
            "volume": 1_000_000.0,
        }

    idx_high = hourly["high"].idxmax()
    idx_low  = hourly["low"].idxmin()
    hour_high = int(hourly.loc[idx_high, "time"].hour)
    hour_low  = int(hourly.loc[idx_low,  "time"].hour)

    latest_close = float(hourly.iloc[-1]["close"])
    day_volume   = float(hourly["volume"].sum())

    now = datetime.now(timezone.utc)
    return {
        "year": now.year,
        "month": now.month,
        "hour_high": hour_high,
        "hour_low": hour_low,
        "close": latest_close,
        "volume": day_volume,
    }

# -------------------------------------------------------------
# Feature builder for model (log transforms + column order)
# -------------------------------------------------------------
def build_feature_row_from_raw(raw: dict) -> pd.DataFrame:
    """
    raw dict keys: year, month, hour_high, hour_low, close, volume
    transforms: log_close=log1p(close), log_volume=log1p(volume)
    orders columns for the trained model
    """
    if any(k not in raw for k in ["year", "month", "hour_high", "hour_low", "close", "volume"]):
        raise HTTPException(status_code=500, detail="Server: missing raw keys to build features.")

    row = {
        "year": int(raw["year"]),
        "month": int(raw["month"]),
        "hour_high": int(raw["hour_high"]),
        "hour_low": int(raw["hour_low"]),
        "log_close": float(np.log1p(float(raw["close"]))),
        "log_volume": float(np.log1p(float(raw["volume"]))),
    }

    cols = FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT
    missing = [c for c in cols if c not in row]
    if missing:
        raise HTTPException(status_code=500, detail=f"Server: missing engineered features {missing}.")

    return pd.DataFrame([row], columns=cols)

# -------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------
app = FastAPI(
    title="AT3 – Next-Day HIGH Predictor (Ethereum API)",
    description="GET endpoints only. Auto-fetches fresh features and predicts t+1 HIGH for Ethereum.",
    version="1.0.0",
)

@app.get("/", tags=["info"])
def root():
    return {
        "project": "AT3 – Crypto Next-Day HIGH Prediction (Student API)",
        "token_supported": list(ALLOWED_TOKENS),
        "endpoints": {
            "/": "GET – project info",
            "/health/": "GET – service healthcheck",
            "/predict/{token}": "GET – return next-day HIGH prediction for the token (no request body)",
        },
        "expected_output_example": {
            "token": "ethereum",
            "predicted_next_day_high": 3456.78,
            "units": "USD",
            "timestamp_utc": "2025-11-01T10:00:00Z",
            "features_used": {
                "year": 2025, "month": 11, "hour_high": 15, "hour_low": 3,
                "close": 3400.56, "volume": 1234567.0
            },
            "n_features": 6,
            "feature_order_used": FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT,
        },
        "repo": "ADD_YOUR_PRIVATE_REPO_LINK_HERE"
    }

@app.get("/health/", tags=["info"])
def health():
    status = model is not None
    return JSONResponse(status_code=200 if status else 503,
                        content={"status": "ok" if status else "model_not_loaded"})

@app.get("/predict/{token}", tags=["prediction"])
def predict(token: str = Path(..., description="Must be 'ethereum' for this student API")):
    """
    GET-only: no request body. The API fetches fresh features itself (Kraken),
    builds the model input, and returns the t+1 HIGH prediction.
    """
    if token.lower() not in ALLOWED_TOKENS:
        raise HTTPException(status_code=400, detail=f"Unsupported token '{token}'. Allowed: {list(ALLOWED_TOKENS)}")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    try:
        fresh_raw = build_fresh_eth_features_for_today()
        X = build_feature_row_from_raw(fresh_raw)
        yhat = float(model.predict(X)[0])
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream data fetch failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": token.lower(),
        "predicted_next_day_high": yhat,
        "units": "USD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_used": fresh_raw,  # raw (pre-log) values that were used
        "n_features": int(X.shape[1]),
        "feature_order_used": list(X.columns),
    }
