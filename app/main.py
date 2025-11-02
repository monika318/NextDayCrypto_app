# app/main.py
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import requests
import joblib
import json
from datetime import datetime, timezone

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
ALLOWED_TOKENS = {"ethereum"}  # ETH only for this student API

MODEL_PATH = "models/ridge_final_on_scaled_features_list_3.pkl"     # your trained model/pipeline
FEATURES_PATH = "models/feature_names.json"         # optional: exact training feature order
FEATURE_ORDER_DEFAULT = [
    "open","close","hour_high","hour_low","marketCap","month","day","hour_open"
]

# -------------------------------------------------------------
# Load model + feature order
# -------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)  # ideally Pipeline(StandardScaler(), Ridge(...))
    print(f"[INFO] Model loaded: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

try:
    with open(FEATURES_PATH, "r") as f:
        FEATURE_ORDER = json.load(f)  # list[str] in the training order
        print(f"[INFO] Feature order loaded: {FEATURES_PATH} -> {FEATURE_ORDER}")
except Exception:
    FEATURE_ORDER = None
    print("[WARN] No feature_names.json found; using default order.")

# -------------------------------------------------------------
# Public data fetchers (no keys)
# -------------------------------------------------------------
def kraken_eth_hourly_today() -> pd.DataFrame:
    """
    Pull today's (UTC) ETH/USD 60-min candles from Kraken.
    Returns columns: time, open, high, low, close, volume
    """
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    since_unix = int(today_utc.timestamp())

    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "ETHUSD", "interval": 60, "since": since_unix}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(f"Kraken error: {js['error']}")

    key = next((k for k in js["result"].keys() if k != "last"), None)
    rows = js["result"].get(key, [])
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    cols = ["ts", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(rows, columns=cols)
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]

def coingecko_eth_marketcap_usd() -> float:
    """
    Fetch current Ethereum market cap in USD from CoinGecko (no auth).
    """
    url = "https://api.coingecko.com/api/v3/coins/ethereum"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    mc = js.get("market_data", {}).get("market_cap", {}).get("usd", None)
    if mc is None:
        raise RuntimeError("CoinGecko: market cap USD not found")
    return float(mc)

# -------------------------------------------------------------
# Build fresh feature payload (GET path)
# -------------------------------------------------------------
def build_fresh_eth_features_for_today() -> dict:
    """
    Build the freshest feature payload for the model:
      - hour_high/hour_low: hours of today's max high / min low (UTC) from Kraken
      - open: today's first hourly open (UTC)
      - close: latest hourly close (UTC)
      - marketCap: current market cap (USD) from CoinGecko
      - month/day: from current UTC date
      - hour_open: hour of the first candle today (UTC)
    """
    hourly = kraken_eth_hourly_today()
    now = datetime.now(timezone.utc)
    
    defaults = {
        "open": 3000.0,
        "close": 3000.0,
        "hour_high": 12,
        "hour_low": 3,
        "marketCap": coingecko_eth_marketcap_usd(),
        "month": now.month,
        "day": now.day,
        "hour_open": 0,
    }

    if hourly.empty:
        return defaults

    idx_high = hourly["high"].idxmax()
    idx_low  = hourly["low"].idxmin()
    hour_high = int(hourly.loc[idx_high, "time"].hour)
    hour_low  = int(hourly.loc[idx_low,  "time"].hour)

    # First hourly 'open' of the UTC day and its hour
    first_row = hourly.iloc[0]
    todays_open = float(first_row["open"])
    hour_open = int(first_row["time"].hour)

    latest_close = float(hourly.iloc[-1]["close"])
    mc = coingecko_eth_marketcap_usd()

    return {
        "open": todays_open,
        "close": latest_close,
        "hour_high": hour_high,
        "hour_low": hour_low,
        "marketCap": mc,
        "month": now.month,
        "day": now.day,
        "hour_open": hour_open,
    }

# -------------------------------------------------------------
# Feature builder (NO transforms, just exact order)
# -------------------------------------------------------------
def to_model_dataframe(raw: dict) -> pd.DataFrame:
    """
    raw keys required: open, close, hour_high, hour_low, marketCap, month, day, hour_open
    No transforms; order must match training.
    """
    required = ["open","close","hour_high","hour_low","marketCap","month","day","hour_open"]
    for k in required:
        if k not in raw:
            raise HTTPException(status_code=400, detail=f"Missing '{k}' in payload.")

    row = {
        "open": float(raw["open"]),
        "close": float(raw["close"]),
        "hour_high": int(raw["hour_high"]),
        "hour_low": int(raw["hour_low"]),
        "marketCap": float(raw["marketCap"]),
        "month": int(raw["month"]),
        "day": int(raw["day"]),
        "hour_open": int(raw["hour_open"]),
    }

    cols = FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT
    missing = [c for c in cols if c not in row]
    if missing:
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: missing features {missing}.")

    return pd.DataFrame([row], columns=cols)

# -------------------------------------------------------------
# Request schema for POST
# -------------------------------------------------------------
class EthInput(BaseModel):
    open: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    hour_high: int = Field(..., ge=0, le=23)
    hour_low: int = Field(..., ge=0, le=23)
    marketCap: float = Field(..., ge=0, description="Market cap in USD")
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour_open: int = Field(..., ge=0, le=23)

# -------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------
app = FastAPI(
    title="AT3 – Next-Day HIGH Predictor (Ethereum API)",
    description=(
        "Predicts next-day HIGH (t+1) for ETH using features: "
        "['open','close','hour_high','hour_low','marketCap','month','day','hour_open'].\n"
        "Use GET to auto-fetch or POST to supply features explicitly. All times are UTC."
    ),
    version="1.2.0",
)

@app.get("/", tags=["info"])
def root():
    return {
        "project": "AT3 – Crypto Next-Day HIGH Prediction (Student API)",
        "student": "Monika Shakya",
        "token_supported": list(ALLOWED_TOKENS),
        "endpoints": {
            "/": "GET – project info",
            "/health/": "GET – service healthcheck",
            "/predict/{token}": "GET – auto-fetch features & predict",
            "/predict/{token} (POST)": "POST – provide features & predict",
        },
        "feature_order_used": FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT,
        "model_artifacts": {"model_path": MODEL_PATH, "feature_order_file": FEATURES_PATH},
        "expected_post_body_example": {
            "open": 3400.56, "close": 3432.11, "hour_high": 15, "hour_low": 3,
            "marketCap": 4.2e11, "month": 11, "day": 1, "hour_open": 0
        },
        "note": "Ensure your trained model was fit with exactly these 8 features in this exact order (UTC-based).",
    }

@app.get("/health/", tags=["info"])
def health():
    status = model is not None
    return JSONResponse(status_code=200 if status else 503,
                        content={"status": "ok" if status else "model_not_loaded"})

# ---------- GET: auto-fetch ----------
@app.get("/predict/{token}", tags=["prediction"])
def predict_get(token: str = Path(..., description="Must be 'ethereum'")):
    if token.lower() not in ALLOWED_TOKENS:
        raise HTTPException(status_code=400, detail=f"Unsupported token '{token}'. Allowed: {list(ALLOWED_TOKENS)}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    try:
        raw = build_fresh_eth_features_for_today()
        X = to_model_dataframe(raw)
        yhat = float(model.predict(X)[0])
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": token.lower(),
        "predicted_next_day_high": yhat,
        "units": "USD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_used": raw,  # raw values fed to model
        "n_features": int(X.shape[1]),
        "feature_order_used": list(X.columns),
    }

# ---------- POST: user-supplied features ----------
@app.post("/predict/{token}", tags=["prediction"])
def predict_post(payload: EthInput, token: str = Path(..., description="Must be 'ethereum'")):
    if token.lower() not in ALLOWED_TOKENS:
        raise HTTPException(status_code=400, detail=f"Unsupported token '{token}'. Allowed: {list(ALLOWED_TOKENS)}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    raw = payload.dict()
    X = to_model_dataframe(raw)
    try:
        yhat = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": token.lower(),
        "predicted_next_day_high": yhat,
        "units": "USD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_used": raw,
        "n_features": int(X.shape[1]),
        "feature_order_used": list(X.columns),
    }
