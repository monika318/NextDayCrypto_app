# app/main.py
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import requests
import joblib
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
ALLOWED_TOKENS = {"ethereum"}  # ETH only for this student API

MODEL_PATH = "models/ridge_final_on_scaled_features_list_3.pkl"  # your trained model/pipeline
FEATURES_PATH = "models/feature_names.json"                      # optional: exact training feature order
FEATURE_ORDER_DEFAULT = [
    "open", "close", "hour_high", "hour_low", "marketCap", "month", "day", "hour_open"
]

# -------------------------------------------------------------
# Robust HTTP session (retries/backoff) + simple caches
# -------------------------------------------------------------
_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=1.5,  # 0s, 1.5s, 3.0s...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.headers.update({"User-Agent": "eth-high-predictor/1.0"})

# Market cap cache (10 min TTL)
_MC_CACHE = {"usd": None, "ts": 0.0}
_MC_TTL = 600

# Feature bundle cache (1 min TTL) to throttle upstream calls
_FEATURES_CACHE = {"data": None, "ts": 0.0}
_FEATURES_TTL = 60

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
# UTC day helpers
# -------------------------------------------------------------
def utc_midnight(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

def utc_today_midnight() -> datetime:
    return utc_midnight(datetime.now(timezone.utc))

def utc_yesterday_midnight() -> datetime:
    return utc_today_midnight() - timedelta(days=1)

# -------------------------------------------------------------
# Public data fetchers (no keys)
# -------------------------------------------------------------
def kraken_eth_hourly_since(since_dt_utc: datetime) -> pd.DataFrame:
    """
    Pull ETH/USD 60-min candles from Kraken since 'since_dt_utc'.
    Returns columns: time, open, high, low, close, volume
    """
    since_unix = int(since_dt_utc.timestamp())
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "ETHUSD", "interval": 60, "since": since_unix}
    r = _session.get(url, params=params, timeout=30)
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

def fallback_marketcap_eth_usd() -> float | None:
    """
    Fallback provider (CoinCap) if CoinGecko is rate-limited & no cache.
    """
    try:
        r = _session.get("https://api.coincap.io/v2/assets/ethereum", timeout=15)
        r.raise_for_status()
        data = r.json().get("data", {})
        mc = data.get("marketCapUsd")
        return float(mc) if mc is not None else None
    except Exception:
        return None

def coingecko_eth_marketcap_usd() -> float:
    """
    Fetch current Ethereum market cap in USD from CoinGecko with retry, cache, and fallback.
    - Serves cached value if fresh (<= _MC_TTL).
    - On 429/5xx returns last cached value when available.
    - If no cache, tries fallback provider before raising.
    """
    now = time.time()
    if (now - _MC_CACHE["ts"] <= _MC_TTL) and (_MC_CACHE["usd"] is not None):
        return _MC_CACHE["usd"]

    url = "https://api.coingecko.com/api/v3/coins/ethereum"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }

    r = _session.get(url, params=params, timeout=15)

    # If rate-limited and we have a cache, use it
    if r.status_code == 429 and _MC_CACHE["usd"] is not None:
        return _MC_CACHE["usd"]

    try:
        r.raise_for_status()
        js = r.json()
        mc = js.get("market_data", {}).get("market_cap", {}).get("usd", None)
        if mc is None:
            # Try fallback or cached value
            alt = fallback_marketcap_eth_usd()
            if alt is not None:
                _MC_CACHE.update({"usd": alt, "ts": now})
                return alt
            if _MC_CACHE["usd"] is not None:
                return _MC_CACHE["usd"]
            raise RuntimeError("CoinGecko: market cap USD not found")
        value = float(mc)
        _MC_CACHE.update({"usd": value, "ts": now})
        return value
    except requests.RequestException:
        # Network/HTTP error: try fallback, else cached, else raise
        alt = fallback_marketcap_eth_usd()
        if alt is not None:
            _MC_CACHE.update({"usd": alt, "ts": now})
            return alt
        if _MC_CACHE["usd"] is not None:
            return _MC_CACHE["usd"]
        raise

# -------------------------------------------------------------
# Build features for the LATEST COMPLETED UTC DAY (D)
# -------------------------------------------------------------
def build_eth_features_for_completed_day():
    """
    Uses the last fully completed UTC day (yesterday 00:00..today 00:00) as D.
    Returns: (features_dict, feature_day_start_utc[D], prediction_day_start_utc[D+1])
    """
    # Serve cached bundle if fresh
    now_ts = time.time()
    cache = _FEATURES_CACHE
    if cache["data"] is not None and (now_ts - cache["ts"] <= _FEATURES_TTL):
        day_start = cache["data"]["__meta"]["feature_day_start_utc"]
        return {k: v for k, v in cache["data"].items() if k != "__meta__"}, day_start, day_start + timedelta(days=1)

    day_start = utc_yesterday_midnight()
    day_end = utc_today_midnight()

    # Pull from Kraken and filter to the exact window [D, D+1)
    hourly_all = kraken_eth_hourly_since(day_start)
    hourly = hourly_all[(hourly_all["time"] >= day_start) & (hourly_all["time"] < day_end)].copy()

    defaults = {
        "open": 3000.0,
        "close": 3000.0,
        "hour_high": 12,
        "hour_low": 3,
        "marketCap": coingecko_eth_marketcap_usd(),
        "month": day_start.month,
        "day": day_start.day,
        "hour_open": 0,
    }

    if hourly.empty:
        bundle = {**defaults, "__meta__": {"feature_day_start_utc": day_start}}
        _FEATURES_CACHE.update({"data": bundle, "ts": now_ts})
        return defaults, day_start, day_start + timedelta(days=1)

    idx_high = hourly["high"].idxmax()
    idx_low = hourly["low"].idxmin()
    hour_high = int(hourly.loc[idx_high, "time"].hour)
    hour_low = int(hourly.loc[idx_low, "time"].hour)

    first_row = hourly.iloc[0]
    todays_open = float(first_row["open"])
    hour_open = int(first_row["time"].hour)

    latest_close = float(hourly.iloc[-1]["close"])

    try:
        mc = coingecko_eth_marketcap_usd()
    except Exception:
        mc = _MC_CACHE["usd"] if _MC_CACHE["usd"] is not None else 4.0e11

    result = {
        "open": todays_open,
        "close": latest_close,
        "hour_high": hour_high,
        "hour_low": hour_low,
        "marketCap": mc,
        "month": day_start.month,
        "day": day_start.day,
        "hour_open": hour_open,
        "__meta__": {"feature_day_start_utc": day_start},
    }
    _FEATURES_CACHE.update({"data": result, "ts": now_ts})
    return {k: v for k, v in result.items() if k != "__meta__"}, day_start, day_start + timedelta(days=1)

# -------------------------------------------------------------
# Feature builder (NO transforms, just exact order)
# -------------------------------------------------------------
def to_model_dataframe(raw: dict) -> pd.DataFrame:
    """
    raw keys required: open, close, hour_high, hour_low, marketCap, month, day, hour_open
    No transforms; order must match training.
    """
    required = ["open", "close", "hour_high", "hour_low", "marketCap", "month", "day", "hour_open"]
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
        "Predicts next-day HIGH (t+1) for ETH using features from the latest COMPLETED UTC day (D).\n"
        "GET builds features for D (yesterday UTC) and predicts for D+1. POST accepts features for a given day D and predicts for D+1.\n"
        "All times are UTC to avoid leakage."
    ),
    version="1.5.0",
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
            "/predict/{token}": "GET – auto-fetch features for latest completed UTC day & predict next day",
            "/predict/{token} (POST)": "POST – provide features (for day D) & predicts for day D+1",
        },
        "feature_order_used": FEATURE_ORDER if FEATURE_ORDER else FEATURE_ORDER_DEFAULT,
        "model_artifacts": {"model_path": MODEL_PATH, "feature_order_file": FEATURES_PATH},
        "expected_post_body_example": {
            "open": 3400.56, "close": 3432.11, "hour_high": 15, "hour_low": 3,
            "marketCap": 4.2e11, "month": 11, "day": 1, "hour_open": 0
        },
        "note": "GET always uses the last completed UTC day (D) and predicts for D+1.",
    }

@app.get("/health/", tags=["info"])
def health():
    status = model is not None
    return JSONResponse(
        status_code=200 if status else 503,
        content={"status": "ok" if status else "model_not_loaded"}
    )

# ---------- GET: ALWAYS use latest completed day D, predict D+1 ----------
@app.get("/predict/{token}", tags=["prediction"])
def predict_get(token: str = Path(..., description="Must be 'ethereum'")):
    if token.lower() not in ALLOWED_TOKENS:
        raise HTTPException(status_code=400, detail=f"Unsupported token '{token}'. Allowed: {list(ALLOWED_TOKENS)}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    try:
        raw, feature_day_start, predict_day_start = build_eth_features_for_completed_day()
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
        "feature_day_utc": feature_day_start.date().isoformat(),          # D
        "prediction_for_day_utc": (predict_day_start.date().isoformat()), # D+1
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_used": raw,
        "n_features": int(X.shape[1]),
        "feature_order_used": list(X.columns),
    }

# ---------- POST: user-supplied features for day D -> predict D+1 ----------
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

    # Infer feature day D from (month, day) using current UTC year (best effort)
    year_utc = datetime.now(timezone.utc).year
    try:
        feature_day = datetime(year_utc, int(raw["month"]), int(raw["day"]), tzinfo=timezone.utc)
        predict_day = feature_day + timedelta(days=1)
        feature_day_iso = feature_day.date().isoformat()
        predict_day_iso = predict_day.date().isoformat()
    except Exception:
        feature_day_iso = None
        predict_day_iso = None

    return {
        "token": token.lower(),
        "predicted_next_day_high": yhat,
        "units": "USD",
        "feature_day_utc": feature_day_iso,          # D
        "prediction_for_day_utc": predict_day_iso,   # D+1
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features_used": raw,
        "n_features": int(X.shape[1]),
        "feature_order_used": list(X.columns),
    }
