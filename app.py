import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def safe_get(series, default=0.0):
    try:
        val = series.iloc[-1]
        if np.isnan(val):
            return default
        return val
    except:
        return default

def get_accumulation_prob(df, support, resistance):

    # ----------------------------
    # HARD SAFETY
    # ----------------------------
    if df is None or len(df) < 50:
        return 0.0

    required_cols = ["Close", "Volume", "SMA50"]
    for col in required_cols:
        if col not in df.columns:
            return 0.0

    last = df.iloc[-1]

    # prevent division crash
    range_den = (resistance - support)
    if abs(range_den) < 1e-9:
        range_den = 1e-9

    features = []

    # ----------------------------
    # 1. Position in range
    # ----------------------------
    try:
        pos = (last["Close"] - support) / range_den
        pos = np.clip(pos, 0, 1)
        features.append(1 - pos)
    except:
        features.append(0.5)

    # ----------------------------
    # 2. Volume pressure
    # ----------------------------
    try:
        vol_mean = safe_get(df["Volume"].rolling(20).mean(), last["Volume"])
        vol = last["Volume"] / (vol_mean + 1e-9)
        features.append(np.clip(vol / 2, 0, 1))
    except:
        features.append(0.5)

    # ----------------------------
    # 3. Compression (ATR)
    # ----------------------------
    try:
        if "ATR_NORM" in df.columns:
            atr_mean = safe_get(df["ATR_NORM"].rolling(50).mean(), 0.01)
            compression = 1 - (last["ATR_NORM"] / (atr_mean + 1e-9))
            features.append(np.clip(compression, 0, 1))
        else:
            features.append(0.5)
    except:
        features.append(0.5)

    # ----------------------------
    # 4. Trend weakness
    # ----------------------------
    try:
        trend = last["Close"] / (last["SMA50"] + 1e-9)
        trend_score = 1 - np.clip(trend - 1, 0, 1)
        features.append(np.clip(trend_score, 0, 1))
    except:
        features.append(0.5)

    # ----------------------------
    # CALCULATION
    # ----------------------------
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    raw_score = np.dot(weights, features)

    # convert to probability space
    prob = sigmoid((raw_score - 0.5) * 6)

    return float(prob * 100)
