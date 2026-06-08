import numpy as np

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

    required_cols = ["Close", "Volume"]
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
        if "ATR_NORM