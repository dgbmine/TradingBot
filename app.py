import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_accumulation_prob(df, support, resistance):
    # הגנה מוחלטת: אם אין נתונים, תחזיר 0 ולא תקרוס
    if df is None or df.empty:
        return 0.0

    # וידוא עמודות קיימות
    required = ["Close", "Volume", "SMA50"]
    if not all(col in df.columns for col in required):
        return 0.0 # או הודעת שגיאה ללוגים

    last = df.iloc[-1]

    # חישוב טווח בטוח
    range_den = (resistance - support) if abs(resistance - support) > 1e-9 else 1e-9
    
    features = []

    # 1. מיקום בטווח
    pos = np.clip((last["Close"] - support) / range_den, 0, 1)
    features.append(1 - pos) 

    # 2. ווליום
    vol_mean = df["Volume"].rolling(20).mean().iloc[-1]
    vol = last["Volume"] / (vol_mean + 1e-9)
    features.append(np.clip(vol / 2, 0, 1))

    # 3. ATR (בדיקה אם העמודה קיימת)
    if "ATR_NORM" in df.columns:
        atr_mean = df["ATR_NORM"].rolling(50).mean().iloc[-1]
        compression = 1 - np.clip(last["ATR_NORM"] / (atr_mean + 1e-9), 0, 1)
        features.append(np.clip(compression, 0, 1))
    else:
        features.append(0.5)

    # 4. מומנטום
    trend = last["Close"] / (last["SMA50"] + 1e-9)
    trend_score = 1 - np.clip(trend - 1, 0, 1)
    features.append(np.clip(trend_score, 0, 1))

    # חישוב
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    raw_score = np.dot(weights, features)
    prob = sigmoid((raw_score - 0.5) * 6)

    return float(prob * 100)
