import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Wyckoff State Machine Engine")

# ----------------------------
# 1. DATA ENGINE (כולל חישובי ATR ו-SMA)
# ----------------------------
@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty: return None
    
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    
    # Volatility & ATR
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_NORM"] = df["ATR"] / (df["Close"] + 1e-9)
    
    # Volume Z-Score
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["VOL_Z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
    
    return df.dropna()

# ----------------------------
# 2. PROBABILITY ENGINE (Sigmoid Logic)
# ----------------------------
def get_accumulation_prob(df, support, resistance):
    try:
        last = df.iloc[-1]
        range_den = max(resistance - support, 1e-9)
        
        # Features
        pos = np.clip((last["Close"] - support) / range_den, 0, 1)
        f1 = 1 - pos # קרוב לתמיכה = טוב
        
        vol = np.clip(last["Volume"] / (df["Volume"].rolling(20).mean().iloc[-1] + 1e-9) / 2, 0, 1)
        f2 = vol
        
        atr_mean = df["ATR_NORM"].rolling(50).mean().iloc[-1]
        f3 = np.clip(1 - (last["ATR_NORM"] / (atr_mean + 1e-9)), 0, 1)
        
        trend = np.clip(1 - (last["Close"] / (last["SMA50"] + 1e-9) - 1), 0, 1)
        f4 = trend
        
        features = np.array([f1, f2, f3, f4])
        weights = np.array([0.35, 0.25, 0.25, 0.15])
        
        raw_score = np.dot(weights, features)
        prob = 1 / (1 + np.exp(-((raw_score - 0.5) * 6)))
        return float(prob * 100)
    except Exception as e:
        return 0.0

# ----------------------------
# 3. UI ENGINE
# ----------------------------
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Wyckoff Analysis"):
    df = get_data(ticker)
    
    if df is None:
        st.error("לא נמצאו נתונים עבור הטיקר הזה.")
    else:
        # פשטתי את חישוב התמיכה/התנגדות כדי למנוע קריסות
        support = df["Low"].rolling(20).min().iloc[-1]
        resistance = df["High"].rolling(20).max().iloc[-1]
        
        prob = get_accumulation_prob(df, support, resistance)
        
        st.markdown("## Wyckoff Accumulation Probability")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            gauge
