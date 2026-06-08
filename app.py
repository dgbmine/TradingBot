import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Wyckoff State Machine")

# ----------------------------
# 1. DATA ENGINE
# ----------------------------
@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty: return None
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
    df["VOL_STD"] = df["Volume"].rolling(20).std()
    df["VOL_Z"] = (df["Volume"] - df["VOL_MEAN"]) / (df["VOL_STD"] + 1e-9)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_NORM"] = df["ATR"] / (df["Close"] + 1e-9)
    return df.dropna()

# ----------------------------
# 2. PROBABILITY ENGINE
# ----------------------------
def get_accumulation_prob(df, support, resistance):
    try:
        last = df.iloc[-1]
        range_den = max(resistance - support, 1e-9)
        
        # חישוב פיצ'רים בצורה מפורקת ובטוחה
        pos = np.clip((last["Close"] - support) / range_den, 0, 1)
        f1 = 1 - pos
        
        vol_ratio = last["Volume"] / (df["VOL_MEAN"].iloc[-1] + 1e-9)
        f2 = np.clip(vol_ratio / 2, 0, 1)
        
        atr_avg = df["ATR_NORM"].rolling(50).mean().iloc[-1]
        f3 = np.clip(1 - (last["ATR_NORM"] / (atr_avg + 1e-9)), 0, 1)
        
        trend = np.clip(1 - (last["Close"] / (last["SMA50"] + 1e-9) - 1), 0, 1)
        f4 = trend
        
        # חישוב הסתברות
        raw_score = (0.35 * f1) + (0.25 * f2) + (0.25 * f3) + (0.15 * f4)
        prob = 1 / (1 + np.exp(-((raw_score - 0.5) * 6)))
        return float(prob * 100)
    except:
        return 0.0

# ----------------------------
# 3. UI
# ----------------------------
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Wyckoff Analysis"):
    df = get_data(ticker)
    if df is None:
        st.error("לא נמצאו נתונים")
    else:
        # שימוש בחישוב בטוח של רמות
        support = df["Low"].rolling(20).min().iloc[-1]
        resistance = df["High"].rolling(20).max().iloc[-1]
        
        prob = get_accumulation_prob(df, support, resistance)
        
        st.markdown("## Wyckoff State Machine")
        
        # בנייה מפורקת של הגרף כדי למנוע טעויות סוגריים
        gauge_val = prob
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            title={'text': "הסתברות לאיסוף (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig)
        
        if prob > 70: st.success("💎 איסוף מוסדי מזוהה")
        elif prob > 40: st.warning("⚠️ דשדוש רגיל")
        else: st.error("❌ לא זוהה איסוף")
