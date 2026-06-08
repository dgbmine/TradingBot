import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Efi Quant Engine V4.2")

# ─────────────────────────────
# DATA & LOGIC LAYER
# ─────────────────────────────
@st.cache_data(ttl=300)
def get_quant_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty: return df
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    return df

def get_structure(df, window=5):
    highs, lows = df["High"], df["Low"]
    pivot_highs, pivot_lows = [], []
    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            pivot_highs.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            pivot_lows.append(lows.iloc[i])
    res = pivot_highs[-1] if pivot_highs else highs.max()
    sup = pivot_lows[-1] if pivot_lows else lows.min()
    return sup, res

def analyze_regime(df):
    price, sma50, sma200 = df["Close"].iloc[-1], df["SMA50"].iloc[-1], df["SMA200"].iloc[-1]
    slope = ((sma50 - df["SMA50"].iloc[-10]) / df["SMA50"].iloc[-10]) * 100
    if price > sma200 and slope > 0.5: return "Markup (Bull)"
    if price < sma200 and slope < -0.5: return "Markdown (Bear)"
    if abs(slope) < 0.3: return "Accumulation / Range"
    return "Transition"

def decision_logic(df, support, resistance):
    price = df["Close"].iloc[-1]
    trend = (1 if price > df["SMA50"].iloc[-1] else 0) + (1 if price > df["SMA200"].iloc[-1] else 0)
    vol_ratio = df["Volume"].iloc[-1] / df["VOL_MA20"].iloc[-1]
    vol_score = min(vol_ratio / 1.5, 1)
    struct_score = 1 if price > resistance else (0 if price < support else 0.5)
    agreement = (trend/2 * 0.4) + (vol_score * 0.3) + (struct_score * 0.3)
    
    if agreement > 0.75: signal = "LONG"
    elif agreement < 0.35: signal = "RISK / SHORT BIAS"
    else: signal = "NO EDGE"
    return signal, agreement

# ─────────────────────────────
# UI LAYER
# ─────────────────────────────
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Quant Analysis"):
    df = get_quant_data(ticker)
    if df.empty:
        st.error("טיקר לא נמצא, בדוק שוב")
    else:
        sup, res = get_structure(df)
        regime = analyze_regime(df)
        signal, conf = decision_logic(df, sup, res)
        
        # גרף
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(color='yellow')))
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # תובנות בעברית
        st.write(f"### מצב השוק: {regime}")
        st.write(f"**החלטה:** {signal} (ציון הסכמה: {conf*100:.0f}%)")
        
        if signal == "LONG":
            st.success("זהו שלב ה-Markup. המניה במגמה חיובית, הווליום תומך. אפשר לחפש כניסה.")
        elif signal == "RISK / SHORT BIAS":
            st.error("זהירות! המניה מתחת לרמות מפתח. תמתין ליציבות או תחפש מניה אחרת.")
        else:
            st.warning("המניה בדשדוש (Accumulation). אין יתרון סטטיסטי כרגע. תמתין לפריצה.")
