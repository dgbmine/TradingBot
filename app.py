import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Wyckoff State Machine")

# ----------------------------
# 1. DATA ENGINE (SAFE)
# ----------------------------
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")

        if df is None or df.empty:
            return None

        df = df.copy()

        # indicators
        df["SMA50"] = df["Close"].rolling(50).mean()

        df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
        df["VOL_STD"] = df["Volume"].rolling(20).std()
        df["VOL_STD"] = df["VOL_STD"].replace(0, np.nan)

        df["VOL_Z"] = (df["Volume"] - df["VOL_MEAN"]) / (df["VOL_STD"] + 1e-9)

        df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ATR_NORM"] = df["ATR"] / (df["Close"] + 1e-9)

        df = df.dropna()

        if len(df) < 60:
            return None

        return df

    except Exception as e:
        st.error(f"Data error: {e}")
        return None


# ----------------------------
# 2. PROBABILITY ENGINE (SAFE)
# ----------------------------
def get_accumulation_prob(df, support, resistance):
    try:
        last = df.iloc[-1]

        range_den = max(resistance - support, 1e-9)

        # 1. position in range
        pos = (last["Close"] - support) / range_den
        pos = np.clip(pos, 0, 1)
        f1 = 1 - pos

        # 2. volume pressure
        vol_mean = df["Volume"].rolling(20).mean().iloc[-1]
        vol_mean = vol_mean if not np.isnan(vol_mean) else last["Volume"]

        f2 = np.clip((last["Volume"] / (vol_mean + 1e-9)) / 2, 0, 1)

        # 3. compression
        atr_mean = df["ATR_NORM"].rolling(50).mean().iloc[-1]
        atr_mean = atr_mean if not np.isnan(atr_mean) else last["ATR_NORM"]

        f3 = np.clip(1 - (last["ATR_NORM"] / (atr_mean + 1e-9)), 0, 1)

        # 4. trend weakness
        sma = last["SMA50"]
        if np.isnan(sma):
            f4 = 0.5
        else:
            f4 = np.clip(1 - (last["Close"] / (sma + 1e-9) - 1), 0, 1)

        # weighted model
        raw = (0.35 * f1) + (0.25 * f2) + (0.25 * f3) + (0.15 * f4)

        # stable sigmoid
        prob = 1 / (1 + np.exp(-(raw - 0.5) * 5))

        return float(np.clip(prob * 100, 0, 100))

    except Exception as e:
        st.error(f"Model error: {e}")
        return 0.0


# ----------------------------
# 3. UI
# ----------------------------
ticker = st.text_input("Enter Ticker", "NVDA").upper()

st.markdown("### Wyckoff Dashboard")

if st.button("Run Analysis"):

    df = get_data(ticker)

    if df is None:
        st.error("❌ אין דאטה מספיק או שהטיקר לא תקין")
        st.stop()

    # safe support/resistance
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    if np.isnan(support) or np.isnan(resistance):
        st.error("❌ לא ניתן לחשב support/resistance")
        st.stop()

    prob = get_accumulation_prob(df, support, resistance)

    st.write("### Debug Info")
    st.write(f"Rows: {len(df)}")
    st.write(f"Support: {support:.2f}")
    st.write(f"Resistance: {resistance:.2f}")
    st.write(f"Probability: {prob:.2f}%")

    # ----------------------------
    # GAUGE
    # ----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Accumulation Probability (%)"},
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

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # SIGNAL TEXT
    # ----------------------------
    if prob > 70:
        st.success("💎 איסוף מוסדי אפשרי (High Probability)")
    elif prob > 40:
        st.warning("⚠️ קונסולידציה / אי ודאות")
    else:
        st.error("❌ אין סימני איסוף")