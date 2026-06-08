import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Efi Quant Engine V4.2")

# ─────────────────────────────
# DATA LAYER
# ─────────────────────────────
@st.cache_data(ttl=300)
def get_quant_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")

    if df.empty:
        return df

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # ATR (true range)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    return df


# ─────────────────────────────
# STRUCTURE ENGINE (FIXED + STABLE)
# ─────────────────────────────
def get_structure(df, window=5):
    """
    Non-repainting swing structure engine
    """

    highs = df["High"]
    lows = df["Low"]

    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(df) - window):
        h = highs.iloc[i]
        l = lows.iloc[i]

        if h == highs.iloc[i-window:i+window+1].max():
            pivot_highs.append((df.index[i], h))

        if l == lows.iloc[i-window:i+window+1].min():
            pivot_lows.append((df.index[i], l))

    resistance = pivot_highs[-1][1] if len(pivot_highs) > 0 else highs.max()
    support = pivot_lows[-1][1] if len(pivot_lows) > 0 else lows.min()

    return support, resistance, pivot_highs, pivot_lows


# ─────────────────────────────
# REGIME CLASSIFIER (clean + normalized)
# ─────────────────────────────
def analyze_regime(df):
    price = df["Close"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]

    if np.isnan(sma50) or np.isnan(sma200):
        return "Insufficient Data"

    slope = (df["SMA50"].iloc[-1] - df["SMA50"].iloc[-10]) / df["SMA50"].iloc[-10]
    slope = slope * 100

    if price > sma200 and slope > 0.5:
        return "Markup (Bull)"
    elif price < sma200 and slope < -0.5:
        return "Markdown (Bear)"
    elif abs(slope) < 0.3:
        return "Accumulation / Range"
    else:
        return "Transition"


# ─────────────────────────────
# AGREEMENT MODEL (real scoring)
# ─────────────────────────────
def decision_logic(df, support, resistance):
    price = df["Close"].iloc[-1]

    # Trend agreement
    trend = 0
    if price > df["SMA50"].iloc[-1]:
        trend += 1
    if price > df["SMA200"].iloc[-1]:
        trend += 1

    trend_score = trend / 2  # 0–1

    # Volume agreement
    vol_ratio = df["Volume"].iloc[-1] / df["VOL_MA20"].iloc[-1]
    vol_score = min(vol_ratio / 1.5, 1)  # capped normalization

    # Structure agreement
    if price > resistance:
        structure_score = 1
    elif price < support:
        structure_score = 0
    else:
        structure_score = 0.5

    # Final agreement (weighted, not fake equal average)
    agreement = (
        trend_score * 0.4 +
        vol_score * 0.3 +
        structure_score * 0.3
    )

    if agreement > 0.75:
        signal = "LONG"
    elif agreement < 0.35:
        signal = "RISK / SHORT BIAS"
    else:
        signal = "NO EDGE"

    return signal, agreement, vol_ratio


# ─────────────────────────────
# UI
# ─────────────────────────────
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Quant Analysis"):
    df = get_quant_data(ticker)

    if df.empty:
        st.error("No data found")
        st.stop()

    support, resistance, piv_high, piv_low = get_structure(df)
    regime = analyze_regime(df)
    signal, conf, vol_ratio = decision_logic(df, support, resistance)

    col1, col2, col3 = st.columns(3)

    col1.metric("Market Regime", regime)
    col2.metric("Signal", signal)
    col3.metric("Agreement Score", f"{conf*100:.1f}%")

    st.divider()

    st.write("### Market Structure")
    st.write(f"- Support: {support:.2f}")
    st.write(f"- Resistance: {resistance:.2f}")
    st.write(f"- Volume Ratio: {vol_ratio:.2f}")

    st.write(f"- Pivot Highs: {len(piv_high)}")
    st.write(f"- Pivot Lows: {len(piv_low)}")

    st.divider()

    if signal == "LONG":
        st.success("Trend + Volume + Structure aligned → bullish regime")
    elif signal == "RISK / SHORT BIAS":
        st.error("Weak structure / distribution conditions")
    else:
        st.warning("No statistical edge detected")