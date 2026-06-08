import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Efi Family Office Engine")

# ----------------------------
# DATA
# ----------------------------
@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")

    df["SMA50"] = df["Close"].rolling(50).mean()

    # robust volume z-score (less sensitive to outliers)
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["VOL_Z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)

    # smoother trend strength (normalized slope)
    df["SMA50_SLOPE"] = df["SMA50"].diff(5) / (df["Close"] + 1e-9)

    # volatility compression (Wyckoff-like idea)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_RATIO"] = df["ATR"] / (df["Close"] + 1e-9)

    df = df.dropna()
    return df


# ----------------------------
# PIVOTS (more robust)
# ----------------------------
def find_pivots(df, left=3, right=3):
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        window_high = highs[i-left:i+right+1]
        window_low = lows[i-left:i+right+1]

        # use tolerance instead of exact equality
        if highs[i] >= np.max(window_high) * 0.999:
            pivot_highs.append(highs[i])

        if lows[i] <= np.min(window_low) * 1.001:
            pivot_lows.append(lows[i])

    return pivot_highs, pivot_lows


# ----------------------------
# SUPPORT / RESISTANCE (clustered)
# ----------------------------
def cluster_levels(levels, tolerance=0.015):
    if len(levels) == 0:
        return []

    levels = sorted(levels)
    clusters = [[levels[0]]]

    for lvl in levels[1:]:
        if abs(lvl - np.mean(clusters[-1])) / np.mean(clusters[-1]) < tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    # return cluster means
    return [np.mean(c) for c in clusters]


# ----------------------------
# CORE SCORING (improved Wyckoff-like logic)
# ----------------------------
def get_transparency_report(df, support, resistance):

    curr = df.iloc[-1]
    report = []

    score = 0.0

    # 1. Trend component
    trend = 0
    if curr["Close"] > curr["SMA50"] and curr["SMA50_SLOPE"] > 0:
        report.append("✅ מגמה חיובית מעל SMA50 עם שיפוע עולה.")
        trend = 1
    else:
        report.append("❌ אין מגמת עלייה ברורה מעל SMA50.")

    # 2. Volume (robust interpretation)
    vol = 0
    if curr["VOL_Z"] > 1.2:
        report.append("✅ ווליום חריג - אפשרות לפעילות מוסדית.")
        vol = 1
    elif curr["VOL_Z"] > 0.3:
        report.append("⚠️ ווליום בינוני - עניין חלקי.")
        vol = 0.5
    else:
        report.append("❌ ווליום חלש - אין הוכחת ביקוש משמעותי.")

    # 3. Structure (breakout / accumulation zone)
    struct = 0.5

    if curr["Close"] > resistance:
        report.append("🚀 פריצה מעל התנגדות - יציאה מטווח צבירה אפשרי.")
        struct = 1
    elif curr["Close"] < support:
        report.append("⚠️ שבירה מתחת תמיכה - מבנה חלש.")
        struct = 0
    else:
        report.append("📦 מחיר בתוך טווח - אפשרות לצבירה (Wyckoff range).")

        # Wyckoff-like compression signal
        if df["ATR_RATIO"].iloc[-1] < df["ATR_RATIO"].rolling(50).mean().iloc[-1]:
            report.append("🧊 תנודתיות יורדת - קומפרסיה (שלב צבירה פוטנציאלי).")
            struct = 0.7

    # FINAL SCORE (less arbitrary weights, more balanced)
    score = (trend * 0.4) + (vol * 0.3) + (struct * 0.3)

    return report, score


# ----------------------------
# APP
# ----------------------------
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("ℹ️ איך האפליקציה עובדת?"):
    st.info("השיטה משלבת Trend, Volume, ו-Structure לזיהוי מצב שוק בסגנון Wyckoff.")

if st.button("Generate Investment Brief"):

    df = get_data(ticker)
    piv_highs, piv_lows = find_pivots(df)

    # clustered levels instead of raw extremes
    resistance_levels = cluster_levels(piv_highs)
    support_levels = cluster_levels(piv_lows)

    resistance = resistance_levels[-1] if len(resistance_levels) > 0 else df["High"].max()
    support = support_levels[-1] if len(support_levels) > 0 else df["Low"].min()

    st.write(f"### 📊 תדריך משימה עבור {ticker}")

    report, score = get_transparency_report(df, support, resistance)

    st.markdown("### האם הנייר באיסוף?")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "red"},
                {'range': [35, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ]
        },
        title={'text': "ציון איסוף מוסדי (%)"}
    ))

    st.plotly_chart(fig_gauge)

    for item in report:
        st.write(item)