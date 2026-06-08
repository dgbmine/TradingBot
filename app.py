import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Efi Family Office Engine")

@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")

    df["SMA50"] = df["Close"].rolling(50).mean()

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()

    df["VOL_Z"] = (df["Volume"] - vol_mean) / vol_std
    df["SMA50_SLOPE"] = df["SMA50"].diff(5)

    # חשוב: מניעת NaN שמפיל לוגיקה
    df = df.dropna()

    return df


def find_pivots(df, left=3, right=3):
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        window_high = highs[i-left:i+right+1]
        window_low = lows[i-left:i+right+1]

        if highs[i] == np.max(window_high):
            pivot_highs.append(highs[i])

        if lows[i] == np.min(window_low):
            pivot_lows.append(lows[i])

    return pivot_highs, pivot_lows


def get_transparency_report(df, support, resistance):
    curr = df.iloc[-1]
    report = []

    trend_score = 0
    vol_score = 0
    struct_score = 0.5

    if curr["Close"] > df["SMA50"].iloc[-1] and curr["SMA50_SLOPE"] > 0:
        report.append("✅ המחיר מעל ממוצע 50 - סימן חיובי לטווח קצר.")
        trend_score = 1
    else:
        report.append("❌ המחיר מתחת לממוצע 50 - אין מומנטום קונים.")

    if curr["VOL_Z"] > 1:
        report.append("✅ ווליום גבוה מהממוצע - יש עניין בשוק.")
        vol_score = 1
    else:
        report.append("❌ ווליום נמוך - השוק 'ישנוני', אין הוכחה לכניסת כסף מוסדי.")

    if curr["Close"] > resistance:
        report.append("✅ פריצת רמת התנגדות - המבנה נפרץ כלפי מעלה.")
        struct_score = 1
    else:
        report.append("⚠️ מחיר בתוך טווח - אנחנו תקועים בין קונים למוכרים. תמתין לפריצה.")

    final_score = (trend_score * 0.4) + (vol_score * 0.3) + (struct_score * 0.3)

    return report, final_score


ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("ℹ️ איך האפליקציה עובדת?"):
    st.info("השיטה משתמשת בשלוש שכבות ניתוח מוסדיות: Trend, Smart Volume, ו-Liquidity Structure.")

if st.button("Generate Investment Brief"):
    df = get_data(ticker)
    piv_highs, piv_lows = find_pivots(df)

    # הגנות קריטיות (זה מה ששובר אצלך לרוב)
    if len(piv_highs) >= 5:
        resistance = float(np.max(piv_highs[-5:]))
    else:
        resistance = float(df["High"].max())

    if len(piv_lows) >= 5:
        support = float(np.min(piv_lows[-5:]))