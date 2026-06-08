import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Efi Family Office Engine")

@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")

    # בסיס
    df["SMA50"] = df["Close"].rolling(50).mean()

    # שיפור ווליום - Z SCORE
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["VOL_Z"] = (df["Volume"] - vol_mean) / vol_std

    # מומנטום של ממוצע (שיפוע)
    df["SMA50_SLOPE"] = df["SMA50"].diff(5)

    return df


def find_pivots(df, left=3, right=3):
    """זיהוי פיבוטים אמיתי (Wyckoff structure בסיסי)"""
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        # pivot high
        if highs[i] == max(highs[i-left:i+right+1]):
            pivot_highs.append(highs[i])

        # pivot low
        if lows[i] == min(lows[i-left:i+right+1]):
            pivot_lows.append(lows[i])

    return pivot_highs, pivot_lows


def get_transparency_report(df, support, resistance):
    """הסבר בשפה פשוטה למה אנחנו ממתינים"""
    curr = df.iloc[-1]

    report = []

    # 1. טרנד משופר (מחיר + שיפוע)
    if curr["Close"] > df["SMA50"].iloc[-1] and curr["SMA50_SLOPE"] > 0:
        report.append("✅ המחיר מעל ממוצע 50 - סימן חיובי לטווח קצר.")
        trend_score = 1
    else:
        report.append("❌ המחיר מתחת לממוצע 50 - אין מומנטום קונים.")
        trend_score = 0

    # 2. ווליום חכם (Z-score)
    if curr["VOL_Z"] > 1:
        report.append("✅ ווליום גבוה מהממוצע - יש עניין בשוק.")
        vol_score = 1
    else:
        report.append("❌ ווליום נמוך - השוק 'ישנוני', אין הוכחה לכניסת כסף מוסדי.")
        vol_score = 0

    # 3. מבנה שוק מבוסס פיבוטים
    if curr["Close"] > resistance:
        report.append("✅ פריצת רמת התנגדות - המבנה נפרץ כלפי מעלה.")
        struct_score = 1
    else:
        report.append("⚠️ מחיר בתוך טווח - אנחנו תקועים בין קונים למוכרים. תמתין לפריצה.")
        struct_score = 0.5

    # חישוב ציון משוקלל ל-Gauge
    final_score = (trend_score * 0.4) + (vol_score * 0.3) + (struct_score * 0.3)
    
    return report, final_score


ticker = st.text_input("Enter Ticker", "NVDA").upper()

# כפתור מידע על השיטה
if st.button("ℹ️ איך האפליקציה עובדת?"):
    st.info("""
    ### השיטה שלנו: Institutional Market Structure
    האפליקציה משתמשת בשילוב של שלוש שכבות ניתוח מוסדיות:
    1. **Trend Analysis (וייקוף):** זיהוי מגמה לפי ממוצעים נעים ושיפועים. המטרה היא להיות בצד הנכון של השוק.
    2. **Smart Volume (VSA):** שימוש ב-Z-Score כדי לבדוק האם הווליום חריג. ווליום גבוה בפריצה מעיד על כניסת כסף גדול (מוסדי).
    3. **Liquidity Structure:** פיבוטים המגדירים תמיכה והתנגדות. אנחנו לא קונים בתוך הטווח, אלא רק בפריצות אישור.
    **המטרה:** להוריד סיכון על ידי המתנה ל'אישור' (Confirmation) של שלושת הפרמטרים במקביל.
    """)

if st.button("Generate Investment Brief"):
    df = get_data(ticker)

    # פיבוטים אמיתיים
    piv_highs, piv_lows = find_pivots(df)

    # fallback אם אין מספיק נתונים
    resistance = max(piv_highs[-5:]) if len(piv_highs) >= 5 else df["High"].rolling(20).max().iloc[-1]
    support = min(piv_lows[-5:]) if len(piv_lows) >= 5 else df["Low"].rolling(20).min().iloc[-1]

    st.write(f"### 📊 תדריך משימה עבור {ticker}")

    report, score = get_transparency_report(df, support, resistance)

    # הצגת שעון המחוגים (Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "darkblue"},
                 'steps' : [{'range': [0, 35], 'color': "red"},
                            {'range': [35, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "green"}]},
        title = {'text': "מדד הסכמה ללונג"}))
    st.plotly_chart(fig_gauge)

    for item in report:
        st.write(item)

    st.divider()

    st.write("### למה אנחנו ממתינים?")
    st.info(
        "כדי להוריד סיכון בניהול התיק, אנחנו לא מחפשים 'ניחוש' של הכיוון. "
        "אנחנו מחפשים **'אישור' (Confirmation)**. כרגע, חסר לנו האישור של הווליום ושל פריצת ההתנגדות. "
        "ברגע שהם יופיעו – הציון יעלה ל-LONG."
    )
