import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Efi Quant Engine V4.3")

# 1. ניהול דאטה (עם טיפול בשגיאות)
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if df.empty: return None
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["VOL_MA20"] = df["Volume"].rolling(20).mean()
        return df
    except: return None

# 2. ניתוח וייקוף עמוק
def get_wyckoff_insights(df):
    curr = df.iloc[-1]
    vol_ratio = curr["Volume"] / df["VOL_MA20"].iloc[-1]
    
    # זיהוי מצב
    if curr["Close"] > df["SMA200"].iloc[-1] and vol_ratio > 1.2:
        return "Markup (שלב D)", "המגמה חזקה. חפש כניסה בתיקונים (Pullbacks). אל תמכור עדיין.", 0.8
    elif curr["Close"] < df["SMA50"].iloc[-1] and vol_ratio > 1.5:
        return "Distribution (שלב C/D)", "סימני חולשה. המוסדיים מחלקים סחורה. צא מהפוזיציה.", 0.2
    else:
        return "Accumulation / Range", "דשדוש. אין כיוון ברור. המתנה לפריצה או ל-Spring.", 0.5

# 3. תצוגה
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Engine"):
    df = get_data(ticker)
    
    if df is None:
        st.error("לא ניתן לטעון נתונים. בדוק את הטיקר.")
    else:
        # גרף
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # ניתוח
        phase, action, score = get_wyckoff_insights(df)
        
        st.write(f"### שלב שוק: {phase}")
        st.write(f"**מה לעשות:** {action}")
        st.progress(score)
        
        if score > 0.7: st.success("סיכוי גבוה ללונג")
        elif score < 0.3: st.error("סיכון גבוה")
        else: st.warning("אין יתרון סטטיסטי (No Edge)")
