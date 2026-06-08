import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ... (פונקציות ה-Data, Structure, Regime, Logic נשארות כפי שכתבת) ...

def get_wyckoff_phase(regime, agreement, price, support, resistance):
    """מתרגם נתונים לשלב וייקוף והנחיה"""
    if regime == "Accumulation / Range":
        phase = "שלב B/C (איסוף)"
        action = "המתנה לפריצה עם ווליום או ל-Spring. לא לקנות בתוך הטווח."
    elif regime == "Markup (Bull)":
        phase = "שלב D/E (עליות)"
        action = "אפשרות לכניסה בתיקונים (Pullbacks) ל-SMA50."
    elif regime == "Distribution":
        phase = "שלב D (פיזור)"
        action = "זהירות: הגיע הזמן לממש רווחים. אל תפתח פוזיציות לונג."
    else:
        phase = "לא ברור"
        action = "חפש מניה אחרת עם מבנה ברור יותר."
    
    return phase, action

# ─────────────────────────────
# UI משודרג עם גרפים והסברים
# ─────────────────────────────
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Quant Analysis"):
    df = get_quant_data(ticker)
    support, resistance, piv_high, piv_low = get_structure(df)
    regime = analyze_regime(df)
    signal, conf, vol_ratio = decision_logic(df, support, resistance)
    
    # 1. גרף מוסדי
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(color='yellow')))
    fig.update_layout(template="plotly_dark", height=500, title=f"Chart for {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    # 2. ניתוח וייקוף והנחיות
    phase, action = get_wyckoff_phase(regime, conf, df['Close'].iloc[-1], support, resistance)
    
    st.write(f"### ניתוח וייקוף: {phase}")
    st.info(f"**מה לעשות:** {action}")
    
    # 3. החלטה סופית
    if signal == "LONG" and conf > 0.7:
        st.success("🟢 זמן קנייה: המערכת מסונכרנת (Trend+Vol+Structure).")
    elif signal == "RISK / SHORT BIAS":
        st.error("🔴 זמן למכור/להימנע: המניה בלחץ פיזור או מגמה שלילית.")
    else:
        st.warning(f"⚪ להמתין: מחכים לאישור (Agreement Score: {int(conf*100)}%).")
