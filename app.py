import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Wyckoff Institutional Scout")

# ----------------------------
# 1. LOGIC & DATA
# ----------------------------
@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if len(df) < 100: return None
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    return df

def analyze_wyckoff(df):
    last = df.iloc[-1]
    # Selling Climax: Vol spike + Long lower tail
    vol_spike = last["Volume"] > (df["VOL_MEAN"].iloc[-1] * 1.5)
    long_shadow = last["LOWER_SHADOW"] > (last["BODY"] * 1.5)
    # No Supply: Low volume relative to mean
    no_supply = last["Volume"] < (df["VOL_MEAN"].iloc[-1] * 0.8)
    
    # Scoring (0-100)
    score = 0
    reasons = []
    if vol_spike: 
        score += 50
        reasons.append("זוהה Selling Climax: ווליום גבוה עם זנב תחתון המעיד על ספיגת פאניקה.")
    if no_supply: 
        score += 50
        reasons.append("זוהה No Supply: רגיעה בווליום המעידה על כך שהיצע המוכרים התייבש.")
    
    return score, reasons

# ----------------------------
# 2. UI & BUTTONS
# ----------------------------
st.title("Wyckoff Institutional Scout")

# Help Modal
with st.expander("ℹ️ איך הבוט עובד? לחץ להסבר"):
    st.write("""
    **מה הבוט מחפש?**
    הבוט מחפש חתימות של 'כסף חכם' (מוסדיים) לפי מתודולוגיית וייקוף:
    1. **Selling Climax (50 נק'):** פאניקה בשוק עם ווליום חריג שנעצרת ע"י קונים גדולים (זנב תחתון בנר).
    2. **No Supply (50 נק'):** לאחר הפאניקה, בוט מחפש רגיעה משמעותית בווליום. זה מעיד שהקונים המוסדיים 'בלעו' את כל המוכרים ואין יותר היצע שמושך את המחיר למטה.
    """)

ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Run Analysis"):
    df = get_data(ticker)
    if df is None:
        st.error("לא נמצא דאטה מספק")
    else:
        score, reasons = analyze_wyckoff(df)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={'text': "Institutional Absorption Score"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}
        ))
        st.plotly_chart(fig)
        
        # Explanation for Score
        st.write("### ניתוח התוצאה:")
        for r in reasons: st.write(f"- {r}")
        if not reasons: st.write("לא זוהו סימני איסוף מובהקים.")
        
        # Decision
        if score >= 100:
            st.success("✅ כן: זוהו שני הסימנים המבניים לאיסוף מוסדי.")
        elif score == 50:
            st.warning("⚠️ לא חד משמעי: זוהה רק חלק מהתהליך (או פאניקה ללא שקט, או שקט ללא פאניקה).")
        else:
            st.error("❌ לא: לא נמצאו אינדיקטורים לפעילות מוסדית.")
