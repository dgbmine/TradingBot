import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Efi's Dashboard", layout="wide")
st.title("📊 Efi's Enhanced Swing Dashboard")

ticker = st.text_input("הקלד טיקר:", "NVDA").upper()
indicator = st.selectbox("בחר אינדיקטור:", ["SMA_20", "SMA_50", "RSI", "MACD"])

def get_rating(val, indicator):
    if indicator == "RSI":
        if val < 20: return "🟢 קניה אגדית"
        if val < 30: return "🟢 קניה"
        if val > 80: return "🔥 לזרוק לפח"
        if val > 70: return "🔴 מכירה"
        return "⚪ נייטרלי"
    return "⚪ נייטרלי" # ברירת מחדל לאינדיקטורים אחרים

if st.button("הפעל סריקה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1h")
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if indicator == "SMA_20":
            val = df['Close'].rolling(window=20).mean().iloc[-1]
            price = df['Close'].iloc[-1]
            rating = "🟢 קניה" if price > val else "🔴 מכירה"
            st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע: {val:.2f}")
            st.write("דירוג:", rating)

        elif indicator == "RSI":
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))
            val = rsi.iloc[-1]
            st.write(f"### ערך RSI: {val:.2f}")
            st.write("דירוג:", get_rating(val, "RSI"))

        # ניתן להמשיך להוסיף לוגיקה דומה ל-MACD ו-SMA_50...
        
        st.line_chart(df['Close'])
