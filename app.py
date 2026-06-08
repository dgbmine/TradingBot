import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Efi's Dashboard", layout="wide")
st.title("📊 Efi's Lean Swing Dashboard")

ticker = st.text_input("הקלד טיקר:", "NVDA").upper()
indicator = st.selectbox("בחר אינדיקטור:", ["SMA_20", "SMA_50", "RSI", "MACD"])

if st.button("הפעל סריקה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1h")
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if indicator == "SMA_20":
            val = df['Close'].rolling(window=20).mean().iloc[-1]
            price = df['Close'].iloc[-1]
            st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע 20: {val:.2f}")
            st.write("המלצה:", "קניה" if price > val else "מכירה")

        elif indicator == "SMA_50":
            val = df['Close'].rolling(window=50).mean().iloc[-1]
            price = df['Close'].iloc[-1]
            st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע 50: {val:.2f}")
            st.write("המלצה:", "קניה" if price > val else "מכירה")

        elif indicator == "RSI":
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            val = rsi.iloc[-1]
            st.write(f"### ערך RSI: {val:.2f}")
            st.write("המלצה:", "מכירה" if val > 70 else ("קניה" if val < 30 else "המתן"))

        elif indicator == "MACD":
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            val = macd.iloc[-1]
            st.write(f"### ערך MACD: {val:.2f}")
            st.write("המלצה:", "קניה" if val > 0 else "מכירה")

        st.line_chart(df['Close'])
