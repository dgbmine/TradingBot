import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Efi's Dashboard", layout="wide")
st.title("📊 Efi's Ultimate Trading Dashboard")

ticker = st.text_input("הקלד טיקר:", "NVDA").upper()
mode = st.radio("בחר מצב עבודה:", ["אינדיקטורים טכניים", "איסוף מוסדי"])

if st.button("הפעל סריקה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1h")
    
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if mode == "אינדיקטורים טכניים":
            indicator = st.selectbox("בחר אינדיקטור:", ["SMA_20", "SMA_50", "RSI", "MACD"])
            
            if indicator == "SMA_20":
                val = df['Close'].rolling(window=20).mean().iloc[-1]
                price = df['Close'].iloc[-1]
                rating = "🟢 קניה" if price > val else "🔴 מכירה"
                st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע 20: {val:.2f}")
                st.write("דירוג:", rating)
            elif indicator == "SMA_50":
                val = df['Close'].rolling(window=50).mean().iloc[-1]
                price = df['Close'].iloc[-1]
                rating = "🟢 קניה" if price > val else "🔴 מכירה"
                st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע 50: {val:.2f}")
                st.write("דירוג:", rating)
            elif indicator == "RSI":
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = 100 - (100 / (1 + (gain / loss)))
                val = rsi.iloc[-1]
                if val < 20: rating = "🟢 קניה אגדית"
                elif val < 30: rating = "🟢 קניה"
                elif val > 80: rating = "🔥 לזרוק לפח"
                elif val > 70: rating = "🔴 מכירה"
                else: rating = "⚪ נייטרלי"
                st.write(f"### ערך RSI: {val:.2f}")
                st.write("דירוג:", rating)
            elif indicator == "MACD":
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                val = macd.iloc[-1]
                rating = "🟢 קניה" if val > 0 else "🔴 מכירה"
                st.write(f"### ערך MACD: {val:.2f}")
                st.write("דירוג:", rating)

        else: # איסוף מוסדי משוכלל
            avg_vol = df['Volume'].rolling(window=50).mean()
            accumulation = df[(df['Close'] < df['Open']) & (df['Volume'] > avg_vol * 2.0)]
            recent_events = accumulation.tail(24)
            count = len(recent_events)
            
            st.write(f"אירועי איסוף ב-24 השעות האחרונות: {count}")
            if count >= 3:
                st.success(f"✅ איסוף מוסדי מרוכז מזוהה! ({count} אירועים)")
            else:
                st.warning(f"❌ לא מזהה ריכוז של איסוף מוסדי.")
        
        st.line_chart(df['Close'])
