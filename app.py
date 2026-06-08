import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Efi's Dashboard", layout="wide")
st.title("📊 Efi's Institutional & Swing Dashboard")

ticker = st.text_input("הקלד טיקר:", "NVDA").upper()
option = st.selectbox("בחר מצב עבודה:", ["אינדיקטורים טכניים", "איסוף מוסדי"])

if st.button("הפעל סריקה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1h")
    
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if option == "אינדיקטורים טכניים":
            indicator = st.selectbox("בחר אינדיקטור:", ["SMA_20", "SMA_50", "RSI", "MACD"])
            
            if indicator == "SMA_20":
                val = df['Close'].rolling(window=20).mean().iloc[-1]
                price = df['Close'].iloc[-1]
                rating = "🟢 קניה" if price > val else "🔴 מכירה"
                st.metric("מחיר נוכחי", f"{price:.2f}", f"ממוצע 20: {val:.2f}")
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
        
        else: # איסוף מוסדי
            avg_vol = df['Volume'].rolling(window=20).mean()
            accumulation = df[(df['Close'] < df['Open']) & (df['Volume'] > avg_vol * 1.5)]
            count = len(accumulation)
            if count >= 3:
                st.success(f"✅ איסוף מוסדי מזוהה! ({count} אירועים)")
            else:
                st.warning(f"❌ לא מזהה איסוף מוסדי ({count} אירועים)")
        
        st.line_chart(df['Close'])
