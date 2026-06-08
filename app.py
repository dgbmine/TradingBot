import streamlit as st
import yfinance as yf
import pandas_ta as ta

# הגדרות עיצוב לדשבורד
st.set_page_config(page_title="Efi's Swing Dashboard", layout="wide")
st.title("📊 Efi's Swing Trading Dashboard")

# קלט מהמשתמש
ticker = st.text_input("הקלד טיקר (למשל: NVDA, BN, CBT):", "NVDA").upper()
indicator = st.selectbox("בחר אינדיקטור לניתוח:", 
                         ["SMA_20", "SMA_50", "RSI", "MACD", "Bollinger Bands"])

if st.button("הפעל סריקה"):
    try:
        # משיכת נתונים
        df = yf.Ticker(ticker).history(period="60d", interval="1h")
        
        if df.empty:
            st.error("לא נמצאו נתונים עבור המניה. בדוק את הטיקר.")
        else:
            st.subheader(f"ניתוח {indicator} עבור {ticker}")
            
            # לוגיקת אינדיקטורים
            if indicator == "SMA_20":
                val = ta.sma(df['Close'], length=20).iloc[-1]
                price = df['Close'].iloc[-1]
                signal = "🟢 קניה" if price > val else "🔴 מכירה"
                st.metric(label="מחיר נוכחי", value=f"{price:.2f}", delta=f"ממוצע: {val:.2f}")
                st.write(f"המלצה: {signal}")

            elif indicator == "SMA_50":
                val = ta.sma(df['Close'], length=50).iloc[-1]
                price = df['Close'].iloc[-1]
                signal = "🟢 קניה" if price > val else "🔴 מכירה"
                st.metric(label="מחיר נוכחי", value=f"{price:.2f}", delta=f"ממוצע: {val:.2f}")
                st.write(f"המלצה: {signal}")

            elif indicator == "RSI":
                val = ta.rsi(df['Close'], length=14).iloc[-1]
                signal = "🔴 מכירה" if val > 70 else ("🟢 קניה" if val < 30 else "⚪ המתנה")
                st.write(f"### ערך RSI: {val:.2f}")
                st.write(f"המלצה: {signal}")

            elif indicator == "MACD":
                macd = ta.macd(df['Close'])
                val = macd.iloc[-1, 0]
                signal = "🟢 קניה" if val > 0 else "🔴 מכירה"
                st.write(f"### ערך MACD: {val:.2f}")
                st.write(f"המלצה: {signal}")

            elif indicator == "Bollinger Bands":
                bb = ta.bbands(df['Close'], length=20)
                low, high = bb.iloc[-1, 0], bb.iloc[-1, 2]
                price = df['Close'].iloc[-1]
                signal = "🟢 קניה" if price < low else ("🔴 מכירה" if price > high else "⚪ המתנה")
                st.write(f"### טווח בולינגר: {low:.2f} - {high:.2f}")
                st.write(f"המלצה: {signal}")
            
            # הצגת גרף מחיר
            st.line_chart(df['Close'])
            
    except Exception as e:
        st.error(f"אירעה שגיאה: {e}")
