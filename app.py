import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Efi's Pro Dashboard", layout="wide")
st.title("📊 Efi's Ultimate Institutional & Swing Bot")

ticker = st.text_input("הקלד טיקר:", "NVDA").upper()
mode = st.radio("בחר אסטרטגיה:", ["אינדיקטורים טכניים", "וייקוף מוסדי (60 יום)"])

if st.button("בצע סריקה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1d")
    
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if mode == "אינדיקטורים טכניים":
            # חישובים
            df['SMA20'] = df['Close'].rolling(20).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # לוגיקה ושורה תחתונה
            rsi = df['RSI'].iloc[-1]
            if rsi < 20: rating = "🚀 להעמיס דחוף!"
            elif rsi < 30: rating = "🟢 קניה"
            elif rsi > 80: rating = "🔥 להעיף לפח!"
            elif rsi > 70: rating = "🔴 מכירה"
            else: rating = "⚪ נייטרלי"
            
            st.subheader("🛠 ניתוח טכני")
            data = {
                "מחיר": f"{df['Close'].iloc[-1]:.2f}",
                "RSI": f"{rsi:.2f}",
                "שורה תחתונה": rating,
                "מגמת SMA20": "קניה" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "מכירה"
            }
            st.table(pd.DataFrame.from_dict(data, orient='index', columns=['מצב']))
            
        else: # וייקוף מוסדי
            low_60 = df['Low'].min()
            vol_avg = df['Volume'].rolling(20).mean()
            climax = df[(df['Low'] == low_60) & (df['Volume'] > vol_avg * 2)]
            is_acc = (df['Close'].max() - df['Close'].min()) < (df['Close'].mean() * 0.2)
            vol_trend = df['Volume'].rolling(10).mean().diff().iloc[-1]
            
            st.subheader("🏛 ניתוח וייקוף מוסדי")
            data = {
                "שלב שוק": "איסוף (Accumulation)" if is_acc else "מגמה פעילה",
                "Selling Climax": "זוהה ✅" if len(climax) > 0 else "לא זוהה ❌",
                "שורה תחתונה": "🚀 להעמיס דחוף!" if (is_acc and vol_trend < 0) else "⚪ המתנה / נייטרלי"
            }
            st.table(pd.DataFrame.from_dict(data, orient='index', columns=['מצב']))

        st.line_chart(df['Close'])
