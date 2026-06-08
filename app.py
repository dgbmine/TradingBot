import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Efi's Pro Dashboard", layout="wide")
st.title("📊 Efi's Ultimate Institutional & Swing Bot")

ticker = st.text_input("הקלד טיקר (NVDA, BN, CBT):", "NVDA").upper()
mode = st.radio("בחר אסטרטגיית ניתוח:", ["אינדיקטורים טכניים", "וייקוף מוסדי (60 יום)"])

if st.button("בצע סריקה"):
    # הורדת נתונים יומיים לניתוח ארוך טווח
    df = yf.Ticker(ticker).history(period="60d", interval="1d")
    
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        if mode == "אינדיקטורים טכניים":
            # חישובים טכניים
            df['SMA20'] = df['Close'].rolling(20).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            st.subheader("🛠 ניתוח טכני")
            data = {
                "מחיר": f"{df['Close'].iloc[-1]:.2f}",
                "RSI": f"{df['RSI'].iloc[-1]:.2f}",
                "סטטוס RSI": "🟢 קניה" if df['RSI'].iloc[-1] < 30 else ("🔴 מכירה" if df['RSI'].iloc[-1] > 70 else "⚪ נייטרלי"),
                "מעל SMA20": "כן" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "לא"
            }
            st.table(pd.DataFrame.from_dict(data, orient='index', columns=['ערך']))
            
        else: # וייקוף מוסדי
            low_60 = df['Low'].min()
            vol_avg = df['Volume'].rolling(20).mean()
            climax = df[(df['Low'] == low_60) & (df['Volume'] > vol_avg * 2)]
            is_acc = (df['Close'].max() - df['Close'].min()) < (df['Close'].mean() * 0.2)
            vol_trend = df['Volume'].rolling(10).mean().diff().iloc[-1]
            
            st.subheader("🏛 ניתוח וייקוף מוסדי (60 יום)")
            data = {
                "שלב שוק": "איסוף (Accumulation)" if is_acc else "מגמה פעילה",
                "Selling Climax": "זוהה בשפל ✅" if len(climax) > 0 else "לא זוהה ❌",
                "מגמת ווליום (Test)": "יורדת (חיובי) ✅" if vol_trend < 0 else "עולה (זהירות) ❌",
                "שורה תחתונה": "✅ סימני איסוף" if (is_acc and vol_trend < 0) else "❌ לא במבנה איסוף"
            }
            st.table(pd.DataFrame.from_dict(data, orient='index', columns=['מצב']))

        st.line_chart(df['Close'])
