import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Efi's Pro Dashboard", layout="wide")
st.title("📊 Efi's Ultimate Institutional Bot")

ticker = st.text_input("הקלד טיקר (NVDA, BN, CBT):", "NVDA").upper()

if st.button("בצע סריקה מלאה"):
    df = yf.Ticker(ticker).history(period="60d", interval="1h")
    
    if df.empty:
        st.error("לא נמצאו נתונים.")
    else:
        # 1. חישובים טכניים
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # 2. ניתוח וייקוף
        vol_avg = df['Volume'].rolling(20).mean()
        is_selling_climax = (df['Close'].iloc[-1] < df['Open'].iloc[-1] * 0.98) & (df['Volume'].iloc[-1] > vol_avg.iloc[-1] * 3)
        is_test = df['Volume'].iloc[-1] < vol_avg.iloc[-1] * 0.5
        
        # 3. ניתוח מוסדי
        acc_events = df[(df['Close'] < df['Open'] * 0.995) & (df['Volume'] > vol_avg * 2.5)].tail(24)
        
        # תצוגה
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠 ניתוח טכני")
            tech_data = {
                "מחיר": f"{df['Close'].iloc[-1]:.2f}",
                "RSI": f"{df['RSI'].iloc[-1]:.2f}",
                "דירוג RSI": "🟢 קניה" if df['RSI'].iloc[-1] < 30 else ("🔴 מכירה" if df['RSI'].iloc[-1] > 70 else "⚪ נייטרלי"),
                "מעל SMA20": "כן" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "לא"
            }
            st.table(pd.DataFrame.from_dict(tech_data, orient='index', columns=['ערך']))
            
        with col2:
            st.subheader("🏛 ניתוח מוסדי ווייקוף")
            inst_data = {
                "Selling Climax": "זוהה ✅" if is_selling_climax else "לא זוהה ❌",
                "Wyckoff Test": "זוהה ✅" if is_test else "לא זוהה ❌",
                "אירועי איסוף (24ש)": len(acc_events),
                "שורה תחתונה": "✅ איסוף פעיל" if (is_selling_climax or len(acc_events) >= 2) else "❌ אין איסוף"
            }
            st.table(pd.DataFrame.from_dict(inst_data, orient='index', columns=['מצב']))

        st.line_chart(df['Close'])
        st.write("---")
        st.write("הערה: 'Selling Climax' ו-'Test' מבוססים על נתוני השעה האחרונה. איסוף מוסדי מבוסס על ריכוז של 24 שעות.")
