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
            df['SMA20'] = df['Close'].rolling(20).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            rsi = df['RSI'].iloc[-1]
            rating = "⚪ נייטרלי"
            if rsi < 20: rating = "🚀 להעמיס דחוף!"
            elif rsi < 30: rating = "🟢 קניה"
            elif rsi > 80: rating = "🔥 להעיף לפח!"
            elif rsi > 70: rating = "🔴 מכירה"
            
            st.subheader("🛠 ניתוח טכני")
            st.table(pd.DataFrame({"מדד": ["RSI", "SMA20"], "ערך": [f"{rsi:.2f}", "מעל הממוצע" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "מתחת לממוצע"], "שורה תחתונה": [rating, "מגמה חיובית" if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else "מגמה שלילית"]}))
            
            with st.expander("לחץ כאן להסבר איך לפעול לפי האינדיקטורים:"):
                st.write("""
                - **קניה/להעמיס:** כשה-RSI נמוך (מתחת ל-30 או 20), זה אומר שהמוכרים התעייפו. שקול כניסה מדורגת.
                - **מכירה/להעיף:** כשה-RSI גבוה (מעל 70 או 80), השוק ב-Overbought. זה הזמן לממש רווחים, לא לקנות.
                - **SMA20:** אם המחיר מעל, המגמה חיובית. אם מתחת, נסה להימנע מלונג.
                """)

        else: # וייקוף
            low_60 = df['Low'].min()
            vol_avg = df['Volume'].rolling(20).mean()
            climax = df[(df['Low'] == low_60) & (df['Volume'] > vol_avg * 2)]
            is_acc = (df['Close'].max() - df['Close'].min()) < (df['Close'].mean() * 0.2)
            vol_trend = df['Volume'].rolling(10).mean().diff().iloc[-1]
            
            st.subheader("🏛 ניתוח וייקוף מוסדי")
            phase = "איסוף (Accumulation)" if is_acc else "מגמה פעילה / פיזור"
            data = {"פרמטר": ["שלב השוק", "Selling Climax", "שורה תחתונה"], 
                    "מצב": [phase, "זוהה ✅" if len(climax) > 0 else "לא זוהה ❌", "🚀 להעמיס דחוף!" if is_acc else "⚪ המתנה"]}
            st.table(pd.DataFrame(data))
            
            with st.expander(f"לחץ כאן להסבר על שלב ה-{phase}:"):
                st.write(f"""
                **מה זה אומר?** המניה כרגע ב-{phase}.
                - **פעולה:** אם 'איסוף', המוסדיים בונים פוזיציה בטווח המחירים הנוכחי. אל תצפה לעלייה חדה מיד.
                - **אסטרטגיה:** חפש את ה-'Test' (ירידה קלה בנפח מסחר נמוך). אם המחיר לא יורד למרות ירידה בווליום, זה סימן אולטימטיבי לקניה.
                """)
        
        st.line_chart(df['Close'])
