import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Efi Family Office Engine")

@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    return df

def get_transparency_report(df, support, resistance):
    """הסבר בשפה פשוטה למה אנחנו ממתינים"""
    curr = df.iloc[-1]
    
    report = []
    # 1. בדיקת טרנד
    if curr["Close"] > df["SMA50"].iloc[-1]:
        report.append("✅ המחיר מעל ממוצע 50 - סימן חיובי לטווח קצר.")
    else:
        report.append("❌ המחיר מתחת לממוצע 50 - אין מומנטום קונים.")
        
    # 2. בדיקת ווליום
    if curr["Volume"] > df["VOL_MA20"].iloc[-1]:
        report.append("✅ ווליום גבוה מהממוצע - יש עניין בשוק.")
    else:
        report.append("❌ ווליום נמוך - השוק 'ישנוני', אין הוכחה לכניסת כסף מוסדי.")
        
    # 3. בדיקת מבנה
    if curr["Close"] > resistance:
        report.append("✅ פריצת רמת התנגדות - המבנה נפרץ כלפי מעלה.")
    else:
        report.append("⚠️ מחיר בתוך טווח - אנחנו תקועים בין קונים למוכרים. תמתין לפריצה.")
        
    return report

ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Generate Investment Brief"):
    df = get_data(ticker)
    # לצורך הדוגמה נניח תמיכה/התנגדות מחושבים
    res = df['High'].rolling(20).max().iloc[-1]
    sup = df['Low'].rolling(20).min().iloc[-1]
    
    st.write(f"### 📊 תדריך משימה עבור {ticker}")
    report = get_transparency_report(df, sup, res)
    
    for item in report:
        st.write(item)
        
    st.divider()
    st.write("### למה אנחנו ממתינים?")
    st.info("כדי להוריד סיכון בניהול התיק, אנחנו לא מחפשים 'ניחוש' של הכיוון. אנחנו מחפשים **'אישור' (Confirmation)**. כרגע, חסר לנו האישור של הווליום ושל פריצת ההתנגדות. ברגע שהם יופיעו – הציון יעלה ל-LONG.")
