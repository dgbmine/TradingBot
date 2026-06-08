import telebot
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# הגדרות בוט
API_TOKEN = '8908180052:AAHci2ZzKkjW2HchGdFL5U68Tqfs7krGB8Q'
bot = telebot.TeleBot(API_TOKEN)
STOCKS = ["GOOG", "AMZN", "NVDA", "TSLA"]

def get_data(ticker):
    df = yf.download(ticker, period="1mo", interval="1h")
    return df

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.reply_to(message, "הבוט פעיל! סורק את: " + ", ".join(STOCKS))

# פונקציית הסריקה שתופעל בלולאה
def scan_markets():
    for ticker in STOCKS:
        df = get_data(ticker)
        # בדיקת נזילות
        if df['Volume'].mean() < 1000000:
            continue
        
        # חישוב RSI
        rsi = ta.rsi(df['Close'], length=14).iloc[-1]
        
        # כאן תהיה הלוגיקה של הראש-כתפיים
        if rsi < 30: # דוגמה לתנאי היפוך
            bot.send_message(CHAT_ID, f"הכנה: {ticker} מתקרב לתבנית היפוך!")

bot.polling()
