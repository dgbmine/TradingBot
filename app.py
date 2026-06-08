import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Efi Trading Engine", layout="wide")
st.title("📊 Institutional Trading Dashboard")

# ─────────────────────────────────────────────
# CACHE DATA (CRITICAL UPGRADE)
# ─────────────────────────────────────────────
@st.cache_data(ttl=60 * 5)
def load_data(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_bollinger(series, period=20, std=2):
    sma = series.rolling(period).mean()
    stdv = series.rolling(period).std()
    return sma + std * stdv, sma, sma - std * stdv


def compute_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    obv = (direction * df['Volume']).cumsum()
    return obv.ewm(span=10).mean()


# ─────────────────────────────────────────────
# SAFE LEVELS
# ─────────────────────────────────────────────
def pivots(df, left=3, right=3):
    highs, lows = [], []

    for i in range(left, len(df) - right):
        if df['High'].iloc[i] == df['High'].iloc[i-left:i+right+1].max():
            highs.append(i)
        if df['Low'].iloc[i] == df['Low'].iloc[i-left:i+right+1].min():
            lows.append(i)

    return highs, lows


def get_levels(df):
    highs, lows = pivots(df)

    if len(highs) < 2 or len(lows) < 2:
        return df['Low'].min(), df['High'].max()

    return df['Low'].iloc[lows[-1]], df['High'].iloc[highs[-1]]


# ─────────────────────────────────────────────
# WYCKOFF ENGINE (PRODUCTION VERSION)
# ─────────────────────────────────────────────
def wyckoff_analysis(df):
    result = {
        "bias": "Unclear",
        "phase": "?",
        "events": [],
        "signal": "⚪ המתנה",
        "confidence": 0,
        "explanation": ""
    }

    if df is None or len(df) < 80:
        result["explanation"] = "לא מספיק דאטה (נדרש 80+ ימים)"
        return result

    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']

    vol_ma = vol.rolling(20).mean()
    vol_ratio = vol / vol_ma

    obv = compute_obv(df)
    obv_trend = obv.iloc[-1] > obv.iloc[-20]

    support, resistance = get_levels(df)

    recent = df.tail(30)
    range_pct = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()
    in_range = range_pct < 0.12

    price = close.iloc[-1]

    vol_ok = vol_ratio.iloc[-1] > 1.25

    # ── EVENTS ──
    spring = low.iloc[-1] < support and price > support and vol_ok
    utad = high.iloc[-1] > resistance and price < resistance and vol_ok
    sos = price > resistance and vol_ok
    sow = price < support and vol_ok

    if spring:
        result["events"].append("🟢 Spring")
    if utad:
        result["events"].append("🔴 UTAD")
    if sos:
        result["events"].append("🚀 SOS")
    if sow:
        result["events"].append("⚠️ SOW")

    # ── CONFIDENCE MODEL (NEW) ──
    confidence = 0

    confidence += 25 if spring else 0
    confidence += 25 if sos else 0
    confidence += 15 if obv_trend else 0
    confidence += 10 if in_range else 0
    confidence -= 20 if utad else 0
    confidence -= 20 if sow else 0

    result["confidence"] = max(0, min(100, confidence))

    # ── BIAS ──
    if confidence >= 40:
        result["bias"] = "Accumulation"
    elif confidence <= -10:
        result["bias"] = "Distribution"

    # ── SIGNAL ──
    if spring:
        result["signal"] = "🟢 Spring — המתנה ל-Test"
        result["phase"] = "C"
    elif sos:
        result["signal"] = "🚀 SOS — אפשר כניסה"
        result["phase"] = "D"
    elif utad:
        result["signal"] = "🟠 UTAD — זהירות"
        result["phase"] = "C"
    elif sow:
        result["signal"] = "🔴 SOW — יציאה"
        result["phase"] = "D"
    else:
        result["signal"] = "⚪ אין יתרון ברור"
        result["phase"] = "B"

    # ── EXPLANATION (RTL SAFE) ──
    result["explanation"] = f"""
<div dir="rtl" style="text-align:right">

**הטיה:** {result['bias']}  
**שלב:** {result['phase']}  
**איכות אות (Confidence):** {result['confidence']}/100  

**מחיר:** {price:.2f}  

**לוגיקת קריאה:**
- Spring + נפח = איסוף
- SOS = התחלת מגמה
- UTAD = מלכודת קונים
- SOW = שבירה

</div>
"""

    return result


# ─────────────────────────────────────────────
# TECHNICAL (STABLE VERSION)
# ─────────────────────────────────────────────
def technical_signal(df):
    rsi = compute_rsi(df['Close']).iloc[-1]
    macd, sig, _ = compute_macd(df['Close'])

    price = df['Close'].iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) > 50 else sma20

    score = 0

    if rsi < 30: score += 20
    elif rsi > 70: score -= 20

    if macd.iloc[-1] > sig.iloc[-1]: score += 15
    else: score -= 15

    if price > sma20 > sma50: score += 15
    elif price < sma20 < sma50: score -= 15

    label = "🟢 חיובי" if score > 20 else "🔴 שלילי" if score < -20 else "⚪ ניטרלי"

    return score, label


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def build_chart(df):
    macd, sig, hist = compute_macd(df['Close'])
    upper, mid, lower = compute_bollinger(df['Close'])
    rsi = compute_rsi(df['Close'])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=upper), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=lower), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=700,
                      xaxis_rangeslider_visible=False)

    return fig


# ─────────────────────────────────────────────
# UI (PROTECTED)
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

ticker = col1.text_input("טיקר", "NVDA").upper()
mode = col2.radio("מצב", ["טכני", "וייקוף"], horizontal=True)
period = col3.selectbox("תקופה", ["60d", "90d", "6mo", "1y"])

if st.button("🔍 סריקה"):
    df = load_data(ticker, period)

    if df is None or len(df) < 50:
        st.error("אין מספיק דאטה לניתוח")
        st.stop()

    st.plotly_chart(build_chart(df), use_container_width=True)

    if mode == "וייקוף":
        wy = wyckoff_analysis(df)

        st.subheader(f"🏛 Wyckoff — {wy['bias']}")
        st.metric("Confidence", f"{wy['confidence']}/100")

        if wy['events']:
            for e in wy['events']:
                st.success(e)
        else:
            st.info("אין אירועי Wyckoff ברורים")

        st.markdown(wy['explanation'], unsafe_allow_html=True)

    else:
        score, label = technical_signal(df)
        st.subheader(label)
        st.write("Score:", score)