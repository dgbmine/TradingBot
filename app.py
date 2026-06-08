import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Efi's Pro Dashboard", layout="wide")
st.title("📊 Efi's Institutional Analysis Dashboard")

# ─────────────────────────────────────────────
# HELPERS — INDICATORS
# ─────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def compute_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def find_swing_highs_lows(series, window=5):
    """Local swing highs and lows with a rolling window."""
    highs = series[(series == series.rolling(window * 2 + 1, center=True).max())]
    lows = series[(series == series.rolling(window * 2 + 1, center=True).min())]
    return highs, lows

# ─────────────────────────────────────────────
# HELPERS — WYCKOFF
# ─────────────────────────────────────────────

def wyckoff_analysis(df):
    """
    Full Wyckoff phase detection.
    Returns a dict with:
      - phase: A/B/C/D/E
      - phase_label: human-readable
      - events: list of detected events (PS, SC, AR, ST, Spring, LPS, SOS, SOW, BC, UT, UTAD)
      - bias: Accumulation / Distribution / Unclear
      - signal: action string
      - explanation: markdown string
    """
    result = {
        "phase": "?",
        "phase_label": "לא ניתן לזהות",
        "events": [],
        "bias": "Unclear",
        "signal": "⚪ המתנה",
        "explanation": "",
    }

    if len(df) < 40:
        result["explanation"] = "נדרשים לפחות 40 ימי מסחר לניתוח וייקוף."
        return result

    close = df['Close']
    volume = df['Volume']
    high = df['High']
    low = df['Low']

    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    atr = compute_atr(df)
    obv = compute_obv(df)

    # ── 1. Find Selling Climax / Buying Climax ──
    # SC: big down candle, huge volume, near recent lows
    price_change_pct = close.pct_change()
    vol_spike = volume > vol_ma20 * 2

    sc_candidates = df[
        (price_change_pct < -0.02) &
        vol_spike &
        (low == low.rolling(20).min())
    ]

    bc_candidates = df[
        (price_change_pct > 0.02) &
        vol_spike &
        (high == high.rolling(20).max())
    ]

    has_sc = len(sc_candidates) > 0
    has_bc = len(bc_candidates) > 0

    if has_sc:
        result["events"].append(f"✅ Selling Climax (SC) — {sc_candidates.index[-1].strftime('%Y-%m-%d')}")
    if has_bc:
        result["events"].append(f"✅ Buying Climax (BC) — {bc_candidates.index[-1].strftime('%Y-%m-%d')}")

    # ── 2. Automatic Rally / Reaction ──
    # After SC: sharp bounce. After BC: sharp drop.
    if has_sc:
        sc_idx = sc_candidates.index[-1]
        post_sc = close[close.index > sc_idx].head(10)
        if len(post_sc) > 2 and post_sc.max() > sc_candidates['Close'].iloc[-1] * 1.03:
            result["events"].append("✅ Automatic Rally (AR)")

    if has_bc:
        bc_idx = bc_candidates.index[-1]
        post_bc = close[close.index > bc_idx].head(10)
        if len(post_bc) > 2 and post_bc.min() < bc_candidates['Close'].iloc[-1] * 0.97:
            result["events"].append("✅ Automatic Reaction (AR — Distribution)")

    # ── 3. Trading Range detection ──
    # Range = max-min < X% of mean, over last 30 bars
    recent = df.tail(30)
    price_range_pct = (recent['Close'].max() - recent['Close'].min()) / recent['Close'].mean()
    in_range = price_range_pct < 0.15

    # ── 4. Spring / Shakeout ──
    # Price dips below range support on LOW volume → quickly recovers
    if in_range and len(df) >= 30:
        support = recent['Low'].quantile(0.1)
        resistance = recent['High'].quantile(0.9)

        spring_bars = df[
            (low < support * 0.995) &
            (close > support) &
            (volume < vol_ma20 * 0.8)
        ].tail(15)
        has_spring = len(spring_bars) > 0

        # Upthrust After Distribution (UTAD): pierces resistance, reverses on high vol
        utad_bars = df[
            (high > resistance * 1.005) &
            (close < resistance) &
            vol_spike
        ].tail(15)
        has_utad = len(utad_bars) > 0

        if has_spring:
            result["events"].append(f"✅ Spring / Shakeout — {spring_bars.index[-1].strftime('%Y-%m-%d')}")
        if has_utad:
            result["events"].append(f"✅ Upthrust / UTAD — {utad_bars.index[-1].strftime('%Y-%m-%d')}")

    else:
        has_spring = False
        has_utad = False
        support = close.min()
        resistance = close.max()

    # ── 5. LPS / SOS (Last Point of Support / Sign of Strength) ──
    # SOS: breakout above resistance on expanding volume
    last_5 = df.tail(5)
    sos = (
        last_5['Close'].iloc[-1] > resistance and
        last_5['Volume'].mean() > vol_ma20.iloc[-1] * 1.3
    ) if in_range else False

    # SOW: breakdown below support on expanding volume
    sow = (
        last_5['Close'].iloc[-1] < support and
        last_5['Volume'].mean() > vol_ma20.iloc[-1] * 1.3
    ) if in_range else False

    # LPS: pullback after SOS, volume dries up, holds above support
    lps = (
        close.iloc[-1] > support and
        close.iloc[-1] < close.iloc[-6] and
        vol_ma5.iloc[-1] < vol_ma20.iloc[-1] * 0.7
    )

    if sos:
        result["events"].append("✅ Sign of Strength (SOS) — פריצה מעל ההתנגדות!")
    if sow:
        result["events"].append("⚠️ Sign of Weakness (SOW) — שבירה מתחת לתמיכה!")
    if lps:
        result["events"].append("✅ Last Point of Support (LPS) — נפח יורד, מחיר מחזיק")

    # ── 6. OBV trend confirmation ──
    obv_trend = "עולה" if obv.iloc[-1] > obv.iloc[-10] else "יורד"
    obv_divergence = ""
    if obv_trend == "עולה" and close.iloc[-1] < close.iloc[-10]:
        obv_divergence = "⚡ Bullish Divergence — OBV עולה בעוד המחיר יורד (מוסדיים צוברים)"
    elif obv_trend == "יורד" and close.iloc[-1] > close.iloc[-10]:
        obv_divergence = "⚡ Bearish Divergence — OBV יורד בעוד המחיר עולה (מוסדיים מפזרים)"

    if obv_divergence:
        result["events"].append(obv_divergence)

    # ── 7. Phase Assignment ──
    acc_score = sum([
        has_sc,
        in_range,
        has_spring,
        sos,
        lps,
        obv_trend == "עולה",
    ])

    dist_score = sum([
        has_bc,
        in_range,
        has_utad,
        sow,
        obv_trend == "יורד",
    ])

    if acc_score > dist_score:
        result["bias"] = "Accumulation"
        if sos:
            result["phase"] = "D"
            result["phase_label"] = "Phase D — SOS זוהה, מגמה עולה מתפתחת"
            result["signal"] = "🚀 כניסה אגרסיבית — פריצת SOS"
        elif has_spring:
            result["phase"] = "C"
            result["phase_label"] = "Phase C — Spring זוהה, בדוק Test"
            result["signal"] = "🟢 כניסה מדורגת — חכה ל-Test של ה-Spring"
        elif in_range and has_sc:
            result["phase"] = "B"
            result["phase_label"] = "Phase B — בנייה בטווח, SC זוהה"
            result["signal"] = "🟡 המתן — המוסדיים עוד בונים פוזיציה"
        else:
            result["phase"] = "A"
            result["phase_label"] = "Phase A — עצירת מגמה, SC אפשרי"
            result["signal"] = "⚪ צפה בלבד — מוקדם מדי"

    elif dist_score > acc_score:
        result["bias"] = "Distribution"
        if sow:
            result["phase"] = "D"
            result["phase_label"] = "Phase D (Distribution) — SOW זוהה, מגמה יורדת"
            result["signal"] = "🔴 צא מפוזיציה / שקול שורט"
        elif has_utad:
            result["phase"] = "C"
            result["phase_label"] = "Phase C (Distribution) — UTAD זוהה"
            result["signal"] = "🟠 הפחת חשיפה — הניסיון לפרוץ נכשל"
        elif in_range and has_bc:
            result["phase"] = "B"
            result["phase_label"] = "Phase B (Distribution) — BC זוהה, פיזור בטווח"
            result["signal"] = "🟡 המתן / הפחת — מוסדיים מפזרים"
        else:
            result["phase"] = "A"
            result["phase_label"] = "Phase A (Distribution) — עצירת עלייה"
            result["signal"] = "⚪ צפה — BC אפשרי"

    else:
        result["phase"] = "?"
        result["phase_label"] = "לא ברור — אין הטיה ברורה"
        result["signal"] = "⚪ המתן לבהירות"

    # ── 8. Explanation ──
    result["explanation"] = f"""
**הטיה:** {result['bias']} | **שלב:** {result['phase_label']}

**OBV:** {obv_trend} | טווח מחירים 30 יום: {price_range_pct:.1%}

**איך לפעול:**
- **Accumulation Phase C (Spring):** כנסי מדורג עם סטופ מתחת ל-Spring. חכה ל-Test בנפח נמוך.
- **Accumulation Phase D (SOS):** כניסה אגרסיבית יותר. פולבק ל-LPS הוא הזדמנות נוספת.
- **Distribution Phase C (UTAD):** הפחת חשיפה, הניסיון לפרוץ נכשל.
- **Distribution Phase D (SOW):** סגור פוזיציות. שבירה עם נפח = אישור מגמה יורדת.
"""

    return result


# ─────────────────────────────────────────────
# HELPERS — TECHNICAL SIGNAL
# ─────────────────────────────────────────────

def technical_signal(df):
    """
    Multi-factor technical signal.
    Returns (score -100 to +100, label, details_dict).
    """
    score = 0
    details = {}

    rsi = compute_rsi(df['Close']).iloc[-1]
    macd_line, signal_line, histogram = compute_macd(df['Close'])
    upper_bb, mid_bb, lower_bb = compute_bollinger(df['Close'])
    atr = compute_atr(df).iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
    close = df['Close'].iloc[-1]
    vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    vol_ratio = last_vol / vol_ma20 if vol_ma20 > 0 else 1

    # RSI (±30 points)
    if rsi < 20:
        score += 30
        rsi_signal = "🚀 קיצוני — Oversold חמור"
    elif rsi < 30:
        score += 20
        rsi_signal = "🟢 Oversold"
    elif rsi < 45:
        score += 5
        rsi_signal = "🟡 ניטרלי-חלש"
    elif rsi < 55:
        score += 0
        rsi_signal = "⚪ ניטרלי"
    elif rsi < 70:
        score -= 5
        rsi_signal = "🟡 ניטרלי-חזק"
    elif rsi < 80:
        score -= 20
        rsi_signal = "🔴 Overbought"
    else:
        score -= 30
        rsi_signal = "🔥 קיצוני — Overbought חמור"

    details["RSI"] = (f"{rsi:.1f}", rsi_signal)

    # MACD (±25 points)
    macd_val = macd_line.iloc[-1]
    macd_sig = signal_line.iloc[-1]
    macd_hist_now = histogram.iloc[-1]
    macd_hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

    if macd_val > macd_sig and macd_hist_now > macd_hist_prev:
        score += 25
        macd_signal_str = "🟢 Bullish — קרוס + מומנטום עולה"
    elif macd_val > macd_sig:
        score += 10
        macd_signal_str = "🟡 Bullish חלש — מעל signal אבל נחלש"
    elif macd_val < macd_sig and macd_hist_now < macd_hist_prev:
        score -= 25
        macd_signal_str = "🔴 Bearish — קרוס + מומנטום יורד"
    else:
        score -= 10
        macd_signal_str = "🟡 Bearish חלש — מתחת signal אבל נחלש"

    details["MACD"] = (f"{macd_val:.3f} / Signal: {macd_sig:.3f}", macd_signal_str)

    # Bollinger Bands (±20 points)
    bb_pos = (close - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1]) if (upper_bb.iloc[-1] - lower_bb.iloc[-1]) > 0 else 0.5
    if bb_pos < 0.1:
        score += 20
        bb_signal = "🟢 ליד הבנד התחתון — Oversold"
    elif bb_pos < 0.3:
        score += 8
        bb_signal = "🟡 רבע תחתון"
    elif bb_pos > 0.9:
        score -= 20
        bb_signal = "🔴 ליד הבנד העליון — Overbought"
    elif bb_pos > 0.7:
        score -= 8
        bb_signal = "🟡 רבע עליון"
    else:
        bb_signal = "⚪ אמצע הטווח"

    details["Bollinger"] = (f"מיקום {bb_pos:.0%} בתוך הבנד", bb_signal)

    # SMA Trend (±15 points)
    if sma50 is not None:
        if close > sma20 > sma50:
            score += 15
            sma_signal = "🟢 מגמה עולה — מחיר > SMA20 > SMA50"
        elif close < sma20 < sma50:
            score -= 15
            sma_signal = "🔴 מגמה יורדת — מחיר < SMA20 < SMA50"
        elif close > sma20:
            score += 5
            sma_signal = "🟡 מעל SMA20 אבל מתחת SMA50"
        else:
            score -= 5
            sma_signal = "🟡 מתחת SMA20"
    else:
        if close > sma20:
            score += 8
            sma_signal = "🟡 מעל SMA20 (אין מספיק היסטוריה ל-SMA50)"
        else:
            score -= 8
            sma_signal = "🟡 מתחת SMA20"

    details["מגמה (SMA)"] = (f"SMA20: {sma20:.2f}" + (f" | SMA50: {sma50:.2f}" if sma50 else ""), sma_signal)

    # Volume confirmation (±10 points)
    if vol_ratio > 1.5 and score > 0:
        score += 10
        vol_signal = f"🟢 נפח גבוה x{vol_ratio:.1f} — מאשר עלייה"
    elif vol_ratio > 1.5 and score < 0:
        score -= 10
        vol_signal = f"🔴 נפח גבוה x{vol_ratio:.1f} — מאשר ירידה"
    elif vol_ratio < 0.5:
        vol_signal = f"⚪ נפח נמוך x{vol_ratio:.1f} — סיגנל חלש"
    else:
        vol_signal = f"⚪ נפח נורמלי x{vol_ratio:.1f}"

    details["נפח"] = (f"{last_vol:,.0f} vs ממוצע {vol_ma20:,.0f}", vol_signal)
    details["ATR (תנודתיות)"] = (f"{atr:.2f} ({atr/close*100:.1f}% ממחיר)", "מדד תנודתיות — שים לב לגודל פוזיציה")

    # Final label
    if score >= 60:
        label = "🚀 קנייה חזקה"
    elif score >= 30:
        label = "🟢 נטייה לקנייה"
    elif score >= -10:
        label = "⚪ ניטרלי"
    elif score >= -30:
        label = "🟠 נטייה למכירה"
    else:
        label = "🔴 מכירה חזקה"

    return score, label, details


# ─────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────

def build_chart(df, mode):
    macd_line, signal_line, histogram = compute_macd(df['Close'])
    upper_bb, mid_bb, lower_bb = compute_bollinger(df['Close'])
    rsi = compute_rsi(df['Close'])
    obv = compute_obv(df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("מחיר", "RSI", "MACD", "OBV")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="מחיר",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=upper_bb, line=dict(color='rgba(150,150,255,0.4)', width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mid_bb, line=dict(color='rgba(150,150,255,0.6)', width=1, dash='dot'), name="BB Mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=lower_bb, line=dict(color='rgba(150,150,255,0.4)', width=1), fill='tonexty', fillcolor='rgba(150,150,255,0.05)', name="BB Lower"), row=1, col=1)

    # SMA20, SMA50
    sma20 = df['Close'].rolling(20).mean()
    sma50 = df['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma20, line=dict(color='#FFB300', width=1.5), name="SMA20"), row=1, col=1)
    if sma50.notna().sum() > 5:
        fig.add_trace(go.Scatter(x=df.index, y=sma50, line=dict(color='#FF7043', width=1.5), name="SMA50"), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#AB47BC', width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in histogram]
    fig.add_trace(go.Bar(x=df.index, y=histogram, marker_color=colors, name="Histogram", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, line=dict(color='#42A5F5', width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, line=dict(color='#EF5350', width=1.5), name="Signal"), row=3, col=1)

    # OBV
    fig.add_trace(go.Scatter(x=df.index, y=obv, line=dict(color='#66BB6A', width=1.5), name="OBV"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=20),
    )

    return fig


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    ticker = st.text_input("טיקר:", "NVDA").upper()
with col2:
    mode = st.radio("אסטרטגיה:", ["אינדיקטורים טכניים", "וייקוף מוסדי"], horizontal=True)
with col3:
    period = st.selectbox("תקופה:", ["60d", "90d", "6mo", "1y"])

scan = st.button("🔍 בצע סריקה", use_container_width=True)

if scan:
    with st.spinner("טוען נתונים..."):
        df = yf.Ticker(ticker).history(period=period, interval="1d")

    if df.empty:
        st.error("לא נמצאו נתונים לטיקר זה.")
        st.stop()

    if len(df) < 30:
        st.warning("מעט מדי נתונים לניתוח מלא. מנסה...")

    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change_pct = (current_price - prev_price) / prev_price * 100
    high_52 = df['High'].max()
    low_52 = df['Low'].min()
    atr_val = compute_atr(df).iloc[-1]

    # Header metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("מחיר נוכחי", f"${current_price:.2f}", f"{change_pct:+.2f}%")
    m2.metric("High (תקופה)", f"${high_52:.2f}")
    m3.metric("Low (תקופה)", f"${low_52:.2f}")
    m4.metric("ATR (תנודתיות)", f"${atr_val:.2f}")
    m5.metric("% מהשיא", f"{(current_price/high_52-1)*100:.1f}%")

    st.divider()

    # ── CHART ──
    st.plotly_chart(build_chart(df, mode), use_container_width=True)

    st.divider()

    if mode == "אינדיקטורים טכניים":
        score, label, details = technical_signal(df)

        st.subheader(f"🛠 ניתוח טכני — {label}")

        # Score bar
        normalized = (score + 100) / 200  # 0 to 1
        bar_color = "#26a69a" if score > 10 else ("#ef5350" if score < -10 else "#FFA726")
        st.markdown(f"""
        <div style="background:#1e1e2e;border-radius:8px;padding:12px;margin-bottom:16px">
            <div style="font-size:13px;color:#aaa;margin-bottom:6px">ציון כולל: <b style="color:{bar_color};font-size:18px">{score:+d} / 100</b></div>
            <div style="background:#333;border-radius:4px;height:10px">
                <div style="background:{bar_color};width:{normalized*100:.0f}%;height:10px;border-radius:4px;transition:width 0.5s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for indicator, (value, signal) in details.items():
            rows.append({"אינדיקטור": indicator, "ערך": value, "סיגנל": signal})

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📖 מדריך פעולה לפי אינדיקטורים"):
            st.markdown("""
**RSI:**
- < 20: Oversold קיצוני — שקול כניסה מדורגת. אבל אל תיכנס סתם כי "זול" — ודא מגמה.
- 20–30: Oversold — סיגנל קנייה בתנאי שיש confirmation נוסף (MACD / OBV).
- 70–80: Overbought — שקול מימוש חלקי.
- > 80: Overbought קיצוני — שקול יציאה. אל תוסיף פוזיציה.

**MACD:**
- קרוס bullish (MACD עולה מעל Signal) = סיגנל כניסה.
- היסטוגרמה גדלה = מומנטום מתחזק.
- קרוס bearish = זהירות, שקול הפחתה.

**Bollinger Bands:**
- מחיר נוגע בבנד התחתון = תנאי Oversold.
- Squeeze (הבנדים מצטמצמים) = עומדת לפרוץ תנועה גדולה.
- מחיר נוגע בבנד העליון = זהירות / תמיכה ב-exit.

**OBV:**
- OBV עולה בעוד מחיר יורד = מוסדיים צוברים = Bullish Divergence.
- OBV יורד בעוד מחיר עולה = מוסדיים מפזרים = Bearish Divergence.

**ATR:**
- משמש לחישוב גודל פוזיציה. סטופ לוס מינימלי = 1.5x–2x ATR ממחיר הכניסה.
            """)

    else:  # וייקוף
        wyckoff = wyckoff_analysis(df)

        st.subheader(f"🏛 ניתוח וייקוף — {wyckoff['phase_label']}")

        w1, w2 = st.columns(2)
        with w1:
            bias_color = "#26a69a" if wyckoff['bias'] == "Accumulation" else ("#ef5350" if wyckoff['bias'] == "Distribution" else "#FFA726")
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px">
                <div style="font-size:13px;color:#aaa">הטיה</div>
                <div style="font-size:22px;font-weight:bold;color:{bias_color}">{wyckoff['bias']}</div>
                <div style="font-size:20px;margin-top:8px">{wyckoff['signal']}</div>
            </div>
            """, unsafe_allow_html=True)

        with w2:
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:16px">
                <div style="font-size:13px;color:#aaa">שלב</div>
                <div style="font-size:20px;font-weight:bold;color:#42A5F5">{wyckoff['phase']}</div>
                <div style="font-size:14px;color:#ccc;margin-top:6px">{wyckoff['phase_label']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### אירועים שזוהו:")
        if wyckoff['events']:
            for event in wyckoff['events']:
                st.markdown(f"- {event}")
        else:
            st.markdown("_לא זוהו אירועי וייקוף ברורים בתקופה זו._")

        with st.expander("📖 הסבר מלא + אסטרטגיית פעולה"):
            st.markdown(wyckoff['explanation'])

            st.markdown("""
---
#### מבנה וייקוף — תרשים מהיר

**Accumulation:**
```
Phase A: עצירת ירידה (PS → SC → AR → ST)
Phase B: בנייה בטווח (Secondary Tests)
Phase C: Shakeout / Spring  ← נקודת הכניסה האידיאלית
Phase D: SOS + LPS          ← כניסה שניה
Phase E: מגמה עולה
```

**Distribution:**
```
Phase A: עצירת עלייה (PSY → BC → AR → ST)
Phase B: בנייה בטווח ברמה גבוהה
Phase C: UTAD               ← מלכודת לקונים
Phase D: SOW + LPSY         ← יציאה / שורט
Phase E: מגמה יורדת
```

**הכלל הכי חשוב:**
> נפח מאמת כוונה. עלייה בנפח נמוך = חלשה. ירידה בנפח נמוך = בריאה. 
> Spring + Test בנפח נמוך = הסיגנל הכי חזק של וייקוף.
            """)

        st.divider()
        st.markdown("""
        > **הערה:** וייקוף הוא שיטה פרשנית. הקוד מזהה סיגנלים אוטומטית, אבל השיפוט הסופי תמיד שלך.
        > שלב C (Spring) הוא הנקודה הכי טובה לכניסה — אבל דורש אישור Test מוצלח לפני ביצוע.
        """)