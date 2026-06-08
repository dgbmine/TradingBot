import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Wyckoff Institutional Scout")

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Hebrew', sans-serif;
        direction: rtl;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; direction: ltr; }

    .header-box {
        background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 100%);
        border: 1px solid #2a4a6a;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 28px;
        color: #e0eaf4;
        direction: rtl;
        line-height: 1.9;
    }
    .header-box h2 {
        color: #4fc3f7;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        margin-bottom: 12px;
        direction: ltr;
    }
    .header-box p { color: #b0c8e0; font-size: 0.92rem; margin: 6px 0; }
    .header-box .tag {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #4fc3f7;
        color: #4fc3f7;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 3px 2px;
    }
    .score-reason-box {
        background: #0d1b2a;
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0;
        direction: rtl;
        color: #cde3f5;
        font-size: 0.88rem;
        line-height: 1.8;
    }
    .score-reason-box.positive { border-left-color: #26a69a; }
    .score-reason-box.neutral  { border-left-color: #ffa726; }
    .score-reason-box.negative { border-left-color: #ef5350; }
    .score-reason-box strong { color: #fff; }
    .criteria-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #1e3040;
        font-size: 0.84rem;
    }
    .criteria-row .hit  { color: #26a69a; font-weight: 600; }
    .criteria-row .miss { color: #ef5350; }
    .disclaimer {
        background: #1a1206;
        border: 1px solid #5a4010;
        border-radius: 8px;
        padding: 10px 16px;
        color: #a08040;
        font-size: 0.78rem;
        direction: rtl;
        margin-top: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER EXPLANATION BOX
# ----------------------------
st.markdown("""
<div class="header-box">
  <h2>⬛ WYCKOFF INSTITUTIONAL SCOUT — אסטרטגיה ומתודולוגיה</h2>
  <p>
    הבוט מחפש חתימות של <strong>איסוף מוסדי מוקדם (Early Accumulation)</strong> לפי מתודולוגיית ריצ'רד וייקוף (Wyckoff).
    הרעיון: מוסדיים לא קונים בבת-אחת — הם בונים פוזיציות בשקט, בשלב שבו הציבור עדיין פוחד.
    הבוט מזהה שלב זה על-ידי שילוב של חמישה קריטריונים עצמאיים.
  </p>
  <p><strong>חמשת הקריטריונים (כל אחד מוסיף נקודות לציון 0–100):</strong></p>
  <p>
    <span class="tag">SC – Selling Climax</span>
    <span class="tag">AR – Automatic Rally</span>
    <span class="tag">No Supply</span>
    <span class="tag">Price–Vol Divergence</span>
    <span class="tag">Trading Range</span>
  </p>
  <p>
    <strong>SC (25 נק'):</strong> זינוק ווליום חריג ≥2× הממוצע עם זנב תחתון ארוך — פאניקה שנבלמת על-ידי קונים גדולים.<br>
    <strong>AR (20 נק'):</strong> עלייה חדה בתוך 3–10 ימים לאחר ה-SC — ראיה ראשונה שהיצע הוסר.<br>
    <strong>No Supply (20 נק'):</strong> ווליום נמוך &lt;0.7× הממוצע בעשרת הימים האחרונים — המוכרים נעלמו.<br>
    <strong>Price–Vol Divergence (20 נק'):</strong> מחיר ממשיך לרדת אבל ווליום קורס — הלחץ מתייבש.<br>
    <strong>Trading Range (15 נק'):</strong> המחיר מתנודד בטווח צר בשלושת השבועות האחרונים — סימן קלאסי של צבירה שקטה.
  </p>
  <p style="color:#607d8b; font-size:0.82rem;">
    ⚠️ תנאי מקדים: הבוט בודק רק ניירות שירדו ≥12% מהשיא ב-3 חודשים — SC רלוונטי רק אחרי ירידה.
  </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOGIC & DATA
# ----------------------------
@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y")
    if len(df) < 100:
        return None
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["RANGE"] = df["High"] - df["Low"]
    return df


def analyze_wyckoff(df):
    score = 0
    criteria = []  # list of dicts: {name, hit, points, earned, explanation}

    # --- PREREQUISITE: Prior downtrend ---
    high_3m = df["Close"].iloc[-65:].max()
    current = df["Close"].iloc[-1]
    drawdown = (high_3m - current) / high_3m
    prereq_met = drawdown >= 0.12

    # --- 1. SELLING CLIMAX (SC) ---
    # Look for SC in last 30 days: vol >= 2x mean AND lower_shadow > body
    sc_window = df.iloc[-30:]
    sc_candidates = sc_window[
        (sc_window["Volume"] >= sc_window["VOL_MEAN"] * 2.0) &
        (sc_window["LOWER_SHADOW"] > sc_window["BODY"] * 1.2)
    ]
    sc_found = len(sc_candidates) > 0
    sc_points = 25 if (sc_found and prereq_met) else 0
    score += sc_points
    sc_idx = sc_candidates.index[-1] if sc_found else None
    criteria.append({
        "name": "Selling Climax (SC)",
        "hit": sc_found and prereq_met,
        "points": 25,
        "earned": sc_points,
        "explanation": (
            f"זוהה SC ב-{sc_idx.strftime('%d/%m/%Y') if sc_idx else '—'}: "
            f"ווליום פי {sc_candidates['Volume'].iloc[-1] / sc_candidates['VOL_MEAN'].iloc[-1]:.1f} מהממוצע עם זנב תחתון ארוך."
            if sc_found and prereq_met else
            "לא זוהה SC. " + (
                "ירידה מהשיא של {:.1f}% — תנאי מקדים לא מתקיים (נדרש ≥12%).".format(drawdown * 100)
                if not prereq_met else
                "לא נמצא נר עם ווליום חריג וזנב תחתון משמעותי ב-30 הימים האחרונים."
            )
        )
    })

    # --- 2. AUTOMATIC RALLY (AR) after SC ---
    ar_found = False
    ar_points = 0
    ar_explanation = "לא זוהה AR — נדרש SC קודם."
    if sc_found and sc_idx is not None:
        post_sc = df.loc[sc_idx:].iloc[1:11]  # up to 10 days after SC
        if len(post_sc) >= 2:
            rally = (post_sc["Close"].max() - df.loc[sc_idx, "Close"]) / df.loc[sc_idx, "Close"]
            ar_found = rally >= 0.04
            ar_points = 20 if ar_found else 0
            score += ar_points
            ar_explanation = (
                f"זוהה AR: עלייה של {rally*100:.1f}% תוך 10 ימים לאחר ה-SC — אישור להסרת היצע."
                if ar_found else
                f"לא זוהה AR ברור: עלייה מקסימלית של {rally*100:.1f}% בלבד לאחר ה-SC (נדרש ≥4%)."
            )
    criteria.append({
        "name": "Automatic Rally (AR)",
        "hit": ar_found,
        "points": 20,
        "earned": ar_points,
        "explanation": ar_explanation
    })

    # --- 3. NO SUPPLY ---
    recent_10 = df.iloc[-10:]
    avg_vol_10 = recent_10["Volume"].mean()
    global_mean = df["VOL_MEAN"].iloc[-1]
    no_supply = avg_vol_10 < global_mean * 0.7
    ns_points = 20 if no_supply else 0
    score += ns_points
    criteria.append({
        "name": "No Supply (יצע מתייבש)",
        "hit": no_supply,
        "points": 20,
        "earned": ns_points,
        "explanation": (
            f"הווליום הממוצע ב-10 ימים האחרונים ({avg_vol_10:,.0f}) הוא {avg_vol_10/global_mean*100:.0f}% מהממוצע — "
            + ("המוכרים נעלמו." if no_supply else "הווליום עדיין גבוה מדי, המוכרים עדיין פעילים.")
        )
    })

    # --- 4. PRICE-VOLUME DIVERGENCE ---
    last_20 = df.iloc[-20:]
    price_change = (last_20["Close"].iloc[-1] - last_20["Close"].iloc[0]) / last_20["Close"].iloc[0]
    vol_change = (last_20["Volume"].iloc[-5:].mean() - last_20["Volume"].iloc[:5].mean()) / last_20["Volume"].iloc[:5].mean()
    divergence = (price_change < 0) and (vol_change < -0.25)
    div_points = 20 if divergence else 0
    score += div_points
    criteria.append({
        "name": "Price–Volume Divergence",
        "hit": divergence,
        "points": 20,
        "earned": div_points,
        "explanation": (
            f"מחיר: {price_change*100:+.1f}% | ווליום: {vol_change*100:+.1f}% (20 ימים). "
            + ("מחיר יורד אבל לחץ המכירה קורס — סימן קלאסי של Smart Money." if divergence
               else "לא נמצאה דיברגנציה ברורה בין מחיר לווליום.")
        )
    })

    # --- 5. TRADING RANGE ---
    last_15 = df.iloc[-15:]
    tr_high = last_15["High"].max()
    tr_low = last_15["Low"].min()
    tr_range_pct = (tr_high - tr_low) / tr_low
    in_range = tr_range_pct < 0.12
    tr_points = 15 if in_range else 0
    score += tr_points
    criteria.append({
        "name": "Trading Range (טווח צר)",
        "hit": in_range,
        "points": 15,
        "earned": tr_points,
        "explanation": (
            f"טווח 15 הימים האחרונים: {tr_range_pct*100:.1f}% "
            + ("— מחיר 'קפוא' בטווח צר: קלאסי לשלב איסוף שקט." if in_range
               else "— טווח רחב מדי (>12%): עדיין תנודתי, לא דפוס איסוף יציב.")
        )
    })

    # --- SUMMARY VERDICT ---
    if score >= 75:
        verdict = "סבירות גבוהה לאיסוף מוסדי מוקדם"
        verdict_color = "#26a69a"
    elif score >= 45:
        verdict = "סימנים חלקיים — נדרשת תשומת לב"
        verdict_color = "#ffa726"
    else:
        verdict = "לא נמצאו ראיות מוסדות לאיסוף"
        verdict_color = "#ef5350"

    return score, criteria, verdict, verdict_color, prereq_met, drawdown


# ----------------------------
# UI
# ----------------------------
st.title("WYCKOFF INSTITUTIONAL SCOUT")

col_input, _ = st.columns([1, 3])
with col_input:
    ticker = st.text_input("טיקר", "NVDA").upper()
    run = st.button("▶ הרץ אנליזה", use_container_width=True)

if run:
    with st.spinner("שולף דאטה ומנתח..."):
        df = get_data(ticker)

    if df is None:
        st.error("לא נמצא דאטה מספיק (נדרש לפחות 100 ימי מסחר).")
    else:
        score, criteria, verdict, verdict_color, prereq_met, drawdown = analyze_wyckoff(df)

        # ---- Row: Gauge + Criteria breakdown ----
        col_gauge, col_reasons = st.columns([1, 1], gap="large")

        with col_gauge:
            # Gauge color zones
            gauge_color = "#26a69a" if score >= 75 else "#ffa726" if score >= 45 else "#ef5350"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"<b>Institutional Absorption Score</b><br><span style='font-size:0.85em;color:{verdict_color}'>{verdict}</span>",
                       'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a6a8a"},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': "#0d1b2a",
                    'borderwidth': 1,
                    'bordercolor': "#2a4a6a",
                    'steps': [
                        {'range': [0, 44],  'color': '#1a0d0d'},
                        {'range': [44, 74], 'color': '#1a1206'},
                        {'range': [74, 100],'color': '#0d1a18'},
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 2},
                        'thickness': 0.75,
                        'value': score
                    }
                },
                number={'font': {'size': 52, 'color': gauge_color}, 'suffix': '/100'}
            ))
            fig.update_layout(
                height=320,
                margin=dict(t=80, b=20, l=30, r=30),
                paper_bgcolor="#0a1520",
                font_color="#e0eaf4"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prerequisite note
            if not prereq_met:
                st.markdown(f"""
                <div class="score-reason-box negative">
                ⚠️ <strong>תנאי מקדים לא מתקיים:</strong> הנייר ירד רק {drawdown*100:.1f}% מהשיא (נדרש ≥12%).
                SC רלוונטי רק לאחר ירידה משמעותית — ייתכן שהניתוח פחות אמין.
                </div>
                """, unsafe_allow_html=True)

        with col_reasons:
            st.markdown("### פירוט הניקוד")
            total_possible = sum(c["points"] for c in criteria)
            for c in criteria:
                hit_label = "✅ הצליח" if c["hit"] else "❌ נכשל"
                hit_class = "hit" if c["hit"] else "miss"
                st.markdown(f"""
                <div class="score-reason-box {'positive' if c['hit'] else 'negative'}">
                  <div class="criteria-row">
                    <strong>{c['name']}</strong>
                    <span>
                      <span class="{hit_class}">{hit_label}</span>
                      &nbsp;|&nbsp; <strong>{c['earned']}/{c['points']} נק'</strong>
                    </span>
                  </div>
                  <div style="margin-top:6px; color:#b0c8e0;">{c['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)

        # ---- Price + Volume Chart ----
        st.markdown("---")
        st.markdown("### גרף מחיר וווליום — 3 חודשים אחרונים")
        df_chart = df.iloc[-65:].copy()

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.7, 0.3], vertical_spacing=0.04)

        # Candlestick
        fig2.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"], close=df_chart["Close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            name="Price"
        ), row=1, col=1)

        # Volume bars colored by day
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(df_chart["Close"], df_chart["Open"])]
        fig2.add_trace(go.Bar(
            x=df_chart.index, y=df_chart["Volume"],
            marker_color=vol_colors, name="Volume", opacity=0.8
        ), row=2, col=1)

        # Volume MA
        fig2.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["VOL_MEAN"],
            line=dict(color="#4fc3f7", width=1.5, dash="dot"),
            name="Vol MA(20)"
        ), row=2, col=1)

        fig2.update_layout(
            height=480,
            paper_bgcolor="#0a1520",
            plot_bgcolor="#0d1b2a",
            font_color="#e0eaf4",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(t=20, b=20)
        )
        fig2.update_xaxes(gridcolor="#1e3040", showgrid=True)
        fig2.update_yaxes(gridcolor="#1e3040", showgrid=True)

        st.plotly_chart(fig2, use_container_width=True)

        # ---- Disclaimer ----
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <strong>לידיעתך:</strong> הבוט מספק אנליזה טכנית אוטומטית בלבד ואינו המלצת השקעה.
        מתודולוגיית וייקוף היא כלי לזיהוי דפוסים ולא ערובה לתוצאות. תמיד בצע Due Diligence עצמאי.
        </div>
        """, unsafe_allow_html=True)