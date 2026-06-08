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
    .overview-card {
        background: #0d1b2a;
        border: 1px solid #2a4a6a;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        direction: ltr;
    }
    .overview-card .ticker-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #4fc3f7;
        margin-bottom: 4px;
    }
    .overview-card .score-big {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 6px 0;
    }
    .overview-card .verdict-label {
        font-size: 0.78rem;
        color: #b0c8e0;
        margin-top: 4px;
    }
    .overview-card .bar-bg {
        background: #1e3040;
        border-radius: 4px;
        height: 8px;
        margin-top: 10px;
        overflow: hidden;
    }
    .overview-card .bar-fill {
        height: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class="header-box">
  <h2>⬛ WYCKOFF INSTITUTIONAL SCOUT — אסטרטגיה ומתודולוגיה</h2>
  <p>
    הבוט מחפש חתימות של <strong>איסוף מוסדי מוקדם (Early Accumulation)</strong> לפי מתודולוגיית ריצ'רד וייקוף.
    מוסדיים בונים פוזיציות בשקט, בשלב שבו הציבור עדיין פוחד. הבוט מזהה שלב זה על-ידי חמישה קריטריונים עצמאיים.
  </p>
  <p><strong>חמשת הקריטריונים (0–100 נקודות סה"כ):</strong></p>
  <p>
    <span class="tag">SC – Selling Climax · 25 נק'</span>
    <span class="tag">AR – Automatic Rally · 20 נק'</span>
    <span class="tag">No Supply · 20 נק'</span>
    <span class="tag">Price–Vol Divergence · 20 נק'</span>
    <span class="tag">Trading Range · 15 נק'</span>
  </p>
  <p>
    <strong>SC:</strong> ווליום ≥2× ממוצע עם זנב תחתון — פאניקה שנבלמת על-ידי קונים גדולים.<br>
    <strong>AR:</strong> עלייה ≥4% תוך 10 ימים לאחר ה-SC — אישור ראשון להסרת היצע.<br>
    <strong>No Supply:</strong> ווליום ממוצע &lt;70% מהממוצע ב-10 ימים — המוכרים נעלמו.<br>
    <strong>Divergence:</strong> מחיר ממשיך לרדת אבל ווליום קורס — לחץ המכירה מתייבש.<br>
    <strong>Trading Range:</strong> טווח 15 ימים &lt;12% — צבירה שקטה קלאסית.
  </p>
  <p style="color:#607d8b; font-size:0.82rem;">⚠️ תנאי מקדים: ירידה ≥12% מהשיא ב-3 חודשים — SC רלוונטי רק אחרי ירידה.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOGIC
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
    criteria = []

    high_3m = df["Close"].iloc[-65:].max()
    current = df["Close"].iloc[-1]
    drawdown = (high_3m - current) / high_3m
    prereq_met = drawdown >= 0.12

    # 1. SC
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
        "name": "Selling Climax (SC)", "hit": sc_found and prereq_met,
        "points": 25, "earned": sc_points,
        "explanation": (
            f"זוהה SC ב-{sc_idx.strftime('%d/%m/%Y') if sc_idx else '—'}: "
            f"ווליום פי {sc_candidates['Volume'].iloc[-1] / sc_candidates['VOL_MEAN'].iloc[-1]:.1f} מהממוצע עם זנב תחתון ארוך."
            if sc_found and prereq_met else
            "לא זוהה SC. " + (
                f"ירידה מהשיא של {drawdown*100:.1f}% — תנאי מקדים לא מתקיים (נדרש ≥12%)."
                if not prereq_met else
                "לא נמצא נר עם ווליום חריג וזנב תחתון משמעותי ב-30 הימים האחרונים."
            )
        )
    })

    # 2. AR
    ar_found = False
    ar_points = 0
    ar_explanation = "לא זוהה AR — נדרש SC קודם."
    if sc_found and sc_idx is not None:
        post_sc = df.loc[sc_idx:].iloc[1:11]
        if len(post_sc) >= 2:
            rally = (post_sc["Close"].max() - df.loc[sc_idx, "Close"]) / df.loc[sc_idx, "Close"]
            ar_found = rally >= 0.04
            ar_points = 20 if ar_found else 0
            score += ar_points
            ar_explanation = (
                f"זוהה AR: עלייה של {rally*100:.1f}% תוך 10 ימים לאחר ה-SC."
                if ar_found else
                f"לא זוהה AR ברור: עלייה מקסימלית של {rally*100:.1f}% בלבד (נדרש ≥4%)."
            )
    criteria.append({
        "name": "Automatic Rally (AR)", "hit": ar_found,
        "points": 20, "earned": ar_points, "explanation": ar_explanation
    })

    # 3. No Supply
    recent_10 = df.iloc[-10:]
    avg_vol_10 = recent_10["Volume"].mean()
    global_mean = df["VOL_MEAN"].iloc[-1]
    no_supply = avg_vol_10 < global_mean * 0.7
    ns_points = 20 if no_supply else 0
    score += ns_points
    criteria.append({
        "name": "No Supply", "hit": no_supply,
        "points": 20, "earned": ns_points,
        "explanation": (
            f"ווליום ממוצע ב-10 ימים: {avg_vol_10/global_mean*100:.0f}% מהממוצע — "
            + ("המוכרים נעלמו." if no_supply else "ווליום עדיין גבוה מדי.")
        )
    })

    # 4. Divergence
    last_20 = df.iloc[-20:]
    price_change = (last_20["Close"].iloc[-1] - last_20["Close"].iloc[0]) / last_20["Close"].iloc[0]
    vol_change = (last_20["Volume"].iloc[-5:].mean() - last_20["Volume"].iloc[:5].mean()) / last_20["Volume"].iloc[:5].mean()
    divergence = (price_change < 0) and (vol_change < -0.25)
    div_points = 20 if divergence else 0
    score += div_points
    criteria.append({
        "name": "Price–Vol Divergence", "hit": divergence,
        "points": 20, "earned": div_points,
        "explanation": (
            f"מחיר: {price_change*100:+.1f}% | ווליום: {vol_change*100:+.1f}% (20 ימים). "
            + ("לחץ מכירה קורס בזמן שמחיר יורד — Smart Money." if divergence
               else "לא נמצאה דיברגנציה ברורה.")
        )
    })

    # 5. Trading Range
    last_15 = df.iloc[-15:]
    tr_range_pct = (last_15["High"].max() - last_15["Low"].min()) / last_15["Low"].min()
    in_range = tr_range_pct < 0.12
    tr_points = 15 if in_range else 0
    score += tr_points
    criteria.append({
        "name": "Trading Range", "hit": in_range,
        "points": 15, "earned": tr_points,
        "explanation": (
            f"טווח 15 ימים: {tr_range_pct*100:.1f}% "
            + ("— טווח צר, קלאסי לאיסוף שקט." if in_range else "— תנודתי מדי (>12%).")
        )
    })

    if score >= 75:
        verdict, verdict_color = "סבירות גבוהה לאיסוף מוסדי", "#26a69a"
    elif score >= 45:
        verdict, verdict_color = "סימנים חלקיים", "#ffa726"
    else:
        verdict, verdict_color = "אין ראיות לאיסוף", "#ef5350"

    return score, criteria, verdict, verdict_color, prereq_met, drawdown


def render_gauge(score, verdict, verdict_color):
    gauge_color = "#26a69a" if score >= 75 else "#ffa726" if score >= 45 else "#ef5350"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text': f"<b>Absorption Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>",
               'font': {'size': 13}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a6a8a"},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "#0d1b2a", 'borderwidth': 1, 'bordercolor': "#2a4a6a",
            'steps': [
                {'range': [0, 44],  'color': '#1a0d0d'},
                {'range': [44, 74], 'color': '#1a1206'},
                {'range': [74, 100],'color': '#0d1a18'},
            ],
            'threshold': {'line': {'color': "#ffffff", 'width': 2}, 'thickness': 0.75, 'value': score}
        },
        number={'font': {'size': 48, 'color': gauge_color}, 'suffix': '/100'}
    ))
    fig.update_layout(
        height=300, margin=dict(t=80, b=10, l=20, r=20),
        paper_bgcolor="#0a1520", font_color="#e0eaf4"
    )
    return fig


def render_chart(df, ticker):
    df_chart = df.iloc[-65:].copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"], close=df_chart["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="Price"
    ), row=1, col=1)
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df_chart["Close"], df_chart["Open"])]
    fig.add_trace(go.Bar(
        x=df_chart.index, y=df_chart["Volume"],
        marker_color=vol_colors, name="Volume", opacity=0.8
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart["VOL_MEAN"],
        line=dict(color="#4fc3f7", width=1.5, dash="dot"), name="Vol MA20"
    ), row=2, col=1)
    fig.update_layout(
        height=420, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
        font_color="#e0eaf4", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=10, b=10)
    )
    fig.update_xaxes(gridcolor="#1e3040")
    fig.update_yaxes(gridcolor="#1e3040")
    return fig


# ----------------------------
# UI — Ticker Input
# ----------------------------
st.markdown("### הכנס טיקרים לניתוח")
col_inp, col_btn = st.columns([4, 1])
with col_inp:
    raw_input = st.text_input(
        "טיקרים (מופרדים בפסיק או רווח)",
        "NVDA, MSFT, AMZN",
        help="לדוגמה: NVDA, MSFT, AMZN, AAPL"
    )
with col_btn:
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    run = st.button("▶ הרץ", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in raw_input.replace(",", " ").split() if t.strip()]
    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    if not tickers:
        st.error("יש להזין לפחות טיקר אחד.")
        st.stop()

    # --- Fetch all data ---
    results = {}
    progress = st.progress(0, text="שולף דאטה...")
    for i, t in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"מנתח {t}...")
        df = get_data(t)
        if df is None:
            results[t] = None
        else:
            score, criteria, verdict, verdict_color, prereq_met, drawdown = analyze_wyckoff(df)
            results[t] = {
                "df": df, "score": score, "criteria": criteria,
                "verdict": verdict, "verdict_color": verdict_color,
                "prereq_met": prereq_met, "drawdown": drawdown
            }
    progress.empty()

    valid = {t: v for t, v in results.items() if v is not None}
    failed = [t for t, v in results.items() if v is None]

    if failed:
        st.warning(f"לא נמצא דאטה מספיק עבור: {', '.join(failed)}")

    if not valid:
        st.error("לא נמצא דאטה תקין לאף טיקר.")
        st.stop()

    # =============================================
    # OVERVIEW PANEL (multi-ticker only)
    # =============================================
    if len(valid) > 1:
        st.markdown("---")
        st.markdown("### סקירה כללית — השוואת כל הניירות")

        # Sort by score descending
        sorted_tickers = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
        cols = st.columns(len(sorted_tickers))

        for col, t in zip(cols, sorted_tickers):
            r = valid[t]
            s = r["score"]
            color = r["verdict_color"]
            bar_pct = s
            with col:
                st.markdown(f"""
                <div class="overview-card">
                  <div class="ticker-label">{t}</div>
                  <div class="score-big" style="color:{color}">{s}</div>
                  <div style="color:#607d8b; font-size:0.72rem; font-family:'IBM Plex Mono',monospace;">/ 100</div>
                  <div class="verdict-label">{r['verdict']}</div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{bar_pct}%; background:{color};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Comparison gauge chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig_cmp = go.Figure()
        bar_colors = [valid[t]["verdict_color"] for t in sorted_tickers]
        fig_cmp.add_trace(go.Bar(
            x=sorted_tickers,
            y=[valid[t]["score"] for t in sorted_tickers],
            marker_color=bar_colors,
            text=[str(valid[t]["score"]) for t in sorted_tickers],
            textposition="outside",
            textfont=dict(color="#e0eaf4", family="IBM Plex Mono", size=14)
        ))
        fig_cmp.update_layout(
            height=280,
            paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
            font_color="#e0eaf4", font_family="IBM Plex Mono",
            yaxis=dict(range=[0, 110], gridcolor="#1e3040", title="Score"),
            xaxis=dict(gridcolor="#1e3040"),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # =============================================
    # INDIVIDUAL TABS
    # =============================================
    st.markdown("---")
    st.markdown("### ניתוח פרטני לכל נייר")

    tabs = st.tabs([f"{'🟢' if valid[t]['score'] >= 75 else '🟡' if valid[t]['score'] >= 45 else '🔴'} {t}" for t in valid])

    for tab, t in zip(tabs, valid):
        with tab:
            r = valid[t]
            df = r["df"]
            score = r["score"]
            criteria = r["criteria"]
            verdict = r["verdict"]
            verdict_color = r["verdict_color"]
            prereq_met = r["prereq_met"]
            drawdown = r["drawdown"]

            col_gauge, col_reasons = st.columns([1, 1], gap="large")

            with col_gauge:
                st.plotly_chart(render_gauge(score, verdict, verdict_color), use_container_width=True)
                if not prereq_met:
                    st.markdown(f"""
                    <div class="score-reason-box negative">
                    ⚠️ <strong>תנאי מקדים לא מתקיים:</strong> הנייר ירד רק {drawdown*100:.1f}% מהשיא (נדרש ≥12%).
                    SC רלוונטי רק לאחר ירידה משמעותית — הניתוח פחות אמין.
                    </div>
                    """, unsafe_allow_html=True)

            with col_reasons:
                st.markdown("#### פירוט הניקוד")
                for c in criteria:
                    hit_class = "hit" if c["hit"] else "miss"
                    hit_label = "✅ הצליח" if c["hit"] else "❌ נכשל"
                    box_class = "positive" if c["hit"] else "negative"
                    st.markdown(f"""
                    <div class="score-reason-box {box_class}">
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

            st.markdown(f"##### גרף מחיר וווליום — {t}")
            st.plotly_chart(render_chart(df, t), use_container_width=True)

    # =============================================
    # DISCLAIMER
    # =============================================
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>לידיעתך:</strong> הבוט מספק אנליזה טכנית אוטומטית בלבד ואינו המלצת השקעה.
    מתודולוגיית וייקוף היא כלי לזיהוי דפוסים ולא ערובה לתוצאות. תמיד בצע Due Diligence עצמאי.
    </div>
    """, unsafe_allow_html=True)