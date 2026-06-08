import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Institutional Scout")

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Hebrew', sans-serif;
        direction: rtl;
    }
    h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; direction: ltr; }

    /* MODE SWITCHER */
    .mode-switcher {
        display: flex;
        gap: 12px;
        margin-bottom: 28px;
        direction: ltr;
    }
    .mode-btn {
        flex: 1;
        padding: 14px 20px;
        border-radius: 10px;
        border: 2px solid #2a4a6a;
        background: #0d1b2a;
        color: #607d8b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        cursor: pointer;
        text-align: center;
        transition: all 0.2s;
    }
    .mode-btn.active-wyckoff {
        border-color: #4fc3f7;
        background: linear-gradient(135deg, #0f2030 0%, #1a3a5a 100%);
        color: #4fc3f7;
    }
    .mode-btn.active-vp {
        border-color: #ab47bc;
        background: linear-gradient(135deg, #1a0f2a 0%, #2a1a3a 100%);
        color: #ce93d8;
    }

    /* HEADER BOX */
    .header-box {
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 28px;
        color: #e0eaf4;
        direction: rtl;
        line-height: 1.9;
    }
    .header-box.wyckoff {
        background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 100%);
        border: 1px solid #2a4a6a;
    }
    .header-box.vp {
        background: linear-gradient(135deg, #160f23 0%, #251535 100%);
        border: 1px solid #4a2a6a;
    }
    .header-box h2 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        margin-bottom: 12px;
        direction: ltr;
    }
    .header-box.wyckoff h2 { color: #4fc3f7; }
    .header-box.vp      h2 { color: #ce93d8; }
    .header-box p { color: #b0c8e0; font-size: 0.92rem; margin: 6px 0; }
    .tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 3px 2px;
    }
    .tag-w { background: #1e3a5f; border: 1px solid #4fc3f7; color: #4fc3f7; }
    .tag-v { background: #2a1a4a; border: 1px solid #ab47bc; color: #ce93d8; }

    /* SCORE BOXES */
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
    .score-reason-box.negative { border-left-color: #ef5350; }
    .score-reason-box.vp-positive { background: #150d20; border-left-color: #ab47bc; }
    .score-reason-box.vp-negative { background: #150d20; border-left-color: #ef5350; }
    .score-reason-box strong { color: #fff; }

    .criteria-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #1e3040;
        font-size: 0.84rem;
    }
    .hit  { color: #26a69a; font-weight: 600; }
    .miss { color: #ef5350; }

    /* OVERVIEW CARDS */
    .overview-card {
        background: #0d1b2a;
        border: 1px solid #2a4a6a;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        direction: ltr;
    }
    .overview-card.vp-card { border-color: #4a2a6a; background: #120d1e; }
    .ticker-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .score-big {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 6px 0;
    }
    .verdict-label { font-size: 0.78rem; color: #b0c8e0; margin-top: 4px; }
    .bar-bg { background: #1e3040; border-radius: 4px; height: 8px; margin-top: 10px; overflow: hidden; }
    .bar-fill { height: 8px; border-radius: 4px; }

    /* DISCLAIMER */
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

# ============================================================
# SESSION STATE — MODE
# ============================================================
if "mode" not in st.session_state:
    st.session_state.mode = "wyckoff"

# ============================================================
# MODE SWITCHER BUTTONS
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT")

col_w, col_v = st.columns(2)
with col_w:
    if st.button("⬛  Wyckoff Accumulation", use_container_width=True,
                 type="primary" if st.session_state.mode == "wyckoff" else "secondary"):
        st.session_state.mode = "wyckoff"
        st.rerun()
with col_v:
    if st.button("🔮  Volume Profile", use_container_width=True,
                 type="primary" if st.session_state.mode == "vp" else "secondary"):
        st.session_state.mode = "vp"
        st.rerun()

st.markdown("---")
# ============================================================
# SHARED GAUGE RENDERER
# ============================================================
def render_gauge(score, verdict, verdict_color, mode="wyckoff"):
    if mode == "wyckoff":
        steps = [
            {'range': [0, 44],  'color': '#1a0d0d'},
            {'range': [44, 74], 'color': '#1a1206'},
            {'range': [74, 100],'color': '#0d1a18'},
        ]
        bar_color = "#26a69a" if score >= 75 else "#ffa726" if score >= 45 else "#ef5350"
    else:
        steps = [
            {'range': [0, 44],  'color': '#1a0d18'},
            {'range': [44, 74], 'color': '#1a0f2a'},
            {'range': [74, 100],'color': '#1a0d25'},
        ]
        bar_color = "#ab47bc" if score >= 75 else "#ffa726" if score >= 45 else "#ef5350"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={
            'text': f"<b>Institutional Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>",
            'font': {'size': 13}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a6a8a"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#0d1b2a", 'borderwidth': 1, 'bordercolor': "#2a4a6a",
            'steps': steps,
            'threshold': {'line': {'color': "#ffffff", 'width': 2}, 'thickness': 0.75, 'value': score}
        },
        number={'font': {'size': 48, 'color': bar_color}, 'suffix': '/100'}
    ))
    fig.update_layout(
        height=300, margin=dict(t=80, b=10, l=20, r=20),
        paper_bgcolor="#0a1520", font_color="#e0eaf4"
    )
    return fig


# ============================================================
# SHARED OVERVIEW BAR CHART
# ============================================================
def render_comparison_chart(valid, accent_color):
    sorted_tickers = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
    bar_colors = [valid[t]["verdict_color"] for t in sorted_tickers]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_tickers,
        y=[valid[t]["score"] for t in sorted_tickers],
        marker_color=bar_colors,
        text=[str(valid[t]["score"]) for t in sorted_tickers],
        textposition="outside",
        textfont=dict(color="#e0eaf4", family="IBM Plex Mono", size=14)
    ))
    fig.update_layout(
        height=280, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
        font_color="#e0eaf4", font_family="IBM Plex Mono",
        yaxis=dict(range=[0, 115], gridcolor="#1e3040", title="Score"),
        xaxis=dict(gridcolor="#1e3040"),
        margin=dict(t=20, b=20, l=20, r=20), showlegend=False
    )
    return fig, sorted_tickers


# ============================================================
# ██╗    ██╗██╗   ██╗ ██████╗██╗  ██╗ ██████╗ ███████╗███████╗
# ██║    ██║╚██╗ ██╔╝██╔════╝██║ ██╔╝██╔═══██╗██╔════╝██╔════╝
# ██║ █╗ ██║ ╚████╔╝ ██║     █████╔╝ ██║   ██║█████╗  █████╗
# ██║███╗██║  ╚██╔╝  ██║     ██╔═██╗ ██║   ██║██╔══╝  ██╔══╝
# ╚███╔███╔╝   ██║   ╚██████╗██║  ██╗╚██████╔╝██║     ██║
#  ╚══╝╚══╝    ╚═╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
# ============================================================

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
    ar_found = False; ar_points = 0
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
    criteria.append({"name": "Automatic Rally (AR)", "hit": ar_found,
                     "points": 20, "earned": ar_points, "explanation": ar_explanation})
# 3. No Supply
    avg_vol_10 = df.iloc[-10:]["Volume"].mean()
    global_mean = df["VOL_MEAN"].iloc[-1]
    no_supply = avg_vol_10 < global_mean * 0.7
    ns_points = 20 if no_supply else 0
    score += ns_points
    criteria.append({
        "name": "No Supply", "hit": no_supply,
        "points": 20, "earned": ns_points,
        "explanation": f"ווליום ממוצע ב-10 ימים: {avg_vol_10/global_mean*100:.0f}% מהממוצע — " +
                       ("המוכרים נעלמו." if no_supply else "ווליום עדיין גבוה מדי.")
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
        "explanation": f"מחיר: {price_change*100:+.1f}% | ווליום: {vol_change*100:+.1f}% (20 ימים). " +
                       ("לחץ מכירה קורס בזמן שמחיר יורד — Smart Money." if divergence
                        else "לא נמצאה דיברגנציה ברורה.")
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
        "explanation": f"טווח 15 ימים: {tr_range_pct*100:.1f}% " +
                       ("— טווח צר, קלאסי לאיסוף שקט." if in_range else "— תנודתי מדי (>12%).")
    })

    if score >= 75:
        verdict, vcolor = "סבירות גבוהה לאיסוף מוסדי", "#26a69a"
    elif score >= 45:
        verdict, vcolor = "סימנים חלקיים", "#ffa726"
    else:
        verdict, vcolor = "אין ראיות לאיסוף", "#ef5350"

    return score, criteria, verdict, vcolor, prereq_met, drawdown


def render_wyckoff_chart(df):
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
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart["Volume"],
                         marker_color=vol_colors, name="Volume", opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["VOL_MEAN"],
                              line=dict(color="#4fc3f7", width=1.5, dash="dot"), name="Vol MA20"),
                  row=2, col=1)
    fig.update_layout(height=420, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
                      font_color="#e0eaf4", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=10, b=10))
    fig.update_xaxes(gridcolor="#1e3040")
    fig.update_yaxes(gridcolor="#1e3040")
    return fig


# ============================================================
# ██╗   ██╗ ██████╗ ██╗     ██╗   ██╗███╗   ███╗███████╗
# ██║   ██║██╔═══██╗██║     ██║   ██║████╗ ████║██╔════╝
# ██║   ██║██║   ██║██║     ██║   ██║██╔████╔██║█████╗
# ╚██╗ ██╔╝██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══╝
#  ╚████╔╝ ╚██████╔╝███████╗╚██████╔╝██║ ╚═╝ ██║███████╗
#   ╚═══╝   ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
# PROFILE
# ============================================================
def build_volume_profile(df, bins=40):
    """Build a price-level volume profile histogram."""
    price_min = df["Low"].min()
    price_max = df["High"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    vol_at_price = np.zeros(bins)

    for _, row in df.iterrows():
        lo, hi, vol = row["Low"], row["High"], row["Volume"]
        if hi == lo:
            continue
        for i in range(bins):
            overlap_lo = max(edges[i], lo)
            overlap_hi = min(edges[i + 1], hi)
            if overlap_hi > overlap_lo:
                fraction = (overlap_hi - overlap_lo) / (hi - lo)
                vol_at_price[i] += vol * fraction

    midpoints = (edges[:-1] + edges[1:]) / 2
    return midpoints, vol_at_price, edges


def analyze_vp(df):
    score = 0
    criteria = []
    current_price = df["Close"].iloc[-1]

    midpoints, vol_at_price, edges = build_volume_profile(df)
    total_vol = vol_at_price.sum()

    # --- POC (Point of Control) ---
    poc_idx = np.argmax(vol_at_price)
    poc_price = midpoints[poc_idx]
    poc_vol_pct = vol_at_price[poc_idx] / total_vol * 100

    # --- Value Area (70% of total volume) ---
    sorted_idx = np.argsort(vol_at_price)[::-1]
    va_vol = 0
    va_indices = []
    for i in sorted_idx:
        if va_vol >= total_vol * 0.70:
            break
        va_vol += vol_at_price[i]
        va_indices.append(i)
    vah = midpoints[max(va_indices)]
    val = midpoints[min(va_indices)]

    # --- Low Volume Node below current price (LVN — vacuum zone) ---
    below_current = midpoints < current_price
    if below_current.any():
        vol_below = vol_at_price[below_current]
        lvn_threshold = np.percentile(vol_at_price, 20)
        lvn_count = np.sum(vol_below < lvn_threshold)
        has_lvn_below = lvn_count >= 2
    else:
        has_lvn_below = False

    # --- High Volume Node above current price (HVN — institutional ceiling) ---
    above_current = midpoints > current_price
    if above_current.any():
        vol_above = vol_at_price[above_current]
        hvn_threshold = np.percentile(vol_at_price, 75)
        hvn_above_count = np.sum(vol_above > hvn_threshold)
        has_hvn_above = hvn_above_count >= 1
    else:
        has_hvn_above = False

    # --- Price near POC (within 3%) ---
    poc_distance = abs(current_price - poc_price) / poc_price
    near_poc = poc_distance <= 0.03

    # --- Price below Value Area Low (cheap vs. institutional fair value) ---
    below_val = current_price < val
    price_to_val_pct = (val - current_price) / val * 100 if below_val else 0

    # --- Recent volume surge at current level ---
    recent_30 = df.iloc[-30:]
    recent_profile_mids, recent_profile_vols, _ = build_volume_profile(recent_30, bins=20)
    closest_recent = np.argmin(np.abs(recent_profile_mids - current_price))
    recent_avg_vol = recent_profile_vols.mean()
    current_level_vol = recent_profile_vols[closest_recent]
    vol_surge = current_level_vol > recent_avg_vol * 1.8

    # ---- SCORING ----

    # 1. Price below VAL (25 pts)
    bval_points = 25 if below_val else 0
    score += bval_points
    criteria.append({
        "name": "מחיר מתחת ל-Value Area Low (VAL)",
        "hit": below_val,
        "points": 25, "earned": bval_points,
        "explanation": (
            f"מחיר נוכחי ({current_price:.2f}) נמצא {price_to_val_pct:.1f}% מתחת ל-VAL ({val:.2f}). "
            "מחיר מתחת לאזור הערך = נייר נסחר בדיסקאונט ביחס לאזור בו מוסדיים ריכזו פעילות."
            if below_val else
            f"מחיר ({current_price:.2f}) בתוך Value Area או מעליו (VAL={val:.2f}, VAH={vah:.2f}). "
            "אין דיסקאונט מובהק ביחס לאזור הערך."
        )
    })

    # 2. LVN below current price (20 pts)
    lvn_points = 20 if has_lvn_below else 0
    score += lvn_points
    criteria.append({
        "name": "LVN — אזור ריק מתחת למחיר",
        "hit": has_lvn_below,
        "points": 20, "earned": lvn_points,
        "explanation": (
            "זוהו Low Volume Nodes (אזורי ריק) מתחת למחיר הנוכחי. "
            "אזורי LVN פועלים כ'מגנטים' — המחיר נוטה לנוע דרכם במהירות, מה שמשאיר מעט 'אוויר' בין המחיר הנוכחי לבין תמיכות משמעותיות."
            if has_lvn_below else
            "לא זוהו LVN משמעותיים מתחת למחיר. הסיכון לירידה חדה קטן יותר, אך גם פוטנציאל ה-snapback מוגבל."
        )
    })
# 3. HVN above price (20 pts) — ceiling that muffles upside BUT signals where institutions loaded
    hvn_points = 20 if has_hvn_above else 0
    score += hvn_points
    criteria.append({
        "name": "HVN — ריכוז פעילות מעל המחיר",
        "hit": has_hvn_above,
        "points": 20, "earned": hvn_points,
        "explanation": (
            f"זוהו {hvn_above_count} High Volume Nodes מעל המחיר הנוכחי. "
            "HVN מעיד על אזורים בהם מוסדיים בנו פוזיציות בעבר — כשהמחיר חוזר לשם, הם עשויים להוסיף."
            if has_hvn_above else
            "לא זוהו HVN מעל המחיר. פחות ראיות לנוכחות מוסדית ביעדים הקרובים."
        )
    })

    # 4. Near POC (20 pts)
    poc_points = 20 if near_poc else 0
    score += poc_points
    criteria.append({
        "name": "מחיר סמוך ל-POC",
        "hit": near_poc,
        "points": 20, "earned": poc_points,
        "explanation": (
            f"מחיר ({current_price:.2f}) נמצא במרחק {poc_distance*100:.1f}% מה-POC ({poc_price:.2f}). "
            "ה-POC הוא רמת המחיר בעלת הנפח הגבוה ביותר — מוסדיים נוטים לצבור שם. קרבה ל-POC מגבירה את ההסתברות לתגובה חיובית."
            if near_poc else
            f"מחיר ({current_price:.2f}) רחוק מה-POC ({poc_price:.2f}) ב-{poc_distance*100:.1f}% — מעל הסף של 3%."
        )
    })

    # 5. Recent vol surge at level (15 pts)
    surge_points = 15 if vol_surge else 0
    score += surge_points
    criteria.append({
        "name": "Volume Surge ברמה הנוכחית",
        "hit": vol_surge,
        "points": 15, "earned": surge_points,
        "explanation": (
            f"נפח ב-30 הימים האחרונים ברמת המחיר הנוכחית גבוה ב-{(current_level_vol/recent_avg_vol - 1)*100:.0f}% מהממוצע. "
            "ספיגת נפח חריגה ברמה ספציפית = חתימת Smart Money טרייה."
            if vol_surge else
            "לא זוהה volume surge חריג ברמת המחיר הנוכחית ב-30 הימים האחרונים."
        )
    })

    if score >= 75:
        verdict, vcolor = "סבירות גבוהה לנוכחות מוסדית", "#ab47bc"
    elif score >= 45:
        verdict, vcolor = "סימנים חלקיים — ניטור מומלץ", "#ffa726"
    else:
        verdict, vcolor = "אין ריכוז מוסדי מובהק", "#ef5350"

    vp_data = {
        "poc": poc_price, "vah": vah, "val": val,
        "midpoints": midpoints, "vol_at_price": vol_at_price,
        "poc_vol_pct": poc_vol_pct
    }
    return score, criteria, verdict, vcolor, vp_data


def render_vp_chart(df, vp_data, ticker):
    """Horizontal Volume Profile + candlestick side by side."""
    current_price = df["Close"].iloc[-1]
    df_chart = df.iloc[-65:].copy()

    midpoints = vp_data["midpoints"]
    vol_at_price = vp_data["vol_at_price"]
    poc = vp_data["poc"]
    vah = vp_data["vah"]
    val = vp_data["val"]
    # Normalize vol bars to fraction of chart width
    max_vol = vol_at_price.max()
    bar_width = 0.12  # fraction of price range
    price_range = df["High"].max() - df["Low"].min()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        shared_yaxes=True,
        horizontal_spacing=0.01
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"], close=df_chart["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="Price"
    ), row=1, col=1)

    # POC / VAH / VAL lines on candle chart
    x_range = [df_chart.index[0], df_chart.index[-1]]
    for level, color, label in [(poc, "#ce93d8", "POC"), (vah, "#4fc3f7", "VAH"), (val, "#4fc3f7", "VAL")]:
        fig.add_trace(go.Scatter(
            x=x_range, y=[level, level],
            mode="lines+text",
            line=dict(color=color, width=1.5, dash="dash"),
            text=["", f" {label}: {level:.2f}"],
            textposition="top right",
            textfont=dict(color=color, size=10),
            name=label, showlegend=True
        ), row=1, col=1)

    # Current price line
    fig.add_trace(go.Scatter(
        x=x_range, y=[current_price, current_price],
        mode="lines", line=dict(color="#ffffff", width=1, dash="dot"),
        name=f"Current: {current_price:.2f}", showlegend=True
    ), row=1, col=1)

    # Volume Profile — horizontal bars
    bar_colors_vp = []
    for m in midpoints:
        if abs(m - poc) < (midpoints[1] - midpoints[0]):
            bar_colors_vp.append("#ce93d8")
        elif val <= m <= vah:
            bar_colors_vp.append("#5c35a0")
        else:
            bar_colors_vp.append("#2a3a5a")

    normalized_vols = vol_at_price / max_vol * 100  # scale to 0-100

    fig.add_trace(go.Bar(
        x=normalized_vols,
        y=midpoints,
        orientation='h',
        marker_color=bar_colors_vp,
        name="Vol Profile",
        opacity=0.9,
        width=(midpoints[1] - midpoints[0]) * 0.85
    ), row=1, col=2)

    fig.update_layout(
        height=500,
        paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a",
        font_color="#e0eaf4",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
        margin=dict(t=20, b=20, l=10, r=10)
    )
    fig.update_xaxes(gridcolor="#1e3040")
    fig.update_yaxes(gridcolor="#1e3040")
    fig.update_xaxes(title_text="Vol %", row=1, col=2)

    return fig


# ============================================================
# ██╗    ██╗██╗   ██╗ ██████╗██╗  ██╗ ██████╗ ███████╗███████╗
# WYCKOFF SCREEN
# ============================================================
def screen_wyckoff():
st.markdown("""
    <div class="header-box wyckoff">
      <h2>⬛ WYCKOFF ACCUMULATION SCOUT</h2>
      <p>מחפש חתימות של <strong>איסוף מוסדי מוקדם</strong> לפי מתודולוגיית ריצ'רד וייקוף.
      מוסדיים בונים פוזיציות בשקט, כשהציבור עדיין פוחד. הבוט מזהה שלב זה ע"י חמישה קריטריונים.</p>
      <p>
        <span class="tag tag-w">SC – Selling Climax · 25 נק'</span>
        <span class="tag tag-w">AR – Automatic Rally · 20 נק'</span>
        <span class="tag tag-w">No Supply · 20 נק'</span>
        <span class="tag tag-w">Price–Vol Divergence · 20 נק'</span>
        <span class="tag tag-w">Trading Range · 15 נק'</span>
      </p>
      <p>
        <strong>SC:</strong> ווליום ≥2× ממוצע + זנב תחתון — פאניקה שנבלמת ע"י קונים גדולים.<br>
        <strong>AR:</strong> עלייה ≥4% תוך 10 ימים לאחר ה-SC — אישור ראשון להסרת היצע.<br>
        <strong>No Supply:</strong> ווליום &lt;70% מהממוצע ב-10 ימים — המוכרים נעלמו.<br>
        <strong>Divergence:</strong> מחיר יורד אבל ווליום קורס — לחץ המכירה מתייבש.<br>
        <strong>Trading Range:</strong> טווח 15 ימים &lt;12% — צבירה שקטה קלאסית.
      </p>
      <p style="color:#607d8b; font-size:0.82rem;">⚠️ תנאי מקדים: ירידה ≥12% מהשיא ב-3 חודשים.</p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        raw = st.text_input("טיקרים (פסיק או רווח)", "NVDA, MSFT, AMZN", key="w_input")
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run = st.button("▶ הרץ", use_container_width=True, key="w_run")

    if not run:
        return

    tickers = list(dict.fromkeys([t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]))
    if not tickers:
        st.error("יש להזין לפחות טיקר אחד.")
        return

    results = {}
    prog = st.progress(0, text="שולף דאטה...")
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"מנתח {t}...")
        df = get_data(t)
        if df is None:
            results[t] = None
        else:
            sc, cr, vd, vc, pm, dd = analyze_wyckoff(df)
            results[t] = {"df": df, "score": sc, "criteria": cr,
                          "verdict": vd, "verdict_color": vc,
                          "prereq_met": pm, "drawdown": dd}
    prog.empty()

    valid = {t: v for t, v in results.items() if v is not None}
    failed = [t for t in results if results[t] is None]
    if failed:
        st.warning(f"לא נמצא דאטה עבור: {', '.join(failed)}")
    if not valid:
        st.error("לא נמצא דאטה תקין.")
        return

    # Overview
    if len(valid) > 1:
        st.markdown("---")
        st.markdown("### סקירה כללית")
        sorted_t = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
        cols = st.columns(len(sorted_t))
        for col, t in zip(cols, sorted_t):
            r = valid[t]; s = r["score"]; c = r["verdict_color"]
            with col:
                st.markdown(f"""
                <div class="overview-card">
                  <div class="ticker-label" style="color:#4fc3f7">{t}</div>
                  <div class="score-big" style="color:{c}">{s}</div>
                  <div style="color:#607d8b;font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">/ 100</div>
                  <div class="verdict-label">{r['verdict']}</div>
                  <div class="bar-bg"><div class="bar-fill" style="width:{s}%;background:{c}"></div></div>
                </div>""", unsafe_allow_html=True)

        fig_cmp, _ = render_comparison_chart(valid, "#4fc3f7")
        st.plotly_chart(fig_cmp, use_container_width=True)
    # Individual tabs
    st.markdown("---")
    st.markdown("### ניתוח פרטני")
    tabs = st.tabs([f"{'🟢' if valid[t]['score']>=75 else '🟡' if valid[t]['score']>=45 else '🔴'} {t}" for t in valid])

    for tab, t in zip(tabs, valid):
        with tab:
            r = valid[t]
            cg, cr = st.columns([1, 1], gap="large")
            with cg:
                st.plotly_chart(render_gauge(r["score"], r["verdict"], r["verdict_color"], "wyckoff"),
                                use_container_width=True)
                if not r["prereq_met"]:
                    st.markdown(f"""<div class="score-reason-box negative">
                    ⚠️ <strong>תנאי מקדים לא מתקיים:</strong> ירידה של {r['drawdown']*100:.1f}% בלבד (נדרש ≥12%).
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown("#### פירוט הניקוד")
                for c in r["criteria"]:
                    box = "positive" if c["hit"] else "negative"
                    lbl = "✅ הצליח" if c["hit"] else "❌ נכשל"
                    cls = "hit" if c["hit"] else "miss"
                    st.markdown(f"""
                    <div class="score-reason-box {box}">
                      <div class="criteria-row">
                        <strong>{c['name']}</strong>
                        <span><span class="{cls}">{lbl}</span> &nbsp;|&nbsp; <strong>{c['earned']}/{c['points']} נק'</strong></span>
                      </div>
                      <div style="margin-top:6px;color:#b0c8e0">{c['explanation']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"##### גרף — {t}")
            st.plotly_chart(render_wyckoff_chart(r["df"]), use_container_width=True)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה. תמיד בצע Due Diligence עצמאי.</div>""",
                unsafe_allow_html=True)


# ============================================================
# VOLUME PROFILE SCREEN
# ============================================================
def screen_vp():
    st.markdown("""
    <div class="header-box vp">
      <h2>🔮 VOLUME PROFILE SCOUT</h2>
      <p>מנתח <strong>פרופיל הנפח (Volume Profile)</strong> — פיזור הנפח לפי רמות מחיר לאורך שנה.
      בניגוד לניתוח וייקוף שמסתכל על זמן, VP מסתכל על <em>מחיר</em>: איפה המוסדיים באמת ישבו.</p>
      <p><strong>ארבעה מושגי יסוד:</strong></p>
      <p>
        <span class="tag tag-v">POC – Point of Control</span>
        <span class="tag tag-v">Value Area (VAH/VAL)</span>
        <span class="tag tag-v">HVN – High Volume Node</span>
        <span class="tag tag-v">LVN – Low Volume Node</span>
      </p>
      <p>
        <strong>POC:</strong> רמת המחיר עם הנפח הגבוה ביותר בכל התקופה — אזור האיזון המוסדי.<br>
        <strong>Value Area:</strong> 70% מסך הנפח — הטווח שבו "הכסף החכם" סחר את רוב הזמן.<br>
        <strong>VAH/VAL:</strong> גבולות אזור הערך. מחיר מתחת ל-VAL = דיסקאונט, מעל VAH = פרמיה.<br>
        <strong>HVN:</strong> ריכוז פעילות = תמיכה/התנגדות חזקה, שם מוסדיים מחזיקים סחורה.<br>
        <strong>LVN:</strong> אזור ריק = המחיר נוטה לנוע דרכו מהר, בלי התנגדות.
      </p>
      <p><strong>חמשת הקריטריונים (0–100):</strong></p>
      <p>
        <span class="tag tag-v">מחיר מתחת ל-VAL · 25 נק'</span>
        <span class="tag tag-v">LVN מתחת למחיר · 20 נק'</span>
        <span class="tag tag-v">HVN מעל המחיר · 20 נק'</span>
        <span class="tag tag-v">קרוב ל-POC · 20 נק'</span>
        <span class="tag tag-v">Volume Surge ברמה · 15 נק'</span>
      </p>
      <p style="color:#607d8b; font-size:0.82rem;">
        ✦ הגרף מציג פרופיל אופקי: כל שורה = רמת מחיר, אורך הבר = כמה נפח נסחר שם.
        סגול כהה = Value Area. סגול בהיר = POC.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        raw = st.text_input("טיקרים (פסיק או רווח)", "NVDA, MSFT, AMZN", key="vp_input")
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run = st.button("▶ הרץ", use_container_width=True, key="vp_run")

    if not run:
        return

    tickers = list(dict.fromkeys([t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]))
    if not tickers:
        st.error("יש להזין לפחות טיקר אחד.")
        return

    results = {}
    prog = st.progress(0, text="בונה פרופיל נפח...")
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"מנתח {t}...")
        df = get_data(t)
        if df is None:
            results[t] = None
        else:
            sc, cr, vd, vc, vpd = analyze_vp(df)
            results[t] = {"df": df, "score": sc, "criteria": cr,
                          "verdict": vd, "verdict_color": vc, "vp_data": vpd}
    prog.empty()

    valid = {t: v for t, v in results.items() if v is not None}
    failed = [t for t in results if results[t] is None]
    if failed:
        st.warning(f"לא נמצא דאטה עבור: {', '.join(failed)}")
    if not valid:
        st.error("לא נמצא דאטה תקין.")
        return
# Overview
    if len(valid) > 1:
        st.markdown("---")
        st.markdown("### סקירה כללית")
        sorted_t = sorted(valid.keys(), key=lambda t: valid[t]["score"], reverse=True)
        cols = st.columns(len(sorted_t))
        for col, t in zip(cols, sorted_t):
            r = valid[t]; s = r["score"]; c = r["verdict_color"]
            vpd = r["vp_data"]
            with col:
                st.markdown(f"""
                <div class="overview-card vp-card">
                  <div class="ticker-label" style="color:#ce93d8">{t}</div>
                  <div class="score-big" style="color:{c}">{s}</div>
                  <div style="color:#607d8b;font-size:0.72rem;font-family:'IBM Plex Mono',monospace;">/ 100</div>
                  <div class="verdict-label">{r['verdict']}</div>
                  <div style="font-size:0.72rem;color:#8a6a9a;margin-top:6px;font-family:'IBM Plex Mono',monospace;">
                    POC {vpd['poc']:.2f} | VAH {vpd['vah']:.2f} | VAL {vpd['val']:.2f}
                  </div>
                  <div class="bar-bg"><div class="bar-fill" style="width:{s}%;background:{c}"></div></div>
                </div>""", unsafe_allow_html=True)

        fig_cmp, _ = render_comparison_chart(valid, "#ab47bc")
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Individual tabs
    st.markdown("---")
    st.markdown("### ניתוח פרטני")
    tabs = st.tabs([f"{'🟣' if valid[t]['score']>=75 else '🟡' if valid[t]['score']>=45 else '🔴'} {t}" for t in valid])

    for tab, t in zip(tabs, valid):
        with tab:
            r = valid[t]
            vpd = r["vp_data"]
            current = r["df"]["Close"].iloc[-1]

            # POC / VAH / VAL summary line
            st.markdown(f"""
            <div style="direction:ltr;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                        color:#b0b0c0;background:#0d1220;border-radius:6px;padding:8px 14px;margin-bottom:12px;">
              Current: <b style="color:#fff">{current:.2f}</b> &nbsp;|&nbsp;
              POC: <b style="color:#ce93d8">{vpd['poc']:.2f}</b> &nbsp;|&nbsp;
              VAH: <b style="color:#4fc3f7">{vpd['vah']:.2f}</b> &nbsp;|&nbsp;
              VAL: <b style="color:#4fc3f7">{vpd['val']:.2f}</b> &nbsp;|&nbsp;
              POC vol share: <b style="color:#ce93d8">{vpd['poc_vol_pct']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

            cg, cr = st.columns([1, 1], gap="large")
            with cg:
                st.plotly_chart(render_gauge(r["score"], r["verdict"], r["verdict_color"], "vp"),
                                use_container_width=True)
            with cr:
                st.markdown("#### פירוט הניקוד")
                for c in r["criteria"]:
                    box = "vp-positive" if c["hit"] else "vp-negative"
                    lbl = "✅ הצליח" if c["hit"] else "❌ נכשל"
                    cls = "hit" if c["hit"] else "miss"
                    st.markdown(f"""
                    <div class="score-reason-box {box}">
                      <div class="criteria-row">
                        <strong>{c['name']}</strong>
                        <span><span class="{cls}">{lbl}</span> &nbsp;|&nbsp; <strong>{c['earned']}/{c['points']} נק'</strong></span>
                      </div>
                      <div style="margin-top:6px;color:#c8b0d8">{c['explanation']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"##### Volume Profile + מחיר — {t} (שנה אחרונה)")
            st.plotly_chart(render_vp_chart(r["df"], vpd, t), use_container_width=True)

    st.markdown("""<div class="disclaimer">⚠️ אנליזה טכנית בלבד, אינה המלצת השקעה. תמיד בצע Due Diligence עצמאי.</div>""",
                unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================
if st.session_state.mode == "wyckoff":
    screen_wyckoff()
else:
    screen_vp()