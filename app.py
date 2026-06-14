# ============================================================
# INSTITUTIONAL SCOUT PRO - FINAL UI V10.9 (HARD REBOOT)
# ============================================================
import sys
import os
import json
import pickle
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import streamlit as st
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ייבוא מפורש מ-scout_core — אין import *
from scout_core import (
    clean_filename,
    get_data,
    calculate_optimal_threshold,
    check_phase_entry_allowed,
    BacktestConfig,
    FactorEngine,
    run_wyckoff_anchored_backtest,
    build_research_ground_truth,
)

# ============================================================
# הגנה על ייבוא הטריינר - מונע קריסה מוחלטת של האפליקציה
# ============================================================
try:
    from trainer_core import run_auto_trainer
    TRAINER_AVAILABLE = True
    TRAINER_ERROR = ""
except ModuleNotFoundError as e:
    run_auto_trainer = None
    TRAINER_AVAILABLE = False
    TRAINER_ERROR = str(e)
except Exception as e:
    run_auto_trainer = None
    TRAINER_AVAILABLE = False
    TRAINER_ERROR = f"שגיאה פנימית בקובץ הטריינר: {str(e)}"

st.set_page_config(layout="wide", page_title="Institutional Scout Pro")

# ============================================================
# קבועים
# ============================================================
AUTO_TRAINER_STATUS_FILE = os.path.join(BASE_DIR, "models", "auto_trainer_status.json")
AUTO_TRAINER_DONE_FLAG   = os.path.join(BASE_DIR, "models", "auto_trainer.done")
LOG_FILE_PATH            = os.path.join(BASE_DIR, "auto_trainer_error.log")

# ============================================================
# פונקציות עזר — שמירה / טעינה
# ============================================================
def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs("models", exist_ok=True)
    safe_name = clean_filename(str(slot_name))
    file_path = f"models/model_{safe_name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata, "phase_encoder": encoder}, f)
    return file_path


def load_all_models_from_disk():
    loaded = {}
    if os.path.exists("models"):
        for filename in os.listdir("models"):
            if filename.endswith(".pkl"):
                filepath = os.path.join("models", filename)
                try:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)
                    slot = data.get("metadata", {}).get("slot", filename)
                    loaded[slot] = data
                except Exception:
                    pass
    return loaded


def load_all_research_dfs_from_disk():
    archive = {}
    if os.path.exists("research_labels"):
        for filename in os.listdir("research_labels"):
            if filename.endswith(".csv"):
                filepath = os.path.join("research_labels", filename)
                try:
                    df  = pd.read_csv(filepath)
                    key = filename.replace("research_", "").replace(".csv", "")
                    archive[key] = df
                except Exception:
                    pass
    return archive


def read_auto_trainer_status():
    default = {
        "state": "idle", "message": "לא רץ כרגע",
        "progress": 0, "current_slot": "N/A",
        "updated_at": "N/A", "started_at": "N/A", "finished_at": "N/A",
    }
    if os.path.exists(AUTO_TRAINER_STATUS_FILE):
        try:
            with open(AUTO_TRAINER_STATUS_FILE, "r", encoding="utf-8") as f:
                default.update(json.load(f))
        except Exception:
            pass
    elif os.path.exists(AUTO_TRAINER_DONE_FLAG):
        default.update({"state": "completed", "message": "האימון הסתיים", "progress": 100})
    return default

# ============================================================
# Universe
# ============================================================
SCAN_UNIVERSE = list(dict.fromkeys([
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","JNJ",
    "V","UNH","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP",
    "KO","AVGO","COST","WMT","LLY","TMO","MCD","ACN","BAC","CRM",
    "NFLX","AMD","ADBE","CSCO","ABT","TXN","NEE","DHR","RTX","QCOM",
    "HON","NKE","INTC","AMGN","PM","IBM","SBUX","INTU","GS","CAT",
    "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","ADI","GILD",
    "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
    "C","MS","CVS","CI","SLB","EOG","OXY","COP","PSX","VLO",
    "AMT","PLD","CCI","EQIX","SPG","O","WELL","DLR",
    "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
    "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
    "UBER","ABNB","COIN","SOFI","UPST",
    "F","GM","RIVN","NIO",
    "ONTO","KLAC","LRCX","AMAT","MRVL","SMCI","DELL","HPQ",
    "DIS","CMCSA","RBLX","U","TTWO","EA",
    "DAL","UAL","AAL","LUV","FDX","UPS","XPO","ODFL",
    "DKNG","MGM","CZR","RCL","CCL","MAR","HLT",
]))

SECTOR_MAP = {
    "הכול (כל השוק האמריקאי)": SCAN_UNIVERSE,
    "צמיחה וטכנולוגיה (Growth)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
        "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
        "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
        "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
        "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA",
    ],
    "ערך ומדד (Value/Index)": [
        "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
        "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
        "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
        "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
        "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
        "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
        "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
        "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT",
    ],
    "סחורות ואנרגיה (Commodities)": [
        "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
        "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
        "GLD","SLV",
    ],
}

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Hebrew', sans-serif; direction: rtl; text-align: right; box-sizing: border-box; }
h1, h2, h3, h4, h5, h6 { direction: rtl; }
.header-box { border-radius: 12px; padding: 24px 32px; margin-bottom: 28px; color: #e0eaf4; line-height: 1.9; }
.header-box.wyckoff   { background: linear-gradient(135deg, #0f1923, #1a2a3a); border: 1px solid #2a4a6a; }
.header-box.vp        { background: linear-gradient(135deg, #160f23, #251535); border: 1px solid #4a2a6a; }
.header-box.vwap      { background: linear-gradient(135deg, #0f2318, #1a3528); border: 1px solid #2a6a4a; }
.header-box.composite { background: linear-gradient(135deg, #1a1208, #2a1e08); border: 1px solid #6a4a1a; }
.header-box.ml        { background: linear-gradient(135deg, #1c0a20, #2e1236); border: 1px solid #7b1fa2; }
.header-box.scanner   { background: linear-gradient(135deg, #0f231f, #1a3a35); border: 1px solid #26a69a; }
.header-box.monitor   { background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #7f8c8d; }
.widget-panel-ai { background: #111922; border: 1px solid #2d3d4f; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
.audit-row { padding: 12px; margin-bottom: 8px; border-radius: 5px; border-right: 4px solid; }
.win  { background: rgba(38, 166, 154, 0.1); border-color: #26a69a; }
.loss { background: rgba(239, 83, 80, 0.1);  border-color: #ef5350; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Cache נתונים
# ============================================================
@st.cache_data(ttl=3600)
def get_cached_data(ticker, period="1y", start=None, end=None):
    try:
        effective_period = None if (start or end) else period
        df = get_data(ticker, effective_period, start, end)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end) if (start or end) else t.history(period=period or "1y")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None

# ============================================================
# Session State
# ============================================================
if "mode"             not in st.session_state: st.session_state.mode             = "wyckoff"
if "ml_model"         not in st.session_state: st.session_state.ml_model         = None
if "ml_metadata"      not in st.session_state: st.session_state.ml_metadata      = None
if "use_ml"           not in st.session_state: st.session_state.use_ml           = False
if "phase_encoder"    not in st.session_state: st.session_state.phase_encoder    = None
if "model_archive"    not in st.session_state: st.session_state.model_archive    = load_all_models_from_disk()
if "research_archive" not in st.session_state: st.session_state.research_archive = load_all_research_dfs_from_disk()

# ============================================================
# UI helpers
# ============================================================
def render_threshold_control(label, key):
    if key not in st.session_state:
        st.session_state[key] = 65
    st.markdown(f"**{label}**")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state[key] = st.slider("", 40, 95, st.session_state[key],
                                           key=f"{key}_slider", label_visibility="collapsed")
    with col2:
        st.number_input("", 40, 95, st.session_state[key],
                        key=f"{key}_num", label_visibility="collapsed")
    return st.session_state[key]


def render_active_ai_selector_widget(screen_identifier):
    st.markdown("<div class='widget-panel-ai'>", unsafe_allow_html=True)
    st.markdown("### 🧠 הגדרות מנוע החלטה AI חכם")
    col_a, col_b, col_c = st.columns([2, 1.5, 1])
    with col_a:
        if st.session_state.model_archive:
            slots_list    = list(st.session_state.model_archive.keys())
            selected_slot = st.selectbox("בחר מודל מוסדי פעיל:", slots_list,
                                         key=f"selector_slot_{screen_identifier}")
            if st.button("✅ טען והפעל מודל", key=f"activate_btn_{screen_identifier}",
                         use_container_width=True):
                target_data = st.session_state.model_archive[selected_slot]
                st.session_state.ml_model      = target_data["model"]
                st.session_state.ml_metadata   = target_data["metadata"]
                st.session_state.phase_encoder = target_data.get("phase_encoder")
                st.session_state.use_ml        = True
                st.success(f"המודל '{selected_slot}' הופעל בהצלחה!")
                st.rerun()
        else:
            st.info("לא נמצאו מודלים בזיכרון. הרץ אימון ידני או אוטומטי.")
    with col_b:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 רענן מודלים מהדיסק", key=f"sync_git_{screen_identifier}",
                     use_container_width=True):
            st.session_state.model_archive = load_all_models_from_disk()
            st.rerun()
    with col_c:
        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
        ai_toggle = st.checkbox("הפעל שימוש ב-AI", value=st.session_state.use_ml,
                                key=f"checkbox_ai_{screen_identifier}")
        if ai_toggle != st.session_state.use_ml:
            st.session_state.use_ml = ai_toggle
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ניווט
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
nav = [
    ("wyckoff","⬛ Wyckoff"), ("vp","🔮 VP"), ("vwap","📊 VWAP"),
    ("composite","📈 Composite"), ("backtest","📊 Backtest"),
    ("ml","🧠 ML Trainer"), ("scanner","🔎 Scanner"), ("monitor","👁️ Monitor"),
]
for col, (mode_key, label) in zip([c1,c2,c3,c4,c5,c6,c7,c8], nav):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.mode == mode_key else "secondary",
                     key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key
            st.rerun()
st.markdown("---")

if st.session_state.use_ml and st.session_state.ml_model is not None:
    metadata = st.session_state.ml_metadata or {}
    acc      = metadata.get("test_acc", metadata.get("train_acc", 0.0))
    rec_th   = metadata.get("recommended_threshold", "לא חושב")
    tr_count = metadata.get("num_trades", "?")
    st.info(
        f"🧠 **מצב AI מופעל:** {metadata.get('slot','כללי')} | "
        f"דיוק OOB אמיתי: {acc*100:.1f}% | "
        f"🎯 **ציון סף מומלץ לכניסה:** {rec_th} | "
        f"מאומן על {tr_count} עסקאות היסטוריות"
    )

# ============================================================
# מסכים
# ============================================================
def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF 3.0 STRUCTURAL ENGINE</h2></div>""",
                unsafe_allow_html=True)
    render_active_ai_selector_widget("wyckoff_screen")
    c1, c2 = st.columns([4, 1])
    with c1:
        ticker = st.text_input("סמל לניתוח:", "NVDA", key="w_ticker")
    with c2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        btn = st.button("▶ הרץ ניתוח", use_container_width=True, type="primary")
    if btn:
        with st.spinner("מנתח דרך FactorEngine..."):
            df = get_cached_data(ticker.upper())
            if df is None or df.empty:
                st.error("אין נתונים.")
                return
            try:
                engine       = FactorEngine(BacktestConfig())
                factors      = engine.compute(df)
                phase_series = engine.get_wyckoff_phase(df)
                cis_series   = engine.composite_cis(factors, df)
                if factors is None or factors.empty or cis_series is None or len(cis_series) == 0:
                    st.warning("לא התקבלה תוצאת פקטורים תקינה מה-Engine.")
                    return
                current_phase = phase_series.iloc[-1] if hasattr(phase_series, "iloc") else phase_series
                current_cis   = float(cis_series.iloc[-1]) if hasattr(cis_series, "iloc") else float(cis_series)
                st.markdown(f"### 📌 סטטוס: **{current_phase}**")
                st.metric("Composite CIS", f"{current_cis:.1f}")
                if st.session_state.use_ml and st.session_state.ml_model is not None:
                    st.info("ניתוח ה-Wyckoff מבוצע ישירות דרך FactorEngine. מודל ה-AI פעיל בשאר המסכים.")
            except Exception as e:
                st.error(f"שגיאה בחישוב המנוע: {e}")


def screen_backtest():
    st.markdown("""<div class="header-box composite"><h2>📊 WYCKOFF-ANCHORED BACKTEST ENGINE</h2></div>""",
                unsafe_allow_html=True)
    render_active_ai_selector_widget("bt_screen")
    col_r1, _ = st.columns([1, 2])
    with col_r1:
        risk_profile = st.selectbox("🎯 Risk Profile:", ["Aggressive", "Balanced", "Conservative"], index=1)
    c1, c2, _ = st.columns([2, 1.5, 1])
    with c1:
        ticker = st.text_input("סמל לבדיקה:", "COST", key="bt_t")
    with c2:
        render_threshold_control("סף ציון CIS", "bt_threshold")
        bt_threshold = st.session_state["bt_threshold"]
    if st.button("▶ הרץ סימולציה", use_container_width=True, type="primary"):
        with st.spinner("מריץ..."):
            try:
                bt_df, audit_df = run_wyckoff_anchored_backtest(
                    ticker.upper(), st.session_state.use_ml, bt_threshold,
                    period="2y", risk_profile=risk_profile,
                )
            except Exception as e:
                st.error(f"שגיאה בהרצת הבק-טסט: {e}")
                return
            if bt_df is None:
                st.error("שגיאה בנתונים.")
                return
            t_count = len(audit_df)
            w_rate  = len(audit_df[audit_df['win'] == True]) / t_count if t_count > 0 else 0
            s_ret   = bt_df['Cum_Strategy'].iloc[-1]
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("מס' עסקאות", t_count)
            c_m2.metric("Win Rate", f"{w_rate:.1%}" if t_count > 0 else "N/A")
            c_m3.metric("תשואה", f"{s_ret:.2%}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Strategy'],
                                     name='Wyckoff Strategy', line=dict(color='#00ff00')))
            fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Baseline'],
                                     name='Baseline', line=dict(color='#888888', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            if not audit_df.empty:
                st.markdown("### 📋 Audit Logs")
                for _, row in audit_df.iterrows():
                    cls   = "win" if row['win'] else "loss"
                    emoji = "✅" if row['win'] else "❌"
                    st.markdown(
                        f"""<div class="audit-row {cls}"><b>{emoji} {row['entry_date']} → {row['exit_date']}</b><br>
                        פאזה: {row['phase_at_entry']} | תשואה: {row['return_pct']}% | יציאה: {row.get('exit_type','N/A')}</div>""",
                        unsafe_allow_html=True,
                    )

    # כפתור הריבוט העוצמתי - מזריק פקודת JavaScript לדפדפן
    st.markdown("---")
    st.markdown("### ⚙️ אתחול כפוי למערכת")
    if st.button("🔄 אתחול מלא (Hard Reboot & Clear Cache)", use_container_width=True):
        # שלב 1: מנקים את הזיכרון של Streamlit
        st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
        st.session_state.clear()
        
        # שלב 2: מזריקים קוד לדפדפן כדי לטעון מחדש את העמוד מאפס לחלוטין
        js_code = "<script>window.parent.location.reload(true);</script>"
        st.components.v1.html(js_code, height=0)


def screen_scanner():
    st.markdown("""<div class="header-box scanner"><h2>🔎 MARKET SCANNER</h2></div>""",
                unsafe_allow_html=True)
    render_active_ai_selector_widget("scan_screen")
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_universe = SECTOR_MAP[
            st.selectbox("📀 בחר סקטור:", list(SECTOR_MAP.keys()), key="scanner_sector")
        ]
    with c2:
        scan_limit = st.slider("כמות מניות:", 5, len(chosen_universe),
                               min(10, len(chosen_universe)), step=5)
    render_threshold_control("סף כניסה (Threshold) לסינון התוצאות:", "scan_threshold")
    scan_th = st.session_state["scan_threshold"]
    if st.button("🚀 התחל סריקה", use_container_width=True, type="primary"):
        results  = []
        engine   = FactorEngine(BacktestConfig())
        progress = st.progress(0)
        for i, ticker in enumerate(chosen_universe[:scan_limit]):
            df = get_cached_data(ticker, period="6mo")
            if df is not None and len(df) > 30:
                try:
                    f     = engine.compute(df)
                    score = engine.composite_cis(f, df).iloc[-1]
                    phase = engine.get_wyckoff_phase(df).iloc[-1]
                    if score >= scan_th:
                        results.append({"Ticker": ticker, "Score": round(score, 1), "Phase": phase})
                except Exception:
                    pass
            progress.progress((i + 1) / scan_limit)
        if results:
            st.success(f"נמצאו {len(results)} מניות שעוברות את רף הציון {scan_th}:")
            st.dataframe(pd.DataFrame(results).sort_values("Score", ascending=False),
                         use_container_width=True)
        else:
            st.warning(f"אף מניה לא חצתה את רף הציון של {scan_th}.")


def screen_vp():
    st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2></div>""",
                unsafe_allow_html=True)

def screen_vwap():
    st.markdown("""<div class="header-box vwap"><h2>📊 VWAP DEVIATION</h2></div>""",
                unsafe_allow_html=True)

def screen_composite():
    st.markdown("""<div class="header-box composite"><h2>📈 COMPOSITE SCORE</h2></div>""",
                unsafe_allow_html=True)


def screen_monitor():
    st.markdown("""<div class="header-box monitor"><h2>👁️ UNDER THE HOOD - Lab Monitor</h2>
    <p>פיקוח בזמן אמת על מה שהמכונה לומדת, הנתונים שהיא צוברת והפקטורים שמניעים אותה.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.model_archive:
        st.warning("אין עדיין מודלים בספרייה. הרץ אימון ידני או אוטומטי קודם.")
        if st.button("🔄 רענן מודלים"):
            st.session_state.model_archive = load_all_models_from_disk()
            st.rerun()
        return

    slot      = st.selectbox("בחר סקטור למעקב:", list(st.session_state.model_archive.keys()))
    safe_slot = clean_filename(str(slot))
    csv_path  = f"models/training_data_{safe_slot}.csv"

    model_data = st.session_state.model_archive[slot]
    model      = model_data['model']
    meta       = model_data['metadata']

    df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("דיוק (OOB Score)",            f"{meta.get('train_acc', 0)*100:.1f}%")
    c2.metric("סה\"כ עסקאות בבסיס הנתונים", len(df) if not df.empty else 0)
    c3.metric("Threshold מומלץ לכניסה",      meta.get('recommended_threshold', 50))
    if not df.empty and 'label' in df.columns:
        c4.metric("Win Rate היסטורי גולמי",  f"{df['label'].mean()*100:.1f}%")
    else:
        c4.metric("Win Rate היסטורי גולמי",  "N/A")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🧬 מה המודל לומד? (Feature Importance)")
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            fi_df = pd.DataFrame({
                'Feature':    model.feature_names_in_,
                'Importance': model.feature_importances_,
            }).sort_values('Importance', ascending=True).tail(10)
            fig = go.Figure(go.Bar(
                x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                marker=dict(color='#26a69a'),
            ))
            fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=350,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("המודל לא מכיל מידע על חשיבות פקטורים.")

    with col_b:
        st.markdown("### 📊 התפלגות מניות בספרייה (Top 10)")
        if not df.empty and 'ticker' in df.columns:
            ticker_counts = df['ticker'].value_counts().head(10)
            fig2 = go.Figure(go.Pie(
                labels=ticker_counts.index, values=ticker_counts.values, hole=0.4,
                marker=dict(colors=['#3498db','#9b59b6','#34495e','#16a085',
                                    '#27ae60','#f1c40f','#e67e22','#e74c3c']),
            ))
            fig2.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=350,
                               paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("אין מספיק נתונים.")

    st.markdown("---")
    st.markdown("### 📈 התפלגות VIX ברגעי עסקאות")
    if not df.empty:
        vix_col = 'f_macro_vix_zscore' if 'f_macro_vix_zscore' in df.columns else \
                  'vix_close' if 'vix_close' in df.columns else None
        if vix_col:
            label_vix = 'VIX Z-Score' if vix_col == 'f_macro_vix_zscore' else 'VIX Close'
            mean_vix  = df[vix_col].mean()
            fig_vix   = go.Figure()
            fig_vix.add_trace(go.Histogram(x=df[vix_col], nbinsx=25,
                                           name=label_vix, marker_color='#7b1fa2', opacity=0.75))
            fig_vix.add_vline(x=mean_vix, line_dash="dash", line_color="#ef5350",
                              annotation_text=f"ממוצע: {mean_vix:.2f}", annotation_position="top right")
            fig_vix.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vix, use_container_width=True)
        else:
            st.info("לא נמצאו נתוני VIX בקובץ האימון.")

    st.markdown("---")
    st.markdown("### 🕒 עסקאות אחרונות שנסרקו")
    if not df.empty:
        cols_ok  = [c for c in ['entry_date','ticker','phase','label'] if c in df.columns]
        show_df  = df[cols_ok].sort_values('entry_date', ascending=False).head(15).copy()
        if 'label' in show_df.columns:
            show_df['label'] = show_df['label'].apply(lambda x: '✅ הצלחה' if x == 1 else '❌ כישלון')
        show_df.rename(columns={
            'entry_date': 'תאריך כניסה', 'ticker': 'מניה',
            'phase': 'פאזת Wyckoff', 'label': 'סטטוס קצה',
        }, inplace=True)
        st.dataframe(show_df, use_container_width=True)


def screen_ml_trainer():
    st.markdown("""<div class="header-box ml">
        <h2>🧠 WYCKOFF-ANCHORED ML TRAINER (Manual Override)</h2>
        <p>מסך זה מאפשר אימון ידני בודד לבדיקות. האימון האוטומטי המלא נמצא למטה.</p>
    </div>""", unsafe_allow_html=True)

    MODEL_SLOTS = ["Growth (צמיחה)", "Value/Index (ערך/מדד)", "Commodities (סחורות)"]

    # ── אימון ידני ─────────────────────────────────────────────
    st.markdown("### 🔬 אימון ידני בודד")
    c1, c2, c3 = st.columns(3)
    with c1: train_ticker = st.text_input("סמל לאימון:", "SPY")
    with c2: target_slot  = st.selectbox("משבצת אסטרטגית:", MODEL_SLOTS)
    with c3: train_risk   = st.selectbox("רמת סיכון:", ["Aggressive", "Balanced", "Conservative"])

    c4, c5, c6 = st.columns(3)
    with c4: start_date = st.date_input("מתאריך:", value=datetime(2020, 1, 1))
    with c5: end_date   = st.date_input("עד תאריך:", value=datetime.today())
    with c6:
        render_threshold_control("סף כניסה בסיסי:", "base_threshold")
        base_th = st.session_state["base_threshold"]

    if st.button("🚀 התחל למידה רציפה (הוסף לספרייה)", use_container_width=True, type="primary"):
        with st.spinner("שואב עסקאות ומאמן מודל..."):
            df = get_cached_data(train_ticker.upper(),
                                 start=start_date.strftime('%Y-%m-%d'),
                                 end=end_date.strftime('%Y-%m-%d'))
            if df is None or len(df) < 60:
                st.error("אין מספיק נתונים.")
                return

            engine = FactorEngine(BacktestConfig())
            try:
                bt_df, audit_df = run_wyckoff_anchored_backtest(
                    train_ticker.upper(), use_ai=False, threshold=base_th, period=None,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    risk_profile=train_risk,
                )
            except Exception as e:
                st.error(f"שגיאה בבק-טסט: {e}")
                return

            if audit_df is None or audit_df.empty:
                st.error("לא היו עסקאות בתקופה. נסה להוריד את הסף.")
                return

            features_list = []
            for _, trade in audit_df.iterrows():
                entry_dt = pd.Timestamp(trade['entry_date'])
                if entry_dt in bt_df.index:
                    window_df = (df.loc[:entry_dt].iloc[-200:]
                                 if len(df.loc[:entry_dt]) > 200 else df.loc[:entry_dt])
                    try:
                        factors = engine.compute(window_df)
                        if len(factors) > 0:
                            feature_row = factors.iloc[-1].to_dict()
                            feature_row['phase']      = bt_df.loc[entry_dt]['wyckoff_phase']
                            feature_row['label']      = 1 if trade['win'] else 0
                            feature_row['ticker']     = train_ticker.upper()
                            feature_row['entry_date'] = trade['entry_date']
                            features_list.append(feature_row)
                    except Exception:
                        continue

            if len(features_list) < 3:
                st.error("מעט מדי עסקאות לאימון.")
                return

            new_df         = pd.DataFrame(features_list)
            os.makedirs("models", exist_ok=True)
            safe_slot_name = clean_filename(str(target_slot))
            history_path   = f"models/training_data_{safe_slot_name}.csv"

            if os.path.exists(history_path):
                hist_df     = pd.read_csv(history_path)
                combined_df = (
                    pd.concat([hist_df, new_df], ignore_index=True)
                    .drop_duplicates(subset=['ticker','entry_date'], keep='last')
                )
            else:
                combined_df = new_df

            combined_df.to_csv(history_path, index=False)

            if combined_df['label'].nunique() < 2:
                st.error("צריך לפחות שתי מחלקות שונות לאימון מודל.")
                return

            y             = combined_df['label'].values
            le            = LabelEncoder()
            phase_encoded = le.fit_transform(combined_df['phase'].fillna("לא בתהליך איסוף"))
            phase_dummies = pd.get_dummies(phase_encoded, prefix='phase').astype(int)
            drop_cols     = ['phase','label','ticker','entry_date']
            tech_factors  = (
                combined_df
                .drop(columns=[c for c in drop_cols if c in combined_df.columns])
                .select_dtypes(include=[np.number])
            )
            X = (
                pd.concat([tech_factors.reset_index(drop=True),
                           phase_dummies.reset_index(drop=True)], axis=1)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            model = RandomForestClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=3,
                oob_score=True, random_state=42, n_jobs=-1,
            )
            model.fit(X, y)
            try:    train_acc = model.oob_score_
            except: train_acc = model.score(X, y)

            optimal_th = calculate_optimal_threshold(model, X, y)
            meta = {
                "train_ticker": "MANUAL_ADDITION", "train_acc": train_acc,
                "test_acc": train_acc, "slot": target_slot,
                "model_type": "Wyckoff-Anchored", "num_trades": len(combined_df),
                "recommended_threshold": optimal_th,
            }
            save_path = save_model_to_disk(target_slot, model, meta, le)
            st.session_state.model_archive  = load_all_models_from_disk()
            st.session_state.ml_model       = model
            st.session_state.ml_metadata    = meta
            st.session_state.phase_encoder  = le
            st.session_state.use_ml         = True
            st.success(f"✅ אימון הושלם! מודל נשמר: {save_path}")
            c_r1, c_r2, c_r3 = st.columns(3)
            c_r1.metric("דיוק OOB",              f"{train_acc*100:.1f}%")
            c_r2.metric("סה\"כ עסקאות בספרייה", len(combined_df))
            c_r3.metric("🎯 Threshold מומלץ",    optimal_th)

    # ── סטטוס Auto-Trainer ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚦 Auto-Trainer Status")
    status = read_auto_trainer_status()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("מצב",           status.get("state", "idle"))
    s2.metric("התקדמות",       f"{status.get('progress', 0)}%")
    s3.metric("סקטור נוכחי",   status.get("current_slot", "N/A"))
    s4.metric("עודכן",         status.get("updated_at", "N/A"))

    # ── אימון אוטומטי מלא ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 אימון אוטומטי מלא (כל הסקטורים)")

    if st.button("🚀 הזנק אימון אוטומטי מלא", type="primary", use_container_width=True):
        if not TRAINER_AVAILABLE:
            st.error(f"❌ שגיאת מערכת: לא ניתן להריץ את האימון.\n\nהמערכת לא מוצאת את הקובץ `trainer_core.py` או שיש בו שגיאת קידוד. אנא ודא שהקובץ נמצא בתיקיית הפרויקט המרכזית בגיטהאב וששמו נכתב בדיוק כך.\n\nפרטי השגיאה הטכנית:\n{TRAINER_ERROR}")
        else:
            with st.spinner("האימון רץ... אל תרענן את הדף."):
                try:
                    run_auto_trainer()
                    st.success("✅ האימון האוטומטי הושלם בהצלחה!")
                    st.session_state.model_archive = load_all_models_from_disk()
                    time.sleep(2)
                    st.rerun()
                except Exception:
                    import traceback
                    st.error("❌ האימון נכשל. הנה השגיאה המדויקת:")
                    st.code(traceback.format_exc(), language="text")

    with st.expander("📝 יומן ריצה ושגיאות", expanded=False):
        if os.path.exists(LOG_FILE_PATH):
            try:
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    logs = f.read()
                st.text_area("היומן המלא:", logs[-5000:], height=300)
                if st.button("🗑️ נקה יומן"):
                    open(LOG_FILE_PATH, 'w').close()
                    st.rerun()
            except Exception as e:
                st.warning(f"לא ניתן לקרוא את קובץ הלוג: {e}")
        else:
            st.info("קובץ היומן עדיין לא נוצר. יופיע כשהאימון יתחיל.")


# ============================================================
# ניתוב
# ============================================================
routes = {
    "wyckoff":   screen_wyckoff,
    "vp":        screen_vp,
    "vwap":      screen_vwap,
    "composite": screen_composite,
    "backtest":  screen_backtest,
    "ml":        screen_ml_trainer,
    "scanner":   screen_scanner,
    "monitor":   screen_monitor,
}
routes[st.session_state.mode]()
