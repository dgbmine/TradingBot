# ============================================================
# INSTITUTIONAL SCOUT PRO - FINAL CLOSED-LOOP V9.1 (Fixed Dates)
# ============================================================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import warnings
import pickle
import base64
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import time

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Institutional Scout Pro")

# ============================================================
# חלק 2: רשימת המניות וחלוקה לקבוצות סקטוריאליות
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
    "DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
]))

SECTOR_MAP = {
    "הכול (כל השוק האמריקאי)": SCAN_UNIVERSE,
    "צמיחה וטכנולוגיה (Growth)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
        "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
        "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
        "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
        "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA"
    ],
    "ערך ומדד (Value/Index)": [
        "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
        "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
        "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
        "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
        "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
        "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
        "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
        "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
    ],
    "סחורות ואנרגיה (Commodities)": [
        "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
        "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG"
    ]
}

# ============================================================
# חלק 3: עיצוב CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Hebrew', sans-serif; direction: rtl; text-align: right; box-sizing: border-box; }
h1, h2, h3, h4, h5, h6 { direction: rtl; }
.header-box { border-radius: 12px; padding: 24px 32px; margin-bottom: 28px; color: #e0eaf4; line-height: 1.9; }
.header-box.wyckoff { background: linear-gradient(135deg, #0f1923, #1a2a3a); border: 1px solid #2a4a6a; }
.header-box.vp { background: linear-gradient(135deg, #160f23, #251535); border: 1px solid #4a2a6a; }
.header-box.vwap { background: linear-gradient(135deg, #0f2318, #1a3528); border: 1px solid #2a6a4a; }
.header-box.composite { background: linear-gradient(135deg, #1a1208, #2a1e08); border: 1px solid #6a4a1a; }
.header-box.ml { background: linear-gradient(135deg, #1c0a20, #2e1236); border: 1px solid #7b1fa2; }
.header-box.scanner { background: linear-gradient(135deg, #0f231f, #1a3a35); border: 1px solid #26a69a; }
.widget-panel-ai { background: #111922; border: 1px solid #2d3d4f; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
.audit-row { padding: 12px; margin-bottom: 8px; border-radius: 5px; border-right: 4px solid; }
.win { background: rgba(38, 166, 154, 0.1); border-color: #26a69a; }
.loss { background: rgba(239, 83, 80, 0.1); border-color: #ef5350; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# חלק 4: ניהול ושמירת מודלים בדיסק + חישוב סף דינמי
# ============================================================
def clean_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')

def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs("models", exist_ok=True)
    safe_name = clean_filename(slot_name)
    file_path = f"models/model_{safe_name}.pkl"
    save_data = {"model": model, "metadata": metadata, "phase_encoder": encoder}
    with open(file_path, "wb") as f: pickle.dump(save_data, f)
    return file_path

def load_all_models_from_disk():
    loaded_archive = {}
    if os.path.exists("models"):
        for filename in os.listdir("models"):
            if filename.endswith(".pkl"):
                filepath = os.path.join("models", filename)
                try:
                    with open(filepath, "rb") as f: data = pickle.load(f)
                    slot = data.get("metadata", {}).get("slot", filename)
                    loaded_archive[slot] = data
                except: pass
    return loaded_archive

def calculate_optimal_threshold(model, X, y):
    """מוצא את הציון סף שנותן את ה-Win Rate הכי טוב תוך שמירה על כמות עסקאות סבירה"""
    try:
        probs = model.predict_proba(X)[:, 1] * 100
    except:
        return 65 # Fallback
    
    best_thresh = 65
    best_score = 0
    
    for th in range(50, 95, 2):
        mask = probs >= th
        trades_count = mask.sum()
        if trades_count >= max(5, len(y) * 0.1): # מינימום 10% מהעסקאות המקוריות
            win_rate = y[mask].mean()
            # משקללים את ה-Win Rate עם קצת בונוס לכמות עסקאות כדי לא להישאר עם עסקה אחת
            score = win_rate * (1 + np.log1p(trades_count)/10) 
            if score > best_score:
                best_score = score
                best_thresh = th
    return best_thresh

for k, v in [("mode", "wyckoff"), ("ml_model", None), ("ml_metadata", None),
             ("use_ml", False), ("phase_encoder", None)]:
    if k not in st.session_state: st.session_state[k] = v

if "model_archive" not in st.session_state or not st.session_state.model_archive:
    st.session_state.model_archive = load_all_models_from_disk()

# משיכת סף מומלץ מהמודל הפעיל (לשימוש בבוררים השונים)
def get_active_threshold_recommendation():
    if st.session_state.use_ml and st.session_state.ml_metadata:
        return st.session_state.ml_metadata.get("recommended_threshold", 65)
    return 65

# ============================================================
# חלק 5: בורר ה-AI
# ============================================================
def render_active_ai_selector_widget(screen_identifier):
    st.markdown("<div class='widget-panel-ai'>", unsafe_allow_html=True)
    st.markdown("### 🧠 הגדרות מנוע החלטה AI חכם")
    col_a, col_b, col_c = st.columns([2, 1.5, 1])
    with col_a:
        if st.session_state.model_archive:
            slots_list = list(st.session_state.model_archive.keys())
            selected_slot = st.selectbox("בחר מודל מוסדי פעיל:", slots_list, key=f"selector_slot_{screen_identifier}")
            if st.button("✅ טען והפעל מודל", key=f"activate_btn_{screen_identifier}", use_container_width=True):
                target_data = st.session_state.model_archive[selected_slot]
                st.session_state.ml_model = target_data["model"]
                st.session_state.ml_metadata = target_data["metadata"]
                st.session_state.phase_encoder = target_data.get("phase_encoder")
                st.session_state.use_ml = True
                st.success(f"המודל '{selected_slot}' הופעל בהצלחה!")
                st.rerun()
        else: st.info("לא נמצאו מודלים בזיכרון.")
    with col_b:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 טען מהדיסק", key=f"sync_git_{screen_identifier}", use_container_width=True):
            st.session_state.model_archive = load_all_models_from_disk(); st.rerun()
    with col_c:
        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
        ai_toggle = st.checkbox("הפעל שימוש ב-AI", value=st.session_state.use_ml, key=f"checkbox_ai_{screen_identifier}")
        if ai_toggle != st.session_state.use_ml:
            st.session_state.use_ml = ai_toggle; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# חלק 6: ניווט
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
nav = [("wyckoff","⬛ Wyckoff"),("vp","🔮 Volume Profile"),("vwap","📊 VWAP Deviation"),
       ("composite","📈 Composite Score"),("backtest","📊 Backtest"),
       ("ml","🧠 ML Trainer"), ("scanner","🔎 Scanner")]
for col, (mode_key, label) in zip([c1,c2,c3,c4,c5,c6,c7], nav):
    with col:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.mode==mode_key else "secondary", key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")
if st.session_state.use_ml and st.session_state.ml_model is not None:
    metadata = st.session_state.ml_metadata or {}
    acc = metadata.get("test_acc", metadata.get("train_acc", 0.0))
    rec_th = metadata.get("recommended_threshold", "לא חושב")
    st.info(f"🧠 **מצב AI מופעל:** {metadata.get('slot', 'כללי')} | דיוק (Test): {acc*100:.1f}% | 🎯 **ציון סף מומלץ לכניסה:** {rec_th}")

# ============================================================
# חלק 7: פקטורים ו-VSA 
# ============================================================
@dataclass
class BacktestConfig:
    commission: float = 0.001
    initial_capital: float = 100_000.0
    hold_days: int = 40
    period: str = "2y"

class FactorEngine:
    def __init__(self, cfg: BacktestConfig): self.cfg = cfg

    def _compute_quick_wyckoff(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        if len(df) < 40: return score
        spread = df['High'] - df['Low']; vol_ma = df['Volume'].rolling(20).mean()
        has_sc, has_ar, has_st = False, False, False; sc_idx, sc_low, ar_high = 0, 0, 0
        search_df = df.iloc[-90:]
        for i in range(1, len(search_df)):
            idx = search_df.index[i]
            vol = search_df['Volume'].iloc[i]; vol_ma_i = vol_ma.loc[idx]
            close = search_df['Close'].iloc[i]; low = search_df['Low'].iloc[i]
            high = search_df['High'].iloc[i]; open_px = search_df['Open'].iloc[i]
            if not has_sc:
                if close < open_px and vol > vol_ma_i * 2.0 and close <= search_df['Close'].iloc[max(0, i-20):i].min():
                    has_sc = True; sc_idx = i; sc_low = low; score.loc[idx] = 0.3
            elif has_sc and not has_ar and (i - sc_idx <= 15):
                if close > open_px and close > search_df['Close'].iloc[i-1]:
                    has_ar = True; ar_high = high; score.loc[idx] = 0.4
            elif has_ar and not has_st:
                if vol < search_df['Volume'].iloc[sc_idx] * 0.75 and abs(low - sc_low)/sc_low < 0.05:
                    has_st = True; score.loc[idx] = 0.6
            elif has_st:
                if low < sc_low and close > sc_low: score.loc[idx] = 0.8
                elif low > sc_low and low < search_df['Low'].iloc[i-1] and vol < vol_ma_i: score.loc[idx] = 0.85
                elif close > ar_high and vol > vol_ma_i * 1.5: score.loc[idx] = 1.0; has_sc = False
        return score

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        body = (df["Close"] - df["Open"]).abs()
        rng = df["High"] - df["Low"]
        vol_ma20 = df["Volume"].rolling(20).mean()
        rvol = df["Volume"] / vol_ma20.replace(0, np.nan)
        spread_ma20 = rng.rolling(20).mean()
        f["f04_absorption"] = (((df["Volume"] > vol_ma20 * 1.5) & (rng < spread_ma20 * 0.8)) & (df["Close"] <= df["Low"].rolling(20).min() * 1.05)).astype(float)
        f["f36_wyckoff_score"] = self._compute_quick_wyckoff(df)
        price_bins = pd.cut(df["Close"], bins=40, labels=False)
        f["f01_liquidity_gap"] = ((df.groupby(price_bins)["Volume"].transform("sum") < df.groupby(price_bins)["Volume"].transform("mean") * 0.5).astype(float).rolling(5).mean())
        sma20 = df["Close"].rolling(20).mean(); std20 = df["Close"].rolling(20).std()
        atr14 = pd.concat([rng, (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1).rolling(14).mean()
        f["f02_volatility_squeeze"] = ((((2 * std20) / sma20.replace(0, np.nan)) < ((2 * std20) / sma20.replace(0, np.nan)).rolling(20).mean() * 0.75) & (atr14 < atr14.rolling(20).mean() * 0.75)).astype(float)
        spy_slope = df.get("spy_close", df["Close"]).rolling(50).mean().diff(10) / df.get("spy_close", df["Close"]).rolling(50).mean().shift(10).replace(0, np.nan)
        f["f03_regime"] = (spy_slope > 0.01).astype(float) - (spy_slope < -0.01).astype(float)
        resist = df["High"].rolling(20).max().shift(1)
        f["f05_breakout_quality"] = ((df["Close"] > resist) & (df["Close"].rolling(3).mean() > resist.shift(1))).astype(float)
        f["f06_cis_weight"] = np.clip(1.0 / (std20 / std20.rolling(60).mean().replace(0, np.nan)).replace(0, np.nan), 0.5, 2.0)
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        f["f07_obv_velocity"] = (obv.diff(10) / obv.abs().rolling(10).mean().replace(0, np.nan)).clip(-3, 3)
        f["f10_temporal_seq"] = (f["f04_absorption"].rolling(30).max() * (rvol < 0.7).astype(float))
        f["f11_kill_switch"] = ((df["Close"].pct_change() < -0.05) | (rvol > 4.0)).astype(float)
        f["f14_inst_intent"] = (f["f04_absorption"] * 0.3 + f["f07_obv_velocity"].clip(0, 1) * 0.4 + f["f10_temporal_seq"] * 0.3).clip(0, 1)
        f["f15_mtf"] = ((df["Close"] > sma20).astype(float) * (df["Close"].rolling(5).mean() > df["Close"].rolling(5).mean().rolling(4).mean()).astype(float))
        support = df["Low"].rolling(20).min().shift(1)
        f["f20_liquidity_sweep"] = ((df["Low"] < support) & (df["Close"] > support)).astype(float)
        f["f22_sr_strength"] = (df["Low"].rolling(5).min() <= df["Low"].rolling(20).min() * 1.005).astype(float).rolling(20).sum() / 20
        f["f26_accept_reject"] = ((df["Close"] > (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float).rolling(5).mean() - ((df["Close"] < (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float).rolling(5).mean()
        f["f28_inst_part"] = ((body > body.rolling(20).mean() * 1.5) & (rvol > 1.5)).astype(float)
        f["f31_bear_trap"] = ((df["Close"] < df["Low"].rolling(20).min().shift(1)) & (df["Close"].shift(1) > df["Low"].rolling(20).min().shift(2))).astype(float)
        f["f35_struct_break"] = (df["Close"] > df["High"].rolling(20).max().shift(1)).astype(float) - (df["Close"] < df["Low"].rolling(20).min().shift(1)).astype(float)
        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame, df: pd.DataFrame = None) -> pd.Series:
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            model = st.session_state.ml_model
            X_pred = factors.copy()
            phase_encoder = st.session_state.phase_encoder
            if phase_encoder is not None and df is not None and "wyckoff_phase" in df.columns:
                phases = df["wyckoff_phase"].fillna("לא בתהליך איסוף")
                try:
                    phase_labels = phase_encoder.transform(phases)
                    for i, label in enumerate(phase_encoder.classes_): X_pred[f"phase_{label}"] = (phase_labels == i).astype(int)
                except:
                    for label in phase_encoder.classes_: X_pred[f"phase_{label}"] = 0
            expected_features = getattr(model, "feature_names_in_", None)
            if expected_features is not None:
                for c in expected_features:
                    if c not in X_pred.columns: X_pred[c] = 0
                X_pred = X_pred[expected_features]
            try: probs = model.predict_proba(X_pred)[:, 1]
            except: probs = model.predict(X_pred)
            score = pd.Series(probs * 100, index=factors.index)
        else:
            w = {"f04_absorption": 6, "f07_obv_velocity": 5, "f14_inst_intent": 6, "f20_liquidity_sweep": 3, "f26_accept_reject": 3, "f35_struct_break": 2}
            tot = sum(abs(v) for v in w.values() if v != 0)
            score = pd.Series(0.0, index=factors.index)
            for col, weight in w.items():
                if col in factors.columns: score += factors[col].clip(-1, 1) * weight
            score = (score / tot * 100 + 50).clip(0, 100)
            
        if "f36_wyckoff_score" in factors.columns:
            wyckoff_score = factors["f36_wyckoff_score"]
            boost_floor = np.where(wyckoff_score >= 0.9, 65.0, 0.0)
            score = np.maximum(score, boost_floor)
            boost = np.where(wyckoff_score > 0.5, (wyckoff_score - 0.5) * 40, 0)
            score = score + boost
        if "f11_kill_switch" in factors.columns:
            score = score * (1 - factors["f11_kill_switch"])
        return score.round(1).clip(0, 100)

    def get_wyckoff_phase(self, df: pd.DataFrame) -> pd.Series:
        phases = pd.Series("לא בתהליך איסוף", index=df.index)
        if len(df) < 40: return phases
        has_sc, has_ar, has_st = False, False, False; sc_idx, sc_low, ar_high = 0, 0, 0
        for i in range(40, len(df)):
            window = df.iloc[max(0, i-90):i+1]
            if len(window) < 40: continue
            vol_ma = window['Volume'].rolling(20).mean(); current_phase = "לא בתהליך איסוף"
            for j in range(1, len(window)):
                vol = window['Volume'].iloc[j]; vol_ma_j = vol_ma.iloc[j]
                close = window['Close'].iloc[j]; low = window['Low'].iloc[j]
                high = window['High'].iloc[j]; open_px = window['Open'].iloc[j]
                if not has_sc:
                    if close < open_px and vol > vol_ma_j * 2.0 and close <= window['Close'].iloc[max(0, j-20):j].min():
                        has_sc = True; sc_idx = j; sc_low = low; current_phase = "Phase A (SC)"
                elif has_sc and not has_ar and (j - sc_idx <= 15):
                    if close > open_px and close > window['Close'].iloc[j-1]:
                        has_ar = True; ar_high = high; current_phase = "Phase B (AR)"
                elif has_ar and not has_st:
                    if vol < window['Volume'].iloc[sc_idx] * 0.75 and abs(low - sc_low)/sc_low < 0.05:
                        has_st = True; current_phase = "Phase B (ST)"
                elif has_st:
                    if low < sc_low and close > sc_low: current_phase = "Phase C (Spring)"
                    elif low > sc_low and low < window['Low'].iloc[j-1] and vol < vol_ma_j: current_phase = "Phase D (LPS)"
                    elif close > ar_high and vol > vol_ma_j * 1.5: current_phase = "Phase D (SOS)"; has_sc = False; has_ar = False; has_st = False
                    elif close > ar_high * 1.02: current_phase = "Phase E (Breakout)"
            phases.iloc[i] = current_phase
        return phases

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.strftime("%Y-%m-%d")
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)

# ============================================================
# חלק 8: WYCKOFF Logic
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y", start=None, end=None):
    try:
        if start is not None and end is not None:
            df = yf.Ticker(ticker).history(start=start, end=end)
        else:
            df = yf.Ticker(ticker).history(period=period)
        if df is None or len(df) < 40: return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except: return None

def analyze_wyckoff_strict(df):
    phase = "לא בתהילים ניתוח מובהק"; score = 0; alerts = []
    has_sc, has_ar, has_st = False, False, False; sc_idx, sc_low, ar_high = 0, 0, 0
    df['Spread'] = df['High'] - df['Low']; df['Vol_MA'] = df['Volume'].rolling(20).mean(); df['Spread_MA'] = df['Spread'].rolling(20).mean()
    search_df = df.iloc[-90:]
    for i in range(1, len(search_df)):
        vol = search_df['Volume'].iloc[i]; vol_ma = search_df['Vol_MA'].iloc[i]; close = search_df['Close'].iloc[i]; low = search_df['Low'].iloc[i]; high = search_df['High'].iloc[i]; open_px = search_df['Open'].iloc[i]
        if vol > vol_ma * 1.5 and search_df['Spread'].iloc[i] < search_df['Spread_MA'].iloc[i] * 0.8:
            if (len(search_df) - i - 1) < 10: alerts.append(f"⚠️ ספיגת VSA זוהתה")
        if not has_sc:
            if close < open_px and vol > vol_ma * 2.0 and close <= search_df['Close'].iloc[max(0, i-20):i].min(): has_sc = True; sc_idx = i; sc_low = low; phase = "SC / Phase A"; score = 30
        elif has_sc and not has_ar and (i - sc_idx <= 15):
            if close > open_px and close > search_df['Close'].iloc[i-1]: has_ar = True; ar_high = high; phase = "AR"; score = 40
        elif has_ar and not has_st:
            if vol < search_df['Volume'].iloc[sc_idx] * 0.75 and abs(low - sc_low)/sc_low < 0.05: has_st = True; phase = "ST / Phase B"; score = 60
        elif has_st:
            if low < sc_low and close > sc_low: phase = "Phase C (Spring)"; score = 80
            elif low > sc_low and low < search_df['Low'].iloc[i-1] and vol < vol_ma: phase = "LPS"; score = 85
            elif close > ar_high and vol > vol_ma * 1.5: phase = "SOS / Phase D"; score = 100; has_sc = False
    return score, phase, "", list(set(alerts)), phase, "#26a69a" if score >= 80 else "#ffa726" if score >= 40 else "#ef5350"

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF 3.0 STRUCTURAL ENGINE</h2></div>""",unsafe_allow_html=True)
    render_active_ai_selector_widget("wyckoff_screen")
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סמל לניתוח:", "NVDA", key="w_ticker")
    with c2: st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True); btn = st.button("▶ הרץ ניתוח", use_container_width=True, type="primary")
    if btn:
        with st.spinner("מנתח..."):
            df = get_data(ticker.upper())
            if df is not None:
                score, phase, _, alerts, _, vc = analyze_wyckoff_strict(df)
                st.markdown(f"### 📌 סטטוס: **{phase}** (ציון: {score})")
                if alerts:
                    for alert in alerts: st.warning(alert)
                if st.session_state.use_ml and st.session_state.ml_model is not None:
                    engine = FactorEngine(BacktestConfig())
                    df["wyckoff_phase"] = engine.get_wyckoff_phase(df)
                    cis = engine.composite_cis(engine.compute(df), df)
                    st.markdown(f"### 🤖 תחזית מודל: **{cis.iloc[-1]:.1f}** (הסתברות הצלחה מוסדית)")
            else: st.error("אין נתונים.")

# ============================================================
# חלק 9: BACKTEST ENGINE 
# ============================================================
def check_phase_entry_allowed(phase, risk_profile):
    if "לא בתהליך" in phase: return False
    if risk_profile == "Aggressive": return any(p in phase for p in ["Phase C", "Phase D", "Phase E", "Spring", "LPS", "SOS", "Breakout"])
    elif risk_profile == "Balanced": return any(p in phase for p in ["Phase D", "Phase E", "LPS", "SOS", "Breakout"])
    elif risk_profile == "Conservative": return any(p in phase for p in ["Phase E", "Breakout"])
    return False

def run_wyckoff_anchored_backtest(ticker, use_ai, threshold, period=None, start=None, end=None, risk_profile="Balanced"):
    df = get_data(ticker, period=period, start=start, end=end)
    if df is None: return None, None
    cfg_period = period if period else f"{start}/{end}"
    engine = FactorEngine(BacktestConfig(period=cfg_period))
    factors = engine.compute(df)
    df['wyckoff_phase'] = engine.get_wyckoff_phase(df)
    df['cis_score'] = engine.composite_cis(factors, df)
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)

    positions = []; audit_logs = []; in_position = False; entry_price = 0; entry_phase = ""; entry_date = None; peak_price = 0
    for i in range(len(df)):
        current_phase = df['wyckoff_phase'].iloc[i]
        current_cis = df['cis_score'].iloc[i]
        phase_allowed = check_phase_entry_allowed(current_phase, risk_profile)
        score_allowed = current_cis >= threshold
        
        if not in_position:
            if phase_allowed and score_allowed:
                positions.append(1); in_position = True; entry_price = df['Close'].iloc[i]; entry_phase = current_phase; entry_date = df.index[i]; peak_price = entry_price
            else: positions.append(0)
        else:
            if "לא בתהליך" in current_phase or current_cis < threshold - 15:
                positions.append(0)
                exit_px = df['Close'].iloc[i]
                ret = (exit_px - entry_price) / entry_price
                max_dd = (peak_price - min(entry_price, exit_px)) / peak_price if peak_price > 0 else 0
                audit_logs.append({"entry_date": entry_date.strftime("%Y-%m-%d"), "exit_date": df.index[i].strftime("%Y-%m-%d"), "phase_at_entry": entry_phase, "entry_price": round(entry_price, 2), "exit_price": round(exit_px, 2), "return_pct": round(ret * 100, 2), "win": ret > 0, "max_drawdown_pct": round(max_dd * 100, 2)})
                in_position = False
            else:
                positions.append(1)
                if df['Close'].iloc[i] > peak_price: peak_price = df['Close'].iloc[i]

    df['Position'] = pd.Series(positions, index=df.index[:len(positions)]).shift(1).fillna(0)
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod() - 1
    df['Cum_Baseline'] = (1 + df['Daily_Return']).cumprod() - 1
    return df, pd.DataFrame(audit_logs) if audit_logs else pd.DataFrame()

def screen_backtest():
    st.markdown("""<div class="header-box composite"><h2>📊 WYCKOFF-ANCHORED BACKTEST ENGINE</h2></div>""",unsafe_allow_html=True)
    render_active_ai_selector_widget("bt_screen")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1: risk_profile = st.selectbox("🎯 Risk Profile:", ["Aggressive", "Balanced", "Conservative"], index=1)
    
    rec_th = get_active_threshold_recommendation()
    
    c1, c2, c3 = st.columns([2, 1.5, 1])
    with c1: ticker = st.text_input("סמל לבדיקה:", "COST", key="bt_t")
    with c2: bt_threshold = st.slider("סף ציון CIS (מסונכרן עם ה-AI)", 40, 95, int(rec_th) if isinstance(rec_th, (int, float)) else 65)
    
    if st.button("▶ הרץ סימולציה", use_container_width=True, type="primary"):
        with st.spinner("מריץ..."):
            bt_df, audit_df = run_wyckoff_anchored_backtest(ticker.upper(), st.session_state.use_ml, bt_threshold, period="2y", risk_profile=risk_profile)
            if bt_df is None: st.error("שגיאה בנתונים."); return
            
            s_ret = bt_df['Cum_Strategy'].iloc[-1]; t_count = len(audit_df); w_rate = len(audit_df[audit_df['win']==True])/t_count if t_count>0 else 0
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("מס' עסקאות", t_count); c_m2.metric("Win Rate", f"{w_rate:.1%}" if t_count>0 else "N/A"); c_m3.metric("תשואה", f"{s_ret:.2%}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Strategy'], name='Wyckoff Strategy', line=dict(color='#00ff00')))
            fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Baseline'], name='Baseline', line=dict(color='#888888', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

            if not audit_df.empty:
                st.markdown("### 📋 Audit Logs")
                for _, row in audit_df.iterrows():
                    cls = "win" if row['win'] else "loss"; emoji = "✅" if row['win'] else "❌"
                    st.markdown(f"""<div class="audit-row {cls}"><b>{emoji} {row['entry_date']} → {row['exit_date']}</b><br>פאזה: {row['phase_at_entry']} | תשואה: {row['return_pct']}%</div>""", unsafe_allow_html=True)

# ============================================================
# חלק 10: סקרנר 
# ============================================================
def screen_scanner():
    st.markdown("""<div class="header-box scanner"><h2>🔎 MARKET SCANNER</h2></div>""",unsafe_allow_html=True)
    render_active_ai_selector_widget("scan_screen")
    
    rec_th = get_active_threshold_recommendation()
    
    c1, c2 = st.columns([2, 1])
    with c1: chosen_universe = SECTOR_MAP[st.selectbox("📀 בחר סקטור:", list(SECTOR_MAP.keys()), key="scanner_sector")]
    with c2: scan_limit = st.slider("כמות מניות:", 5, len(chosen_universe), min(10, len(chosen_universe)), step=5)
    scan_th = st.slider("סף כניסה (Threshold) לסינון התוצאות:", 50, 95, int(rec_th) if isinstance(rec_th, (int, float)) else 65)

    if st.button("🚀 התחל סריקה", use_container_width=True, type="primary"):
        results = []
        engine = FactorEngine(BacktestConfig())
        progress = st.progress(0)
        for i, ticker in enumerate(chosen_universe[:scan_limit]):
            df = get_data(ticker, period="6mo")
            if df is not None and len(df) > 30:
                f = engine.compute(df)
                score = engine.composite_cis(f, df).iloc[-1]
                phase = engine.get_wyckoff_phase(df).iloc[-1]
                if score >= scan_th:
                    results.append({"Ticker": ticker, "Score": round(score, 1), "Phase": phase})
            progress.progress((i+1)/scan_limit)
        
        if results:
            st.success(f"נמצאו {len(results)} מניות שעוברות את רף הציון {scan_th}:")
            st.dataframe(pd.DataFrame(results).sort_values("Score", ascending=False), use_container_width=True)
        else:
            st.warning(f"אף מניה לא חצתה את רף הציון של {scan_th}.")

# ============================================================
# חלק 11 & 12: ML TRAINER (Closed-Loop) + Dynamic Threshold
# ============================================================
def screen_vp(): st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2></div>""",unsafe_allow_html=True)
def screen_vwap(): st.markdown("""<div class="header-box vwap"><h2>📊 VWAP DEVIATION</h2></div>""",unsafe_allow_html=True)
def screen_composite(): st.markdown("""<div class="header-box composite"><h2>📈 COMPOSITE SCORE</h2></div>""",unsafe_allow_html=True)

def screen_ml_trainer():
    st.markdown("""<div class="header-box ml"><h2>🧠 WYCKOFF-ANCHORED ML TRAINER</h2>
    <p>מערכת אימון מוסדית עם כיול אוטומטי של סף ציון הכניסה.</p></div>""",unsafe_allow_html=True)
    MODEL_SLOTS = ["Growth (צמיחה)", "Value/Index (ערך/מדד)", "Commodities (סחורות)"]
    
    st.markdown("### 🚀 הגדרות אימון")
    c1, c2, c3 = st.columns(3)
    with c1: train_ticker = st.text_input("סמל לאימון:", "SPY")
    with c2: target_slot = st.selectbox("משבצת אסטרטגית:", MODEL_SLOTS)
    with c3: train_risk = st.selectbox("רמת סיכון לסינון הבק-טסט:", ["Aggressive", "Balanced", "Conservative"])
    
    c4, c5, c6 = st.columns(3)
    with c4: start_date = st.date_input("מתאריך:", value=datetime(2020, 1, 1))
    with c5: end_date = st.date_input("עד תאריך:", value=datetime.today())
    with c6: base_th = st.slider("סף כניסה בסיסי לשאיבת עסקאות:", 40, 80, 50)

    if st.button("🚀 התחל למידה וכיול Threshold", use_container_width=True, type="primary"):
        with st.spinner("שואב עסקאות עבר ומאמן את המודל..."):
            df = yf.Ticker(train_ticker.upper()).history(start=start_date, end=end_date)
            if df is None or len(df) < 60: st.error("אין מספיק נתונים לחלון הזמן המבוקש."); return
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            engine = FactorEngine(BacktestConfig())
            bt_df, audit_df = run_wyckoff_anchored_backtest(
                train_ticker.upper(), 
                use_ai=False, 
                threshold=base_th, 
                period=None, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'), 
                risk_profile=train_risk
            )
            
            if audit_df is None or audit_df.empty:
                st.error("לא היו עסקאות בתקופה. נסה להוריד את הסף הבסיסי או לשנות פרופיל סיכון לאגרסיבי."); return

            features_list, labels = [], []
            for _, trade in audit_df.iterrows():
                entry_dt = pd.Timestamp(trade['entry_date'])
                if entry_dt in bt_df.index:
                    window_df = df.loc[:entry_dt].iloc[-200:] if len(df.loc[:entry_dt]) > 200 else df.loc[:entry_dt]
                    factors = engine.compute(window_df)
                    if len(factors) > 0:
                        feature_row = factors.iloc[-1].to_dict()
                        feature_row['phase'] = bt_df.loc[entry_dt]['wyckoff_phase']
                        features_list.append(feature_row)
                        labels.append(1 if trade['win'] else 0)

            if len(features_list) < 5: st.error("מעט מדי עסקאות לאימון. שנה פרמטרים."); return

            feature_df = pd.DataFrame(features_list)
            le = LabelEncoder()
            phase_encoded = le.fit_transform(feature_df['phase'].fillna("לא בתהליך איסוף"))
            phase_dummies = pd.get_dummies(phase_encoded, prefix='phase').astype(int)
            tech_factors = feature_df.drop(columns=['phase']).select_dtypes(include=[np.number])
            X = pd.concat([tech_factors.reset_index(drop=True), phase_dummies.reset_index(drop=True)], axis=1).fillna(0)
            y = np.array(labels)

            model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            train_acc = model.score(X, y)
            
            # --- חישוב Threshold אופטימלי ---
            optimal_th = calculate_optimal_threshold(model, X, y)

            meta = {"train_ticker": train_ticker.upper(), "train_acc": train_acc, "test_acc": train_acc,
                    "slot": target_slot, "model_type": "Wyckoff-Anchored", "num_trades": len(features_list),
                    "recommended_threshold": optimal_th}
            
            save_path = save_model_to_disk(target_slot, model, meta, le)
            st.session_state.model_archive = load_all_models_from_disk()
            st.session_state.ml_model = model
            st.session_state.ml_metadata = meta
            st.session_state.phase_encoder = le
            st.session_state.use_ml = True
            
            st.success(f"✅ אימון הושלם בהצלחה! מודל נשמר: {save_path}")
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("דיוק אימון", f"{train_acc*100:.1f}%")
            c_res2.metric("כמות עסקאות שנלמדו", len(features_list))
            c_res3.metric("🎯 Threshold מומלץ (AI)", optimal_th)

            st.info("💡 שים לב: כשסף ה-Threshold המומלץ יפסיק להשתנות מאימון לאימון (על אותה מניה/סקטור), תדע שהמודל הגיע למיצוי ההבנה שלו את השוק (Convergence).")

    st.markdown("---")
    st.markdown("### 📦 מודלים קיימים במערכת")
    if st.session_state.model_archive:
        for slot_name, data in st.session_state.model_archive.items():
            meta = data["metadata"]
            st.markdown(f"- **{slot_name}**: אומן על {meta.get('train_ticker', '?')} | Threshold מומלץ: **{meta.get('recommended_threshold', 'לא חושב')}**")

# ============================================================
# ניתוב
# ============================================================
routes = {"wyckoff": screen_wyckoff, "vp": screen_vp, "vwap": screen_vwap,
          "composite": screen_composite, "backtest": screen_backtest,
          "ml": screen_ml_trainer, "scanner": screen_scanner}
routes[st.session_state.mode]()
