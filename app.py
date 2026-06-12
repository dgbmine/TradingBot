# ============================================================
# חלק 1: ייבוא ספריות והגדרות עמוד בסיסיות
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
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Institutional Scout Pro")


# ============================================================
# חלק 2: רשימת המניות וחלוקה לקבוצות סקטוריאליות (SCAN UNIVERSE)
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
    "צמיחה וטכנולוגיה (Growth - NVDA, AMD, PLTR, AAPL...)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
        "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
        "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
        "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
        "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA"
    ],
    "ערך ומדדים (Value/Index - JPM, BRK-B, WMT, COST...)": [
        "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
        "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
        "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
        "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
        "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
        "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
        "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
        "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
    ],
    "סחורות ואנרגיה (Commodities - XOM, CVX, COP, GLD...)": [
        "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
        "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG"
    ]
}


# ============================================================
# חלק 3: עיצוב CSS מתקדם ומותאם ל-RTL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Hebrew', sans-serif;
    direction: rtl;
    text-align: right;
    box-sizing: border-box;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Sans Hebrew', sans-serif;
    direction: rtl;
}
.mono-text {
    font-family: 'IBM Plex Mono', monospace;
    direction: ltr;
    text-align: left;
}
.header-box {
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 28px;
    color: #e0eaf4;
    line-height: 1.9;
}
.header-box.wyckoff   { background: linear-gradient(135deg, #0f1923, #1a2a3a); border: 1px solid #2a4a6a; }
.header-box.vp         { background: linear-gradient(135deg, #160f23, #251535); border: 1px solid #4a2a6a; }
.header-box.vwap       { background: linear-gradient(135deg, #0f2318, #1a3528); border: 1px solid #2a6a4a; }
.header-box.composite  { background: linear-gradient(135deg, #1a1208, #2a1e08); border: 1px solid #6a4a1a; }
.header-box.ml         { background: linear-gradient(135deg, #1c0a20, #2e1236); border: 1px solid #7b1fa2; }
.header-box.scanner    { background: linear-gradient(135deg, #0f231f, #1a3a35); border: 1px solid #26a69a; }

.widget-panel-ai {
    background: #111922;
    border: 1px solid #2d3d4f;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
}
.factor-box {
    background: #111b26; 
    border: 1px solid #1e3040; 
    border-radius: 8px; 
    padding: 12px; 
    margin-bottom: 10px;
}
.factor-title {
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.9rem; 
    font-weight: 600;
}
.hit { color: #26a69a; font-weight: 600; }
.miss { color: #ef5350; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# חלק 4: ניהול משתני זיכרון גלובליים (SESSION STATE)
# ============================================================
for k, v in [("mode", "wyckoff"), ("ml_model", None), ("ml_metadata", None), ("use_ml", False), ("model_archive", {})]:
    if k not in st.session_state: 
        st.session_state[k] = v


# ============================================================
# חלק 5: פונקציות תשתית לפענוח וטעינת מודלים (מניעת באגים)
# ============================================================
def clean_and_unpack_archive(archive_dict):
    """מוודא שכל המודלים בארכיון עברו דה-סריאליזציה מלאה מאובייקט בייטס"""
    unpacked = {}
    for slot_name, data in archive_dict.items():
        model_obj = data["model"]
        if isinstance(model_obj, bytes):
            try:
                model_obj = pickle.loads(model_obj)
            except:
                pass
        unpacked[slot_name] = {"model": model_obj, "metadata": data["metadata"]}
    return unpacked

def trigger_auto_load_from_file():
    """טוען אוטומטית את קובץ הארכיון מתיקיית המודלים המשותפת"""
    file_path = "models/batch_archive_v1.txt"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                encoded_batch = f.read().strip()
            if encoded_batch and "כאן יודבק" not in encoded_batch and len(encoded_batch) > 50:
                decoded = base64.b64decode(encoded_batch.encode("utf-8"))
                raw_archive = pickle.loads(decoded)
                st.session_state.model_archive = clean_and_unpack_archive(raw_archive)
                return True
        except:
            pass
    return False


# ============================================================
# חלק 6: רכיב ממשק משותף - בורר ה-AI החכם (גלובלי)
# ============================================================
def render_active_ai_selector_widget(screen_identifier):
    """מציג פאנל שליטה חכם ואינטראקטיבי לבחירה והפעלת מודלים ישירות מכל מסך"""
    trigger_auto_load_from_file()
    
    st.markdown("<div class='widget-panel-ai'>", unsafe_allow_html=True)
    st.markdown("### 🧠 הגדרות מנוע החלטה AI חכם")
    
    col_a, col_b, col_c = st.columns([2, 1.5, 1])
    
    with col_a:
        if st.session_state.model_archive:
            slots_list = list(st.session_state.model_archive.keys())
            selected_slot = st.selectbox(
                "בחר מודל מוסדי פעיל:", 
                slots_list, 
                key=f"selector_slot_{screen_identifier}"
            )
            
            if st.button("⚡ טען והפעל מודל נבחר", key=f"activate_btn_{screen_identifier}", use_container_width=True):
                target_data = st.session_state.model_archive[selected_slot]
                model_instance = target_data["model"]
                if isinstance(model_instance, bytes):
                    model_instance = pickle.loads(model_instance)
                
                st.session_state.ml_model = model_instance
                st.session_state.ml_metadata = target_data["metadata"]
                st.session_state.use_ml = True
                st.success(f"המודל המשויך למשבצת '{selected_slot}' הופעל בהצלחה!")
                st.rerun()
        else:
            st.info("לא נמצאו מודלים טעונים בזיכרון. אנא עבור ל-ML Trainer לאימון או טעינה.")
            
    with col_b:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 סנכרון מהיר מגיטהאב", key=f"sync_git_{screen_identifier}", use_container_width=True):
            if trigger_auto_load_from_file():
                st.success("✅ סנכרון הקובץ מגיטהאב הושלם בהצלחה!")
                st.rerun()
            else:
                st.error("קובץ המאגר ריק או לא נמצא בנתיב היעד: models/batch_archive_v1.txt")
                
    with col_c:
        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
        ai_toggle = st.checkbox(
            "הפעל שימוש ב-AI בחישובים", 
            value=st.session_state.use_ml, 
            key=f"checkbox_ai_{screen_identifier}"
        )
        if ai_toggle != st.session_state.use_ml:
            st.session_state.use_ml = ai_toggle
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# חלק 7: תפריט ניווט עליון
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
nav = [
    ("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
    ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
    ("backtest","📈  Backtest"), ("ml","🧠  ML Trainer"), ("scanner","🔎  Scanner")
]
cols = [c1,c2,c3,c4,c5,c6,c7]
for col, (mode_key, label) in zip(cols, nav):
    with col:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.mode==mode_key else "secondary", key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key
            st.rerun()
st.markdown("---")

if st.session_state.use_ml and st.session_state.ml_model is not None:
    metadata = st.session_state.ml_metadata or {}
    acc = metadata.get("train_acc", 0.0)
    train_ticker = metadata.get("train_ticker", "???")
    period = metadata.get("period", "???")
    slot = metadata.get("slot", "כללי")
    st.info(f"🤖 **מצב AI מופעל באופן גלובלי:** מודל פעיל - {slot} (אומן על {train_ticker}, {period}) | דיוק פנימי: {acc*100:.1f}%")


# ============================================================
# חלק 8: הגדרות מנוע הבק-טסט והפקטורים
# ============================================================
@dataclass
class BacktestConfig:
    commission: float = 0.001
    slippage: float = 0.0005
    initial_capital: float = 100_000.0
    position_size: float = 0.10
    hold_days: int = 40
    min_score: int = 65
    exit_score: int = 35
    period: str = "2y"
    regime_ticker: str = "SPY"

class FactorEngine:
    def __init__(self, cfg: BacktestConfig): 
        self.cfg = cfg
        
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        body = (df["Close"] - df["Open"]).abs()
        rng = df["High"] - df["Low"]
        lower_shadow = df[["Open","Close"]].min(axis=1) - df["Low"]
        vol_ma20 = df["Volume"].rolling(20).mean()
        vol_ma5 = df["Volume"].rolling(5).mean()
        rvol = df["Volume"] / vol_ma20.replace(0, np.nan)

        price_bins = pd.cut(df["Close"], bins=40, labels=False)
        f["f01_liquidity_gap"] = ((df.groupby(price_bins)["Volume"].transform("sum") < df.groupby(price_bins)["Volume"].transform("mean") * 0.5).astype(float).rolling(5).mean())
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        atr14 = pd.concat([rng, (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1).rolling(14).mean()
        f["f02_volatility_squeeze"] = ((((2 * std20) / sma20.replace(0, np.nan)) < ((2 * std20) / sma20.replace(0, np.nan)).rolling(20).mean() * 0.75) & (atr14 < atr14.rolling(20).mean() * 0.75)).astype(float)
        spy_slope = df.get("spy_close", df["Close"]).rolling(50).mean().diff(10) / df.get("spy_close", df["Close"]).rolling(50).mean().shift(10).replace(0, np.nan)
        f["f03_regime"] = (spy_slope > 0.01).astype(float) - (spy_slope < -0.01).astype(float)
        f["f04_absorption"] = ((df["Close"] < (df["Low"] + rng * 0.35)) & (lower_shadow > body * 1.5) & (rvol > 1.5)).astype(float)
        resist = df["High"].rolling(20).max().shift(1)
        f["f05_breakout_quality"] = ((df["Close"] > resist) & (df["Close"].rolling(3).mean() > resist.shift(1))).astype(float)
        f["f06_cis_weight"] = np.clip(1.0 / (std20 / std20.rolling(60).mean().replace(0, np.nan)).replace(0, np.nan), 0.5, 2.0)
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        f["f07_obv_velocity"] = (obv.diff(10) / obv.abs().rolling(10).mean().replace(0, np.nan)).clip(-3, 3)
        f["f08_fft"] = ((df["Close"] > df["Close"].shift(1)) & (df["Close"].shift(-1) < df["Close"]) & (rvol > 1.5)).astype(float)
        f["f09_dependency"] = f["f04_absorption"].rolling(10).corr(f["f07_obv_velocity"]).clip(-1, 1)
        f["f10_temporal_seq"] = (f["f04_absorption"].rolling(30).max() * (rvol < 0.7).astype(float))
        f["f11_kill_switch"] = ((df["Close"].pct_change() < -0.05) | (rvol > 4.0)).astype(float)
        f["f12_distribution"] = ((df["High"] > df["High"].rolling(20).max().shift(1)) & (df["Close"] < df["High"] - rng * 0.7)).astype(float)
        
        f["f13_confidence_decay"] = np.exp(-f["f04_absorption"].replace(0, np.nan).ffill().isna().astype(int) / 10.0).clip(0, 1)
        f["f14_inst_intent"] = (f["f04_absorption"] * 0.3 + f["f07_obv_velocity"].clip(0, 1) * 0.4 + f["f10_temporal_seq"] * 0.3).clip(0, 1)
        f["f15_mtf"] = ((df["Close"] > sma20).astype(float) * (df["Close"].rolling(5).mean() > df["Close"].rolling(5).mean().rolling(4).mean()).astype(float))
        vwap_full = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        f["f16_anchor_conflict"] = (((df["Close"] > vwap_full).astype(float).rolling(3).sum() * (df["Close"] < vwap_full).astype(float).rolling(3).sum()) > 0).astype(float)
        f["f17_vol_cluster"] = (atr14 > atr14.shift(5) * 1.3).astype(float)
        f["f18_sector_breadth"] = (df["Close"] > df["Close"].shift(1)).astype(float).rolling(10).mean()
        f["f19_order_flow"] = (((df["Close"] - df["Low"]) / rng.replace(0, np.nan)) - ((df["High"] - df["Close"]) / rng.replace(0, np.nan))).rolling(5).mean()
        support = df["Low"].rolling(20).min().shift(1)
        f["f20_liquidity_sweep"] = ((df["Low"] < support) & (df["Close"] > support)).astype(float)
        range_20 = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
        f["f21_break_auth"] = ((df["Close"] - df["Close"].shift(1)).abs() / range_20.replace(0, np.nan)).clip(0, 1)
        f["f22_sr_strength"] = (df["Low"].rolling(5).min() <= df["Low"].rolling(20).min() * 1.005).astype(float).rolling(20).sum() / 20
        f["f23_gap_structure"] = (df["Open"] > df["Close"].shift(1) * 1.005).astype(float) - (df["Open"] < df["Close"].shift(1) * 0.995).astype(float)
        f["f24_event_shock"] = 1.0 - (df["Close"].pct_change().abs() > 0.04).astype(float).rolling(3).sum().clip(0, 1)
        
        f["f25_rvol_anomaly"] = ((rvol - rvol.rolling(60).mean()) / rvol.rolling(60).std().replace(0, np.nan)).clip(-3, 3)
        f["f26_accept_reject"] = ((df["Close"] > (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float).rolling(5).mean() - ((df["Close"] < (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float).rolling(5).mean()
        f["f27_vol_regime"] = ((atr14 / atr14.rolling(60).mean().replace(0, np.nan)) < 0.8).astype(float) - ((atr14 / atr14.rolling(60).mean().replace(0, np.nan)) > 1.2).astype(float)
        f["f28_inst_part"] = ((body > body.rolling(20).mean() * 1.5) & (rvol > 1.5)).astype(float)
        sma50 = df["Close"].rolling(50).mean(); sma200 = df["Close"].rolling(200).mean()
        f["f29_trend_integrity"] = ((df["Close"] > sma20).astype(int) + (sma20 > sma50).astype(int) + (sma50 > sma200).astype(int)) / 3
        f["f30_mean_rev"] = (-((df["Close"] - sma20) / std20.replace(0, np.nan))).clip(-3, 3)
        f["f31_bear_trap"] = ((df["Close"] < df["Low"].rolling(20).min().shift(1)) & (df["Close"].shift(1) > df["Low"].rolling(20).min().shift(2))).astype(float)
        dist_ath = (df["Close"].rolling(252).max() - df["Close"]) / df["Close"].rolling(252).max().replace(0, np.nan)
        f["f32_accum_type"] = (dist_ath > 0.25).astype(float) * 1.0 + ((dist_ath < 0.15) & (dist_ath > 0.05)).astype(float) * 0.6
        f["f33_liq_exhaust"] = ((vol_ma5 < vol_ma5.shift(10)) & (df["Close"].pct_change(5).abs() < 0.02)).astype(float)
        f["f34_corr_stress"] = df["Close"].pct_change().rolling(20).corr(df.get("spy_close", df["Close"]).pct_change()).clip(-1, 1)
        f["f34_corr_stress"] = f["f34_corr_stress"].fillna(0)
        f["f35_struct_break"] = (df["Close"] > df["High"].rolling(20).max().shift(1)).astype(float) - (df["Close"] < df["Low"].rolling(20).min().shift(1)).astype(float)
        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame, df: pd.DataFrame = None) -> pd.Series:
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            model = st.session_state.ml_model
            if isinstance(model, bytes):
                model = pickle.loads(model)
            try:
                probs = model.predict_proba(factors)[:, 1]
            except:
                probs = model.predict(factors)
            score = pd.Series(probs * 100, index=factors.index)
            if df is not None:
                sma50 = df["Close"].rolling(50).mean()
                boost = ((score > 10) & (df["Close"] > sma50)).astype(float) * 20
                score = (score + boost).clip(0, 100)
        else:
            w = {
                "f01_liquidity_gap": 3, "f02_volatility_squeeze": 4, "f03_regime": 5, "f04_absorption": 6,
                "f05_breakout_quality": 3, "f06_cis_weight": 2, "f07_obv_velocity": 5, "f08_fft": -2,
                "f09_dependency": 2, "f10_temporal_seq": 5, "f11_kill_switch": 0, "f12_distribution": -4,
                "f13_confidence_decay": 3, "f14_inst_intent": 6, "f15_mtf": 4, "f16_anchor_conflict": -2,
                "f17_vol_cluster": -1, "f18_sector_breadth": 3, "f19_order_flow": 4, "f20_liquidity_sweep": 3,
                "f21_break_auth": 2, "f22_sr_strength": 2, "f23_gap_structure": 2, "f24_event_shock": 2,
                "f25_rvol_anomaly": 2, "f26_accept_reject": 3, "f27_vol_regime": 3, "f28_inst_part": 3,
                "f29_trend_integrity": 3, "f30_mean_rev": 3, "f31_bear_trap": 2, "f32_accum_type": 2,
                "f33_liq_exhaust": -1, "f34_corr_stress": 1, "f35_struct_break": 2,
            }
            tot = sum(abs(v) for v in w.values() if v != 0)
            score = pd.Series(0.0, index=factors.index)
            for col, weight in w.items():
                if col in factors.columns and col != "f11_kill_switch":
                    score += factors[col].clip(-1, 1) * weight
            score = (score / tot * 100 + 50).clip(0, 100)
        if "f11_kill_switch" in factors.columns: 
            score = score * (1 - factors["f11_kill_switch"])
        return score.round(1)

class SignalDebugger:
    LABELS = {
        "f01_liquidity_gap": "Liquidity Gap (LVN)", "f02_volatility_squeeze": "Volatility Squeeze",
        "f03_regime": "Market Regime", "f04_absorption": "Absorption Signature",
        "f05_breakout_quality": "Breakout Quality", "f06_cis_weight": "Dynamic Weights",
        "f07_obv_velocity": "OBV Accumulation Velocity", "f08_fft": "Failure to Follow Through",
        "f09_dependency": "Signal Dependency", "f10_temporal_seq": "Temporal Sequencing",
        "f12_distribution": "Distribution Mirror", "f13_confidence_decay": "Confidence Decay",
        "f14_inst_intent": "Institutional Intent", "f15_mtf": "MTF Confirmation",
        "f16_anchor_conflict": "Anchor Conflict", "f17_vol_cluster": "Vol Cluster Expansion",
        "f18_sector_breadth": "Sector Breadth", "f19_order_flow": "Order Flow Imbalance",
        "f20_liquidity_sweep": "Liquidity Sweep", "f21_break_auth": "Range Break Auth",
        "f22_sr_strength": "S/R Strength", "f23_gap_structure": "Gap Structure",
        "f24_event_shock": "Event Shock Normalization", "f25_rvol_anomaly": "Relative Volume Anomaly",
        "f26_accept_reject": "Price Accept vs Reject", "f27_vol_regime": "Vol Regime Transition",
        "f28_inst_part": "Institutional Participation", "f29_trend_integrity": "Trend Integrity",
        "f30_mean_rev": "Mean Reversion Pressure", "f31_bear_trap": "False Support Breakdown",
        "f32_accum_type": "Accumulation Differentiation", "f33_liq_exhaust": "Liquidity Exhaustion",
        "f34_corr_stress": "Correlation Stress", "f35_struct_break": "Structural Break"
    }
    def audit(self, factors: pd.DataFrame, cis: pd.Series) -> list:
        row = factors.iloc[-1]; res = []
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            model = st.session_state.ml_model
            if isinstance(model, bytes):
                model = pickle.loads(model)
            importances = model.feature_importances_
            for i, col in enumerate(factors.columns):
                if col in self.LABELS and importances[i] > 0.01:
                    direction = 1 if row[col] > 0 else -1
                    res.append({"factor": self.LABELS[col], "impact": importances[i] * direction * 100})
        else:
            for col, val in row.items():
                if col in self.LABELS and val != 0: 
                    res.append({"factor": self.LABELS[col], "impact": val})
        return sorted(res, key=lambda x: x["impact"], reverse=True)


# ============================================================
# חלק 9: מנוע סימולציית העבר (BACKTEST ENGINE)
# ============================================================
class BacktestEngine:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.factors = FactorEngine(self.cfg)
        self.debugger = SignalDebugger()
        
    def run(self, ticker: str):
        try:
            df = yf.Ticker(ticker).history(period=self.cfg.period)
            if df is None or len(df) < 50: 
                return {"error": "נתוני שוק לא זמינים או חסרים עבור הטיקר המבוקש."}
            df.index = pd.to_datetime(df.index).tz_localize(None)
            try: 
                df["spy_close"] = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)["Close"].reindex(df.index).ffill()
            except: 
                df["spy_close"] = df["Close"]
            f = self.factors.compute(df)
            cis = self.factors.composite_cis(f, df)
            entry = (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score)
            exit_ = (cis < self.cfg.exit_score)
            closes = df["Close"].values; dates = df.index; trades = []; in_trade = False; entry_px = 0; hold = 0
            for i in range(1, len(closes)):
                if not in_trade and entry.iloc[i]:
                    entry_px = closes[i] * (1 + self.cfg.commission); in_trade = True; hold = 0; ent_d = dates[i]
                elif in_trade:
                    hold += 1
                    if exit_.iloc[i] or hold >= self.cfg.hold_days:
                        ext_px = closes[i] * (1 - self.cfg.commission)
                        trades.append({"entry_date": ent_d, "exit_date": dates[i], "return": (ext_px - entry_px)/entry_px})
                        in_trade = False
            trades_df = pd.DataFrame(trades); equity = [self.cfg.initial_capital]
            if not trades_df.empty:
                for r in trades_df["return"]: 
                    equity.append(equity[-1] * (1 + r))
            wr = (trades_df["return"] > 0).mean() if not trades_df.empty else 0
            ret = (equity[-1] - self.cfg.initial_capital) / self.cfg.initial_capital
            equity_arr = np.array(equity); peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak; max_dd = drawdown.min() if len(drawdown) > 0 else 0
            return {"df": df, "cis": cis, "audit": self.debugger.audit(f, cis), "trades": len(trades_df), "wr": wr, "ret": ret, "max_dd": max_dd}
        except Exception as e: 
            return {"error": str(e)}


# ============================================================
# חלק 10: מודול WYCKOFF STRUCTURAL ENGINE
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y"):
    try: 
        df = yf.Ticker(ticker).history(period=period)
    except: 
        return None
    if df is None or len(df) < 40: 
        return None
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["SPREAD"] = df["High"] - df["Low"]
    df["VOL_YEAR_MEAN"] = df["Volume"].rolling(252, min_periods=20).mean()
    df["SPREAD_YEAR_MEAN"] = df["SPREAD"].rolling(252, min_periods=20).mean()
    return df

def render_gauge(score, verdict, verdict_color):
    bc = "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, 
        title={'text':f"<b>Wyckoff Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>"}, 
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':bc}, 'bgcolor':"#0d1b2a"}, number={'font':{'color':bc}}
    ))
    fig.update_layout(height=300, margin=dict(t=80,b=10,l=20,r=20), paper_bgcolor="#0a1520", font_color="#e0eaf4")
    return fig

def analyze_wyckoff(df):
    score = 0; alerts = []; current_phase = "לא בתהליך איסוף"; phase_explanation = "המניה לא הראתה סימני בלימה משמעותיים ב-90 הימים האחרונים."
    last_30 = df.iloc[-30:]
    for i in range(len(last_30)):
        day = last_30.iloc[i]; days_ago = len(last_30) - i - 1
        if day["Volume"] > day["VOL_YEAR_MEAN"] * 1.8:
            spread_ratio = day["SPREAD"] / day["SPREAD_YEAR_MEAN"]
            if spread_ratio < 1.2:
                direction = "ירידות" if day["Close"] < day["Open"] else "עליות"
                alerts.append(f"⚠️ שים לב: לפני {days_ago} ימי מסחר נצפה מחזור חריג ב{direction}, אך תנועת המחיר נבלמה (ספיגת סחורה מוסדית / התנגדות).")
            elif spread_ratio >= 1.4:
                alerts.append(f"✅ שים לב: לפני {days_ago} ימי מסחר זוהה מחזור חריג שתורגם ישירות למהלך מחיר רחב. השתתפות מוסדית ברורה.")
    last_90 = df.iloc[-90:]
    sc_candidates = last_90[(last_90["Volume"] > last_90["VOL_YEAR_MEAN"] * 2.2) & (last_90["Close"] < last_90["Open"])]
    if not sc_candidates.empty:
        sc_idx = sc_candidates.index[0]; sc_low = df.loc[sc_idx, "Low"]; post_sc = df.loc[sc_idx:]; days_since_sc = len(post_sc)
        if days_since_sc < 7:
            current_phase = "Phase A (Stopping the Trend)"
            phase_explanation = f"המניה חוותה בלימה מוסדית אגרסיבית לפני {days_since_sc} ימים. התחלת תהליך בניית תחתית."
            score += 35
        else:
            spring_candidates = post_sc[(post_sc["Low"] < sc_low) & (post_sc["Close"] > sc_low)]
            if not spring_candidates.empty and (len(post_sc) - post_sc.index.get_loc(spring_candidates.index[-1])) <= 15:
                days_since_spring = len(post_sc) - post_sc.index.get_loc(spring_candidates.index[-1]) - 1
                current_phase = "Phase C (Spring / Shakeout)"
                phase_explanation = f"לפני {days_since_spring} ימים המניה ביצעה ניעור של קצוות נזילות ושאבה פקודות סטופ תחת התחתית ההיסטורית."
                score += 85
            else:
                current_phase = "Phase B (Building a Cause)"
                phase_explanation = f"המניה בונה סיבה בתוך טווח הדשדוש הנוכחי מזה {days_since_sc} ימי מסחר רצופים."
                score += 60
    vd = current_phase; vc = "#26a69a" if score >= 75 else "#ffa726" if score >= 40 else "#ef5350"
    return score, current_phase, phase_explanation, alerts, vd, vc

def render_wyckoff_chart(df):
    dc = df.iloc[-120:].copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"], name="Price"), row=1, col=1)
    colors = ['#ef5350' if row['Open'] > row['Close'] else '#26a69a' for _, row in dc.iterrows()]
    fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], name="Volume", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=dc.index, y=dc["VOL_YEAR_MEAN"], mode='lines', name="1-Year Avg Vol", line=dict(color='#ffa726', width=1.5, dash='dot')), row=2, col=1)
    fig.update_layout(height=450, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a", font_color="#e0eaf4", xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
    return fig

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF 3.0 STRUCTURAL ENGINE</h2><p>מערכת ניתוח מוסדית המבוססת על חוקי מאמץ מול תוצאה (VSA) וזיהוי שלבי איסוף/פיזור בזמן אמת, בשילוב אינטגרציית מודלים חכמים.</p></div>""",unsafe_allow_html=True)
    
    # הוספת הוידג'ט החכם המבוקש ישירות למסך וויקוף לפני החיפוש
    render_active_ai_selector_widget("wyckoff_screen")
    
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("הזן סימול מניה לניתוח מבנה (למשל: NVDA, PLTR, AMZN):", "NVDA", key="wyckoff_ticker_input")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        btn = st.button("▶ הרץ ניתוח VSA מוסדי", use_container_width=True, type="primary")
        
    if btn:
        with st.spinner("מנתח חתימות מחיר ומחזור..."):
            df = get_data(ticker.upper())
            if df is not None:
                score, current_phase, phase_exp, alerts, vd, vc = analyze_wyckoff(df)
                col1, col2 = st.columns([1, 2])
                with col1: 
                    st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
                with col2:
                    st.markdown(f"### 📍 סטטוס מבני נוכחי: **{current_phase}**")
                    st.markdown(f"*{phase_exp}*")
                    st.markdown("---")
                    st.markdown("#### ניתוח זיהוי מאמץ מול תוצאה:")
                    if alerts:
                        for alert in alerts:
                            color = "#ef5350" if "⚠️" in alert else "#26a69a"
                            st.markdown(f"<div style='background:#111b26; border-right:4px solid {color}; padding:10px; margin-bottom:10px; border-radius:5px;'>{alert}</div>", unsafe_allow_html=True)
                    else: 
                        st.info("לא נצפו חריגות נפח מסחר קיצוניות ביחס לממוצע השנתי במהלך 30 ימי המסחר האחרונים.")
                st.plotly_chart(render_wyckoff_chart(df), use_container_width=True)
            else:
                st.error("לא נמצאו מספיק נתוני היסטוריה עבור הטיקר שהוזן. ודא שהסימול תקין.")


# ============================================================
# חלק 11: שומרי מקום למסכים קבועים
# ============================================================
def screen_vp(): 
    st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE (מנוע פרופיל ווליום)</h2><p>מודול פרופיל הנפח וזיהוי אזורי ערך (Value Areas) פועל ומחושב כעת באופן אוטומטי כחלק מ-35 הפקטורים המובנים במערכת הבק-טסט והסורק.</p></div>""",unsafe_allow_html=True)
def screen_vwap(): 
    st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION (סטיות VWAP מוסדיות)</h2><p>מנוע סטיות נפח המחיר המשוקלל (Anchor Conflict) פעיל ומחושב דינמית כחלק מפקטורי המערכת המרכזיים.</p></div>""",unsafe_allow_html=True)
def screen_composite(): 
    st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE (שקלול CIS מקיף)</h2><p>מערכת הניקוד המשוקללת פועלת כעת במלואה בתוך מנוע הבק-טסט ומציגה שרשרת סימנים חיוביים ושליליים בזמן אמת.</p></div>""",unsafe_allow_html=True)

def screen_backtest():
    st.markdown("""<div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;"><h2>📈 BACKTEST ENGINE (מנוע בק-טסט מונחה 35 פקטורים)</h2><p>הרצת סימולציית מסחר היסטורית מלאה המשלבת הגדרות עמלות, השפעת מחיר, וחישובי הסתברות מבוססי AI.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("הזן סימול מניה לבדיקה היסטורית מקיפה:", "NVDA", key="bt_input_field")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ הפעל סימולציית 35 פקטורים", use_container_width=True, type="primary")
        
    if run_btn:
        with st.spinner(f"מריץ סימולציה ומחשב משקלים עבור {ticker.upper()}..."):
            res = BacktestEngine().run(ticker.upper())
            if "error" in res: 
                st.error(f"⚠️ שגיאה בעיבוד: {res['error']}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("סה״כ עסקאות", res["trades"])
                col2.metric("אחוז הצלחה (Win Rate)", f"{res['wr']*100:.1f}%")
                col3.metric("תשואה מצטברת בסימולציה", f"{res['ret']*100:.1f}%")
                col4.metric("דרודאון מקסימלי (Max DD)", f"{res['max_dd']*100:.1f}%")
                st.markdown("---")
                st.markdown(f"### 📊 ציון CIS נוכחי: **{res['cis'].iloc[-1]:.1f} / 100**")
                audit = res["audit"]
                positives = [x for x in audit if x['impact'] > 0]
                negatives = [x for x in audit if x['impact'] < 0]
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.success("✅ **פקטורים שתמכו ותדלקו את הציון מעלה:**")
                    for p in positives[:5]: 
                        st.markdown(f"<div class='factor-box'><span class='hit'>+</span> <span class='factor-title'>{p['factor']}</span></div>", unsafe_allow_html=True)
                with pc2:
                    st.error("❌ **פקטורים שהפעילו לחץ והורידו את הציון מטה:**")
                    for n in sorted(negatives, key=lambda x: x['impact'])[:5]: 
                        st.markdown(f"<div class='factor-box'><span class='miss'>-</span> <span class='factor-title'>{n['factor']}</span></div>", unsafe_allow_html=True)


# ============================================================
# חלק 12: פונקציות עזר וסיכום מודל (תיקון באג feature_importances_)
# ============================================================
def get_model_summary(model, metadata):
    """מחלץ בבטחה את חשיבות הפקטורים מהמודל ומונע קריסות של אובייקטי בייטס"""
    if isinstance(model, bytes):
        try:
            model = pickle.loads(model)
        except Exception as e:
            return {"error_summary": f"נכשלה פתיחת אריזת המודל: {str(e)}"}
            
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {
            "train_ticker": metadata.get("train_ticker", "???"),
            "train_acc": metadata.get("train_acc", 0.0),
            "period": metadata.get("period", "???"),
            "slot": metadata.get("slot", "כללי"),
            "top_factors": [{"name": "לא זמין - מודל לא מאומן כראוי", "importance": 0.0}]
        }
        
    top_factors = sorted(zip(SignalDebugger.LABELS.keys(), importances), key=lambda x: x[1], reverse=True)[:5]
    summary = {
        "train_ticker": metadata.get("train_ticker", "???"),
        "train_acc": metadata.get("train_acc", 0.0),
        "period": metadata.get("period", "???"),
        "slot": metadata.get("slot", "כללי"),
        "top_factors": [{"name": SignalDebugger.LABELS.get(f, f), "importance": imp} for f, imp in top_factors]
    }
    return summary


# ============================================================
# חלק 13: מודול MACHINE LEARNING TRAINER & ARCHIVE
# ============================================================
def screen_ml_trainer():
    st.markdown("""<div class="header-box ml"><h2>🧠 MACHINE LEARNING TRAINER & ARCHIVE</h2><p>מסך אימון וניהול מודלים מבוססי Random Forest לחלוקה אסטרטגית לפי 3 משבצות סקטוריאליות קבועות.</p></div>""",unsafe_allow_html=True)
    
    MODEL_SLOTS = ["Growth (צמיחה)", "Value/Index (ערך/מדדים)", "Commodities (סחורות)"]
    
    st.markdown("### 📥 טעינה וניהול המאגר")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 טען מאגר אוטומטית מגיטהאב (Auto-Load)", use_container_width=True, type="primary"):
            if trigger_auto_load_from_file():
                st.success("✅ כל המודלים נטענו בהצלחה מתוך קובץ הארכיון בגיטהאב!")
                st.rerun()
            else:
                st.warning("לא נמצא קובץ תקין בנתיב היעד, או שהקובץ ריק.")
                
        with st.expander("אפשרויות טעינה ידנית (הדבקת קוד Base64)"):
            encoded_paste = st.text_area("הדבק קוד מאגר מוצפן:", height=80)
            if st.button("בצע טעינה ידנית"):
                try:
                    decoded = base64.b64decode(encoded_paste.strip().encode("utf-8"))
                    raw_archive = pickle.loads(decoded)
                    st.session_state.model_archive = clean_and_unpack_archive(raw_archive)
                    st.success("✅ המאגר הידני נטען ופוענח בהצלחה!"); st.rerun()
                except: 
                    st.error("❌ הקוד שהוזן אינו תקין או פגום.")

    with col2:
        if st.session_state.model_archive:
            available_slots = list(st.session_state.model_archive.keys())
            st.markdown(f"**📚 משבצות פעילות ומאוכלסות כרגע במערכת:** {len(available_slots)} / 3")
            selected_model = st.selectbox("בחר מודל להפעלה במנוע החישוב הראשי של השוק:", available_slots)
            if st.button("✅ הפעל מודל נבחר לשימוש"):
                target_data = st.session_state.model_archive[selected_model]
                model_obj = target_data["model"]
                if isinstance(model_obj, bytes):
                    model_obj = pickle.loads(model_obj)
                    
                st.session_state.ml_model = model_obj
                st.session_state.ml_metadata = target_data["metadata"]
                st.session_state.use_ml = True
                st.success(f"המודל '{selected_model}' הוגדר בהצלחה כמודל השולט במערכת!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 🚀 אימון מודל סקטוריאלי חדש")
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
    with c1: train_ticker = st.text_input("הזן סימול מניה מובילה לאימון (למשל: AMD, COP, IWM):", "SPY", key="ml_train_ticker")
    with c2: target_slot = st.selectbox("שייך למשבצת אסטרטגית (ידרוס מודל קודם במשבצת):", MODEL_SLOTS)
    with c3: start_date = st.date_input("תאריך תחילת אימון:", value=datetime(2023, 1, 1))
    with c4: end_date = st.date_input("תאריך סיום אימון:", value=datetime(2023, 12, 31))

    if st.button("🚀 התחל תהליך למידת מכונה ושמור למשבצת המוגדרת", use_container_width=True, type="primary"):
        with st.spinner(f"שואב נתונים ומאמן מודל משבצת '{target_slot}' על בסיס נתוני {train_ticker}..."):
            df = yf.Ticker(train_ticker.upper()).history(start=start_date, end=end_date)
            if df is not None and len(df) >= 40:
                engine = FactorEngine(BacktestConfig())
                factors = engine.compute(df)
                target = (df["Close"].shift(-10) / df["Close"] - 1 > 0.02).astype(int)
                valid_idx = target.notna()
                X = factors[valid_idx].copy(); y = target[valid_idx].values
                
                model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_split=50, random_state=42, n_jobs=-1)
                model.fit(X, y)
                
                acc = model.score(X, y)
                meta = {
                    "train_ticker": train_ticker.upper(), 
                    "train_acc": acc, 
                    "period": f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}", 
                    "slot": target_slot
                }
                
                st.session_state.model_archive[target_slot] = {"model": model, "metadata": meta}
                st.session_state.ml_model = model
                st.session_state.ml_metadata = meta
                st.session_state.use_ml = True
                st.success(f"✅ תהליך האימון הושלם בדיוק פנימי של {acc*100:.1f}%. המודל נשמר בהצלחה במשבצת: {target_slot}")
                st.rerun()
            else: 
                st.error("❌ שגיאה: לא נמצאו מספיק נתוני מסחר היסטוריים בטווח התאריכים המבוקש לאימון מודל יציב.")

    st.markdown("---")
    st.markdown("### 📤 פעולות ייצוא והפקת דוחות AI")
    ca, cb = st.columns(2)
    with ca:
        if st.session_state.model_archive:
            st.markdown("#### 📦 אריזת המאגר לקובץ")
            archive_export = {}
            for k, v in st.session_state.model_archive.items(): 
                archive_export[k] = {"model": pickle.dumps(v["model"]), "metadata": v["metadata"]}
            encoded_all = base64.b64encode(pickle.dumps(archive_export)).decode("utf-8")
            
            st.download_button(
                label="💾 הורד קובץ ארכיון מעודכן (batch_archive_v1.txt)",
                data=encoded_all,
                file_name="batch_archive_v1.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary"
            )
            st.markdown("<small>*לחץ להורדת הקובץ למחשב, ולאחר מכן העלה אותו לתיקיית models בגיטהאב (ע״י Upload files) כדי לבצע דריסה אוטומטית.*</small>", unsafe_allow_html=True)
                
    with cb:
        if st.session_state.model_archive:
            st.markdown("#### 🤖 הפקת דוח התקדמות עבור יועץ ה-AI")
            if st.button("הפק דוח למידה מפורט ללא שגיאות"):
                report = "### דוח התקדמות למידה (מעודכן לפי משבצות)" + "\n\n"
                for name, data in st.session_state.model_archive.items():
                    summ = get_model_summary(data['model'], data['metadata'])
                    if "error_summary" in summ:
                        report += "- **משבצת:** `" + name + "`\n  - שגיאה בחילוץ נתוני המודל.\n\n"
                        continue
                    report += "- **משבצת:** `" + name + "`\n"
                    report += "  - **טיקר אימון:** " + str(summ['train_ticker']) + "\n"
                    report += "  - **תקופה:** " + str(summ['period']) + "\n"
                    report += "  - **דיוק:** " + f"{summ['train_acc']*100:.1f}%\n"
                    report += "  - **פקטורים מובילים:** " + ', '.join([f['name'] for f in summ['top_factors'][:3]]) + "\n\n"
                report += "--- \n*העתק והדבק בצ'אט עם ה-AI.*"
                st.text_area("📋 טקסט הדו״ח המוסדי מוכן להעתקה:", value=report, height=160)


# ============================================================
# חלק 14: מודול MARKET SCANNER (סורק שוק חכם מבוסס סקטורים ומודלים)
# ============================================================
def screen_scanner():
    st.markdown("""<div class="header-box scanner"><h2>🔎 MARKET SCANNER - סורק שוק מוסדי מתקדמת</h2><p>סריקה מבוססת סקטורים ייעודיים המאפשרת התאמה מלאה ומדויקת בין קבוצת המניות הנבחרת למודל ה-AI הפעיל.</p></div>""",unsafe_allow_html=True)
    
    # הוספת הוידג'ט החכם המבוקש ישירות למסך הסורק לפני החיפוש והסינון
    render_active_ai_selector_widget("scanner_screen")
    
    st.markdown("### ⚙️ הגדרות סינון ומיקוד לסריקה")
    col_x, col_y, col_z = st.columns([2, 1, 1])
    
    with col_x:
        # הגולל הבורר המבוקש המאפשר למקד את הסורק בסקטור ספציפי ביחס למודל
        selected_sector_label = st.selectbox(
            "🎯 בחר סקטור / קבוצת מניות להתמקד בה בסריקה:", 
            list(SECTOR_MAP.keys()),
            key="scanner_sector_selector"
        )
        chosen_universe = SECTOR_MAP[selected_sector_label]
        
    with col_y:
        scan_limit = st.slider("כמות מניות לסריקה מקסימלית:", min_value=5, max_value=len(chosen_universe), value=min(20, len(chosen_universe)), step=5)
    with col_z:
        engine_choice = st.radio("מנוע ציון לסריקה:", ["Wyckoff Structural Engine", "Composite CIS Score (ML/Static)"])
        show_all = st.checkbox("הצג את כל תוצאות הסריקה", value=True)
        
    if st.button("🚀 התחל סריקת שוק ממוקדת סקטור", use_container_width=True, type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers_to_scan = chosen_universe[:scan_limit]
        
        for i, ticker in enumerate(tickers_to_scan):
            status_text.text(f"מנתח ומחשב פקטורים עבור {ticker} ({i+1} מתוך {len(tickers_to_scan)})...")
            df = get_data(ticker, period="1y")
            if df is not None and len(df) > 30:
                if "Wyckoff" in engine_choice:
                    score, current_phase, _, _, vd, _ = analyze_wyckoff(df)
                    results.append({"Ticker": ticker, "Score": score, "Engine": "Wyckoff", "Verdict": vd})
                else:
                    engine = FactorEngine(BacktestConfig())
                    factors = engine.compute(df)
                    cis_score = engine.composite_cis(factors, df).iloc[-1]
                    verdict = "Strong Signal" if cis_score >= 75 else "Watch" if cis_score >= 60 else "Wait"
                    results.append({"Ticker": ticker, "Score": cis_score, "Engine": "CIS", "Verdict": verdict})
            progress_bar.progress((i + 1) / len(tickers_to_scan))
            time.sleep(0.05)
            
        status_text.text("✅ הסריקה הסקטוריאלית הושלמה בהצלחה!")
        st.markdown("---")
        
        if results:
            df_results = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
            threshold = 40 if "Wyckoff" in engine_choice else 60
            if not show_all: 
                df_results = df_results[df_results["Score"] >= threshold]
                
            if not df_results.empty: 
                st.success(f"📊 מציג {len(df_results)} תוצאות עבור קבוצת: {selected_sector_label}")
                st.dataframe(df_results, use_container_width=True)
            else: 
                st.warning(f"לא נמצאו מניות בסקטור זה שעברו את רף הסינון המינימלי ({threshold}).")
        else: 
            st.error("הסריקה נכשלה או שלא נאספו נתונים תקינים.")


# ============================================================
# חלק 15: ניתוב הראוטר המרכזי של האפליקציה
# ============================================================
routes = {
    "wyckoff": screen_wyckoff, "vp": screen_vp, "vwap": screen_vwap, 
    "composite": screen_composite, "backtest": screen_backtest, 
    "ml": screen_ml_trainer, "scanner": screen_scanner
}
routes[st.session_state.mode]()
