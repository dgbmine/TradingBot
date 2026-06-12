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
import json
from sklearn.ensemble import RandomForestClassifier
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
# חלק 3: עיצוב CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Hebrew', sans-serif; direction: rtl; text-align: right; box-sizing: border-box; }
h1, h2, h3, h4, h5, h6 { direction: rtl; }
.header-box { border-radius: 12px; padding: 24px 32px; margin-bottom: 28px; color: #e0eaf4; line-height: 1.9; }
.header-box.wyckoff   { background: linear-gradient(135deg, #0f1923, #1a2a3a); border: 1px solid #2a4a6a; }
.header-box.vp         { background: linear-gradient(135deg, #160f23, #251535); border: 1px solid #4a2a6a; }
.header-box.vwap       { background: linear-gradient(135deg, #0f2318, #1a3528); border: 1px solid #2a6a4a; }
.header-box.composite  { background: linear-gradient(135deg, #1a1208, #2a1e08); border: 1px solid #6a4a1a; }
.header-box.ml         { background: linear-gradient(135deg, #1c0a20, #2e1236); border: 1px solid #7b1fa2; }
.header-box.scanner    { background: linear-gradient(135deg, #0f231f, #1a3a35); border: 1px solid #26a69a; }
.widget-panel-ai { background: #111922; border: 1px solid #2d3d4f; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
.factor-box { background: #111b26; border: 1px solid #1e3040; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
.factor-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; }
.hit { color: #26a69a; font-weight: 600; }
.miss { color: #ef5350; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# חלק 4: פונקציות תשתית לפענוח וטעינת מודלים (אוטומטי)
# ============================================================
def clean_and_unpack_archive(archive_dict):
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

# אתחול משתני זיכרון
for k, v in [("mode", "wyckoff"), ("ml_model", None), ("ml_metadata", None), ("use_ml", False), ("model_archive", {})]:
    if k not in st.session_state: 
        st.session_state[k] = v

# --- פתרון המחיקה: טעינה אוטומטית שקופה ברקע בעליית האפליקציה ---
if not st.session_state.model_archive:
    trigger_auto_load_from_file()


# ============================================================
# חלק 5: רכיב ממשק משותף - בורר ה-AI החכם
# ============================================================
def render_active_ai_selector_widget(screen_identifier):
    st.markdown("<div class='widget-panel-ai'>", unsafe_allow_html=True)
    st.markdown("### 🧠 הגדרות מנוע החלטה AI חכם")
    col_a, col_b, col_c = st.columns([2, 1.5, 1])
    
    with col_a:
        if st.session_state.model_archive:
            slots_list = list(st.session_state.model_archive.keys())
            selected_slot = st.selectbox("בחר מודל מוסדי פעיל:", slots_list, key=f"selector_slot_{screen_identifier}")
            if st.button("⚡ טען והפעל מודל נבחר", key=f"activate_btn_{screen_identifier}", use_container_width=True):
                target_data = st.session_state.model_archive[selected_slot]
                model_instance = target_data["model"]
                if isinstance(model_instance, bytes):
                    model_instance = pickle.loads(model_instance)
                st.session_state.ml_model = model_instance
                st.session_state.ml_metadata = target_data["metadata"]
                st.session_state.use_ml = True
                st.success(f"המודל '{selected_slot}' הופעל בהצלחה!")
                st.rerun()
        else:
            st.info("לא נמצאו מודלים טעונים בזיכרון. עבור ל-ML Trainer.")
            
    with col_b:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 סנכרון מהיר מגיטהאב", key=f"sync_git_{screen_identifier}", use_container_width=True):
            if trigger_auto_load_from_file():
                st.success("✅ סנכרון הקובץ מגיטהאב הושלם בהצלחה!")
                st.rerun()
            else:
                st.error("קובץ המאגר ריק או לא נמצא.")
                
    with col_c:
        st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
        ai_toggle = st.checkbox("הפעל שימוש ב-AI", value=st.session_state.use_ml, key=f"checkbox_ai_{screen_identifier}")
        if ai_toggle != st.session_state.use_ml:
            st.session_state.use_ml = ai_toggle
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# חלק 6: תפריט ניווט עליון
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
    acc = metadata.get("test_acc", metadata.get("train_acc", 0.0))
    train_ticker = metadata.get("train_ticker", "???")
    period = metadata.get("period", "???")
    slot = metadata.get("slot", "כללי")
    st.info(f"🤖 **מצב AI מופעל באופן גלובלי:** מודל פעיל - {slot} (אומן על {train_ticker}, {period}) | דיוק בדיקה (Test): {acc*100:.1f}%")


# ============================================================
# חלק 7: הגדרות מנוע הבק-טסט והפקטורים
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
            if isinstance(model, bytes): model = pickle.loads(model)
            try: probs = model.predict_proba(factors)[:, 1]
            except: probs = model.predict(factors)
            score = pd.Series(probs * 100, index=factors.index)
            if df is not None:
                sma50 = df["Close"].rolling(50).mean()
                boost = ((score > 10) & (df["Close"] > sma50)).astype(float) * 20
                score = (score + boost).clip(0, 100)
        else:
            w = {"f01_liquidity_gap": 3, "f02_volatility_squeeze": 4, "f03_regime": 5, "f04_absorption": 6, "f05_breakout_quality": 3, "f06_cis_weight": 2, "f07_obv_velocity": 5, "f08_fft": -2, "f09_dependency": 2, "f10_temporal_seq": 5, "f11_kill_switch": 0, "f12_distribution": -4, "f13_confidence_decay": 3, "f14_inst_intent": 6, "f15_mtf": 4, "f16_anchor_conflict": -2, "f17_vol_cluster": -1, "f18_sector_breadth": 3, "f19_order_flow": 4, "f20_liquidity_sweep": 3, "f21_break_auth": 2, "f22_sr_strength": 2, "f23_gap_structure": 2, "f24_event_shock": 2, "f25_rvol_anomaly": 2, "f26_accept_reject": 3, "f27_vol_regime": 3, "f28_inst_part": 3, "f29_trend_integrity": 3, "f30_mean_rev": 3, "f31_bear_trap": 2, "f32_accum_type": 2, "f33_liq_exhaust": -1, "f34_corr_stress": 1, "f35_struct_break": 2}
            tot = sum(abs(v) for v in w.values() if v != 0)
            score = pd.Series(0.0, index=factors.index)
            for col, weight in w.items():
                if col in factors.columns and col != "f11_kill_switch": score += factors[col].clip(-1, 1) * weight
            score = (score / tot * 100 + 50).clip(0, 100)
        if "f11_kill_switch" in factors.columns: score = score * (1 - factors["f11_kill_switch"])
        return score.round(1)

class SignalDebugger:
    LABELS = {"f01_liquidity_gap": "Liquidity Gap (LVN)", "f02_volatility_squeeze": "Volatility Squeeze", "f03_regime": "Market Regime", "f04_absorption": "Absorption Signature", "f05_breakout_quality": "Breakout Quality", "f06_cis_weight": "Dynamic Weights", "f07_obv_velocity": "OBV Accumulation Velocity", "f08_fft": "Failure to Follow Through", "f09_dependency": "Signal Dependency", "f10_temporal_seq": "Temporal Sequencing", "f12_distribution": "Distribution Mirror", "f13_confidence_decay": "Confidence Decay", "f14_inst_intent": "Institutional Intent", "f15_mtf": "MTF Confirmation", "f16_anchor_conflict": "Anchor Conflict", "f17_vol_cluster": "Vol Cluster Expansion", "f18_sector_breadth": "Sector Breadth", "f19_order_flow": "Order Flow Imbalance", "f20_liquidity_sweep": "Liquidity Sweep", "f21_break_auth": "Range Break Auth", "f22_sr_strength": "S/R Strength", "f23_gap_structure": "Gap Structure", "f24_event_shock": "Event Shock Normalization", "f25_rvol_anomaly": "Relative Volume Anomaly", "f26_accept_reject": "Price Accept vs Reject", "f27_vol_regime": "Vol Regime Transition", "f28_inst_part": "Institutional Participation", "f29_trend_integrity": "Trend Integrity", "f30_mean_rev": "Mean Reversion Pressure", "f31_bear_trap": "False Support Breakdown", "f32_accum_type": "Accumulation Differentiation", "f33_liq_exhaust": "Liquidity Exhaustion", "f34_corr_stress": "Correlation Stress", "f35_struct_break": "Structural Break"}
    def audit(self, factors: pd.DataFrame, cis: pd.Series) -> list:
        row = factors.iloc[-1]; res = []
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            model = st.session_state.ml_model
            if isinstance(model, bytes): model = pickle.loads(model)
            importances = model.feature_importances_
            for i, col in enumerate(factors.columns):
                if col in self.LABELS and importances[i] > 0.01:
                    direction = 1 if row[col] > 0 else -1
                    res.append({"factor": self.LABELS[col], "impact": importances[i] * direction * 100})
        else:
            for col, val in row.items():
                if col in self.LABELS and val != 0: res.append({"factor": self.LABELS[col], "impact": val})
        return sorted(res, key=lambda x: x["impact"], reverse=True)


# ============================================================
# חלק 8: מנוע סימולציית העבר (BACKTEST ENGINE)
# ============================================================
class BacktestEngine:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.factors = FactorEngine(self.cfg)
        self.debugger = SignalDebugger()
        
    def run(self, ticker: str):
        try:
            df = yf.Ticker(ticker).history(period=self.cfg.period)
            if df is None or len(df) < 50: return {"error": "נתוני שוק לא זמינים."}
            df.index = pd.to_datetime(df.index).tz_localize(None)
            try: df["spy_close"] = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)["Close"].reindex(df.index).ffill()
            except: df["spy_close"] = df["Close"]
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
                for r in trades_df["return"]: equity.append(equity[-1] * (1 + r))
            wr = (trades_df["return"] > 0).mean() if not trades_df.empty else 0
            ret = (equity[-1] - self.cfg.initial_capital) / self.cfg.initial_capital
            equity_arr = np.array(equity); peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak; max_dd = drawdown.min() if len(drawdown) > 0 else 0
            return {"df": df, "cis": cis, "audit": self.debugger.audit(f, cis), "trades": len(trades_df), "wr": wr, "ret": ret, "max_dd": max_dd}
        except Exception as e: return {"error": str(e)}


# ============================================================
# חלק 9: מודול WYCKOFF
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y"):
    try: df = yf.Ticker(ticker).history(period=period)
    except: return None
    if df is None or len(df) < 40: return None
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["SPREAD"] = df["High"] - df["Low"]
    df["VOL_YEAR_MEAN"] = df["Volume"].rolling(252, min_periods=20).mean()
    df["SPREAD_YEAR_MEAN"] = df["SPREAD"].rolling(252, min_periods=20).mean()
    return df

def render_gauge(score, verdict, verdict_color):
    bc = "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={'text':f"<b>Wyckoff Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':bc}, 'bgcolor':"#0d1b2a"}, number={'font':{'color':bc}}))
    fig.update_layout(height=300, margin=dict(t=80,b=10,l=20,r=20), paper_bgcolor="#0a1520", font_color="#e0eaf4")
    return fig

def analyze_wyckoff(df):
    score = 0; alerts = []; current_phase = "לא בתהליך איסוף"; phase_explanation = "אין סימני בלימה ב-90 ימים אחרונים."
    last_30 = df.iloc[-30:]
    for i in range(len(last_30)):
        day = last_30.iloc[i]; days_ago = len(last_30) - i - 1
        if day["Volume"] > day["VOL_YEAR_MEAN"] * 1.8:
            spread_ratio = day["SPREAD"] / day["SPREAD_YEAR_MEAN"]
            if spread_ratio < 1.2: alerts.append(f"⚠️ מחזור חריג לפני {days_ago} ימים ללא תוצאת מחיר תואמת (ספיגה/התנגדות).")
            elif spread_ratio >= 1.4: alerts.append(f"✅ מחזור חריג לפני {days_ago} ימים שהפך למהלך רחב. השתתפות מוסדית.")
    last_90 = df.iloc[-90:]
    sc_candidates = last_90[(last_90["Volume"] > last_90["VOL_YEAR_MEAN"] * 2.2) & (last_90["Close"] < last_90["Open"])]
    if not sc_candidates.empty:
        sc_idx = sc_candidates.index[0]; sc_low = df.loc[sc_idx, "Low"]; post_sc = df.loc[sc_idx:]; days_since_sc = len(post_sc)
        if days_since_sc < 7: current_phase = "Phase A (Stopping the Trend)"; phase_explanation = "בלימה מוסדית טרייה."; score += 35
        else:
            spring_candidates = post_sc[(post_sc["Low"] < sc_low) & (post_sc["Close"] > sc_low)]
            if not spring_candidates.empty and (len(post_sc) - post_sc.index.get_loc(spring_candidates.index[-1])) <= 15:
                current_phase = "Phase C (Spring / Shakeout)"; phase_explanation = "ניעור קצוות נזילות (Spring)."; score += 85
            else: current_phase = "Phase B (Building a Cause)"; phase_explanation = "בניית סיבה בדשדוש."; score += 60
    vd = current_phase; vc = "#26a69a" if score >= 75 else "#ffa726" if score >= 40 else "#ef5350"
    return score, current_phase, phase_explanation, alerts, vd, vc

def render_wyckoff_chart(df):
    dc = df.iloc[-120:].copy(); fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"], name="Price"), row=1, col=1)
    colors = ['#ef5350' if row['Open'] > row['Close'] else '#26a69a' for _, row in dc.iterrows()]
    fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], name="Volume", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=dc.index, y=dc["VOL_YEAR_MEAN"], mode='lines', name="1Y Avg Vol", line=dict(color='#ffa726', width=1.5, dash='dot')), row=2, col=1)
    fig.update_layout(height=450, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a", font_color="#e0eaf4", xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
    return fig

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF 3.0 STRUCTURAL ENGINE</h2><p>ניתוח מוסדי ע"פ VSA ושלבי איסוף.</p></div>""",unsafe_allow_html=True)
    render_active_ai_selector_widget("wyckoff_screen")
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול לניתוח:", "NVDA", key="wyckoff_ticker_input")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        btn = st.button("▶ הרץ ניתוח מוסדי", use_container_width=True, type="primary")
    if btn:
        with st.spinner("מנתח חתימות מחיר..."):
            df = get_data(ticker.upper())
            if df is not None:
                score, current_phase, phase_exp, alerts, vd, vc = analyze_wyckoff(df)
                col1, col2 = st.columns([1, 2])
                with col1: st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
                with col2:
                    st.markdown(f"### 📍 סטטוס: **{current_phase}**\n*{phase_exp}*\n---")
                    if alerts:
                        for alert in alerts:
                            color = "#ef5350" if "⚠️" in alert else "#26a69a"
                            st.markdown(f"<div style='background:#111b26; border-right:4px solid {color}; padding:10px; margin-bottom:10px; border-radius:5px;'>{alert}</div>", unsafe_allow_html=True)
                    else: st.info("לא נצפו חריגות נפח משמעותיות החודש.")
                st.plotly_chart(render_wyckoff_chart(df), use_container_width=True)
            else: st.error("לא נמצאו מספיק נתונים לסימול זה.")


# ============================================================
# חלק 10: שומרי מקום למסכים
# ============================================================
def screen_vp(): st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2><p>פעיל כחלק מפקטורי המערכת.</p></div>""",unsafe_allow_html=True)
def screen_vwap(): st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION</h2><p>פעיל כחלק מפקטורי המערכת.</p></div>""",unsafe_allow_html=True)
def screen_composite(): st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2><p>משוקלל בתוך ה-Backtest Engine.</p></div>""",unsafe_allow_html=True)

def screen_backtest():
    st.markdown("""<div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;"><h2>📈 BACKTEST ENGINE</h2><p>בחינה היסטורית מלאה לפי 35 פקטורים.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול לבדיקה:", "NVDA", key="bt_input_field")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ הרץ סימולציה", use_container_width=True, type="primary")
    if run_btn:
        with st.spinner(f"מעבד... {ticker.upper()}"):
            res = BacktestEngine().run(ticker.upper())
            if "error" in res: st.error(res['error'])
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("עסקאות", res["trades"]); col2.metric("Win Rate", f"{res['wr']*100:.1f}%"); col3.metric("תשואה", f"{res['ret']*100:.1f}%"); col4.metric("Max DD", f"{res['max_dd']*100:.1f}%")
                st.markdown("---"); st.markdown(f"### 📊 ציון CIS נוכחי: **{res['cis'].iloc[-1]:.1f} / 100**")
                audit = res["audit"]; positives = [x for x in audit if x['impact'] > 0]; negatives = [x for x in audit if x['impact'] < 0]
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.success("✅ **פקטורים חיוביים:**")
                    for p in positives[:5]: st.markdown(f"<div class='factor-box'><span class='hit'>+</span> <span class='factor-title'>{p['factor']}</span></div>", unsafe_allow_html=True)
                with pc2:
                    st.error("❌ **פקטורים שליליים:**")
                    for n in sorted(negatives, key=lambda x: x['impact'])[:5]: st.markdown(f"<div class='factor-box'><span class='miss'>-</span> <span class='factor-title'>{n['factor']}</span></div>", unsafe_allow_html=True)


# ============================================================
# חלק 11: עזר סיכום למודל
# ============================================================
def get_model_summary(model, metadata):
    if isinstance(model, bytes):
        try: model = pickle.loads(model)
        except Exception as e: return {"error_summary": f"שגיאת חילוץ: {str(e)}"}
    try: importances = model.feature_importances_
    except AttributeError: return {"train_ticker": metadata.get("train_ticker", "?"), "train_acc": metadata.get("train_acc", 0.0), "test_acc": metadata.get("test_acc", 0.0), "period": metadata.get("period", "?"), "slot": metadata.get("slot", "כללי"), "top_factors": [{"name": "לא מאומן", "importance": 0.0}]}
    top_factors = sorted(zip(SignalDebugger.LABELS.keys(), importances), key=lambda x: x[1], reverse=True)[:5]
    summary = {
        "train_ticker": metadata.get("train_ticker", "?"), "train_acc": metadata.get("train_acc", 0.0), "test_acc": metadata.get("test_acc", 0.0), "period": metadata.get("period", "?"), "slot": metadata.get("slot", "כללי"),
        "top_factors": [{"name": SignalDebugger.LABELS.get(f, f), "importance": imp} for f, imp in top_factors]
    }
    return summary


# ============================================================
# חלק 12: ML TRAINER
# ============================================================
def screen_ml_trainer():
    st.markdown("""<div class="header-box ml"><h2>🧠 MACHINE LEARNING TRAINER & ARCHIVE</h2><p>מנוע אימון עם פיצול סדרתי.</p></div>""",unsafe_allow_html=True)
    MODEL_SLOTS = ["Growth (צמיחה)", "Value/Index (ערך/מדדים)", "Commodities (סחורות)"]
    st.markdown("### 📥 ניהול המאגר")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 רענן מגיטהאב", use_container_width=True, type="primary"):
            if trigger_auto_load_from_file(): st.success("✅ נטען בהצלחה!")
            else: st.warning("הקובץ ריק או חסר.")
        with st.expander("הדבקה ידנית"):
            encoded_paste = st.text_area("קוד Base64:"); 
            if st.button("טען ידנית"):
                try: 
                    st.session_state.model_archive = clean_and_unpack_archive(pickle.loads(base64.b64decode(encoded_paste.strip().encode("utf-8"))))
                    st.success("✅ נטען!"); st.rerun()
                except: st.error("❌ שגיאה בטעינה.")
    with col2:
        if st.session_state.model_archive:
            available_slots = list(st.session_state.model_archive.keys())
            st.markdown(f"**📚 משבצות במערכת:** {len(available_slots)} / 3")
            selected_model = st.selectbox("בחר מודל להפעלה:", available_slots)
            if st.button("✅ הפעל מודל"):
                target_data = st.session_state.model_archive[selected_model]
                model_obj = target_data["model"]
                if isinstance(model_obj, bytes): model_obj = pickle.loads(model_obj)
                st.session_state.ml_model = model_obj; st.session_state.ml_metadata = target_data["metadata"]; st.session_state.use_ml = True
                st.success("✅ מופעל!"); st.rerun()

    st.markdown("---"); st.markdown("### 🚀 אימון מודל (Train/Test Split)")
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
    with c1: train_ticker = st.text_input("סימול לאימון:", "SPY", key="ml_train_ticker")
    with c2: target_slot = st.selectbox("משבצת אסטרטגית:", MODEL_SLOTS)
    with c3: start_date = st.date_input("מתאריך:", value=datetime(2022, 1, 1))
    with c4: end_date = st.date_input("עד תאריך:", value=datetime(2024, 12, 31))

    if st.button("🚀 התחל למידת מכונה", use_container_width=True, type="primary"):
        with st.spinner("מפצל נתונים ומאמן..."):
            try:
                df = yf.Ticker(train_ticker.upper()).history(start=start_date, end=end_date)
                if df is not None and len(df) >= 50:
                    engine = FactorEngine(BacktestConfig())
                    factors = engine.compute(df)
                    target = (df["Close"].shift(-10) / df["Close"] - 1 > 0.02).astype(int)
                    valid_idx = target.notna()
                    X = factors[valid_idx].copy(); y = target[valid_idx].values
                    
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_split=50, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test) if len(X_test) > 0 else 0.0
                    gap = train_acc - test_acc
                    
                    meta = {"train_ticker": train_ticker.upper(), "train_acc": train_acc, "test_acc": test_acc, "period": f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}", "slot": target_slot}
                    st.session_state.model_archive[target_slot] = {"model": model, "metadata": meta}
                    st.session_state.ml_model = model; st.session_state.ml_metadata = meta; st.session_state.use_ml = True
                    
                    col_tr, col_te, col_g = st.columns(3)
                    col_tr.metric("דיוק אימון (Train)", f"{train_acc*100:.1f}%"); col_te.metric("דיוק בדיקה (Test)", f"{test_acc*100:.1f}%"); col_g.metric("פער הכללה", f"{gap*100:.1f}%")
                    if gap > 0.10: st.error("⚠️ Overfitting: פער מעל 10%.")
                    else: st.success("✅ מודל יציב.")
                else: st.error("❌ לא נמצאו מספיק נתונים.")
            except Exception as e: st.error(f"שגיאה: {str(e)}")

    st.markdown("---"); st.markdown("### 📤 פעולות ייצוא ודוחות")
    ca, cb = st.columns(2)
    with ca:
        if st.session_state.model_archive:
            archive_export = {k: {"model": pickle.dumps(v["model"]), "metadata": v["metadata"]} for k, v in st.session_state.model_archive.items()}
            st.download_button(label="💾 הורד קובץ ארכיון מעודכן", data=base64.b64encode(pickle.dumps(archive_export)).decode("utf-8"), file_name="batch_archive_v1.txt", mime="text/plain", use_container_width=True, type="primary")
    with cb:
        if st.session_state.model_archive:
            if st.button("הפק אובייקט דוח מלא (JSON)"):
                report_data = []
                for name, data in st.session_state.model_archive.items():
                    summ = get_model_summary(data['model'], data['metadata'])
                    if "error_summary" not in summ:
                        gap = summ.get('train_acc', 0.0) - summ.get('test_acc', 0.0)
                        report_data.append({"Slot": name, "Asset": summ['train_ticker'], "Period": summ['period'], "Metrics": {"Train_Acc": round(summ.get('train_acc',0.0)*100,1), "Test_Acc": round(summ.get('test_acc',0.0)*100,1), "Gap": round(gap*100,1), "Status": "Overfit" if gap > 0.10 else "Stable"}, "Top_Factors": {f['name']: round(f['importance'], 3) for f in summ['top_factors']}})
                st.text_area("📋 JSON להעתקה:", value=json.dumps(report_data, indent=2, ensure_ascii=False), height=250)


# ============================================================
# חלק 13: MARKET SCANNER
# ============================================================
def screen_scanner():
    st.markdown("""<div class="header-box scanner"><h2>🔎 MARKET SCANNER</h2><p>סריקה סקטוריאלית חכמה.</p></div>""",unsafe_allow_html=True)
    render_active_ai_selector_widget("scanner_screen")
    col_x, col_y, col_z = st.columns([2, 1, 1])
    with col_x: chosen_universe = SECTOR_MAP[st.selectbox("🎯 בחר סקטור להתמקדות:", list(SECTOR_MAP.keys()), key="scanner_sector_selector")]
    with col_y: scan_limit = st.slider("מניות לסריקה:", min_value=5, max_value=len(chosen_universe), value=min(20, len(chosen_universe)), step=5)
    with col_z: engine_choice = st.radio("מנוע:", ["Wyckoff Structural Engine", "Composite CIS Score (ML/Static)"]); show_all = st.checkbox("הצג הכל", value=True)
    if st.button("🚀 התחל סריקה", use_container_width=True, type="primary"):
        results = []; progress_bar = st.progress(0); status_text = st.empty(); tickers = chosen_universe[:scan_limit]
        for i, ticker in enumerate(tickers):
            status_text.text(f"מנתח {ticker} ({i+1}/{len(tickers)})..."); df = get_data(ticker, period="1y")
            if df is not None and len(df) > 30:
                if "Wyckoff" in engine_choice:
                    score, _, _, _, vd, _ = analyze_wyckoff(df)
                    results.append({"Ticker": ticker, "Score": score, "Engine": "Wyckoff", "Verdict": vd})
                else:
                    engine = FactorEngine(BacktestConfig()); cis_score = engine.composite_cis(engine.compute(df), df).iloc[-1]
                    results.append({"Ticker": ticker, "Score": cis_score, "Engine": "CIS", "Verdict": "Strong Signal" if cis_score >= 75 else "Watch" if cis_score >= 60 else "Wait"})
            progress_bar.progress((i + 1) / len(tickers)); time.sleep(0.05)
        status_text.text("✅ הושלם!"); st.markdown("---")
        if results:
            df_results = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
            if not show_all: df_results = df_results[df_results["Score"] >= (40 if "Wyckoff" in engine_choice else 60)]
            if not df_results.empty: st.success(f"📊 נמצאו {len(df_results)} תוצאות:"); st.dataframe(df_results, use_container_width=True)
            else: st.warning("אין תוצאות שעברו את הרף.")
        else: st.error("הסריקה נכשלה.")


# ============================================================
# ניתוב
# ============================================================
routes = {"wyckoff": screen_wyckoff, "vp": screen_vp, "vwap": screen_vwap, "composite": screen_composite, "backtest": screen_backtest, "ml": screen_ml_trainer, "scanner": screen_scanner}
routes[st.session_state.mode]()
