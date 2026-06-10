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
from typing import Optional
import warnings
import pickle
import base64
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import time

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Institutional Scout Pro")


# ============================================================
# חלק 2: רשימת המניות לסריקה (SCAN UNIVERSE)
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
    "DIS","CMCSA","NFLX","RBLX","U","TTWO","EA",
    "DAL","UAL","AAL","LUV","FDX","UPS","XPO","ODFL",
    "DKNG","MGM","CZR","RCL","CCL","MAR","HLT",
]))


# ============================================================
# חלק 3: עיצוב CSS (מראה הממשק)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Hebrew:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans Hebrew',sans-serif;direction:rtl;}
h1,h2,h3,h4{font-family:'IBM Plex Mono',monospace;direction:ltr;}
.header-box{border-radius:12px;padding:24px 32px;margin-bottom:28px;color:#e0eaf4;direction:rtl;line-height:1.9;}
.header-box.wyckoff{background:linear-gradient(135deg,#0f1923,#1a2a3a);border:1px solid #2a4a6a;}
.header-box.vp     {background:linear-gradient(135deg,#160f23,#251535);border:1px solid #4a2a6a;}
.header-box.vwap   {background:linear-gradient(135deg,#0f2318,#1a3528);border:1px solid #2a6a4a;}
.header-box.composite{background:linear-gradient(135deg,#1a1208,#2a1e08);border:1px solid #6a4a1a;}
.header-box.ml     {background:linear-gradient(135deg,#1c0a20,#2e1236);border:1px solid #7b1fa2;}
.header-box.scanner{background:linear-gradient(135deg,#0f231f,#1a3a35);border:1px solid #26a69a;}
.header-box h2{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;margin-bottom:12px;direction:ltr;}
.header-box.wyckoff   h2{color:#4fc3f7;}
.header-box.vp        h2{color:#ce93d8;}
.header-box.vwap      h2{color:#4caf7d;}
.header-box.composite h2{color:#ffa726;}
.header-box.ml        h2{color:#e1bee7;}
.header-box.scanner   h2{color:#80cbc4;}
.header-box p{color:#b0c8e0;font-size:0.92rem;margin:6px 0;}
.score-reason-box{background:#0d1b2a;border-left:4px solid #4fc3f7;border-radius:8px;padding:18px 22px;margin:10px 0;direction:rtl;color:#cde3f5;font-size:0.88rem;line-height:1.8;}
.score-reason-box.positive{border-left-color:#26a69a;}
.score-reason-box.negative{border-left-color:#ef5350;}
.criteria-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #1e3040;font-size:0.84rem;}
.hit {color:#26a69a;font-weight:600;}
.miss{color:#ef5350;}
.factor-box{background:#111b26; border:1px solid #1e3040; border-radius:8px; padding:12px; margin-bottom:10px;}
.factor-title{font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:600;}
.model-stats{background:#0d1b2a; border:1px solid #2a4a6a; border-radius:8px; padding:16px; margin:12px 0;}
.model-stats.success{border-left:4px solid #26a69a;}
.model-stats.warning{border-left:4px solid #ffa726;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# חלק 4: ניהול משתני זיכרון (SESSION STATE)
# ============================================================
for k,v in [("mode","wyckoff"), ("ml_model", None), ("ml_metadata", None), ("use_ml", False)]:
    if k not in st.session_state: st.session_state[k] = v


# ============================================================
# חלק 5: תפריט ניווט עליון
# ============================================================
st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
nav = [("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
       ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
       ("backtest","📈  Backtest"), ("ml","🧠  ML Trainer"), ("scanner","🔎  Scanner")]
cols = [c1,c2,c3,c4,c5,c6,c7]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.mode==mode_key else "secondary", key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# חיווי להפעלת מודל ה-ML
if st.session_state.use_ml and st.session_state.ml_model is not None:
    metadata = st.session_state.ml_metadata or {}
    acc = metadata.get("test_acc", 0.0)
    train_ticker = metadata.get("train_ticker", "???")
    st.info(f"🤖 **מצב AI מופעל:** מודל מאומן על {train_ticker} (Test Accuracy: {acc*100:.1f}%)")


# ============================================================
# חלק 6: הגדרות מנוע הבק-טסט (BacktestConfig)
# ============================================================
@dataclass
class BacktestConfig:
    commission: float = 0.001
    slippage: float = 0.0005
    initial_capital: float = 100_000.0
    position_size: float = 0.10
    hold_days: int = 20
    min_score: int = 65
    exit_score: int = 35
    period: str = "2y"
    regime_ticker: str = "SPY"


# ============================================================
# חלק 7: פקטורים - אתחול והכנת נתונים בסיסיים
# ============================================================
class FactorEngine:
    def __init__(self, cfg: BacktestConfig): self.cfg = cfg
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        body = (df["Close"] - df["Open"]).abs()
        rng = df["High"] - df["Low"]
        lower_shadow = df[["Open","Close"]].min(axis=1) - df["Low"]
        vol_ma20 = df["Volume"].rolling(20).mean()
        vol_ma5 = df["Volume"].rolling(5).mean()
        rvol = df["Volume"] / vol_ma20.replace(0, np.nan)


# ============================================================
# חלק 8: חישוב פקטורים 1 עד 12
# ============================================================
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


# ============================================================
# חלק 9: חישוב פקטורים 13 עד 24
# ============================================================
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


# ============================================================
# חלק 10: חישוב פקטורים 25 עד 35 והחזרת המטריצה
# ============================================================
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
        f["f35_struct_break"] = (df["Close"] > df["High"].rolling(20).max().shift(1)).astype(float) - (df["Close"] < df["Low"].rolling(20).min().shift(1)).astype(float)
        return f.fillna(0)


# ============================================================
# חלק 11: חישוב הציון המוסדי (CIS) - מתוקן
# ============================================================
    def composite_cis(self, factors: pd.DataFrame) -> pd.Series:
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            model = st.session_state.ml_model
            # ניסיון חילוץ הסתברויות, אם לא מצליח - שימוש בתוצאה בינארית
            try:
                probs = model.predict_proba(factors)[:, 1]
            except:
                probs = model.predict(factors)
            score = pd.Series(probs * 100, index=factors.index)
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

        if "f11_kill_switch" in factors.columns: score = score * (1 - factors["f11_kill_switch"])
        return score.round(1)
 ============================================================
# חלק 12: פונקציית הדיבאג והוצאת הפקטורים הדומיננטיים
# ============================================================
class SignalDebugger:
    LABELS = {
        "f01_liquidity_gap": "Liquidity Gap (LVN Proxy)", "f02_volatility_squeeze": "Volatility Squeeze", "f03_regime": "Market Regime",
        "f04_absorption": "Absorption Signature", "f05_breakout_quality": "Breakout Quality", "f06_cis_weight": "Dynamic Weights",
        "f07_obv_velocity": "OBV Accumulation Velocity", "f08_fft": "Failure to Follow Through", "f09_dependency": "Signal Dependency",
        "f10_temporal_seq": "Temporal Sequencing", "f12_distribution": "Distribution Mirror", "f13_confidence_decay": "Confidence Decay",
        "f14_inst_intent": "Institutional Intent", "f15_mtf": "MTF Confirmation", "f16_anchor_conflict": "Anchor Point Conflict",
        "f17_vol_cluster": "Vol Cluster Expansion", "f18_sector_breadth": "Sector Breadth", "f19_order_flow": "Order Flow Imbalance",
        "f20_liquidity_sweep": "Liquidity Sweep (Stop Hunt)", "f21_break_auth": "Range Break Authenticity", "f22_sr_strength": "S/R Strength",
        "f23_gap_structure": "Gap Structure", "f24_event_shock": "Event Shock Normalization", "f25_rvol_anomaly": "Relative Volume Anomaly",
        "f26_accept_reject": "Price Accept vs Reject", "f27_vol_regime": "Vol Regime Transition", "f28_inst_part": "Institutional Participation",
        "f29_trend_integrity": "Trend Integrity", "f30_mean_rev": "Mean Reversion Pressure", "f31_bear_trap": "False Support Breakdown",
        "f32_accum_type": "Accumulation Differentiation", "f33_liq_exhaust": "Liquidity Exhaustion", "f34_corr_stress": "Correlation Stress",
        "f35_struct_break": "Structural Break"
    }
    def audit(self, factors: pd.DataFrame, cis: pd.Series) -> list:
        row = factors.iloc[-1]
        res = []
        if st.session_state.use_ml and st.session_state.ml_model is not None:
            importances = st.session_state.ml_model.feature_importances_
            for i, col in enumerate(factors.columns):
                if col in self.LABELS and importances[i] > 0.01:
                    direction = 1 if row[col] > 0 else -1
                    res.append({"factor": self.LABELS[col], "impact": importances[i] * direction * 100})
        else:
            for col, val in row.items():
                if col in self.LABELS and val != 0: res.append({"factor": self.LABELS[col], "impact": val})
        return sorted(res, key=lambda x: x["impact"], reverse=True)


# ============================================================
# חלק 13: מנוע הבק-טסט - אתחול ומשיכת נתונים
# ============================================================
class BacktestEngine:
    def __init__(self):
        self.cfg = BacktestConfig()
        self.factors = FactorEngine(self.cfg)
        self.debugger = SignalDebugger()

    def run(self, ticker: str):
        try:
            df = yf.Ticker(ticker).history(period=self.cfg.period)
            if df is None or len(df) < 100: return {"error": "Data missing"}
            df.index = pd.to_datetime(df.index).tz_localize(None)
            try: df["spy_close"] = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)["Close"].reindex(df.index).ffill()
            except: df["spy_close"] = df["Close"]

            f = self.factors.compute(df)
            cis = self.factors.composite_cis(f)


# ============================================================
# חלק 14: מנוע הבק-טסט - סימולציית מסחר ותוצאות
# ============================================================
            entry = (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score)
            exit_ = (cis < self.cfg.exit_score)
            
            closes = df["Close"].values; dates = df.index; trades = []
            in_trade = False; entry_px = 0; hold = 0
            
            for i in range(1, len(closes)):
                if not in_trade and entry.iloc[i]:
                    entry_px = closes[i] * (1 + self.cfg.commission)
                    in_trade = True; hold = 0; ent_d = dates[i]
                elif in_trade:
                    hold += 1
                    if exit_.iloc[i] or hold >= self.cfg.hold_days:
                        ext_px = closes[i] * (1 - self.cfg.commission)
                        trades.append({"entry_date": ent_d, "exit_date": dates[i], "return": (ext_px - entry_px)/entry_px})
                        in_trade = False
            
            trades_df = pd.DataFrame(trades)
            equity = [self.cfg.initial_capital]
            if not trades_df.empty:
                for r in trades_df["return"]: equity.append(equity[-1] * (1 + r))
            
            wr = (trades_df["return"] > 0).mean() if not trades_df.empty else 0
            ret = (equity[-1] - self.cfg.initial_capital) / self.cfg.initial_capital
            
            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0
            
            return {"df": df, "cis": cis, "audit": self.debugger.audit(f, cis), "trades": len(trades_df), "wr": wr, "ret": ret, "max_dd": max_dd}
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# חלק 15: עזרי ממשק ותצוגה משותפים
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y"):
    try: df = yf.Ticker(ticker).history(period=period)
    except: return None
    if df is None or len(df) < 100: return None
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["RANGE"] = df["High"] - df["Low"]
    return df

def render_gauge(score, verdict, verdict_color):
    bc = "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        title={'text':f"<b>Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>"},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':bc}, 'bgcolor':"#0d1b2a"}, number={'font':{'color':bc}}))
    fig.update_layout(height=300, margin=dict(t=80,b=10,l=20,r=20), paper_bgcolor="#0a1520", font_color="#e0eaf4")
    return fig

def _render_criteria(criteria):
    for c in criteria:
        box_class = "positive" if c["hit"] else "negative"
        lbl = "✅ הצליח" if c["hit"] else "❌ נכשל"
        cls = "hit" if c["hit"] else "miss"
        st.markdown(f"""
        <div class="score-reason-box {box_class}">
            <div class="criteria-row">
                <strong>{c['name']}</strong>
                <span><span class="{cls}">{lbl}</span> | <strong>{c['earned']}/{c['points']}</strong></span>
            </div>
            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 5px;">{c['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# חלק 16: מודול 1 - מנוע Wyckoff מתקדם ומבני (Structural Engine)
# ============================================================
def analyze_wyckoff(df):
    score = 0; criteria = []
    
    high_3m = df["Close"].iloc[-65:].max()
    cur = df["Close"].iloc[-1]
    dd = (high_3m - cur) / high_3m
    prereq = dd >= 0.12 

    sc_win = df.iloc[-30:]
    sc_c = sc_win[(sc_win["Volume"] >= sc_win["VOL_MEAN"] * 2.0) & (sc_win["LOWER_SHADOW"] > sc_win["BODY"] * 1.5)]
    sc_found = len(sc_c) > 0
    pts1 = 20 if (sc_found and prereq) else 0; score += pts1
    criteria.append({"name":"Phase A: Selling Climax (SC)","hit":sc_found and prereq,"points":20,"earned":pts1,"explanation":"זיהוי מאמץ מוסדי מסיבי בבלימת הירידה (ווליום חריג עליון + זנב קונים)."})

    ar_st_found = False
    if sc_found:
        sc_idx = sc_c.index[-1]
        post_sc = df.loc[sc_idx:].iloc[1:15]
        if len(post_sc) >= 3:
            ar_rally = post_sc["High"].max() > df.loc[sc_idx, "High"] * 1.02
            st_test = post_sc["Low"].min() >= df.loc[sc_idx, "Low"] * 0.98 
            st_vol = post_sc["Volume"].mean() < sc_c["Volume"].iloc[-1] 
            ar_st_found = ar_rally and st_test and st_vol
    pts2 = 20 if ar_st_found else 0; score += pts2
    criteria.append({"name":"Phase B: AR & Secondary Test","hit":ar_st_found,"points":20,"earned":pts2,"explanation":"תגובה אוטומטית למעלה ובדיקה חוזרת של אזור השפל בווליום נמוך (היצע דליל)."})

    low_20 = df["Low"].rolling(20).min().iloc[-2] 
    is_spring = (df["Low"].iloc[-1] < low_20) and (df["Close"].iloc[-1] > low_20)
    pts3 = 30 if is_spring else 0; score += pts3
    criteria.append({"name":"Phase C: Spring / Shakeout","hit":is_spring,"points":30,"earned":pts3,"explanation":"ניקוי סטופים אגרסיבי: שבירה של שפל התמיכה המדומה וחזרה מיידית למעלה."})

    recent_vol_stability = df["Volume"].iloc[-10:].std() / df["Volume"].iloc[-10:].mean()
    is_absorbed = recent_vol_stability < 0.45 and (df["Close"].iloc[-1] > df["Close"].rolling(20).mean().iloc[-1])
    pts4 = 30 if is_absorbed else 0; score += pts4
    criteria.append({"name":"Phase C/D: Supply Absorption","hit":is_absorbed,"points":30,"earned":pts4,"explanation":"התייבשות ווליום המוכרים וסגירה מעל ממוצע 20 - המוסדיים ספגו את כל ההיצע."})

    vd = "איסוף מוסדי מובהק (Wyckoff)" if score >= 70 else "סימני התבססות חלקיים" if score >= 40 else "אין תבנית Wyckoff חוקית"
    vc = "#26a69a" if score >= 70 else "#ffa726" if score >= 40 else "#ef5350"
    return score, criteria, vd, vc

def render_wyckoff_chart(df):
    dc = df.iloc[-90:].copy() 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], name="Volume", marker_color="#4fc3f7"), row=2, col=1)
    fig.update_layout(height=450, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a", font_color="#e0eaf4", xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10))
    return fig

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF 2.0 - STRUCTURAL ENGINE</h2><p>מנוע Wyckoff מבני מתקדם לזיהוי שלבי איסוף: Selling Climax, Secondary Test, Spring, ו-Supply Absorption.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (Wyckoff)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("▶ נתח שלבי Wyckoff", use_container_width=True):
            df = get_data(ticker.upper())
            if df is not None:
                score, criteria, vd, vc = analyze_wyckoff(df)
                col1, col2 = st.columns([1, 2])
                with col1: st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
                with col2: _render_criteria(criteria)
                st.plotly_chart(render_wyckoff_chart(df), use_container_width=True)


# ============================================================
# חלק 17: מודולים 2, 3, 4 - מקומות שומרים למסכי הניתוח
# ============================================================
def screen_vp(): st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2><p>פעיל במנוע הראשי של הבק-טסט.</p></div>""",unsafe_allow_html=True)
def screen_vwap(): st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION</h2><p>פעיל במנוע הראשי של הבק-טסט.</p></div>""",unsafe_allow_html=True)
def screen_composite(): st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2><p>שקלול מלא פועל כעת ב-Backtest Engine.</p></div>""",unsafe_allow_html=True)


# ============================================================
# חלק 18: מודול 5 - מסך הבק-טסט והתוצאות
# ============================================================
def screen_backtest():
    st.markdown("""
    <div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;">
      <h2>📈 BACKTEST ENGINE (Powered by 35 Factors)</h2>
      <p>המנוע מנתח את נתוני העבר של המניה על בסיס 35 פקטורים מוסדיים שרצים מתחת למכסה המנוע.</p>
    </div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה לבק-טסט:", "NVDA", key="bt_input")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ הרץ מנוע 35 פקטורים", use_container_width=True, type="primary")
        
    if run_btn:
        with st.spinner(f"מעבד נתונים על {ticker}..."):
            engine = BacktestEngine()
            res = engine.run(ticker.upper())
            if "error" in res:
                st.error(f"⚠️ שגיאה: {res['error']}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("סה״כ עסקאות", res["trades"])
                col2.metric("Win Rate", f"{res['wr']*100:.1f}%")
                col3.metric("תשואה כוללת", f"{res['ret']*100:.1f}%")
                col4.metric("דרודאון מקסימלי", f"{res['max_dd']*100:.1f}%")
                st.markdown("---")
                st.markdown(f"**ציון CIS (לפי {'מודל ML' if st.session_state.use_ml else 'משקלים קבועים'}): {res['cis'].iloc[-1]:.1f}/100**")
                
                audit = res["audit"]
                positives = [x for x in audit if x['impact'] > 0]
                negatives = [x for x in audit if x['impact'] < 0]
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.success("✅ **דחפו למעלה:**")
                    for p in positives[:4]: st.markdown(f"<div class='factor-box'><span class='hit'>+</span> <span class='factor-title'>{p['factor']}</span></div>", unsafe_allow_html=True)
                with pc2:
                    st.error("❌ **הורידו ציון:**")
                    for n in sorted(negatives, key=lambda x: x['impact'])[:4]: st.markdown(f"<div class='factor-box'><span class='miss'>-</span> <span class='factor-title'>{n['factor']}</span></div>", unsafe_allow_html=True)


# ============================================================
# חלק 19: פונקציות ML - Export ו-Load של המודל
# ============================================================
def export_model_to_base64(model, metadata):
    model_data = {"model": pickle.dumps(model), "metadata": metadata, "timestamp": datetime.now().isoformat()}
    serialized = pickle.dumps(model_data)
    encoded = base64.b64encode(serialized).decode("utf-8")
    return encoded

def load_model_from_base64(encoded_str):
    try:
        decoded = base64.b64decode(encoded_str.encode("utf-8"))
        model_data = pickle.loads(decoded)
        return pickle.loads(model_data["model"]), model_data["metadata"]
    except Exception as e:
        st.error(f"❌ שגיאה בטעינת המודל: {e}")
        return None, None

def get_model_summary(model, metadata):
    importances = model.feature_importances_
    top_factors = sorted(zip(SignalDebugger.LABELS.keys(), importances), key=lambda x: x[1], reverse=True)[:5]
    summary = {
        "train_ticker": metadata.get("train_ticker", "???"),
        "train_acc": metadata.get("train_acc", 0.0),
        "test_acc": metadata.get("test_acc", 0.0),
        "overfit_gap": metadata.get("train_acc", 0.0) - metadata.get("test_acc", 0.0),
        "timestamp": metadata.get("timestamp", "???"),
        "n_trees": model.n_estimators,
        "max_depth": model.max_depth,
        "top_factors": [{"name": SignalDebugger.LABELS.get(f, f), "importance": imp} for f, imp in top_factors]
    }
    return summary


# ============================================================
# חלק 20: מודול 6 - מסך אימון למידת המכונה (ML Trainer)
# ============================================================
def screen_ml_trainer():
    st.markdown("""
    <div class="header-box ml">
      <h2>🧠 MACHINE LEARNING TRAINER</h2>
      <p>אימון מודל RandomForest לניבוי תנועות שוק ויצירת משקלים דינמיים.</p>
    </div>""",unsafe_allow_html=True)
    
    st.markdown("### 📥 טעינת מודל קיים")
    col1, col2 = st.columns(2)
    with col1:
        encoded_model = st.text_area("הדבק קוד מודל מקודד (Base64):", height=100, key="model_paste")
        if st.button("⬆️ טעין מודל מקופסה", use_container_width=True):
            if encoded_model.strip():
                model, metadata = load_model_from_base64(encoded_model.strip())
                if model is not None:
                    st.session_state.ml_model = model; st.session_state.ml_metadata = metadata; st.session_state.use_ml = True
                    st.success("✅ המודל טעון בהצלחה!"); st.rerun()
    with col2:
        if st.session_state.ml_model is not None:
            summary = get_model_summary(st.session_state.ml_model, st.session_state.ml_metadata or {})
            st.markdown(f"**📊 מודל פעיל כעת:**\n- טיקר הדרכה: {summary['train_ticker']}\n- Test Accuracy: {summary['test_acc']*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### 🚀 אימון מודל חדש")
    c1, c2 = st.columns([3, 1])
    with c1: train_ticker = st.text_input("הזן סימול לאימון המודל (לדוגמה: SPY, QQQ, NVDA):", "SPY")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        train_btn = st.button("🚀 התחל אימון", use_container_width=True, type="primary")

    if train_btn:
        with st.spinner(f"שואב 5 שנות היסטוריה על {train_ticker}..."):
            df = get_data(train_ticker.upper(), period="5y")
            if df is not None and len(df) >= 500:
                try: df["spy_close"] = yf.Ticker("SPY").history(period="5y")["Close"].reindex(df.index).ffill()
                except: df["spy_close"] = df["Close"]
                
                engine = FactorEngine(BacktestConfig())
                factors = engine.compute(df)
                
                spy_return = df["spy_close"].shift(-10) / df["spy_close"] - 1
                stock_return = df["Close"].shift(-10) / df["Close"] - 1
                alpha = stock_return - spy_return
                valid_idx = alpha.notna()
                X = factors[valid_idx].copy()
                y = (alpha[valid_idx] > 0.02).astype(int).values 
                
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                with st.spinner("מאמן מודל RandomForest עם regularization..."):
                    model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_split=50, min_samples_leaf=25, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                    train_acc, test_acc = model.score(X_train, y_train), model.score(X_test, y_test)
                    st.session_state.ml_model = model
                    st.session_state.ml_metadata = {"train_ticker": train_ticker.upper(), "train_acc": train_acc, "test_acc": test_acc, "timestamp": datetime.now().isoformat()}
                    st.session_state.use_ml = True
                    st.success("✅ אימון הסתיים בהצלחה!")
            else: st.error("❌ לא נמצאו מספיק נתונים לאימון מודל.")

    st.markdown("---")
    if st.session_state.ml_model is not None:
        summary = get_model_summary(st.session_state.ml_model, st.session_state.ml_metadata or {})
        col1, col2 = st.columns(2)
        with col1:
            use_ml_toggle = st.toggle("✅ השתמש במודל זה לקביעת הציונים", value=st.session_state.use_ml)
            if use_ml_toggle != st.session_state.use_ml:
                st.session_state.use_ml = use_ml_toggle; st.rerun()
        with col2:
            encoded = export_model_to_base64(st.session_state.ml_model, summary)
            st.text_area("✂️ ייצוא קוד (העתק):", value=encoded, height=100, disabled=True)
            if st.button("🗑️ מחק מודל מהזיכרון", use_container_width=True, type="secondary"):
                st.session_state.ml_model = None; st.session_state.ml_metadata = None; st.session_state.use_ml = False; st.rerun()


# ============================================================
# חלק 21: מודול 7 - סורק שוק חכם (Market Scanner)
# ============================================================
def screen_scanner():
    st.markdown("""
    <div class="header-box scanner">
      <h2>🔎 MARKET SCANNER</h2>
      <p>סורק אוטומטית את רשימת המניות. הציונים מוצגים מהגבוה לנמוך כדי לזהות מי מתקרבת לשלב האיסוף.</p>
    </div>""",unsafe_allow_html=True)
    
    st.markdown("### הגדרות סריקה")
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        engine_choice = st.radio("בחר מנוע סריקה:", ["Wyckoff Structural Engine", "Composite CIS Score (ML/Static)"])
    with col2:
        scan_limit = st.slider("מספר מניות לסריקה (מהרשימה):", min_value=5, max_value=len(SCAN_UNIVERSE), value=20, step=5)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_all = st.checkbox("הצג את כל התוצאות (כולל חלשות)", value=True)
    
    if st.button("🚀 התחל סריקת שוק", use_container_width=True, type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers_to_scan = SCAN_UNIVERSE[:scan_limit]
        
        for i, ticker in enumerate(tickers_to_scan):
            status_text.text(f"מנתח את {ticker} ({i+1}/{scan_limit})...")
            df = get_data(ticker, period="1y")
            
            if df is not None and len(df) > 50:
                if "Wyckoff" in engine_choice:
                    score, criteria, vd, vc = analyze_wyckoff(df)
                    results.append({"Ticker": ticker, "Score": score, "Engine": "Wyckoff", "Verdict": vd})
                else:
                    engine = FactorEngine(BacktestConfig())
                    factors = engine.compute(df)
                    cis_score = engine.composite_cis(factors).iloc[-1]
                    verdict = "Strong Signal" if cis_score >= 75 else "Watch" if cis_score >= 60 else "Wait"
                    results.append({"Ticker": ticker, "Score": cis_score, "Engine": "CIS", "Verdict": verdict})
            
            progress_bar.progress((i + 1) / scan_limit)
            time.sleep(0.1) 
            
        status_text.text("✅ הסריקה הושלמה!")
        st.markdown("---")
        
        if results:
            df_results = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
            threshold = 40 if "Wyckoff" in engine_choice else 60
            
            if not show_all:
                df_results = df_results[df_results["Score"] >= threshold]
            
            if not df_results.empty:
                st.success(f"📊 מציג {len(df_results)} תוצאות (ציונים ירוקים מעידים על קרבה לאיסוף מוסדי):")
                st.dataframe(
                    df_results.style.background_gradient(cmap="Greens", subset=["Score"]),
                    use_container_width=True
                )
            else:
                st.warning(f"לא נמצאו מניות שעברו את הרף הקריטי ({threshold}). סמן 'הצג הכל' כדי לראות את הציונים הנמוכים.")
        else:
            st.error("הסריקה נכשלה, לא התקבלו נתונים מהשרת.")


# ============================================================
# חלק 22: ניתוב הראוטר (הרצת העמוד שנבחר)
# ============================================================
routes = {
    "wyckoff": screen_wyckoff, "vp": screen_vp, "vwap": screen_vwap,
    "composite": screen_composite, "backtest": screen_backtest, 
    "ml": screen_ml_trainer, "scanner": screen_scanner
}
routes[st.session_state.mode]()
