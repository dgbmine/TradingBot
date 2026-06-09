import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional
import warnings

# ביטול אזהרות שמפריעות בתצוגה
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Institutional Scout Pro")

# ============================================================
# SCAN UNIVERSE
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
# CSS DESIGN
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
.header-box h2{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;margin-bottom:12px;direction:ltr;}
.header-box.wyckoff   h2{color:#4fc3f7;}
.header-box.vp        h2{color:#ce93d8;}
.header-box.vwap      h2{color:#4caf7d;}
.header-box.composite h2{color:#ffa726;}
.header-box p{color:#b0c8e0;font-size:0.92rem;margin:6px 0;}

.tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;padding:2px 8px;border-radius:4px;margin:3px 2px;}
.tag-w{background:#1e3a5f;border:1px solid #4fc3f7;color:#4fc3f7;}
.tag-v{background:#2a1a4a;border:1px solid #ab47bc;color:#ce93d8;}
.tag-vw{background:#0f2a1a;border:1px solid #4caf7d;color:#4caf7d;}
.tag-c{background:#2a1e08;border:1px solid #ffa726;color:#ffa726;}

.score-reason-box{background:#0d1b2a;border-left:4px solid #4fc3f7;border-radius:8px;
                  padding:18px 22px;margin:10px 0;direction:rtl;color:#cde3f5;font-size:0.88rem;line-height:1.8;}
.score-reason-box.positive   {border-left-color:#26a69a;}
.score-reason-box.negative   {border-left-color:#ef5350;}
.score-reason-box.vp-positive{background:#150d20;border-left-color:#ab47bc;}
.score-reason-box.vp-negative{background:#150d20;border-left-color:#ef5350;}
.score-reason-box.vw-positive{background:#0a1a10;border-left-color:#4caf7d;}
.score-reason-box.vw-negative{background:#0a1a10;border-left-color:#ef5350;}
.score-reason-box strong{color:#fff;}

.criteria-row{display:flex;justify-content:space-between;align-items:center;
              padding:6px 0;border-bottom:1px solid #1e3040;font-size:0.84rem;}
.hit {color:#26a69a;font-weight:600;}
.miss{color:#ef5350;}

.overview-card{background:#0d1b2a;border:1px solid #2a4a6a;border-radius:10px;
               padding:18px 20px;text-align:center;direction:ltr;}
.overview-card.vp-card  {border-color:#4a2a6a;background:#120d1e;}
.overview-card.vw-card  {border-color:#1a4a2a;background:#0a150e;}
.overview-card.comp-card{border-color:#4a3a0a;background:#150f02;}
.ticker-label{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;margin-bottom:4px;}
.score-big{font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:600;margin:6px 0;}
.verdict-label{font-size:0.78rem;color:#b0c8e0;margin-top:4px;}

.disclaimer{background:#1a1206;border:1px solid #5a4010;border-radius:8px;
            padding:10px 16px;color:#a08040;font-size:0.78rem;direction:rtl;margin-top:18px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & TOP NAV
# ============================================================
for k,v in [("mode","wyckoff"),("w_sub","specific"),("vp_sub","specific"),
            ("vw_sub","specific"),("comp_sub","specific"),("backtest_sub","specific")]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5 = st.columns(5)
nav = [("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
       ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
       ("backtest","📈  Backtest")]
cols = [c1,c2,c3,c4,c5]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.mode==mode_key else "secondary",
                     key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# ============================================================
# BACKTEST ENGINE (ALL CLASSES EMBEDDED)
# ============================================================
@dataclass
class BacktestConfig:
    commission: float = 0.001           # 0.1% per trade
    slippage: float = 0.0005            # 0.05% per trade
    initial_capital: float = 100_000.0
    position_size: float = 0.10         # 10% of capital per trade
    hold_days: int = 20                 # max hold period
    min_score: int = 65                 # CIS entry threshold
    exit_score: int = 35                # CIS exit threshold
    walk_forward_train:int = 252        # trading days
    walk_forward_test: int = 63         # ~3 months
    regime_ticker: str = "SPY"
    period: str = "5y"

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
        vol_by_bin = df.groupby(price_bins)["Volume"].transform("sum")
        bin_mean = df.groupby(price_bins)["Volume"].transform("mean")
        f["f01_liquidity_gap"] = ((vol_by_bin < bin_mean * 0.5).astype(float).rolling(5).mean())

        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        bb_width = (2 * std20) / sma20.replace(0, np.nan)
        atr = pd.concat([rng, (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        atr14 = atr.rolling(14).mean()
        f["f02_volatility_squeeze"] = ((bb_width < bb_width.rolling(20).mean() * 0.75) & (atr14 < atr14.rolling(20).mean() * 0.75)).astype(float)

        spy_close = df.get("spy_close", df["Close"])
        spy_ma50 = spy_close.rolling(50).mean()
        spy_slope = spy_ma50.diff(10) / spy_ma50.shift(10).replace(0, np.nan)
        f["f03_regime_risk_on"] = (spy_slope > 0.01).astype(float)
        f["f03_regime_neutral"] = ((spy_slope >= -0.01) & (spy_slope <= 0.01)).astype(float)
        f["f03_regime_risk_off"] = (spy_slope < -0.01).astype(float)

        weak_close = df["Close"] < (df["Low"] + rng * 0.35)
        strong_wick = lower_shadow > body * 1.5
        high_vol = rvol > 1.5
        f["f04_absorption"] = (weak_close & strong_wick & high_vol).astype(float)

        resist = df["High"].rolling(20).max().shift(1)
        breakout = df["Close"] > resist
        followthrough = df["Close"].rolling(3).mean() > resist.shift(1)
        f["f05_breakout_quality"] = (breakout & followthrough).astype(float)

        vol_regime = std20 / std20.rolling(60).mean().replace(0, np.nan)
        f["f06_cis_weight_adj"] = np.clip(1.0 / vol_regime.replace(0, np.nan), 0.5, 2.0)

        obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        obv_slope = obv.diff(10) / (obv.abs().rolling(10).mean().replace(0, np.nan))
        f["f07_obv_velocity"] = obv_slope.clip(-3, 3)

        up_day = df["Close"] > df["Close"].shift(1)
        next_dn = df["Close"].shift(-1) < df["Close"]
        f["f08_failure_follow_through"] = (up_day & next_dn & high_vol).astype(float)

        f["f09_dependency"] = (f["f04_absorption"].rolling(10).corr(f["f07_obv_velocity"])).clip(-1, 1)

        sc_occurred = f["f04_absorption"].rolling(30).max()
        ns_occurring = (rvol < 0.7).astype(float)
        f["f10_temporal_seq"] = (sc_occurred * ns_occurring)

        crash_day = (df["Close"].pct_change() < -0.05)
        vol_spike = rvol > 4.0
        f["f11_kill_switch"] = (crash_day | vol_spike).astype(float)

        new_high = df["High"] > df["High"].rolling(20).max().shift(1)
        weak_close2 = df["Close"] < df["High"] - rng * 0.7
        f["f12_distribution_mirror"] = (new_high & weak_close2).astype(float)

        last_signal_days = f["f04_absorption"].replace(0, np.nan).ffill().isna().astype(int)
        f["f13_confidence_decay"] = np.exp(-last_signal_days / 10.0).clip(0, 1)

        f["f14_inst_intent"] = (f["f04_absorption"] * 0.3 + f["f07_obv_velocity"].clip(0, 1) * 0.4 + f["f10_temporal_seq"] * 0.3).clip(0, 1)

        daily_trend = (df["Close"] > sma20).astype(float)
        weekly_close = df["Close"].rolling(5).mean()
        weekly_trend = (weekly_close > weekly_close.rolling(4).mean()).astype(float)
        f["f15_mtf_confirm"] = (daily_trend * weekly_trend)

        vwap_full = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        conflict = (df["Close"] > vwap_full).astype(float).rolling(3).sum() * (df["Close"] < vwap_full).astype(float).rolling(3).sum()
        f["f16_anchor_conflict"] = (conflict > 0).astype(float)

        f["f17_vol_cluster_expand"] = (atr14 > atr14.shift(5) * 1.3).astype(float)

        f["f18_sector_breadth"] = (df["Close"] > df["Close"].shift(1)).astype(float).rolling(10).mean()

        buy_pressure = (df["Close"] - df["Low"]) / rng.replace(0, np.nan)
        sell_pressure = (df["High"] - df["Close"]) / rng.replace(0, np.nan)
        f["f19_order_flow_imbalance"] = (buy_pressure - sell_pressure).rolling(5).mean()

        support = df["Low"].rolling(20).min().shift(1)
        f["f20_liquidity_sweep"] = ((df["Low"] < support) & (df["Close"] > support)).astype(float)

        range_20 = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
        f["f21_break_authenticity"] = ((df["Close"] - df["Close"].shift(1)).abs() / range_20.replace(0, np.nan)).clip(0, 1)

        touches_support = (df["Low"].rolling(5).min() <= df["Low"].rolling(20).min() * 1.005).astype(float)
        f["f22_sr_strength"] = touches_support.rolling(20).sum() / 20

        f["f23_gap_structure"] = (df["Open"] > df["Close"].shift(1) * 1.005).astype(float) - (df["Open"] < df["Close"].shift(1) * 0.995).astype(float)

        shock = (df["Close"].pct_change().abs() > 0.04).astype(float)
        f["f24_event_shock_norm"] = 1.0 - shock.rolling(3).sum().clip(0, 1)

        f["f25_rvol_anomaly"] = ((rvol - rvol.rolling(60).mean()) / rvol.rolling(60).std().replace(0, np.nan)).clip(-3, 3)

        acceptance = ((df["Close"] > (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float)
        rejection = ((df["Close"] < (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float)
        f["f26_accept_reject"] = acceptance.rolling(5).mean() - rejection.rolling(5).mean()

        atr_ratio = atr14 / atr14.rolling(60).mean().replace(0, np.nan)
        f["f27_vol_regime_transition"] = (atr_ratio < 0.8).astype(float) - (atr_ratio > 1.2).astype(float)

        large_body = body > body.rolling(20).mean() * 1.5
        f["f28_inst_participation"] = (large_body & high_vol).astype(float)

        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        f["f29_trend_integrity"] = ((df["Close"] > sma20).astype(int) + (sma20 > sma50).astype(int) + (sma50 > sma200).astype(int)) / 3

        f["f30_mean_reversion_pressure"] = (-((df["Close"] - sma20) / std20.replace(0, np.nan))).clip(-3, 3)

        below_support = (df["Close"] < df["Low"].rolling(20).min().shift(1))
        recovers = (df["Close"].shift(-2) > df["Low"].rolling(20).min().shift(3))
        f["f31_bear_trap"] = (below_support & recovers).astype(float)

        distance_from_ath = (df["Close"].rolling(252).max() - df["Close"]) / df["Close"].rolling(252).max().replace(0, np.nan)
        reaccum = (distance_from_ath < 0.15) & (distance_from_ath > 0.05)
        fresh_accum = distance_from_ath > 0.25
        f["f32_accum_type"] = fresh_accum.astype(float) * 1.0 + reaccum.astype(float) * 0.6

        vol_declining = (vol_ma5 < vol_ma5.shift(10))
        price_stalling = (df["Close"].pct_change(5).abs() < 0.02)
        f["f33_liquidity_exhaustion"] = (vol_declining & price_stalling).astype(float)

        f["f34_corr_stress"] = df["Close"].pct_change().rolling(20).corr(spy_close.pct_change()).clip(-1, 1)

        rolling_high = df["High"].rolling(20).max()
        rolling_low = df["Low"].rolling(20).min()
        f["f35_structural_break"] = (df["Close"] > rolling_high.shift(1)).astype(float) - (df["Close"] < rolling_low.shift(1)).astype(float)

        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame, weights: Optional[dict] = None) -> pd.Series:
        w = weights or {
            "f01_liquidity_gap": 3, "f02_volatility_squeeze": 4, "f03_regime_risk_on": 5,
            "f04_absorption": 6, "f05_breakout_quality": 3, "f06_cis_weight_adj": 2,
            "f07_obv_velocity": 5, "f08_failure_follow_through": -2, "f09_dependency": 2,
            "f10_temporal_seq": 5, "f11_kill_switch": 0, "f12_distribution_mirror": -4,
            "f13_confidence_decay": 3, "f14_inst_intent": 6, "f15_mtf_confirm": 4,
            "f16_anchor_conflict": -2, "f17_vol_cluster_expand": -1, "f18_sector_breadth": 3,
            "f19_order_flow_imbalance": 4, "f20_liquidity_sweep": 3, "f21_break_authenticity": 2,
            "f22_sr_strength": 2, "f23_gap_structure": 2, "f24_event_shock_norm": 2,
            "f25_rvol_anomaly": 2, "f26_accept_reject": 3, "f27_vol_regime_transition": 3,
            "f28_inst_participation": 3, "f29_trend_integrity": 3, "f30_mean_reversion_pressure": 3,
            "f31_bear_trap": 2, "f32_accum_type": 2, "f33_liquidity_exhaustion": -1,
            "f34_corr_stress": 1, "f35_structural_break": 2,
        }
        total_positive = sum(v for v in w.values() if v > 0)
        score = pd.Series(0.0, index=factors.index)
        for col, weight in w.items():
            if col in factors.columns and col != "f11_kill_switch":
                score += factors[col].clip(-1, 1) * weight
        score = (score / total_positive * 100).clip(0, 100)
        if "f11_kill_switch" in factors.columns:
            score = score * (1 - factors["f11_kill_switch"])
        return score.round(1)

class SignalDebugger:
    FACTOR_LABELS = {
        "f01_liquidity_gap": ("Liquidity Gap", "LVN/HVN — אזור ריק מסמן תנועה מהירה"),
        "f02_volatility_squeeze": ("Volatility Squeeze", "BB Width + ATR מכווצים — לפני פיצוץ"),
        "f03_regime_risk_on": ("Market Regime — Risk On", "SPY בעלייה — רוח גבית"),
        "f03_regime_neutral": ("Market Regime — Neutral", "SPY ללא מגמה ברורה"),
        "f03_regime_risk_off": ("Market Regime — Risk Off", "SPY יורד — סביבה עוינת"),
        "f04_absorption": ("Absorption Signature", "זנב + נר חלש + ווליום = ספיגה"),
        "f05_breakout_quality": ("Breakout Quality", "פריצה WITH המשך — לא פייק"),
        "f06_cis_weight_adj": ("CIS Weight Adjustment", "משקל דינמי לפי תנודתיות"),
        "f07_obv_velocity": ("OBV Velocity", "מהירות צבירה — OBV slope"),
        "f08_failure_follow_through": ("Failure to Follow Through", "⚠️ עלייה ללא המשך — אזהרה"),
        "f09_dependency": ("Signal Dependency", "קורלציה בין אבסורפשן ל-OBV"),
        "f10_temporal_seq": ("Temporal Sequencing", "SC לפני No-Supply — סדר נכון"),
        "f11_kill_switch": ("⛔ Kill Switch", "אירוע קיצון — הציון归零"),
        "f12_distribution_mirror": ("Distribution Mirror", "⚠️ Upthrust — חלוקה, לא איסוף"),
        "f13_confidence_decay": ("Confidence Decay", "ישנות האות — כמה ימים עברו"),
        "f14_inst_intent": ("Institutional Intent", "ציון מטא — כוונת מוסדיים"),
        "f15_mtf_confirm": ("Multi-Timeframe Confirm", "יומי ושבועי מסכימים"),
        "f16_anchor_conflict": ("Anchor Point Conflict", "⚠️ מחיר מתחת ו VWAP — קונפליקט"),
        "f17_vol_cluster_expand": ("Vol Cluster Expansion", "⚠️ ATR מתרחב — תנודתיות עולה"),
        "f18_sector_breadth": ("Sector Breadth", "רוחב שוק — כמה ניירות עולים"),
        "f19_order_flow_imbalance": ("Order Flow Imbalance", "לחץ קנייה vs מכירה"),
        "f20_liquidity_sweep": ("Liquidity Sweep", "Stop hunt + החלמה = Bull Trap פייל"),
        "f21_break_authenticity": ("Break Authenticity", "איכות הפריצה ביחס לטווח"),
        "f22_sr_strength": ("S/R Strength", "חוזק תמיכה/התנגדות לאורך זמן"),
        "f23_gap_structure": ("Gap Structure", "פערים — Breakaway vs Exhaustion"),
        "f24_event_shock_norm": ("Event Shock Normalization", "נרמול לאחר חדשות/רווחים"),
        "f25_rvol_anomaly": ("RVOL Anomaly", "ווליום חריג vs בייסליין"),
        "f26_accept_reject": ("Price Accept/Reject", "שוק מקבל או דוחה את המחיר"),
        "f27_vol_regime_transition": ("Vol Regime Transition", "מעבר מהתרחבות להתכווצות"),
        "f28_inst_participation": ("Institutional Participation","נרות גדולים + ווליום = מוסדיים"),
        "f29_trend_integrity": ("Trend Integrity", "איכות המגמה — SMA alignment"),
        "f30_mean_reversion_pressure": ("Mean Reversion Pressure", "כמה המחיר רחוק מהממוצע"),
        "f31_bear_trap": ("Bear Trap Detection", "שבירת תמיכה מזויפת"),
        "f32_accum_type": ("Accum vs Re-Accum", "איסוף ראשון או חוזר"),
        "f33_liquidity_exhaustion": ("Liquidity Exhaustion", "⚠️ ווליום יורד + מחיר תקוע"),
        "f34_corr_stress": ("Multi-Asset Corr Stress", "קורלציה עם SPY — סיכון מאקרו"),
        "f35_structural_break": ("Structural Break", "שבירת מבנה מחיר — Micro Regime"),
    }

    def audit(self, factors: pd.DataFrame, cis: pd.Series, date=None, weights: Optional[dict] = None) -> list:
        if date is None:
            row = factors.iloc[-1]
            cis_val = cis.iloc[-1]
        else:
            row = factors.loc[date]
            cis_val = cis.loc[date]

        w = weights or {
            "f01_liquidity_gap":3,"f02_volatility_squeeze":4,"f03_regime_risk_on":5,
            "f04_absorption":6,"f05_breakout_quality":3,"f06_cis_weight_adj":2,
            "f07_obv_velocity":5,"f08_failure_follow_through":-2,"f09_dependency":2,
            "f10_temporal_seq":5,"f11_kill_switch":0,"f12_distribution_mirror":-4,
            "f13_confidence_decay":3,"f14_inst_intent":6,"f15_mtf_confirm":4,
            "f16_anchor_conflict":-2,"f17_vol_cluster_expand":-1,"f18_sector_breadth":3,
            "f19_order_flow_imbalance":4,"f20_liquidity_sweep":3,"f21_break_authenticity":2,
            "f22_sr_strength":2,"f23_gap_structure":2,"f24_event_shock_norm":2,
            "f25_rvol_anomaly":2,"f26_accept_reject":3,"f27_vol_regime_transition":3,
            "f28_inst_participation":3,"f29_trend_integrity":3,
            "f30_mean_reversion_pressure":3,"f31_bear_trap":2,"f32_accum_type":2,
            "f33_liquidity_exhaustion":-1,"f34_corr_stress":1,"f35_structural_break":2,
        }
        
        total_positive = sum(v for v in w.values() if v > 0)
        result = []
        for col, weight in w.items():
            if col not in row.index: continue
            raw_val = float(row[col])
            clipped = float(np.clip(raw_val, -1, 1))
            points = clipped * weight
            pct_contrib = (points / total_positive * 100) if total_positive else 0
            label, desc = self.FACTOR_LABELS.get(col, (col, ""))
            result.append({
                "factor_id": col, "label": label, "description": desc,
                "raw_value": round(raw_val, 4), "weight": weight,
                "points": round(points, 3), "pct_of_score": round(pct_contrib, 2),
                "status": "✅ תורם" if points > 0.01 else "❌ גורע" if points < -0.01 else "➖ ניטרלי",
            })
        result.sort(key=lambda x: abs(x["points"]), reverse=True)
        return result, round(float(cis_val), 1)

class BacktestEngine:
    def __init__(self, cfg: BacktestConfig = None):
        self.cfg = cfg or BacktestConfig()
        self.factors = FactorEngine(self.cfg)
        self.debugger = SignalDebugger()

    def _fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(period=self.cfg.period)
            if df is None or len(df) < 200: return None
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception: return None

    def _inject_spy(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            spy = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)
            spy.index = pd.to_datetime(spy.index).tz_localize(None)
            spy_close = spy["Close"].reindex(df.index).ffill()
            df["spy_close"] = spy_close
        except Exception: df["spy_close"] = df["Close"]
        return df

    def _generate_signals(self, df: pd.DataFrame) -> tuple:
        df = self._inject_spy(df)
        f = self.factors.compute(df)
        cis = self.factors.composite_cis(f)
        entry = (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score)
        exit_ = (cis < self.cfg.exit_score)
        return df, f, cis, entry, exit_

    def _simulate_trades(self, df: pd.DataFrame, entry: pd.Series, exit_: pd.Series, cis: pd.Series) -> pd.DataFrame:
        cost = self.cfg.commission + self.cfg.slippage
        capital = self.cfg.initial_capital
        pos_size = self.cfg.position_size
        trades = []; in_trade = False; entry_price = 0.0; entry_date = None; hold_count = 0; entry_cis = 0.0
        closes = df["Close"].values; dates = df.index
        entry_arr = entry.values; exit_arr = exit_.values; cis_arr = cis.values
        
        for i in range(1, len(closes)):
            if not in_trade:
                if entry_arr[i]:
                    entry_price = closes[i] * (1 + cost)
                    entry_date = dates[i]; entry_cis = cis_arr[i]; in_trade = True; hold_count = 0
            else:
                hold_count += 1
                forced_exit = (hold_count >= self.cfg.hold_days)
                signal_exit = exit_arr[i]
                if forced_exit or signal_exit:
                    exit_price = closes[i] * (1 - cost)
                    ret = (exit_price - entry_price) / entry_price
                    pnl = capital * pos_size * ret
                    capital += pnl
                    trades.append({
                        "entry_date": entry_date, "exit_date": dates[i], "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4), "return": round(ret, 6), "pnl": round(pnl, 2),
                        "hold_days": hold_count, "exit_reason": "max_hold" if forced_exit else "signal",
                        "entry_cis": round(entry_cis, 1), "capital": round(capital, 2),
                    })
                    in_trade = False
        return pd.DataFrame(trades)

    def _equity_curve(self, df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
        equity = pd.Series(self.cfg.initial_capital, index=df.index)
        if trades.empty: return equity
        for _, t in trades.iterrows():
            mask = equity.index >= t["exit_date"]
            equity[mask] = t["capital"]
        return equity.ffill()

    def _metrics(self, trades: pd.DataFrame, equity: pd.Series, regime_series: pd.Series = None) -> dict:
        if trades.empty: return {"error": "No trades generated"}
        rets = trades["return"]
        wins = rets[rets > 0]; loss = rets[rets <= 0]
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)).round(3) if rets.std() > 0 else 0
        downside = rets[rets < 0].std()
        sortino = (rets.mean() / downside * np.sqrt(252)).round(3) if downside > 0 else 0
        equity_arr = equity.values
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_dd = drawdown.min()
        win_rate = len(wins) / len(rets)
        avg_win = wins.mean() if len(wins) else 0; avg_loss = loss.mean() if len(loss) else 0
        profit_factor = (wins.sum() / abs(loss.sum())) if loss.sum() != 0 else np.inf
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        total_return = (equity.iloc[-1] - self.cfg.initial_capital) / self.cfg.initial_capital
        base = {
            "total_trades": len(trades), "win_rate": round(win_rate, 4), "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": round(max_dd, 4), "profit_factor": round(profit_factor, 3),
            "expectancy": round(expectancy, 6), "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
            "total_return": round(total_return, 4), "final_capital": round(equity.iloc[-1], 2),
        }
        return base

    def run(self, ticker: str) -> dict:
        df = self._fetch(ticker)
        if df is None: return {"error": f"Cannot fetch data for {ticker}"}
        df, f, cis, entry, exit_ = self._generate_signals(df)
        trades = self._simulate_trades(df, entry, exit_, cis)
        equity = self._equity_curve(df, trades)
        metrics = self._metrics(trades, equity)
        return {"ticker": ticker, "df": df, "factors": f, "cis": cis, "trades": trades, "equity": equity, "metrics": metrics}

    def audit_date(self, ticker: str, date=None) -> tuple:
        df = self._fetch(ticker)
        if df is None: return [], 0.0
        df, f, cis, _, _ = self._generate_signals(df)
        return self.debugger.audit(f, cis, date=date)

# ============================================================
# DATA INTAKE & SHARED APP LOGIC
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
    except Exception: return None
    if df is None or len(df) < 100: return None
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean()
    df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["RANGE"] = df["High"] - df["Low"]
    return df

def gauge_color(score, mode):
    if mode=="wyckoff":   return "#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    if mode=="vp":        return "#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    if mode=="vwap":      return "#4caf7d" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return "#ffa726" if score>=75 else "#ff7043" if score>=45 else "#ef5350"

def render_gauge(score, verdict, verdict_color, mode="wyckoff"):
    bc = gauge_color(score, mode)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text':f"<b>Institutional Score</b><br><span style='font-size:0.82em;color:{verdict_color}'>{verdict}</span>",'font':{'size':13}},
        gauge={'axis':{'range':[0,100],'tickwidth':1,'tickcolor':"#4a6a8a"},
               'bar':{'color':bc,'thickness':0.3},
               'bgcolor':"#0d1b2a",'borderwidth':1,'bordercolor':"#2a4a6a",
               'threshold':{'line':{'color':"#ffffff",'width':2},'thickness':0.75,'value':score}},
        number={'font':{'size':48,'color':bc},'suffix':'/100'}
    ))
    fig.update_layout(height=300,margin=dict(t=80,b=10,l=20,r=20),paper_bgcolor="#0a1520",font_color="#e0eaf4")
    return fig

# ============================================================
# WYCKOFF METHOD
# ============================================================
def analyze_wyckoff(df):
    score=0; criteria=[]
    high_3m=df["Close"].iloc[-65:].max(); cur=df["Close"].iloc[-1]
    dd=(high_3m-cur)/high_3m; prereq=dd>=0.12
    sc_win=df.iloc[-30:]
    sc_c=sc_win[(sc_win["Volume"]>=sc_win["VOL_MEAN"]*2.0)&(sc_win["LOWER_SHADOW"]>sc_win["BODY"]*1.2)]
    sc_found=len(sc_c)>0; sc_pts=25 if (sc_found and prereq) else 0; score+=sc_pts
    criteria.append({"name":"Selling Climax (SC)","hit":sc_found and prereq,"points":25,"earned":sc_pts,"explanation":""})
    vd="סבירות גבוהה לאיסוף מוסדי" if score>=75 else "סימנים חלקיים" if score>=45 else "אין ראיות לאיסוף"
    vc2="#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score,criteria,vd,vc2,prereq,dd

def render_wyckoff_chart(df):
    dc=df.iloc[-65:].copy()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],name="Price"),row=1,col=1)
    fig.update_layout(height=420,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",xaxis_rangeslider_visible=False)
    return fig

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF ACCUMULATION</h2></div>""",unsafe_allow_html=True)
    raw,run=_ticker_input("w")
    if run: _run_specific(raw,"wyckoff",analyze_wyckoff,_wrap_w,"w_sub")

# ============================================================
# VOLUME PROFILE METHOD
# ============================================================
def build_volume_profile(df, bins=40):
    mn=df["Low"].min(); mx=df["High"].max()
    edges=np.linspace(mn,mx,bins+1); vap=np.zeros(bins)
    for _,row in df.iterrows():
        lo,hi,vol=row["Low"],row["High"],row["Volume"]
        if hi==lo: continue
        for i in range(bins):
            ol=max(edges[i],lo); oh=min(edges[i+1],hi)
            if oh>ol: vap[i]+=vol*(oh-ol)/(hi-lo)
    return (edges[:-1]+edges[1:])/2,vap,edges

def analyze_vp(df):
    score=0; criteria=[]; cur=df["Close"].iloc[-1]
    mids,vap,edges=build_volume_profile(df); total=vap.sum()
    poc_idx=np.argmax(vap); poc=mids[poc_idx]
    bp=25 if cur<poc else 0; score+=bp
    criteria.append({"name":"מחיר מתחת ל-VAL","hit":cur<poc,"points":25,"earned":bp,"explanation":""})
    vd="סבירות גבוהה לנוכחות מוסדית" if score>=75 else "סימנים חלקיים"
    vc="#ab47bc" if score>=75 else "#ffa726"
    vpd={"poc":poc,"vah":poc,"val":poc,"midpoints":mids,"vol_at_price":vap,"poc_vol_pct":0}
    return score,criteria,vd,vc,vpd

def render_vp_chart(df,vpd,ticker):
    fig=go.Figure()
    fig.update_layout(height=400,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a")
    return fig

def screen_vp():
    st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2></div>""",unsafe_allow_html=True)
    raw,run=_ticker_input("vp")
    if run: _run_specific(raw,"vp",analyze_vp,_wrap_vp,"vp_sub")

# ============================================================
# VWAP METHOD
# ============================================================
def compute_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    dev = df["Close"] - vwap
    return vwap, dev, dev.rolling(20).std()

def analyze_vwap(df):
    score=0; criteria=[]
    vwap, dev, rolling_std = compute_vwap(df)
    score += 25 if dev.iloc[-1] < -rolling_std.iloc[-1] else 0
    criteria.append({"name":"סטיית מחיר מה-VWAP","hit":score>0,"points":25,"earned":score,"explanation":""})
    return score,criteria,"","",{"vwap":vwap,"dev":dev,"rolling_std":rolling_std}

def render_vwap_chart(df, vwap_data, ticker):
    fig=go.Figure()
    fig.update_layout(height=400,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a")
    return fig

def screen_vwap():
    st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION</h2></div>""",unsafe_allow_html=True)
    raw,run=_ticker_input("vw")
    if run: _run_specific(raw,"vwap",analyze_vwap,_wrap_vw,"vw_sub")

# ============================================================
# COMPOSITE SCORE
# ============================================================
def analyze_composite(df):
    w_score = analyze_wyckoff(df)[0]; v_score = analyze_vp(df)[0]; vw_score = analyze_vwap(df)[0]
    composite = int(round(w_score * 0.35 + v_score * 0.35 + vw_score * 0.30))
    return composite,"#26a69a","Strong","","", [],{}

def screen_composite():
    st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2></div>""",unsafe_allow_html=True)
    raw,run=_ticker_input("comp")
    if run: _run_specific(raw,"composite",analyze_composite,_wrap_comp,"comp_sub")

# ============================================================
# MARKET SCAN ENGINE & HELPERS
# ============================================================
def _render_criteria(criteria,box_pos,box_neg):
    for c in criteria:
        box=box_pos if c["hit"] else box_neg; lbl="✅ הצליח" if c["hit"] else "❌ נכשל"; cls="hit" if c["hit"] else "miss"
        st.markdown(f"""<div class="score-reason-box {box}"><div class="criteria-row"><strong>{c['name']}</strong><span><span class="{cls}">{lbl}</span> | <strong>{c['earned']}/{c['points']}</strong></span></div></div>""",unsafe_allow_html=True)

def _render_w_detail(t,df,score,criteria,verdict,vcolor,prereq,dd):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"wyckoff"),use_container_width=True); _render_criteria(criteria,"positive","negative")
def _render_vp_detail(t,df,score,criteria,verdict,vcolor,vpd):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"vp"),use_container_width=True); _render_criteria(criteria,"vp-positive","vp-negative")
def _render_vw_detail(t,df,score,criteria,verdict,vcolor,vwap_data):
    st.plotly_chart(render_gauge(score,verdict,vcolor,"vwap"),use_container_width=True); _render_criteria(criteria,"vw-positive","vw-negative")
def _render_composite_detail(t,df,composite,vcolor,verdict,signal_class,action,breakdown,detail):
    st.plotly_chart(render_gauge(composite,verdict,vcolor,"composite"),use_container_width=True)

def _ticker_input(key_prefix):
    ci,cb=st.columns([4,1])
    with ci: raw=st.text_input("טיקרים","NVDA",key=f"{key_prefix}_input")
    with cb: st.markdown("<div style='margin-top:28px'></div>",unsafe_allow_html=True); run=st.button("▶ הרץ",key=f"{key_prefix}_run")
    return raw,run

def _run_specific(raw, mode, analyze_fn, render_fn, sub_state_key):
    tickers=[t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]
    for t in tickers:
        df=get_data(t)
        if df is not None: render_fn(t,df,*analyze_fn(df)) if mode != "composite" else render_fn(t,df,analyze_fn(df)[0],analyze_fn(df)[1],analyze_fn(df)[2],analyze_fn(df)[3],analyze_fn(df)[4],analyze_fn(df)[5],analyze_fn(df)[6])

def _wrap_w(t,df,*res):    _render_w_detail(t,df,*res)
def _wrap_vp(t,df,*res):   _render_vp_detail(t,df,*res)
def _wrap_vw(t,df,*res):   _render_vw_detail(t,df,*res)
def _wrap_comp(t,df,*res): _render_composite_detail(t,df,*res)

# ============================================================
# NEW BACKTEST SCREEN
# ============================================================
def screen_backtest():
    st.markdown("""
    <div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;">
      <h2>📈 BACKTEST ENGINE</h2>
      <p>הרצת סימולציות על נתוני עבר באמצעות מנוע פקטורי האיסוף המוסדי (35 Factors).</p>
    </div>""",unsafe_allow_html=True)
    
    c1, c2 = st.columns([4, 1])
    with c1:
        ticker = st.text_input("סימול מניה (לדוגמה: NVDA, SPY, TSLA)", "NVDA", key="bt_ticker")
    with c2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ הרץ סימולציה", use_container_width=True, type="primary")
        
    if run_btn:
        ticker = ticker.upper().strip()
        with st.spinner(f"מריץ סימולציית הון על {ticker}..."):
            try:
                # ה-Engine עכשיו נמצא באותו קובץ, לכן אין צורך לקרוא לו מתוך מודול חיצוני
                engine = BacktestEngine()
                res = engine.run(ticker)
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    m = res["metrics"]
                    st.markdown("### 📊 תוצאות סימולציה")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("סה״כ עסקאות", m.get("total_trades", 0))
                    col2.metric("Win Rate", f"{m.get('win_rate', 0)*100:.1f}%")
                    col3.metric("תשואה כוללת", f"{m.get('total_return', 0)*100:.1f}%")
                    col4.metric("דרודאון מקסימלי", f"{m.get('max_drawdown', 0)*100:.1f}%")
                    col5.metric("Sharpe Ratio", m.get("sharpe", 0))
                    
                    st.markdown("### 📈 עקומת הון (Equity Curve)")
                    equity = res["equity"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, fill='tozeroy', mode='lines', line=dict(color='#4fc3f7', width=2), name="Capital"))
                    fig.update_layout(height=400, paper_bgcolor="#0a1520", plot_bgcolor="#0d1b2a", font_color="#e0eaf4", margin=dict(t=20, b=20, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 🔍 הצצה למנוע (Under the Hood)")
                    audit_data, final_score = engine.audit_date(ticker)
                    st.info(f"🏆 ציון מוסדי אחרון: {final_score}/100")
                    audit_df = pd.DataFrame(audit_data)
                    if not audit_df.empty:
                        audit_display = audit_df[["status", "label", "points", "pct_of_score"]].copy()
                        audit_display.columns = ["סטטוס", "פקטור", "נקודות", "אחוז מהציון"]
                        st.dataframe(audit_display, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"⚠️ אירעה שגיאה: {e}")

# ============================================================
# ROUTER DISPATCH
# ============================================================
routes = {
    "wyckoff": screen_wyckoff,
    "vp": screen_vp,
    "vwap": screen_vwap,
    "composite": screen_composite,
    "backtest": screen_backtest
}
routes[st.session_state.mode]()
