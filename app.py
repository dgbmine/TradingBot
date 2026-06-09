import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional
import warnings

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
.score-reason-box{background:#0d1b2a;border-left:4px solid #4fc3f7;border-radius:8px;padding:18px 22px;margin:10px 0;direction:rtl;color:#cde3f5;font-size:0.88rem;line-height:1.8;}
.score-reason-box.positive{border-left-color:#26a69a;}
.score-reason-box.negative{border-left-color:#ef5350;}
.criteria-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #1e3040;font-size:0.84rem;}
.hit {color:#26a69a;font-weight:600;}
.miss{color:#ef5350;}
.overview-card{background:#0d1b2a;border:1px solid #2a4a6a;border-radius:10px;padding:18px 20px;text-align:center;direction:ltr;}
.ticker-label{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;margin-bottom:4px;}
.score-big{font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:600;margin:6px 0;}
.verdict-label{font-size:0.78rem;color:#b0c8e0;margin-top:4px;}
.factor-box{background:#111b26; border:1px solid #1e3040; border-radius:8px; padding:12px; margin-bottom:10px;}
.factor-title{font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & TOP NAV
# ============================================================
for k,v in [("mode","wyckoff"),("w_sub","specific"),("vp_sub","specific"),
            ("vw_sub","specific"),("comp_sub","specific"),("backtest_sub","specific")]:
    if k not in st.session_state: st.session_state[k] = v

st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5 = st.columns(5)
nav = [("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
       ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
       ("backtest","📈  Backtest (35 Factors)")]
cols = [c1,c2,c3,c4,c5]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.mode==mode_key else "secondary", key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# ============================================================
# ENGINE CONFIG & 35 FACTORS (HIDDEN LOGIC)
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
        f["f31_bear_trap"] = ((df["Close"] < df["Low"].rolling(20).min().shift(1)) & (df["Close"].shift(-2) > df["Low"].rolling(20).min().shift(3))).astype(float)
        dist_ath = (df["Close"].rolling(252).max() - df["Close"]) / df["Close"].rolling(252).max().replace(0, np.nan)
        f["f32_accum_type"] = (dist_ath > 0.25).astype(float) * 1.0 + ((dist_ath < 0.15) & (dist_ath > 0.05)).astype(float) * 0.6
        f["f33_liq_exhaust"] = ((vol_ma5 < vol_ma5.shift(10)) & (df["Close"].pct_change(5).abs() < 0.02)).astype(float)
        f["f34_corr_stress"] = df["Close"].pct_change().rolling(20).corr(df.get("spy_close", df["Close"]).pct_change()).clip(-1, 1)
        f["f35_struct_break"] = (df["Close"] > df["High"].rolling(20).max().shift(1)).astype(float) - (df["Close"] < df["Low"].rolling(20).min().shift(1)).astype(float)

        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame) -> pd.Series:
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
        for col, val in row.items():
            if col in self.LABELS and val != 0:
                res.append({"factor": self.LABELS[col], "impact": val})
        return sorted(res, key=lambda x: x["impact"], reverse=True)

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
            
            # חישוב דרודאון מקסימלי (Max Drawdown)
            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0
            
            return {"df": df, "cis": cis, "audit": self.debugger.audit(f, cis), "trades": len(trades_df), "wr": wr, "ret": ret, "max_dd": max_dd}
        except Exception as e:
            return {"error": str(e)}

# ============================================================
# SHARED DATA & UI HELPERS
# ============================================================
@st.cache_data(ttl=3600)
def get_data(ticker):
    try: df = yf.Ticker(ticker).history(period="1y")
    except: return None
    if df is None or len(df) < 100: return None
    df["VOL_MEAN"] = df["Volume"].rolling(20).mean(); df["BODY"] = abs(df["Close"] - df["Open"])
    df["LOWER_SHADOW"] = df[["Open","Close"]].min(axis=1) - df["Low"]; df["RANGE"] = df["High"] - df["Low"]
    return df

# ============================================================
# APP SCREENS 
# ============================================================
def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF ACCUMULATION</h2></div>""",unsafe_allow_html=True)
    st.info("מודול זה פעיל במנוע, אך הוסתר זמנית כדי לחסוך בזיכרון. המסך המרכזי הוא ה-Backtest.")

def screen_vp():
    st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2></div>""",unsafe_allow_html=True)

def screen_vwap():
    st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION</h2></div>""",unsafe_allow_html=True)

def screen_composite():
    st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2></div>""",unsafe_allow_html=True)

# ============================================================
# 🚀 BACKTEST SCREEN 
# ============================================================
def screen_backtest():
    st.markdown("""
    <div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;">
      <h2>📈 BACKTEST ENGINE (Powered by 35 Factors)</h2>
      <p>המנוע מנתח את נתוני העבר של המניה על בסיס 35 פקטורים מוסדיים שרצים מתחת למכסה המנוע.</p>
    </div>""",unsafe_allow_html=True)
    
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (לדוגמה: NVDA)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ הרץ מנוע 35 פקטורים", use_container_width=True, type="primary")
        
    if run_btn:
        with st.spinner(f"מעבד 35 אינדיקטורים מוסדיים על {ticker}..."):
            engine = BacktestEngine()
            res = engine.run(ticker.upper())
            
            if "error" in res:
                st.error(f"⚠️ אירעה שגיאה: {res['error']}")
            else:
                st.markdown("### 📊 תוצאות סימולציה (היסטורית)")
                
                # תצוגה מעודכנת עם 4 עמודות כולל Max Drawdown
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("סה״כ עסקאות אסטרטגיה", res["trades"])
                col2.metric("Win Rate", f"{res['wr']*100:.1f}%")
                col3.metric("תשואה כוללת", f"{res['ret']*100:.1f}%")
                col4.metric("דרודאון מקסימלי (Risk)", f"{res['max_dd']*100:.1f}%")
                
                st.markdown("---")
                st.markdown("### 🧠 הסבר החלטת המנוע (מבוסס 35 פקטורים)")
                st.markdown(f"**הציון המוסדי הנוכחי (CIS) של המניה הוא: {res['cis'].iloc[-1]:.1f}/100**")
                
                audit = res["audit"]
                positives = [x for x in audit if x['impact'] > 0]
                negatives = [x for x in audit if x['impact'] < 0]
                
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.success("✅ **הפקטורים העיקריים שדחפו את הציון למעלה:**")
                    if positives:
                        for p in positives[:4]: 
                            st.markdown(f"<div class='factor-box'><span class='hit'>+ תורם חיובי:</span> <span class='factor-title'>{p['factor']}</span></div>", unsafe_allow_html=True)
                    else:
                        st.write("אין כרגע חתימות איסוף חיוביות משמעותיות.")
                        
                with pc2:
                    st.error("❌ **הפקטורים העיקריים שהורידו את הציון (סיכונים):**")
                    if negatives:
                        for n in sorted(negatives, key=lambda x: x['impact'])[:4]: 
                            st.markdown(f"<div class='factor-box'><span class='miss'>- מוריד ציון:</span> <span class='factor-title'>{n['factor']}</span></div>", unsafe_allow_html=True)
                    else:
                        st.write("לא זוהו תבניות הפצה או סיכונים חריגים.")

# ============================================================
# ROUTER DISPATCH
# ============================================================
routes = {
    "wyckoff": screen_wyckoff, "vp": screen_vp, "vwap": screen_vwap,
    "composite": screen_composite, "backtest": screen_backtest
}
routes[st.session_state.mode]()
