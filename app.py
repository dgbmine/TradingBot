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
# CSS DESIGN & STYLING
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
.factor-box{background:#111b26; border:1px solid #1e3040; border-radius:8px; padding:12px; margin-bottom:10px;}
.factor-title{font-family:'IBM Plex Mono',monospace; font-size:0.9rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & TOP NAV
# ============================================================
for k,v in [("mode","wyckoff")]:
    if k not in st.session_state: st.session_state[k] = v

st.markdown("# INSTITUTIONAL SCOUT PRO")
c1,c2,c3,c4,c5 = st.columns(5)
nav = [("wyckoff","⬛  Wyckoff"),("vp","🔮  Volume Profile"),
       ("vwap","📐  VWAP Deviation"),("composite","🏆  Composite Score"),
       ("backtest","📈  Backtest Engine")]
cols = [c1,c2,c3,c4,c5]
for col,(mode_key,label) in zip(cols,nav):
    with col:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.mode==mode_key else "secondary", key=f"nav_{mode_key}"):
            st.session_state.mode = mode_key; st.rerun()
st.markdown("---")

# ============================================================
# ENGINE CONFIG & 35 FACTORS (BACKTESTING)
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
            if col in self.LABELS and val != 0: res.append({"factor": self.LABELS[col], "impact": val})
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
# MODULE 1: WYCKOFF
# ============================================================
def analyze_wyckoff(df):
    score=0; criteria=[]
    high_3m=df["Close"].iloc[-65:].max(); cur=df["Close"].iloc[-1]
    dd=(high_3m-cur)/high_3m; prereq=dd>=0.12

    sc_win=df.iloc[-30:]
    sc_c=sc_win[(sc_win["Volume"]>=sc_win["VOL_MEAN"]*2.0)&(sc_win["LOWER_SHADOW"]>sc_win["BODY"]*1.2)]
    sc_found=len(sc_c)>0; sc_pts=25 if (sc_found and prereq) else 0; score+=sc_pts
    criteria.append({"name":"Selling Climax (SC)","hit":sc_found and prereq,"points":25,"earned":sc_pts,
                     "explanation":"נרמול הירידות באמצעות זיהוי זנב קונים וווליום חריג בשפל."})

    ar_found=False; ar_pts=0
    if sc_found:
        post=df.loc[sc_c.index[-1]:].iloc[1:11]
        if len(post)>=2:
            rally=(post["Close"].max()-df.loc[sc_c.index[-1],"Close"])/df.loc[sc_c.index[-1],"Close"]
            ar_found=rally>=0.04; ar_pts=20 if ar_found else 0; score+=ar_pts
    criteria.append({"name":"Automatic Rally (AR)","hit":ar_found,"points":20,"earned":ar_pts,
                     "explanation":"עלייה אוטומטית עוקבת שמעידה על בלימת המומנטום השלילי."})

    ns=df.iloc[-10:]["Volume"].mean()<df["VOL_MEAN"].iloc[-1]*0.7; ns_pts=20 if ns else 0; score+=ns_pts
    criteria.append({"name":"No Supply","hit":ns,"points":20,"earned":ns_pts,
                     "explanation":"ירידה משמעותית בווליום (היצע דליל) המאפשרת למחיר לעלות."})

    l20=df.iloc[-20:]; pc=(l20["Close"].iloc[-1]-l20["Close"].iloc[0])/l20["Close"].iloc[0]
    vc=(l20["Volume"].iloc[-5:].mean()-l20["Volume"].iloc[:5].mean())/l20["Volume"].iloc[:5].mean()
    div=(pc<0)and(vc<-0.25); div_pts=20 if div else 0; score+=div_pts
    criteria.append({"name":"Price/Vol Divergence","hit":div,"points":20,"earned":div_pts,
                     "explanation":"סתירה בין כיוון המחיר להיקף המסחר — לחץ המוכרים קורס."})

    tr=(df.iloc[-15:]["High"].max()-df.iloc[-15:]["Low"].min())/df.iloc[-15:]["Low"].min()
    inr=tr<0.12; tr_pts=15 if inr else 0; score+=tr_pts
    criteria.append({"name":"Trading Range","hit":inr,"points":15,"earned":tr_pts,
                     "explanation":"טווח מסחר צר (התבססות) — חתימה קלאסית של איסוף מוקדם."})

    vd="סבירות גבוהה לאיסוף מוסדי" if score>=75 else "סימנים חלקיים לאיסוף" if score>=45 else "אין ראיות לאיסוף"
    vc="#26a69a" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score, criteria, vd, vc

def render_wyckoff_chart(df):
    dc=df.iloc[-65:].copy()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=dc.index,open=dc["Open"],high=dc["High"],low=dc["Low"],close=dc["Close"],name="Price"),row=1,col=1)
    fig.add_trace(go.Bar(x=dc.index,y=dc["Volume"],name="Volume"),row=2,col=1)
    fig.update_layout(height=420,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a",font_color="#e0eaf4",xaxis_rangeslider_visible=False)
    return fig

def screen_wyckoff():
    st.markdown("""<div class="header-box wyckoff"><h2>⬛ WYCKOFF ACCUMULATION</h2><p>זיהוי טביעת האצבע של 'הכסף החכם' במבנה המחיר והווליום.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (Wyckoff)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ נתח Wyckoff", use_container_width=True)
    if run_btn:
        df = get_data(ticker.upper())
        if df is not None:
            score, criteria, vd, vc = analyze_wyckoff(df)
            col1, col2 = st.columns([1, 2])
            with col1: st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
            with col2: _render_criteria(criteria)
            st.plotly_chart(render_wyckoff_chart(df), use_container_width=True)

# ============================================================
# MODULE 2: VOLUME PROFILE
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
    return (edges[:-1]+edges[1:])/2,vap

def analyze_vp(df):
    score=0; criteria=[]; cur=df["Close"].iloc[-1]
    mids, vap = build_volume_profile(df)
    poc_idx = np.argmax(vap); poc = mids[poc_idx]
    
    total = vap.sum(); si = np.argsort(vap)[::-1]; va_vol = 0; va_idx = []
    for i in si:
        if va_vol>=total*0.70: break
        va_vol+=vap[i]; va_idx.append(i)
    val = mids[min(va_idx)]

    below_val = cur < val; bp = 25 if below_val else 0; score+=bp
    criteria.append({"name":"מחיר מתחת ל-VAL","hit":below_val,"points":25,"earned":bp,"explanation":"המחיר נסחר מתחת לאזור הערך, מה שמהווה 'הנחה' מבחינה מוסדית."})
    
    near_poc = abs(cur-poc)/poc <= 0.03; pp = 25 if near_poc else 0; score+=pp
    criteria.append({"name":"סמיכות ל-POC","hit":near_poc,"points":25,"earned":pp,"explanation":"המחיר סמוך לנקודת השליטה (Point of Control) – אזור הוגן בו בוצעה מרבית הפעילות."})
    
    lvn_below = (mids < cur) & (vap < np.percentile(vap, 20))
    has_lvn = np.any(lvn_below); lp = 25 if has_lvn else 0; score+=lp
    criteria.append({"name":"LVN מתחת למחיר","hit":has_lvn,"points":25,"earned":lp,"explanation":"קיום Low Volume Nodes מתחת למחיר משמש כמקפצה לתנועה למעלה."})
    
    hvn_above = (mids > cur) & (vap > np.percentile(vap, 75))
    has_hvn = np.any(hvn_above); hp = 25 if has_hvn else 0; score+=hp
    criteria.append({"name":"HVN מעל המחיר","hit":has_hvn,"points":25,"earned":hp,"explanation":"High Volume Nodes מעל המחיר מספנים אזורי איסוף היסטוריים."})

    vd="סבירות גבוהה לנוכחות מוסדית" if score>=75 else "סימנים חלקיים ב-VP" if score>=45 else "אין ריכוז מוסדי מובהק"
    vc="#ab47bc" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score, criteria, vd, vc

def render_vp_chart(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", line=dict(color="#ab47bc", width=2)))
    fig.update_layout(height=400,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a", font_color="#e0eaf4")
    return fig

def screen_vp():
    st.markdown("""<div class="header-box vp"><h2>🔮 VOLUME PROFILE</h2><p>ניתוח הפעילות האנכית — איפה המוסדיים השקיעו את הכסף שלהם במחיר מסוים.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (Volume Profile)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ נתח VP", use_container_width=True)
    if run_btn:
        df = get_data(ticker.upper())
        if df is not None:
            score, criteria, vd, vc = analyze_vp(df)
            col1, col2 = st.columns([1, 2])
            with col1: st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
            with col2: _render_criteria(criteria)
            st.plotly_chart(render_vp_chart(df), use_container_width=True)

# ============================================================
# MODULE 3: VWAP DEVIATION
# ============================================================
def compute_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    dev = df["Close"] - vwap
    return vwap, dev, dev.rolling(20).std()

def analyze_vwap(df):
    score=0; criteria=[]
    vwap, dev, rolling_std = compute_vwap(df)
    cur = df["Close"].iloc[-1]; cur_vwap = vwap.iloc[-1]; cur_std = rolling_std.iloc[-1]; cur_dev = dev.iloc[-1]
    
    below_1std = cur_dev <= -cur_std; pts1 = 25 if below_1std else 0; score+=pts1
    criteria.append({"name":"סטיית מחיר מ-VWAP","hit":below_1std,"points":25,"earned":pts1,"explanation":"המחיר נסחר בסטיית תקן אחת לפחות מתחת ל-VWAP - אזור דיסקאונט קלאסי."})
    
    vwap_20 = vwap.iloc[-20:]
    slope_early = (vwap_20.iloc[10] - vwap_20.iloc[0]) / vwap_20.iloc[0]
    slope_late = (vwap_20.iloc[-1] - vwap_20.iloc[-10]) / vwap_20.iloc[-10]
    flattening = (slope_early < -0.005) and (slope_late > slope_early * 0.5)
    pts2 = 25 if flattening else 0; score+=pts2
    criteria.append({"name":"VWAP Slope מתיישב","hit":flattening,"points":25,"earned":pts2,"explanation":"בלימה של מומנטום המכירות, הממוצע מתחיל להתאזן אופקית."})
    
    support = df["Low"].rolling(65).min().iloc[-1]
    above_support = cur > support; pts3 = 25 if above_support else 0; score+=pts3
    criteria.append({"name":"שמירה על מבנה השפל","hit":above_support,"points":25,"earned":pts3,"explanation":"למרות הסטייה למטה, המניה מחזיקה את השפל המקומי ולא קורסת."})
    
    std_early = rolling_std.iloc[-20:-10].mean(); std_late = rolling_std.iloc[-10:].mean()
    contracting = (std_late < std_early * 0.85) if std_early > 0 else False
    pts4 = 25 if contracting else 0; score+=pts4
    criteria.append({"name":"סטיית תקן מתכווצת","hit":contracting,"points":25,"earned":pts4,"explanation":"התכווצות בתנודתיות שמעידה על התבססות מחירים שקטה סביב הרמה."})

    vd="מיקום VWAP אופטימלי לכניסה" if score>=75 else "סטייה חלקית" if score>=45 else "אין הזדמנות VWAP"
    vc="#4caf7d" if score>=75 else "#ffa726" if score>=45 else "#ef5350"
    return score, criteria, vd, vc

def render_vwap_chart(df):
    vwap, dev, std = compute_vwap(df)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="#e0eaf4")))
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name="VWAP", line=dict(color="#ffa726", width=2)))
    fig.update_layout(height=400,paper_bgcolor="#0a1520",plot_bgcolor="#0d1b2a", font_color="#e0eaf4")
    return fig

def screen_vwap():
    st.markdown("""<div class="header-box vwap"><h2>📐 VWAP DEVIATION SCOUT</h2><p>מנתח סטיית תקן כדי לזהות מתי המניה "זולה מדי" ביחס לממוצע הכסף החכם.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (VWAP)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ נתח VWAP", use_container_width=True)
    if run_btn:
        df = get_data(ticker.upper())
        if df is not None:
            score, criteria, vd, vc = analyze_vwap(df)
            col1, col2 = st.columns([1, 2])
            with col1: st.plotly_chart(render_gauge(score, vd, vc), use_container_width=True)
            with col2: _render_criteria(criteria)
            st.plotly_chart(render_vwap_chart(df), use_container_width=True)

# ============================================================
# MODULE 4: COMPOSITE SCORE
# ============================================================
def analyze_composite(df):
    w_score = analyze_wyckoff(df)[0]
    v_score = analyze_vp(df)[0]
    vw_score = analyze_vwap(df)[0]
    composite = int(round(w_score * 0.35 + v_score * 0.35 + vw_score * 0.30))
    methods_above_75 = sum(1 for s in [w_score,v_score,vw_score] if s >= 75)
    
    if composite >= 75 and methods_above_75 >= 2:
        verdict = "Strong Signal"; vcolor = "#26a69a"
    elif composite >= 60:
        verdict = "Watch"; vcolor = "#ffa726"
    else:
        verdict = "Wait"; vcolor = "#ef5350"
        
    return composite, verdict, vcolor, w_score, v_score, vw_score

def screen_composite():
    st.markdown("""<div class="header-box composite"><h2>🏆 COMPOSITE SCORE</h2><p>שקלול של כל 3 האלגוריתמים הויזואלים להחלטה אחת סופית.</p></div>""",unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (Composite)", "NVDA")
    with c2: 
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ צור ציון משולב", use_container_width=True)
    if run_btn:
        df = get_data(ticker.upper())
        if df is not None:
            composite, verdict, vcolor, w_score, v_score, vw_score = analyze_composite(df)
            col1, col2 = st.columns([1, 1])
            with col1: st.plotly_chart(render_gauge(composite, verdict, vcolor), use_container_width=True)
            with col2:
                st.markdown(f"### התפלגות ציונים עבור {ticker.upper()}:")
                st.markdown(f"- **Wyckoff Accumulation:** {w_score}/100")
                st.markdown(f"- **Volume Profile:** {v_score}/100")
                st.markdown(f"- **VWAP Deviation:** {vw_score}/100")

# ============================================================
# MODULE 5: 🚀 BACKTEST SCREEN (WITH THE 35 FACTORS)
# ============================================================
def screen_backtest():
    st.markdown("""
    <div class="header-box composite" style="background:linear-gradient(135deg,#121a24,#1a2636);border:1px solid #2a4a6a;">
      <h2>📈 BACKTEST ENGINE (Powered by 35 Factors)</h2>
      <p>המנוע מנתח את נתוני העבר של המניה על בסיס 35 פקטורים מוסדיים שרצים מתחת למכסה המנוע.</p>
    </div>""",unsafe_allow_html=True)
    
    c1, c2 = st.columns([4, 1])
    with c1: ticker = st.text_input("סימול מניה (לדוגמה: NVDA)", "NVDA", key="bt_input")
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
