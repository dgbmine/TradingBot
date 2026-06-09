import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
@dataclass
class BacktestConfig:
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% per trade
    initial_capital: float = 100_000.0
    position_size: float = 0.10 # 10% of capital per trade
    hold_days: int = 20        # max hold period
    min_score: int = 65        # CIS entry threshold
    exit_score: int = 35       # CIS exit threshold
    walk_forward_train: int = 252 
    walk_forward_test: int = 63 
    regime_ticker: str = "SPY"
    period: str = "5y"

# ============================================================
# 35 FACTORS ENGINE
# ============================================================
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

        f["f01_liquidity_gap"] = ((df.groupby(pd.cut(df["Close"], bins=40, labels=False))["Volume"].transform("sum") < 
                                   df.groupby(pd.cut(df["Close"], bins=40, labels=False))["Volume"].transform("mean") * 0.5).astype(float).rolling(5).mean())
        
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        bb_width = (2 * std20) / sma20.replace(0, np.nan)
        atr = pd.concat([rng, (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        atr14 = atr.rolling(14).mean()
        f["f02_volatility_squeeze"] = ((bb_width < bb_width.rolling(20).mean() * 0.75) & (atr14 < atr14.rolling(20).mean() * 0.75)).astype(float)

        spy_close = df.get("spy_close", df["Close"])
        spy_slope = spy_close.rolling(50).mean().diff(10) / spy_close.rolling(50).mean().shift(10).replace(0, np.nan)
        f["f03_regime_risk_on"] = (spy_slope > 0.01).astype(float)
        f["f03_regime_neutral"] = ((spy_slope >= -0.01) & (spy_slope <= 0.01)).astype(float)
        f["f03_regime_risk_off"] = (spy_slope < -0.01).astype(float)

        f["f04_absorption"] = ((df["Close"] < (df["Low"] + rng * 0.35)) & (lower_shadow > body * 1.5) & (rvol > 1.5)).astype(float)
        resist = df["High"].rolling(20).max().shift(1)
        f["f05_breakout_quality"] = ((df["Close"] > resist) & (df["Close"].rolling(3).mean() > resist.shift(1))).astype(float)
        f["f06_cis_weight_adj"] = np.clip(1.0 / (std20 / std20.rolling(60).mean()).replace(0, np.nan), 0.5, 2.0).fillna(1.0)
        
        obv_slope = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum().diff(10) / (np.sign(df["Close"].diff()) * df["Volume"]).cumsum().abs().rolling(10).mean().replace(0, np.nan)
        f["f07_obv_velocity"] = obv_slope.clip(-3, 3).fillna(0)
        f["f08_failure_follow_through"] = ((df["Close"] > df["Close"].shift(1)) & (df["Close"].shift(-1) < df["Close"]) & (rvol > 1.5)).astype(float)
        f["f09_dependency"] = f["f04_absorption"].rolling(10).corr(f["f07_obv_velocity"]).clip(-1, 1).fillna(0)
        f["f10_temporal_seq"] = (f["f04_absorption"].rolling(30).max() * (rvol < 0.7).astype(float))
        f["f11_kill_switch"] = ((df["Close"].pct_change() < -0.05) | (rvol > 4.0)).astype(float)
        f["f12_distribution_mirror"] = ((df["High"] > df["High"].rolling(20).max().shift(1)) & (df["Close"] < df["High"] - rng * 0.7)).astype(float)
        f["f13_confidence_decay"] = np.exp(-f["f04_absorption"].replace(0, np.nan).ffill().isna().astype(int) / 10.0).clip(0, 1)
        f["f14_inst_intent"] = (f["f04_absorption"] * 0.3 + f["f07_obv_velocity"].clip(0, 1) * 0.4 + f["f10_temporal_seq"] * 0.3).clip(0, 1)
        f["f15_mtf_confirm"] = ((df["Close"] > sma20).astype(float) * (df["Close"].rolling(5).mean() > df["Close"].rolling(20).mean()).astype(float))
        
        vwap_full = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        f["f16_anchor_conflict"] = ((df["Close"] > vwap_full).rolling(3).sum() * (df["Close"] < vwap_full).rolling(3).sum() > 0).astype(float)
        f["f17_vol_cluster_expand"] = (atr14 > atr14.shift(5) * 1.3).astype(float)
        f["f18_sector_breadth"] = (df["Close"] > df["Close"].shift(1)).astype(float).rolling(10).mean()
        f["f19_order_flow_imbalance"] = (((df["Close"] - df["Low"]) / rng.replace(0, np.nan)) - ((df["High"] - df["Close"]) / rng.replace(0, np.nan))).rolling(5).mean()
        f["f20_liquidity_sweep"] = ((df["Low"] < df["Low"].rolling(20).min().shift(1)) & (df["Close"] > df["Low"].rolling(20).min().shift(1))).astype(float)
        f["f21_break_authenticity"] = ((df["Close"] - df["Close"].shift(1)).abs() / (df["High"].rolling(20).max() - df["Low"].rolling(20).min()).replace(0, np.nan)).clip(0, 1)
        f["f22_sr_strength"] = ((df["Low"].rolling(5).min() <= df["Low"].rolling(20).min() * 1.005).astype(float)).rolling(20).sum() / 20
        f["f23_gap_structure"] = (df["Open"] > df["Close"].shift(1) * 1.005).astype(float) - (df["Open"] < df["Close"].shift(1) * 0.995).astype(float)
        f["f24_event_shock_norm"] = 1.0 - (df["Close"].pct_change().abs() > 0.04).astype(float).rolling(3).sum().clip(0, 1)
        f["f25_rvol_anomaly"] = ((rvol - rvol.rolling(60).mean()) / rvol.rolling(60).std().replace(0, np.nan)).clip(-3, 3)
        f["f26_accept_reject"] = (((df["Close"] > (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float) - ((df["Close"] < (df["High"] + df["Low"]) / 2) & (df["Volume"] > vol_ma20)).astype(float)).rolling(5).mean()
        f["f27_vol_regime_transition"] = ((atr14 / atr14.rolling(60).mean().replace(0, np.nan)) < 0.8).astype(float) - ((atr14 / atr14.rolling(60).mean().replace(0, np.nan)) > 1.2).astype(float)
        f["f28_inst_participation"] = ((body > body.rolling(20).mean() * 1.5) & (rvol > 1.5)).astype(float)
        f["f29_trend_integrity"] = ((df["Close"] > sma20).astype(int) + (sma20 > sma20.rolling(50).mean()).astype(int) + (sma20.rolling(50).mean() > sma20.rolling(200).mean()).astype(int)) / 3
        f["f30_mean_reversion_pressure"] = (-(df["Close"] - sma20) / std20.replace(0, np.nan)).clip(-3, 3)
        f["f31_bear_trap"] = ((df["Close"] < df["Low"].rolling(20).min().shift(1)) & (df["Close"].shift(-2) > df["Low"].rolling(20).min().shift(3))).astype(float)
        f["f32_accum_type"] = (((df["Close"].rolling(252).max() - df["Close"]) / df["Close"].rolling(252).max().replace(0, np.nan)) > 0.25).astype(float)
        f["f33_liquidity_exhaustion"] = ((vol_ma5 < vol_ma5.shift(10)) & (df["Close"].pct_change(5).abs() < 0.02)).astype(float)
        f["f34_corr_stress"] = df["Close"].pct_change().rolling(20).corr(spy_close.pct_change()).clip(-1, 1)
        f["f35_structural_break"] = (df["Close"] > df["High"].rolling(20).max().shift(1)).astype(float) - (df["Close"] < df["Low"].rolling(20).min().shift(1)).astype(float)
        
        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame) -> pd.Series:
        weights = { "f01_liquidity_gap": 3, "f02_volatility_squeeze": 4, "f03_regime_risk_on": 5, "f04_absorption": 6, 
                    "f05_breakout_quality": 3, "f06_cis_weight_adj": 2, "f07_obv_velocity": 5, "f08_failure_follow_through": -2, 
                    "f09_dependency": 2, "f10_temporal_seq": 5, "f12_distribution_mirror": -4, "f13_confidence_decay": 3, 
                    "f14_inst_intent": 6, "f15_mtf_confirm": 4, "f16_anchor_conflict": -2, "f17_vol_cluster_expand": -1, 
                    "f18_sector_breadth": 3, "f19_order_flow_imbalance": 4, "f20_liquidity_sweep": 3, "f21_break_authenticity": 2, 
                    "f22_sr_strength": 2, "f23_gap_structure": 2, "f24_event_shock_norm": 2, "f25_rvol_anomaly": 2, 
                    "f26_accept_reject": 3, "f27_vol_regime_transition": 3, "f28_inst_participation": 3, "f29_trend_integrity": 3, 
                    "f30_mean_reversion_pressure": 3, "f31_bear_trap": 2, "f32_accum_type": 2, "f33_liquidity_exhaustion": -1, 
                    "f34_corr_stress": 1, "f35_structural_break": 2 }
        
        score = sum(factors[col].clip(-1, 1) * w for col, w in weights.items() if col in factors.columns)
        score = (score / sum(v for v in weights.values() if v > 0) * 100).clip(0, 100)
        return (score * (1 - factors.get("f11_kill_switch", 0))).round(1)

# ============================================================
# BACKTESTER & MAIN
# ============================================================
class BacktestEngine:
    def __init__(self, cfg: BacktestConfig = None):
        self.cfg = cfg or BacktestConfig()
        self.factors = FactorEngine(self.cfg)

    def run(self, ticker: str):
        df = yf.Ticker(ticker).history(period=self.cfg.period)
        if len(df) < 200: return {"error": "Insufficient data"}
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        spy = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)
        df["spy_close"] = spy["Close"].reindex(df.index).ffill()
        
        f = self.factors.compute(df)
        cis = self.factors.composite_cis(f)
        
        entry = (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score)
        exit_ = (cis < self.cfg.exit_score)
        
        trades = []
        in_trade, entry_price, entry_date, hold = False, 0.0, None, 0
        for i in range(1, len(df)):
            if not in_trade and entry.iloc[i]:
                entry_price = df["Close"].iloc[i] * (1 + self.cfg.commission + self.cfg.slippage)
                entry_date = df.index[i]
                in_trade, hold = True, 0
            elif in_trade:
                hold += 1
                if hold >= self.cfg.hold_days or exit_.iloc[i]:
                    ret = (df["Close"].iloc[i] * (1 - self.cfg.commission - self.cfg.slippage) - entry_price) / entry_price
                    trades.append({"return": ret})
                    in_trade = False
        
        return {"trades": pd.DataFrame(trades), "metrics": {"win_rate": (pd.DataFrame(trades)["return"] > 0).mean() if trades else 0}}

if __name__ == "__main__":
    print("Engine Test:", BacktestEngine().run("NVDA")["metrics"])
