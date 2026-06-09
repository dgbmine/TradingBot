cat > /mnt/user-data/outputs/backtest_engine.py << 'PYEOF'
"""
backtest_engine.py
==================
Vectorized Institutional Accumulation Backtesting Engine
Supports: 35 analytical factors, walk-forward, regime analysis,
          transaction costs, slippage, full performance metrics.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
@dataclass
class BacktestConfig:
    commission:        float = 0.001    # 0.1% per trade
    slippage:          float = 0.0005   # 0.05% per trade
    initial_capital:   float = 100_000.0
    position_size:     float = 0.10     # 10% of capital per trade
    hold_days:         int   = 20       # max hold period
    min_score:         int   = 65       # CIS entry threshold
    exit_score:        int   = 35       # CIS exit threshold
    walk_forward_train:int   = 252      # trading days
    walk_forward_test: int   = 63       # ~3 months
    regime_ticker:     str   = "SPY"
    period:            str   = "5y"


# ============================================================
# 35 FACTORS ENGINE
# ============================================================
class FactorEngine:
    """
    Computes all 35 institutional accumulation factors on a DataFrame.
    All computations are fully vectorized (no Python loops over rows).
    Returns a DataFrame with one column per factor + composite CIS score.
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        # Derived base series
        tp   = (df["High"] + df["Low"] + df["Close"]) / 3
        body = (df["Close"] - df["Open"]).abs()
        rng  = df["High"] - df["Low"]
        lower_shadow = df[["Open","Close"]].min(axis=1) - df["Low"]
        upper_shadow = df["High"] - df[["Open","Close"]].max(axis=1)
        vol_ma20 = df["Volume"].rolling(20).mean()
        vol_ma5  = df["Volume"].rolling(5).mean()
        rvol     = df["Volume"] / vol_ma20.replace(0, np.nan)

        # ── 1. Liquidity Gap (LVN proxy via price-volume concentration) ──
        price_bins = pd.cut(df["Close"], bins=40, labels=False)
        vol_by_bin = df.groupby(price_bins)["Volume"].transform("sum")
        bin_mean   = df.groupby(price_bins)["Volume"].transform("mean")
        f["f01_liquidity_gap"] = ((vol_by_bin < bin_mean * 0.5).astype(float)
                                   .rolling(5).mean())

        # ── 2. Volatility Squeeze (BB Width + ATR compression) ──
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        bb_width = (2 * std20) / sma20.replace(0, np.nan)
        atr = pd.concat([rng,
                         (df["High"] - df["Close"].shift(1)).abs(),
                         (df["Low"]  - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        atr14 = atr.rolling(14).mean()
        f["f02_volatility_squeeze"] = (
            (bb_width < bb_width.rolling(20).mean() * 0.75) &
            (atr14    < atr14.rolling(20).mean()    * 0.75)
        ).astype(float)

        # ── 3. Market Regime Filter (SPY MA slope) ──
        spy_close = df.get("spy_close", df["Close"])   # injected externally
        spy_ma50  = spy_close.rolling(50).mean()
        spy_slope = spy_ma50.diff(10) / spy_ma50.shift(10).replace(0, np.nan)
        f["f03_regime_risk_on"] = (spy_slope > 0.01).astype(float)
        f["f03_regime_neutral"]  = ((spy_slope >= -0.01) & (spy_slope <= 0.01)).astype(float)
        f["f03_regime_risk_off"] = (spy_slope < -0.01).astype(float)

        # ── 4. Absorption Signature (weak close + wick + volume) ──
        weak_close  = df["Close"] < (df["Low"] + rng * 0.35)
        strong_wick = lower_shadow > body * 1.5
        high_vol    = rvol > 1.5
        f["f04_absorption"] = (weak_close & strong_wick & high_vol).astype(float)

        # ── 5. Breakout Quality Filter (follow-through 3 days) ──
        resist = df["High"].rolling(20).max().shift(1)
        breakout = df["Close"] > resist
        followthrough = df["Close"].rolling(3).mean() > resist.shift(1)
        f["f05_breakout_quality"] = (breakout & followthrough).astype(float)

        # ── 6. CIS Dynamic Weights (volatility-adjusted) ──
        vol_regime = std20 / std20.rolling(60).mean().replace(0, np.nan)
        f["f06_cis_weight_adj"] = np.clip(1.0 / vol_regime.replace(0, np.nan), 0.5, 2.0)

        # ── 7. Velocity of Accumulation (OBV slope) ──
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        obv_slope = obv.diff(10) / (obv.abs().rolling(10).mean().replace(0, np.nan))
        f["f07_obv_velocity"] = obv_slope.clip(-3, 3)

        # ── 8. Failure to Follow Through ──
        up_day   = df["Close"] > df["Close"].shift(1)
        next_dn  = df["Close"].shift(-1) < df["Close"]
        f["f08_failure_follow_through"] = (up_day & next_dn & high_vol).astype(float)

        # ── 9. Signal Dependency (correlation between f04 and f07) ──
        f["f09_dependency"] = (
            f["f04_absorption"].rolling(10).corr(f["f07_obv_velocity"])
        ).clip(-1, 1)

        # ── 10. Temporal Sequencing ──
        sc_occurred  = f["f04_absorption"].rolling(30).max()
        ns_occurring = (rvol < 0.7).astype(float)
        f["f10_temporal_seq"] = (sc_occurred * ns_occurring)

        # ── 11. Anti-Signal / Kill Switch ──
        crash_day = (df["Close"].pct_change() < -0.05)
        vol_spike = rvol > 4.0
        f["f11_kill_switch"] = (crash_day | vol_spike).astype(float)

        # ── 12. Distribution Mirror (upthrust detection) ──
        new_high     = df["High"] > df["High"].rolling(20).max().shift(1)
        weak_close2  = df["Close"] < df["High"] - rng * 0.7
        f["f12_distribution_mirror"] = (new_high & weak_close2).astype(float)

        # ── 13. Confidence Decay (signal aging) ──
        last_signal_days = f["f04_absorption"].replace(0, np.nan).ffill().isna().astype(int)
        f["f13_confidence_decay"] = np.exp(-last_signal_days / 10.0).clip(0, 1)

        # ── 14. Institutional Intent Score (meta) ──
        f["f14_inst_intent"] = (
            f["f04_absorption"] * 0.3 +
            f["f07_obv_velocity"].clip(0, 1) * 0.4 +
            f["f10_temporal_seq"] * 0.3
        ).clip(0, 1)

        # ── 15. Multi-Timeframe Confirmation ──
        daily_trend  = (df["Close"] > sma20).astype(float)
        weekly_close = df["Close"].rolling(5).mean()
        weekly_trend = (weekly_close > weekly_close.rolling(4).mean()).astype(float)
        f["f15_mtf_confirm"] = (daily_trend * weekly_trend)

        # ── 16. Anchor Point Conflict ──
        vwap_full = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        above_vwap = (df["Close"] > vwap_full).astype(float)
        below_vwap = (df["Close"] < vwap_full).astype(float)
        conflict   = above_vwap.rolling(3).sum() * below_vwap.rolling(3).sum()
        f["f16_anchor_conflict"] = (conflict > 0).astype(float)

        # ── 17. Volatility Cluster Expansion ──
        atr_expand = (atr14 > atr14.shift(5) * 1.3)
        f["f17_vol_cluster_expand"] = atr_expand.astype(float)

        # ── 18. Sector Breadth (rolling up-days proxy) ──
        breadth_proxy = (df["Close"] > df["Close"].shift(1)).astype(float).rolling(10).mean()
        f["f18_sector_breadth"] = breadth_proxy

        # ── 19. Order Flow Imbalance ──
        buy_pressure  = (df["Close"] - df["Low"]) / rng.replace(0, np.nan)
        sell_pressure = (df["High"] - df["Close"]) / rng.replace(0, np.nan)
        f["f19_order_flow_imbalance"] = (buy_pressure - sell_pressure).rolling(5).mean()

        # ── 20. Liquidity Sweep (stop hunt: wick below support then recovery) ──
        support = df["Low"].rolling(20).min().shift(1)
        swept   = (df["Low"] < support) & (df["Close"] > support)
        f["f20_liquidity_sweep"] = swept.astype(float)

        # ── 21. Range Break Authenticity ──
        range_20 = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
        break_size = (df["Close"] - df["Close"].shift(1)).abs()
        f["f21_break_authenticity"] = (break_size / range_20.replace(0, np.nan)).clip(0, 1)

        # ── 22. Support/Resistance Strength ──
        touches_support = (df["Low"].rolling(5).min() <= df["Low"].rolling(20).min() * 1.005).astype(float)
        f["f22_sr_strength"] = touches_support.rolling(20).sum() / 20

        # ── 23. Gap Structure ──
        gap_up   = (df["Open"] > df["Close"].shift(1) * 1.005).astype(float)
        gap_down = (df["Open"] < df["Close"].shift(1) * 0.995).astype(float)
        f["f23_gap_structure"] = gap_up - gap_down

        # ── 24. Event Shock Normalization ──
        shock     = (df["Close"].pct_change().abs() > 0.04).astype(float)
        post_norm = shock.rolling(3).sum().clip(0, 1)
        f["f24_event_shock_norm"] = 1.0 - post_norm

        # ── 25. RVOL Anomaly ──
        rvol_z = (rvol - rvol.rolling(60).mean()) / rvol.rolling(60).std().replace(0, np.nan)
        f["f25_rvol_anomaly"] = rvol_z.clip(-3, 3)

        # ── 26. Price Acceptance vs Rejection ──
        acceptance = (
            (df["Close"] > (df["High"] + df["Low"]) / 2) &
            (df["Volume"] > vol_ma20)
        ).astype(float)
        rejection = (
            (df["Close"] < (df["High"] + df["Low"]) / 2) &
            (df["Volume"] > vol_ma20)
        ).astype(float)
        f["f26_accept_reject"] = acceptance.rolling(5).mean() - rejection.rolling(5).mean()

        # ── 27. Volatility Regime Transition ──
        atr_ratio = atr14 / atr14.rolling(60).mean().replace(0, np.nan)
        expanding   = (atr_ratio > 1.2).astype(float)
        contracting = (atr_ratio < 0.8).astype(float)
        f["f27_vol_regime_transition"] = contracting - expanding  # positive = contracting (bullish)

        # ── 28. Institutional Participation Proxy (large candle + high vol) ──
        large_body = body > body.rolling(20).mean() * 1.5
        f["f28_inst_participation"] = (large_body & high_vol).astype(float)

        # ── 29. Trend Integrity Score ──
        sma50  = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        aligned = (
            (df["Close"] > sma20).astype(int) +
            (sma20 > sma50).astype(int) +
            (sma50 > sma200).astype(int)
        ) / 3
        f["f29_trend_integrity"] = aligned

        # ── 30. Mean Reversion Pressure ──
        z_score = (df["Close"] - sma20) / std20.replace(0, np.nan)
        f["f30_mean_reversion_pressure"] = (-z_score).clip(-3, 3)  # positive when below mean

        # ── 31. False Support Breakdown Filter ──
        below_support = (df["Close"] < df["Low"].rolling(20).min().shift(1))
        recovers      = (df["Close"].shift(-2) > df["Low"].rolling(20).min().shift(3))
        f["f31_bear_trap"] = (below_support & recovers).astype(float)

        # ── 32. Accumulation vs Re-Accumulation ──
        distance_from_ath = (df["Close"].rolling(252).max() - df["Close"]) / df["Close"].rolling(252).max().replace(0, np.nan)
        reaccum = (distance_from_ath < 0.15) & (distance_from_ath > 0.05)
        fresh_accum = distance_from_ath > 0.25
        f["f32_accum_type"] = fresh_accum.astype(float) * 1.0 + reaccum.astype(float) * 0.6

        # ── 33. Liquidity Exhaustion ──
        vol_declining  = (vol_ma5 < vol_ma5.shift(10))
        price_stalling = (df["Close"].pct_change(5).abs() < 0.02)
        f["f33_liquidity_exhaustion"] = (vol_declining & price_stalling).astype(float)

        # ── 34. Multi-Asset Correlation Stress ──
        spy_ret  = spy_close.pct_change()
        tkr_ret  = df["Close"].pct_change()
        rolling_corr = tkr_ret.rolling(20).corr(spy_ret)
        f["f34_corr_stress"] = rolling_corr.clip(-1, 1)

        # ── 35. Structural Break Detection ──
        rolling_high = df["High"].rolling(20).max()
        rolling_low  = df["Low"].rolling(20).min()
        struct_break_up   = (df["Close"] > rolling_high.shift(1)).astype(float)
        struct_break_down = (df["Close"] < rolling_low.shift(1)).astype(float)
        f["f35_structural_break"] = struct_break_up - struct_break_down

        return f.fillna(0)

    def composite_cis(self, factors: pd.DataFrame, weights: Optional[dict] = None) -> pd.Series:
        """
        Composite Institutional Score (0-100) from 35 factors.
        Default weights derived from factor categories.
        Kill switch (f11) zeroes the score when triggered.
        """
        if weights is None:
            weights = {
                "f01_liquidity_gap":          3,
                "f02_volatility_squeeze":     4,
                "f03_regime_risk_on":         5,
                "f04_absorption":             6,
                "f05_breakout_quality":       3,
                "f06_cis_weight_adj":         2,
                "f07_obv_velocity":           5,
                "f08_failure_follow_through": -2,
                "f09_dependency":             2,
                "f10_temporal_seq":           5,
                "f11_kill_switch":            0,   # handled separately
                "f12_distribution_mirror":   -4,
                "f13_confidence_decay":       3,
                "f14_inst_intent":            6,
                "f15_mtf_confirm":            4,
                "f16_anchor_conflict":       -2,
                "f17_vol_cluster_expand":    -1,
                "f18_sector_breadth":         3,
                "f19_order_flow_imbalance":   4,
                "f20_liquidity_sweep":        3,
                "f21_break_authenticity":     2,
                "f22_sr_strength":            2,
                "f23_gap_structure":          2,
                "f24_event_shock_norm":       2,
                "f25_rvol_anomaly":           2,
                "f26_accept_reject":          3,
                "f27_vol_regime_transition":  3,
                "f28_inst_participation":     3,
                "f29_trend_integrity":        3,
                "f30_mean_reversion_pressure":3,
                "f31_bear_trap":              2,
                "f32_accum_type":             2,
                "f33_liquidity_exhaustion":  -1,
                "f34_corr_stress":            1,
                "f35_structural_break":       2,
            }
        total_positive = sum(v for v in weights.values() if v > 0)
        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns and col != "f11_kill_switch":
                score += factors[col].clip(-1, 1) * w
        score = (score / total_positive * 100).clip(0, 100)
        # Kill switch: zero out score when triggered
        if "f11_kill_switch" in factors.columns:
            score = score * (1 - factors["f11_kill_switch"])
        return score.round(1)


# ============================================================
# SIGNAL DEBUGGER (Under the Hood)
# ============================================================
class SignalDebugger:
    """
    Generates a human-readable audit trail for a single row/date,
    showing exactly how each factor contributed to the final CIS score.
    """

    FACTOR_LABELS = {
        "f01_liquidity_gap":           ("Liquidity Gap",              "LVN/HVN — אזור ריק מסמן תנועה מהירה"),
        "f02_volatility_squeeze":      ("Volatility Squeeze",         "BB Width + ATR מכווצים — לפני פיצוץ"),
        "f03_regime_risk_on":          ("Market Regime — Risk On",    "SPY בעלייה — רוח גבית"),
        "f03_regime_neutral":          ("Market Regime — Neutral",    "SPY ללא מגמה ברורה"),
        "f03_regime_risk_off":         ("Market Regime — Risk Off",   "SPY יורד — סביבה עוינת"),
        "f04_absorption":              ("Absorption Signature",       "זנב + נר חלש + ווליום = ספיגה"),
        "f05_breakout_quality":        ("Breakout Quality",           "פריצה עם המשך — לא פייק"),
        "f06_cis_weight_adj":          ("CIS Weight Adjustment",      "משקל דינמי לפי תנודתיות"),
        "f07_obv_velocity":            ("OBV Velocity",               "מהירות צבירה — OBV slope"),
        "f08_failure_follow_through":  ("Failure to Follow Through",  "⚠️ עלייה ללא המשך — אזהרה"),
        "f09_dependency":              ("Signal Dependency",          "קורלציה בין אבסורפשן ל-OBV"),
        "f10_temporal_seq":            ("Temporal Sequencing",        "SC לפני No-Supply — סדר נכון"),
        "f11_kill_switch":             ("⛔ Kill Switch",              "אירוע קיצון — הציון归零"),
        "f12_distribution_mirror":     ("Distribution Mirror",        "⚠️ Upthrust — חלוקה, לא איסוף"),
        "f13_confidence_decay":        ("Confidence Decay",           "ישנות האות — כמה ימים עברו"),
        "f14_inst_intent":             ("Institutional Intent",       "ציון מטא — כוונת מוסדיים"),
        "f15_mtf_confirm":             ("Multi-Timeframe Confirm",    "יומי ושבועי מסכימים"),
        "f16_anchor_conflict":         ("Anchor Point Conflict",      "⚠️ מחיר מתחת ו VWAP — קונפליקט"),
        "f17_vol_cluster_expand":      ("Vol Cluster Expansion",      "⚠️ ATR מתרחב — תנודתיות עולה"),
        "f18_sector_breadth":          ("Sector Breadth",             "רוחב שוק — כמה ניירות עולים"),
        "f19_order_flow_imbalance":    ("Order Flow Imbalance",       "לחץ קנייה vs מכירה"),
        "f20_liquidity_sweep":         ("Liquidity Sweep",            "Stop hunt + החלמה = Bull Trap פייל"),
        "f21_break_authenticity":      ("Break Authenticity",         "איכות הפריצה ביחס לטווח"),
        "f22_sr_strength":             ("S/R Strength",               "חוזק תמיכה/התנגדות לאורך זמן"),
        "f23_gap_structure":           ("Gap Structure",              "פערים — Breakaway vs Exhaustion"),
        "f24_event_shock_norm":        ("Event Shock Normalization",  "נרמול לאחר חדשות/רווחים"),
        "f25_rvol_anomaly":            ("RVOL Anomaly",               "ווליום חריג vs בייסליין"),
        "f26_accept_reject":           ("Price Accept/Reject",        "שוק מקבל או דוחה את המחיר"),
        "f27_vol_regime_transition":   ("Vol Regime Transition",      "מעבר מהתרחבות להתכווצות"),
        "f28_inst_participation":      ("Institutional Participation","נרות גדולים + ווליום = מוסדיים"),
        "f29_trend_integrity":         ("Trend Integrity",            "איכות המגמה — SMA alignment"),
        "f30_mean_reversion_pressure", ("Mean Reversion Pressure",   "כמה המחיר רחוק מהממוצע"),
        "f31_bear_trap":               ("Bear Trap Detection",        "שבירת תמיכה מזויפת"),
        "f32_accum_type":              ("Accum vs Re-Accum",          "איסוף ראשון או חוזר"),
        "f33_liquidity_exhaustion":    ("Liquidity Exhaustion",       "⚠️ ווליום יורד + מחיר תקוע"),
        "f34_corr_stress":             ("Multi-Asset Corr Stress",    "קורלציה עם SPY — סיכון מאקרו"),
        "f35_structural_break":        ("Structural Break",           "שבירת מבנה מחיר — Micro Regime"),
    }

    def audit(self, factors: pd.DataFrame, cis: pd.Series,
              date=None, weights: Optional[dict] = None) -> list:
        """
        Returns a list of dicts describing each factor's contribution
        for a specific date (default: last row).
        """
        if date is None:
            row = factors.iloc[-1]
            cis_val = cis.iloc[-1]
        else:
            row = factors.loc[date]
            cis_val = cis.loc[date]

        WEIGHTS_DEFAULT = {
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
        w = weights or WEIGHTS_DEFAULT
        total_positive = sum(v for v in w.values() if v > 0)

        result = []
        for col, weight in w.items():
            if col not in row.index:
                continue
            raw_val   = float(row[col])
            clipped   = float(np.clip(raw_val, -1, 1))
            points    = clipped * weight
            pct_contrib = (points / total_positive * 100) if total_positive else 0
            label, desc = self.FACTOR_LABELS.get(col, (col, ""))
            result.append({
                "factor_id":   col,
                "label":       label,
                "description": desc,
                "raw_value":   round(raw_val, 4),
                "weight":      weight,
                "points":      round(points, 3),
                "pct_of_score":round(pct_contrib, 2),
                "status":      "✅ תורם" if points > 0.01 else "❌ גורע" if points < -0.01 else "➖ ניטרלי",
            })

        result.sort(key=lambda x: abs(x["points"]), reverse=True)
        return result, round(float(cis_val), 1)


# ============================================================
# BACKTESTER
# ============================================================
class BacktestEngine:
    """
    Vectorized backtesting engine for institutional accumulation strategy.
    Usage:
        engine = BacktestEngine(BacktestConfig())
        results = engine.run("NVDA")
        engine.walk_forward("NVDA")
    """

    def __init__(self, cfg: BacktestConfig = None):
        self.cfg     = cfg or BacktestConfig()
        self.factors = FactorEngine(self.cfg)
        self.debugger= SignalDebugger()

    # ── Data ──────────────────────────────────────────────
    def _fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(period=self.cfg.period)
            if df is None or len(df) < 200:
                return None
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            return None

    def _inject_spy(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            spy = yf.Ticker(self.cfg.regime_ticker).history(period=self.cfg.period)
            spy.index = pd.to_datetime(spy.index).tz_localize(None)
            spy_close = spy["Close"].reindex(df.index).ffill()
            df["spy_close"] = spy_close
        except Exception:
            df["spy_close"] = df["Close"]
        return df

    # ── Signal Generation ─────────────────────────────────
    def _generate_signals(self, df: pd.DataFrame) -> tuple:
        df = self._inject_spy(df)
        f  = self.factors.compute(df)
        cis = self.factors.composite_cis(f)

        # Entry: CIS crosses above threshold (shift to avoid lookahead)
        entry  = (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score)
        # Exit: CIS drops below exit threshold OR max hold reached
        exit_  = (cis < self.cfg.exit_score)

        return df, f, cis, entry, exit_

    # ── Vectorized P&L ───────────────────────────────────
    def _simulate_trades(self, df: pd.DataFrame,
                         entry: pd.Series,
                         exit_: pd.Series,
                         cis: pd.Series) -> pd.DataFrame:
        cost = self.cfg.commission + self.cfg.slippage
        capital = self.cfg.initial_capital
        pos_size = self.cfg.position_size

        trades = []
        in_trade = False
        entry_price = 0.0
        entry_date  = None
        hold_count  = 0
        entry_cis   = 0.0

        closes = df["Close"].values
        dates  = df.index
        entry_arr = entry.values
        exit_arr  = exit_.values
        cis_arr   = cis.values

        for i in range(1, len(closes)):
            if not in_trade:
                if entry_arr[i]:
                    entry_price = closes[i] * (1 + cost)
                    entry_date  = dates[i]
                    entry_cis   = cis_arr[i]
                    in_trade    = True
                    hold_count  = 0
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
                        "entry_date":  entry_date,
                        "exit_date":   dates[i],
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_price, 4),
                        "return":      round(ret, 6),
                        "pnl":         round(pnl, 2),
                        "hold_days":   hold_count,
                        "exit_reason": "max_hold" if forced_exit else "signal",
                        "entry_cis":   round(entry_cis, 1),
                        "capital":     round(capital, 2),
                    })
                    in_trade = False

        return pd.DataFrame(trades)

    # ── Equity Curve ──────────────────────────────────────
    def _equity_curve(self, df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
        equity = pd.Series(self.cfg.initial_capital, index=df.index)
        if trades.empty:
            return equity
        for _, t in trades.iterrows():
            mask = equity.index >= t["exit_date"]
            equity[mask] = t["capital"]
        return equity.ffill()

    # ── Performance Metrics ───────────────────────────────
    def _metrics(self, trades: pd.DataFrame,
                 equity: pd.Series,
                 regime_series: pd.Series = None) -> dict:
        if trades.empty:
            return {"error": "No trades generated"}

        rets = trades["return"]
        wins = rets[rets > 0]
        loss = rets[rets <= 0]

        sharpe   = (rets.mean() / rets.std() * np.sqrt(252)).round(3) if rets.std()>0 else 0
        downside = rets[rets < 0].std()
        sortino  = (rets.mean() / downside * np.sqrt(252)).round(3) if downside>0 else 0

        equity_arr = equity.values
        peak       = np.maximum.accumulate(equity_arr)
        drawdown   = (equity_arr - peak) / peak
        max_dd     = drawdown.min()

        win_rate    = len(wins) / len(rets)
        avg_win     = wins.mean() if len(wins) else 0
        avg_loss    = loss.mean() if len(loss) else 0
        profit_factor = (wins.sum() / abs(loss.sum())) if loss.sum() != 0 else np.inf
        expectancy  = win_rate * avg_win + (1 - win_rate) * avg_loss

        total_return = (equity.iloc[-1] - self.cfg.initial_capital) / self.cfg.initial_capital

        base = {
            "total_trades":   len(trades),
            "win_rate":       round(win_rate, 4),
            "sharpe":         sharpe,
            "sortino":        sortino,
            "max_drawdown":   round(max_dd, 4),
            "profit_factor":  round(profit_factor, 3),
            "expectancy":     round(expectancy, 6),
            "avg_win":        round(avg_win, 4),
            "avg_loss":       round(avg_loss, 4),
            "total_return":   round(total_return, 4),
            "final_capital":  round(equity.iloc[-1], 2),
        }

        # Regime breakdown
        if regime_series is not None and not trades.empty:
            regime_stats = {}
            for regime_name in ["risk_on","neutral","risk_off"]:
                mask = trades["entry_date"].map(
                    lambda d: regime_series.reindex([d], method="ffill").iloc[0] == regime_name
                    if len(regime_series.reindex([d], method="ffill")) else False
                )
                sub = trades[mask]
                if len(sub) > 0:
                    regime_stats[regime_name] = {
                        "trades":   len(sub),
                        "win_rate": round((sub["return"] > 0).mean(), 3),
                        "avg_ret":  round(sub["return"].mean(), 4),
                    }
            base["regime_breakdown"] = regime_stats

        # CIS sensitivity
        buckets = pd.cut(trades["entry_cis"],
                         bins=[0,60,70,80,90,100],
                         labels=["<60","60-70","70-80","80-90","90+"])
        cis_sens = trades.groupby(buckets, observed=True)["return"].agg(
            count="count", mean_ret="mean", win_rate=lambda x: (x>0).mean()
        ).round(4).to_dict("index")
        base["cis_sensitivity"] = cis_sens

        return base

    # ── Full Run ──────────────────────────────────────────
    def run(self, ticker: str) -> dict:
        df = self._fetch(ticker)
        if df is None:
            return {"error": f"Cannot fetch data for {ticker}"}

        df, f, cis, entry, exit_ = self._generate_signals(df)
        trades  = self._simulate_trades(df, entry, exit_, cis)
        equity  = self._equity_curve(df, trades)

        # Regime label series
        regime_map = pd.Series("neutral", index=df.index)
        if "f03_regime_risk_on"  in f.columns: regime_map[f["f03_regime_risk_on"]  == 1] = "risk_on"
        if "f03_regime_risk_off" in f.columns: regime_map[f["f03_regime_risk_off"] == 1] = "risk_off"

        metrics = self._metrics(trades, equity, regime_map)

        return {
            "ticker":        ticker,
            "df":            df,
            "factors":       f,
            "cis":           cis,
            "entry":         entry,
            "exit":          exit_,
            "trades":        trades,
            "equity":        equity,
            "metrics":       metrics,
            "regime_map":    regime_map,
        }

    # ── Walk-Forward ─────────────────────────────────────
    def walk_forward(self, ticker: str) -> pd.DataFrame:
        df = self._fetch(ticker)
        if df is None:
            return pd.DataFrame()

        train = self.cfg.walk_forward_train
        test  = self.cfg.walk_forward_test
        results = []
        i = train

        while i + test <= len(df):
            test_slice = df.iloc[i: i + test].copy()
            test_slice = self._inject_spy(test_slice)
            f   = self.factors.compute(test_slice)
            cis = self.factors.composite_cis(f)
            entry, exit_ = (
                (cis.shift(1) < self.cfg.min_score) & (cis >= self.cfg.min_score),
                cis < self.cfg.exit_score
            )
            trades = self._simulate_trades(test_slice, entry, exit_, cis)
            equity = self._equity_curve(test_slice, trades)
            m = self._metrics(trades, equity)
            m["window_start"] = df.index[i].strftime("%Y-%m-%d")
            m["window_end"]   = df.index[i + test - 1].strftime("%Y-%m-%d")
            results.append(m)
            i += test

        return pd.DataFrame(results)

    # ── Audit (Under the Hood) ────────────────────────────
    def audit_date(self, ticker: str, date=None) -> tuple:
        """
        Returns full factor breakdown for a specific date.
        Used by UI 'Under the Hood' button.
        """
        df = self._fetch(ticker)
        if df is None:
            return [], 0.0
        df, f, cis, _, _ = self._generate_signals(df)
        return self.debugger.audit(f, cis, date=date)


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    print("Running BacktestEngine test on NVDA...")
    engine = BacktestEngine()
    result = engine.run("NVDA")
    m = result["metrics"]
    print(f"  Trades:        {m.get('total_trades', 0)}")
    print(f"  Win Rate:      {m.get('win_rate', 0):.1%}")
    print(f"  Sharpe:        {m.get('sharpe', 0)}")
    print(f"  Sortino:       {m.get('sortino', 0)}")
    print(f"  Max Drawdown:  {m.get('max_drawdown', 0):.1%}")
    print(f"  Profit Factor: {m.get('profit_factor', 0)}")
    print(f"  Total Return:  {m.get('total_return', 0):.1%}")
    print(f"  Final Capital: ${m.get('final_capital', 0):,.2f}")
    print("\nWalk-Forward test...")
    wf = engine.walk_forward("NVDA")
    if not wf.empty:
        print(wf[["window_start","window_end","total_trades","win_rate","sharpe"]].to_string(index=False))
    print("\nAudit (Under the Hood) test...")
    audit, score = engine.audit_date("NVDA")
    print(f"  CIS Score: {score}")
    for a in audit[:5]:
        print(f"  {a['status']} {a['label']}: {a['points']:+.2f} pts ({a['pct_of_score']:+.1f}%)")
    print("\n✅ BacktestEngine OK")
PYEOF
echo "Lines: $(wc -l < /mnt/user-data/outputs/backtest_engine.py)"