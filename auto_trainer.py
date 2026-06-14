# auto_trainer.py - FULLY ISOLATED, NO EXTERNAL clean_filename
import os
import sys
import json
import time
import pickle
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import yfinance as yf
except ImportError:
    yf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

LOG_FILE = os.path.join(BASE_DIR, "auto_trainer_error.log")


def log_message(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")


# ---------- העתק מקומי של clean_filename ----------
def clean_filename(name):
    name = str(name)
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')


from scout_core import (
    FactorEngine,
    BacktestConfig,
    run_wyckoff_anchored_backtest,
    calculate_optimal_threshold,
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODEL_DIR, "auto_trainer_status.json")
DONE_FLAG = os.path.join(MODEL_DIR, "auto_trainer.done")

# ---------- סקטורים מוגדרים ישירות ----------
GROWTH_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
    "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
    "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
    "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
    "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA"
]

VALUE_TICKERS = [
    "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
    "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
    "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
    "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
    "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
    "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
    "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
    "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
]

COMMODITIES_TICKERS = [
    "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
    "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
    "GLD","SLV"
]

# [FIX] רשימת tuples — לא dict ולא set
SECTORS_LIST = [
    ("Growth (צמיחה)",        GROWTH_TICKERS),
    ("Value/Index (ערך/מדד)", VALUE_TICKERS),
    ("Commodities (סחורות)",  COMMODITIES_TICKERS),
]


def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = clean_filename(slot_name)
    file_path = os.path.join(MODEL_DIR, f"model_{safe_name}.pkl")
    save_data = {"model": model, "metadata": metadata, "phase_encoder": encoder}
    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    return file_path


def write_status(state, message="", progress=0, current_slot="N/A",
                 started_at=None, finished_at=None, error=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "state":        state,
        "message":      message,
        "progress":     int(progress),
        "current_slot": current_slot,
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
        "started_at":   started_at or datetime.now().isoformat(timespec="seconds"),
        "finished_at":  finished_at or "N/A",
    }
    if error:
        payload["error"] = str(error)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_sector(slot, tickers, start_date, end_date,
                 base_threshold=50, risk_profile="Aggressive"):
    features_list = []
    errors        = 0
    added_trades  = 0
    engine        = FactorEngine(BacktestConfig())

    # הורדת מאקרו (SPY + VIX) אחת לסקטור
    macro = None
    if yf is not None:
        try:
            spy = yf.download("SPY", start=start_date, end=end_date,
                              progress=False)["Close"].rename("SPY_Close")
            vix = yf.download("^VIX", start=start_date, end=end_date,
                              progress=False)["Close"].rename("VIX_Close")
            macro = pd.concat([spy, vix], axis=1).ffill().bfill()
            macro.index = pd.to_datetime(macro.index).date
        except Exception:
            macro = None

    for ticker in tickers:
        time.sleep(0.3)
        try:
            bt_df, audit_df = run_wyckoff_anchored_backtest(
                ticker, use_ai=False, threshold=base_threshold,
                period=None, start=start_date, end=end_date,
                risk_profile=risk_profile
            )
            if audit_df is None or audit_df.empty:
                continue

            df = bt_df.copy()

            # צירוף מאקרו לפי תאריך
            if macro is not None and not df.empty:
                df["date_key"] = df.index.date
                df = df.merge(macro, left_on="date_key", right_index=True, how="left")
                df.drop(columns="date_key", inplace=True)
                for col in ["SPY_Close", "VIX_Close"]:
                    if col in df.columns:
                        df[col] = df[col].ffill().bfill().fillna(0)

            for _, trade in audit_df.iterrows():
                entry_dt = pd.Timestamp(trade["entry_date"])
                if entry_dt not in df.index:
                    continue
                window_df = (df.loc[:entry_dt].iloc[-200:]
                             if len(df.loc[:entry_dt]) > 200
                             else df.loc[:entry_dt])
                try:
                    factors = engine.compute(window_df)
                    if len(factors) == 0:
                        continue
                    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
                    feature_row = factors.iloc[-1].to_dict()

                    raw_phase = df.loc[entry_dt]["wyckoff_phase"]
                    if isinstance(raw_phase, pd.Series):
                        raw_phase = raw_phase.iloc[-1]

                    feature_row["phase"]      = raw_phase
                    feature_row["label"]      = 1 if trade["win"] else 0
                    feature_row["ticker"]     = ticker
                    feature_row["entry_date"] = trade["entry_date"]
                    features_list.append(feature_row)
                    added_trades += 1
                except Exception:
                    continue

        except Exception:
            errors += 1
            continue

    return features_list, added_trades, errors


def run_auto_trainer():
    log_message("=== run_auto_trainer START ===")

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(DONE_FLAG):
        os.remove(DONE_FLAG)

    started_at = datetime.now().isoformat(timespec="seconds")
    write_status(state="running", message="האימון האוטומטי התחיל",
                 progress=0, started_at=started_at)

    end_date_dt   = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=6 * 365)
    start_date    = start_date_dt.strftime("%Y-%m-%d")
    end_date      = end_date_dt.strftime("%Y-%m-%d")
    base_threshold = 50
    total_sectors  = len(SECTORS_LIST)

    try:
        # [FIX] enumerate על list של tuples — ללא .items(), ללא range()
        for sector_idx, (slot, tickers) in enumerate(SECTORS_LIST, start=1):

            slot_str = str(slot)
            log_message(f"Processing sector {sector_idx}/{total_sectors}: {slot_str}")

            write_status(
                state="running",
                message=f"מעבד סקטור: {slot_str}",
                progress=int(((sector_idx - 1) / total_sectors) * 100),
                current_slot=slot_str,
                started_at=started_at,
            )

            features_list, added_trades, errors = train_sector(
                slot=slot_str, tickers=tickers,
                start_date=start_date, end_date=end_date,
                base_threshold=base_threshold,
            )

            safe_slot_name = clean_filename(slot_str)
            history_path   = os.path.join(MODEL_DIR, f"training_data_{safe_slot_name}.csv")
            log_message(f"history_path: {history_path}")

            new_df = pd.DataFrame(features_list) if features_list else pd.DataFrame()

            if os.path.exists(history_path):
                try:
                    hist_df = pd.read_csv(history_path)
                    if not new_df.empty:
                        combined_df = (
                            pd.concat([hist_df, new_df], ignore_index=True)
                            .drop_duplicates(subset=["ticker", "entry_date"], keep="last")
                        )
                    else:
                        combined_df = hist_df
                except Exception:
                    combined_df = new_df
            else:
                combined_df = new_df

            if combined_df.empty:
                log_message(f"[{slot_str}] No trades — skipping.")
                continue

            combined_df.to_csv(history_path, index=False)
            log_message(f"[{slot_str}] {added_trades} new trades | total: {len(combined_df)} | errors: {errors}")

            if combined_df["label"].nunique() < 2:
                log_message(f"[{slot_str}] Less than 2 classes — skipping model training.")
                continue

            # בניית X, y
            y  = combined_df["label"].values
            le = LabelEncoder()
            phase_encoded = le.fit_transform(
                combined_df["phase"].fillna("לא בתהליך איסוף")
            )
            phase_dummies = pd.get_dummies(phase_encoded, prefix="phase").astype(int)

            drop_cols    = ["phase", "label", "ticker", "entry_date"]
            tech_factors = (
                combined_df
                .drop(columns=[c for c in drop_cols if c in combined_df.columns])
                .select_dtypes(include=[np.number])
            )
            X = (
                pd.concat(
                    [tech_factors.reset_index(drop=True),
                     phase_dummies.reset_index(drop=True)],
                    axis=1
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            model = RandomForestClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=3,
                oob_score=True, random_state=42, n_jobs=-1
            )
            model.fit(X, y)

            try:
                train_acc = model.oob_score_
            except Exception:
                train_acc = model.score(X, y)

            optimal_th = calculate_optimal_threshold(model, X, y)
            meta = {
                "train_ticker":          "AUTO_TRAINER_MASTER_LIBRARY",
                "train_acc":             train_acc,
                "test_acc":              train_acc,
                "slot":                  slot_str,
                "model_type":            "Wyckoff-Anchored",
                "num_trades":            len(combined_df),
                "recommended_threshold": optimal_th,
            }

            save_model_to_disk(slot_str, model, meta, le)
            log_message(f"[{slot_str}] Model saved. OOB: {train_acc:.3f} | Threshold: {optimal_th}")

            write_status(
                state="running",
                message=f"סקטור {slot_str} הושלם",
                progress=int((sector_idx / total_sectors) * 100),
                current_slot=slot_str,
                started_at=started_at,
            )
            time.sleep(1)

        # סיום תקין
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_status(
            state="completed",
            message="האימון האוטומטי הסתיים בהצלחה",
            progress=100,
            started_at=started_at,
            finished_at=finished_at,
        )
        with open(DONE_FLAG, "w", encoding="utf-8") as f:
            f.write(f"completed_at={finished_at}\n")

        log_message("=== run_auto_trainer DONE ===")

    except Exception as e:
        error_msg = traceback.format_exc()
        log_message(f"Critical error: {error_msg}")
        write_status(state="error", message="האימון נכשל",
                     progress=0, error=str(e))
        raise


if __name__ == "__main__":
    run_auto_trainer()