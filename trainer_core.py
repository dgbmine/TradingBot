# ============================================================
# AUTO TRAINER - ISOLATED PROCESS WITH LOCK MECHANISM
# מריץ כתהליך עצמאי, לא נטען על ידי Streamlit
# ============================================================

import os
import sys
import json
import time
import pickle
import signal
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

# ── נתיב בסיס: תמיד לפי מיקום הקובץ הזה ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ── ייבוא מפורש בלבד — אין import * ──────────────────────
from scout_core import (
    FactorEngine,
    BacktestConfig,
    run_wyckoff_anchored_backtest,
    calculate_optimal_threshold,
    clean_filename,
)

# ── קבועי נתיבים ──────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODEL_DIR, "auto_trainer_status.json")
DONE_FLAG = os.path.join(MODEL_DIR, "auto_trainer.done")
LOCK_FILE = os.path.join(MODEL_DIR, "auto_trainer.lock")
STOP_FILE = os.path.join(MODEL_DIR, "auto_trainer.stop")
PID_FILE = os.path.join(MODEL_DIR, "auto_trainer.pid")
LOG_FILE = os.path.join(BASE_DIR, "auto_trainer_error.log")

# ── סקטורים — list of tuples, לא dict ────────────────────
GROWTH_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
    "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
    "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
    "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
    "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA",
]
VALUE_TICKERS = [
    "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
    "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
    "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
    "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
    "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
    "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
    "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
    "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT",
]
COMMODITIES_TICKERS = [
    "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
    "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
    "GLD","SLV",
]
SECTORS_LIST = [
    ("Growth (צמיחה)", GROWTH_TICKERS),
    ("Value/Index (ערך/מדד)", VALUE_TICKERS),
    ("Commodities (סחורות)", COMMODITIES_TICKERS),
]

# ── משתנה גלובלי לטיפול ב-SIGTERM ────────────────────────
_stop_requested = False

def _handle_sigterm(signum, frame):
    global _stop_requested
    _stop_requested = True
    log_message("SIGTERM received — will stop after current ticker.")

# ── רישום בלוג ────────────────────────────────────────────
def log_message(msg: str):
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)

# ── כתיבת סטטוס JSON ──────────────────────────────────────
def write_status(state, message="", progress=0, current_slot="N/A",
                 started_at=None, finished_at=None, error=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "state": state,
        "message": message,
        "progress": int(progress),
        "current_slot": str(current_slot),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at or datetime.now().isoformat(timespec="seconds"),
        "finished_at": finished_at or "N/A",
        "pid": os.getpid(),
    }
    if error:
        payload["error"] = str(error)
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ── שמירת מודל ────────────────────────────────────────────
def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = clean_filename(str(slot_name))
    file_path = os.path.join(MODEL_DIR, f"model_{safe_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata, "phase_encoder": encoder}, f)
    return file_path

# ── בניית X, y ────────────────────────────────────────────
def _build_X_y(combined_df: pd.DataFrame):
    y = combined_df["label"].values
    le = LabelEncoder()
    phase_encoded = le.fit_transform(
        combined_df["phase"].fillna("לא בתהליך איסוף")
    )
    phase_dummies = pd.get_dummies(phase_encoded, prefix="phase").astype(int)
    drop_cols = ["phase", "label", "ticker", "entry_date"]
    tech_factors = (
        combined_df
        .drop(columns=[c for c in drop_cols if c in combined_df.columns])
        .select_dtypes(include=[np.number])
    )
    X = (
        pd.concat(
            [tech_factors.reset_index(drop=True),
             phase_dummies.reset_index(drop=True)],
            axis=1,
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return X, y, le

# ── אימון סקטור בודד ──────────────────────────────────────
def train_sector(slot_str, tickers, start_date, end_date,
                 base_threshold=50, risk_profile="Aggressive"):
    global _stop_requested
    features_list = []
    errors = 0
    added_trades = 0
    engine = FactorEngine(BacktestConfig())

    macro = None
    if yf is not None:
        try:
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"].rename("SPY_Close")
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)["Close"].rename("VIX_Close")
            macro = pd.concat([spy, vix], axis=1).ffill().bfill()
            macro.index = pd.to_datetime(macro.index).date
        except Exception:
            macro = None

    for ticker in tickers:
        # ── בדיקת עצירה אחרי כל מניה ──
        if _stop_requested or os.path.exists(STOP_FILE):
            log_message(f"Stop requested — aborting sector {slot_str} after {added_trades} trades.")
            break
        time.sleep(0.3)
        try:
            bt_df, audit_df = run_wyckoff_anchored_backtest(
                ticker, use_ai=False, threshold=base_threshold,
                period=None, start=start_date, end=end_date,
                risk_profile=risk_profile,
            )
            if audit_df is None or audit_df.empty:
                continue
            df = bt_df.copy()
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
                    feature_row["phase"] = raw_phase
                    feature_row["label"] = 1 if trade["win"] else 0
                    feature_row["ticker"] = ticker
                    feature_row["entry_date"] = trade["entry_date"]
                    features_list.append(feature_row)
                    added_trades += 1
                except Exception:
                    continue
        except Exception:
            errors += 1
            continue

    return features_list, added_trades, errors

# ── MAIN ──────────────────────────────────────────────────
def run_auto_trainer():
    global _stop_requested
    _stop_requested = False

    # רישום handlers לסיגנלים
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, _handle_sigterm)
    except Exception:
        pass

    log_message("=== run_auto_trainer START ===")
    log_message(f"PID: {os.getpid()} | BASE_DIR: {BASE_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── מנגנון נעילה (Lock) ────────────────────────────────
    if os.path.exists(LOCK_FILE):
        # בדוק אם המנעול ישן (מעל 6 שעות) — אם כן, מחק אותו
        try:
            age = time.time() - os.path.getmtime(LOCK_FILE)
            if age > 6 * 3600:
                os.remove(LOCK_FILE)
                log_message("Stale lock removed.")
            else:
                log_message("Lock file exists and is fresh — another instance may be running. Aborting.")
                return
        except Exception:
            pass

    # צור מנעול
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}, f)
    except Exception as e:
        log_message(f"Cannot create lock file: {e}")
        return

    # כתוב PID
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # נקה דגלים ישנים
    for f in [DONE_FLAG, STOP_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    started_at = datetime.now().isoformat(timespec="seconds")
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=6 * 365)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    base_threshold = 50
    total_sectors = len(SECTORS_LIST)
    write_status(state="running", message="האימון האוטומטי התחיל", progress=0, started_at=started_at)

    try:
        for sector_idx, (slot, tickers) in enumerate(SECTORS_LIST, start=1):
            # ── בדיקת עצירה בין סקטורים ──
            if _stop_requested or os.path.exists(STOP_FILE):
                log_message("Stop requested between sectors — exiting gracefully.")
                write_status(state="stopped", message="האימון עצר לפי בקשת המשתמש",
                             progress=int(((sector_idx - 1) / total_sectors) * 100),
                             started_at=started_at)
                break
            slot_str = str(slot)
            log_message(f"--- Sector {sector_idx}/{total_sectors}: {slot_str} ---")
            write_status(
                state="running",
                message=f"מעבד סקטור: {slot_str}",
                progress=int(((sector_idx - 1) / total_sectors) * 100),
                current_slot=slot_str,
                started_at=started_at,
            )

            features_list, added_trades, errors = train_sector(
                slot_str, tickers, start_date, end_date, base_threshold,
            )

            safe_name = clean_filename(slot_str)
            history_path = os.path.join(MODEL_DIR, f"training_data_{safe_name}.csv")
            new_df = pd.DataFrame(features_list) if features_list else pd.DataFrame()

            if os.path.exists(history_path):
                try:
                    hist_df = pd.read_csv(history_path)
                    combined_df = (
                        pd.concat([hist_df, new_df], ignore_index=True)
                        .drop_duplicates(subset=["ticker", "entry_date"], keep="last")
                        if not new_df.empty else hist_df
                    )
                except Exception:
                    combined_df = new_df
            else:
                combined_df = new_df

            if combined_df.empty:
                log_message(f"[{slot_str}] No trades — skipping.")
                continue

            combined_df.to_csv(history_path, index=False)
            log_message(f"[{slot_str}] +{added_trades} trades | total: {len(combined_df)} | errors: {errors}")

            if combined_df["label"].nunique() < 2:
                log_message(f"[{slot_str}] < 2 classes — skipping model fit.")
                continue

            X, y, le = _build_X_y(combined_df)
            model = RandomForestClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=3,
                oob_score=True, random_state=42, n_jobs=-1,
            )
            model.fit(X, y)
            try:
                train_acc = model.oob_score_
            except:
                train_acc = model.score(X, y)

            optimal_th = calculate_optimal_threshold(model, X, y)
            meta = {
                "train_ticker": "AUTO_TRAINER_MASTER_LIBRARY",
                "train_acc": train_acc,
                "test_acc": train_acc,
                "slot": slot_str,
                "model_type": "Wyckoff-Anchored",
                "num_trades": len(combined_df),
                "recommended_threshold": optimal_th,
            }
            save_model_to_disk(slot_str, model, meta, le)
            log_message(f"[{slot_str}] Model saved | OOB: {train_acc:.3f} | Threshold: {optimal_th}")

            write_status(
                state="running",
                message=f"סקטור {slot_str} הושלם ({sector_idx}/{total_sectors})",
                progress=int((sector_idx / total_sectors) * 100),
                current_slot=slot_str,
                started_at=started_at,
            )
            time.sleep(1)
        else:
            # הלולאה הסתיימה ללא break — הצלחה מלאה
            finished_at = datetime.now().isoformat(timespec="seconds")
            write_status(
                state="completed",
                message="האימון האוטומטי הסתיים בהצלחה",
                progress=100,
                started_at=started_at,
                finished_at=finished_at,
            )
            try:
                with open(DONE_FLAG, "w", encoding="utf-8") as f:
                    f.write(f"completed_at={finished_at}\n")
            except Exception:
                pass
            log_message("=== run_auto_trainer DONE ===")
    except Exception as e:
        log_message(f"CRITICAL ERROR: {traceback.format_exc()}")
        write_status(state="error", message="האימון נכשל", progress=0, error=str(e), started_at=started_at)
        raise
    finally:
        # ── תמיד: שחרר מנעול ו-PID ─────────────────────────
        for f in [LOCK_FILE, PID_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        # נקה STOP_FILE אם נשאר
        if os.path.exists(STOP_FILE):
            try:
                os.remove(STOP_FILE)
            except Exception:
                pass

# ── כניסה ישירה מ-CLI ─────────────────────────────────────
if __name__ == "__main__":
    run_auto_trainer()