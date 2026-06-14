# trainer_core.py - FULLY ISOLATED
import os
import sys
import json
import time
import gc
import atexit
import signal
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
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

LOG_FILE = os.path.join(BASE_DIR, "auto_trainer_error.log")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODEL_DIR, "auto_trainer_status.json")
DONE_FLAG = os.path.join(MODEL_DIR, "auto_trainer.done")
LOCK_FILE = os.path.join(MODEL_DIR, "auto_trainer.lock")

LOCK_STALE_AFTER_SECONDS = 6 * 60 * 60
LOCK_WAIT_SLEEP_SECONDS = 2
LOCK_MAX_WAIT_SECONDS = 0

_LOCK_HELD = False
_LOCK_OWNER = None


# ============================================================
# Logging
# ============================================================
def log_message(msg):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")


# ============================================================
# Explicit imports only
# ============================================================
from scout_core import (
    FactorEngine,
    BacktestConfig,
    run_wyckoff_anchored_backtest,
    calculate_optimal_threshold,
    clean_filename,
)

# ============================================================
# Sectors
# ============================================================
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


# ============================================================
# Locking
# ============================================================
def _read_lock_payload():
    if not os.path.exists(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_lock_stale(lock_payload):
    if not lock_payload:
        return True

    created_at = lock_payload.get("created_at")
    if not created_at:
        return True

    try:
        created_dt = datetime.fromisoformat(created_at)
        age_seconds = (datetime.now() - created_dt).total_seconds()
        return age_seconds > LOCK_STALE_AFTER_SECONDS
    except Exception:
        try:
            mtime = os.path.getmtime(LOCK_FILE)
            age_seconds = time.time() - mtime
            return age_seconds > LOCK_STALE_AFTER_SECONDS
        except Exception:
            return True


def acquire_lock(wait=False, timeout_seconds=LOCK_MAX_WAIT_SECONDS):
    """
    Atomic lock using O_EXCL file creation.
    If lock exists and is stale, it gets replaced.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    start_wait = time.time()
    owner = {
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "script": os.path.basename(__file__),
    }

    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(owner, f, ensure_ascii=False, indent=2)
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
            return owner

        except FileExistsError:
            existing = _read_lock_payload()

            if _is_lock_stale(existing):
                try:
                    os.remove(LOCK_FILE)
                    continue
                except Exception:
                    pass

            if not wait:
                raise RuntimeError(
                    "Auto-trainer is already running. Lock file exists and is active."
                )

            if timeout_seconds and (time.time() - start_wait) > timeout_seconds:
                raise RuntimeError(
                    "Timed out waiting for existing auto-trainer lock to be released."
                )

            time.sleep(LOCK_WAIT_SLEEP_SECONDS)

        except Exception as e:
            raise RuntimeError(f"Failed to acquire trainer lock: {e}")


def release_lock():
    global _LOCK_HELD, _LOCK_OWNER
    if not _LOCK_HELD:
        return
    try:
        if os.path.exists(LOCK_FILE):
            try:
                existing = _read_lock_payload()
                if existing is None or existing.get("pid") == os.getpid():
                    os.remove(LOCK_FILE)
            except Exception:
                pass
    finally:
        _LOCK_HELD = False
        _LOCK_OWNER = None


def _signal_handler(signum, frame):
    try:
        release_lock()
    finally:
        raise SystemExit(0)


atexit.register(release_lock)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass


# ============================================================
# Helpers
# ============================================================
def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = clean_filename(str(slot_name))
    file_path = os.path.join(MODEL_DIR, f"model_{safe_name}.pkl")
    save_data = {"model": model, "metadata": metadata, "phase_encoder": encoder}
    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    return file_path


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
        "lock_file": os.path.basename(LOCK_FILE),
        "pid": os.getpid(),
    }
    if error:
        payload["error"] = str(error)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _download_macro(start_date, end_date):
    macro = None
    if yf is None:
        return None

    try:
        spy_df = yf.download(
            "SPY",
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        vix_df = yf.download(
            "^VIX",
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if spy_df is None or spy_df.empty or vix_df is None or vix_df.empty:
            return None

        spy = spy_df["Close"].rename("SPY_Close")
        vix = vix_df["Close"].rename("VIX_Close")
        macro = pd.concat([spy, vix], axis=1).ffill().bfill()
        macro.index = pd.to_datetime(macro.index).date
        return macro

    except Exception as e:
        log_message(f"Macro download failed: {e}")
        return None


def _safe_get_phase_value(df, entry_dt):
    try:
        raw_phase = df.loc[entry_dt]["wyckoff_phase"]
        if isinstance(raw_phase, pd.Series):
            raw_phase = raw_phase.iloc[-1]
        return raw_phase
    except Exception:
        return "Unknown"


def train_sector(slot, tickers, start_date, end_date, base_threshold=50, risk_profile="Aggressive"):
    features_list = []
    errors = 0
    added_trades = 0
    engine = FactorEngine(BacktestConfig())

    macro = _download_macro(start_date, end_date)

    for ticker in tickers:
        time.sleep(0.15)
        try:
            bt_df, audit_df = run_wyckoff_anchored_backtest(
                ticker,
                use_ai=False,
                threshold=base_threshold,
                period=None,
                start=start_date,
                end=end_date,
                risk_profile=risk_profile,
            )

            if bt_df is None or bt_df.empty or audit_df is None or audit_df.empty:
                continue

            df = bt_df.copy()

            if macro is not None and not df.empty:
                df["date_key"] = df.index.date
                df = df.merge(macro, left_on="date_key", right_index=True, how="left")
                df.drop(columns=["date_key"], inplace=True)
                for col in ["SPY_Close", "VIX_Close"]:
                    if col in df.columns:
                        df[col] = df[col].ffill().bfill().fillna(0)

            for _, trade in audit_df.iterrows():
                try:
                    entry_dt = pd.Timestamp(trade["entry_date"])
                    if entry_dt not in df.index:
                        continue

                    history_slice = df.loc[:entry_dt]
                    if history_slice.empty:
                        continue

                    window_df = history_slice.iloc[-200:] if len(history_slice) > 200 else history_slice

                    factors = engine.compute(window_df)
                    if factors is None or len(factors) == 0:
                        continue

                    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
                    feature_row = factors.iloc[-1].to_dict()

                    feature_row["phase"] = _safe_get_phase_value(df, entry_dt)
                    feature_row["label"] = 1 if bool(trade.get("win", False)) else 0
                    feature_row["ticker"] = ticker
                    feature_row["entry_date"] = trade["entry_date"]

                    features_list.append(feature_row)
                    added_trades += 1

                except Exception:
                    continue

        except Exception as e:
            errors += 1
            log_message(f"[{slot}] ticker {ticker} failed: {e}")
            continue

    return features_list, added_trades, errors


def _build_X_y(combined_df):
    """בניית X, y, le מוכנים לאימון."""
    y = combined_df["label"].values
    le = LabelEncoder()
    phase_encoded = le.fit_transform(combined_df["phase"].fillna("לא בתהליך איסוף"))
    phase_dummies = pd.get_dummies(phase_encoded, prefix="phase").astype(int)

    drop_cols = ["phase", "label", "ticker", "entry_date"]
    tech_factors = (
        combined_df
        .drop(columns=[c for c in drop_cols if c in combined_df.columns])
        .select_dtypes(include=[np.number])
    )

    X = pd.concat(
        [tech_factors.reset_index(drop=True), phase_dummies.reset_index(drop=True)],
        axis=1,
    )
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, le


# ============================================================
# Main trainer
# ============================================================
def run_auto_trainer(wait_for_lock=False, lock_timeout_seconds=0):
    """
    Atomic auto-trainer:
    - acquires a file lock
    - writes progress after each sector
    - releases lock on exit or crash
    - prevents concurrent runs from Streamlit reruns
    """
    global _LOCK_HELD, _LOCK_OWNER

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(DONE_FLAG):
        try:
            os.remove(DONE_FLAG)
        except Exception:
            pass

    try:
        _LOCK_OWNER = acquire_lock(wait=wait_for_lock, timeout_seconds=lock_timeout_seconds)
        _LOCK_HELD = True
    except Exception as e:
        write_status(
            state="locked",
            message="האימון כבר רץ במופע אחר",
            progress=0,
            error=str(e),
        )
        log_message(f"Lock acquisition failed: {e}")
        raise

    started_at = datetime.now().isoformat(timespec="seconds")
    write_status(
        state="running",
        message="האימון האוטומטי התחיל",
        progress=0,
        started_at=started_at,
        current_slot="INIT",
    )
    log_message("=== run_auto_trainer START ===")
    log_message(f"Lock acquired by pid={os.getpid()} owner={_LOCK_OWNER}")

    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=6 * 365)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    base_threshold = 50
    total_sectors = len(SECTORS_LIST)

    try:
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
                slot=slot_str,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                base_threshold=base_threshold,
            )

            safe_slot_name = clean_filename(slot_str)
            history_path = os.path.join(MODEL_DIR, f"training_data_{safe_slot_name}.csv")
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
                except Exception as e:
                    log_message(f"[{slot_str}] failed reading history_path, using new_df only: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df

            if combined_df.empty:
                log_message(f"[{slot_str}] No trades — skipping.")
                write_status(
                    state="running",
                    message=f"סקטור {slot_str} הסתיים ללא עסקאות",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot_str,
                    started_at=started_at,
                )
                gc.collect()
                continue

            combined_df.to_csv(history_path, index=False)
            log_message(
                f"[{slot_str}] {added_trades} new trades | total: {len(combined_df)} | errors: {errors}"
            )

            if combined_df["label"].nunique() < 2:
                log_message(f"[{slot_str}] Less than 2 classes — skipping model training.")
                write_status(
                    state="running",
                    message=f"סקטור {slot_str} נשמר, אבל אין מספיק מחלקות לאימון",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot_str,
                    started_at=started_at,
                )
                gc.collect()
                continue

            X, y, le = _build_X_y(combined_df)

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)

            try:
                train_acc = model.oob_score_
            except Exception:
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
            log_message(
                f"[{slot_str}] Model saved. OOB: {train_acc:.3f} | Threshold: {optimal_th}"
            )

            write_status(
                state="running",
                message=f"סקטור {slot_str} הושלם",
                progress=int((sector_idx / total_sectors) * 100),
                current_slot=slot_str,
                started_at=started_at,
            )

            del X, y, le, model, combined_df, new_df, features_list
            gc.collect()
            time.sleep(0.5)

        finished_at = datetime.now().isoformat(timespec="seconds")
        write_status(
            state="completed",
            message="האימון האוטומטי הסתיים בהצלחה",
            progress=100,
            started_at=started_at,
            finished_at=finished_at,
            current_slot="DONE",
        )

        with open(DONE_FLAG, "w", encoding="utf-8") as f:
            f.write(f"completed_at={finished_at}\n")

        log_message("=== run_auto_trainer DONE ===")

    except Exception as e:
        error_msg = traceback.format_exc()
        log_message(f"Critical error: {error_msg}")
        write_status(
            state="error",
            message="האימון נכשל",
            progress=0,
            error=str(e),
            started_at=started_at,
            current_slot="ERROR",
        )
        raise

    finally:
        release_lock()
        gc.collect()


if __name__ == "__main__":
    run_auto_trainer()