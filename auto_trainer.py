# ============================================================
# auto_trainer.py
# ============================================================
import os
import sys
import json
import time
import pickle
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# NEW: import yfinance for macro data download
try:
    import yfinance as yf
except ImportError:
    yf = None

# מוסיף את התיקייה הנוכחית ל-Path כדי לוודא ש-scout_core יימצא
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from scout_core import *  # noqa: F401,F403

MODEL_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODEL_DIR, "auto_trainer_status.json")
DONE_FLAG = os.path.join(MODEL_DIR, "auto_trainer.done")

# ... (SCAN_UNIVERSE, SECTOR_MAP, TRAINING_UNIVERSE – ללא שינוי) ...
SCAN_UNIVERSE = list(dict.fromkeys([...]))  # (הרשימה נשארת כפי שהייתה)
SECTOR_MAP = {...}
TRAINING_UNIVERSE = {...}

# NEW: list of required new features
REQUIRED_NEW_FEATURES = ["Regime_Filter", "VIX_ZScore", "Relative_Strength"]


def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = clean_filename(slot_name)
    file_path = os.path.join(MODEL_DIR, f"model_{safe_name}.pkl")
    save_data = {"model": model, "metadata": metadata, "phase_encoder": encoder}
    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    return file_path


def write_status(state, message="", progress=0, current_slot="N/A", started_at=None, finished_at=None, error=None):
    # (ללא שינוי)
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "state": state,
        "message": message,
        "progress": int(progress),
        "current_slot": current_slot,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at or datetime.now().isoformat(timespec="seconds"),
        "finished_at": finished_at or "N/A",
    }
    if error:
        payload["error"] = str(error)

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def send_webhook(payload):
    # (ללא שינוי)
    webhook_url = os.getenv("AUTO_TRAINER_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=15).read()
    except Exception:
        pass


def train_sector(slot, tickers, start_date, end_date, base_threshold=50, risk_profile="Aggressive"):
    features_list = []
    errors = 0
    added_trades = 0
    engine = FactorEngine(BacktestConfig())

    # NEW: Download macro data for the entire period
    macro = None
    if yf is not None:
        try:
            print(f"[{slot}] Downloading SPY and VIX data...")
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"].rename("SPY_Close")
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)["Close"].rename("VIX_Close")
            macro = pd.concat([spy, vix], axis=1).ffill().bfill()
            macro.index = pd.to_datetime(macro.index).date  # use date only for merging
        except Exception as e:
            print(f"[{slot}] Warning: Could not download macro data: {e}")
            macro = None
    else:
        print(f"[{slot}] yfinance not available, skipping macro data.")

    for ticker in tickers:
        time.sleep(0.2)
        try:
            bt_df, audit_df = run_wyckoff_anchored_backtest(
                ticker,
                use_ai=False,
                threshold=base_threshold,
                period=None,
                start=start_date,
                end=end_date,
                risk_profile=risk_profile
            )

            if audit_df is None or audit_df.empty:
                continue

            df = bt_df.copy()
            # NEW: merge macro data if available
            if macro is not None and not df.empty:
                df["date_key"] = df.index.date
                df = df.merge(macro, left_on="date_key", right_index=True, how="left")
                df.drop(columns="date_key", inplace=True)
                # forward fill any remaining gaps
                for col in ["SPY_Close", "VIX_Close"]:
                    if col in df.columns:
                        df[col] = df[col].ffill().bfill()

            for _, trade in audit_df.iterrows():
                entry_dt = pd.Timestamp(trade["entry_date"])
                if entry_dt in df.index:
                    window_df = df.loc[:entry_dt].iloc[-200:] if len(df.loc[:entry_dt]) > 200 else df.loc[:entry_dt]
                    factors = engine.compute(window_df)
                    if len(factors) > 0:
                        feature_row = factors.iloc[-1].to_dict()
                        feature_row["phase"] = df.loc[entry_dt]["wyckoff_phase"]
                        feature_row["label"] = 1 if trade["win"] else 0
                        feature_row["ticker"] = ticker
                        feature_row["entry_date"] = trade["entry_date"]
                        features_list.append(feature_row)
                        added_trades += 1
        except Exception:
            errors += 1
            continue

    return features_list, added_trades, errors


def run_auto_trainer():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(DONE_FLAG):
        try:
            os.remove(DONE_FLAG)
        except Exception:
            pass

    started_at = datetime.now().isoformat(timespec="seconds")
    write_status(
        state="running",
        message="האימון האוטומטי התחיל",
        progress=0,
        current_slot="N/A",
        started_at=started_at
    )
    send_webhook({"state": "running", "message": "auto_trainer started", "started_at": started_at})

    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=6 * 365)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    base_threshold = 50
    total_sectors = len(TRAINING_UNIVERSE)
    results_summary = {}

    try:
        for sector_idx, (slot, tickers) in enumerate(TRAINING_UNIVERSE.items(), start=1):
            write_status(
                state="running",
                message=f"מעבד סקטור: {slot}",
                progress=int(((sector_idx - 1) / total_sectors) * 100),
                current_slot=slot,
                started_at=started_at
            )

            features_list, added_trades, errors = train_sector(
                slot=slot,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                base_threshold=base_threshold,
                risk_profile="Aggressive"
            )

            safe_slot_name = clean_filename(slot)
            history_path = os.path.join(MODEL_DIR, f"training_data_{safe_slot_name}.csv")

            new_df = pd.DataFrame(features_list) if features_list else pd.DataFrame()
            if os.path.exists(history_path):
                hist_df = pd.read_csv(history_path)
                if not new_df.empty:
                    combined_df = pd.concat([hist_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=["ticker", "entry_date"], keep="last")
                else:
                    combined_df = hist_df
            else:
                combined_df = new_df

            if combined_df.empty:
                write_status(
                    state="running",
                    message=f"אין עסקאות לסקטור {slot}, מדלג",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot,
                    started_at=started_at
                )
                continue

            # NEW: Ensure required new features exist in the combined data
            missing_features = [f for f in REQUIRED_NEW_FEATURES if f not in combined_df.columns]
            if missing_features:
                raise ValueError(
                    f"חסרים פיצ׳רים חדשים ב־CSV של סקטור {slot}: {missing_features}. "
                    "ייתכן ש-FactorEngine.compute לא חישב אותם (בדוק נתוני מאקרו / יישום compute)."
                )

            combined_df.to_csv(history_path, index=False)

            if combined_df["label"].nunique() < 2:
                write_status(
                    state="running",
                    message=f"לא מספיק מגוון תוויות בסקטור {slot}, מדלג על אימון",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot,
                    started_at=started_at
                )
                continue

            y = combined_df["label"].values
            le = LabelEncoder()
            phase_encoded = le.fit_transform(combined_df["phase"].fillna("לא בתהליך איסוף"))
            phase_dummies = pd.get_dummies(phase_encoded, prefix="phase").astype(int)

            drop_cols = ["phase", "label", "ticker", "entry_date"]
            tech_factors = combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns]).select_dtypes(include=[np.number])
            X = pd.concat([tech_factors.reset_index(drop=True), phase_dummies.reset_index(drop=True)], axis=1).fillna(0)

            # The new features are numeric → automatically included in X ✅

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                oob_score=True,
                random_state=42,
                n_jobs=-1
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
                "slot": slot,
                "model_type": "Wyckoff-Anchored",
                "num_trades": len(combined_df),
                "recommended_threshold": optimal_th
            }

            save_model_to_disk(slot, model, meta, le)

            results_summary[slot] = {
                "tickers": len(tickers),
                "oob": train_acc * 100,
                "th": optimal_th,
                "trades": len(combined_df),
                "errors": errors,
                "added": added_trades
            }

            write_status(
                state="running",
                message=f"הושלם סקטור {slot}",
                progress=int((sector_idx / total_sectors) * 100),
                current_slot=slot,
                started_at=started_at
            )

        finished_at = datetime.now().isoformat(timespec="seconds")
        write_status(
            state="completed",
            message="האימון האוטומטי הסתיים בהצלחה",
            progress=100,
            current_slot="N/A",
            started_at=started_at,
            finished_at=finished_at
        )

        with open(DONE_FLAG, "w", encoding="utf-8") as f:
            f.write(f"completed_at={finished_at}\n")

        send_webhook({
            "state": "completed",
            "message": "auto_trainer completed",
            "started_at": started_at,
            "finished_at": finished_at,
            "results_summary": results_summary
        })

    except Exception as e:
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_status(
            state="error",
            message="האימון האוטומטי נכשל",
            progress=0,
            current_slot="N/A",
            started_at=started_at,
            finished_at=finished_at,
            error=str(e)
        )
        send_webhook({
            "state": "error",
            "message": "auto_trainer failed",
            "error": str(e),
            "started_at": started_at,
            "finished_at": finished_at
        })
        raise


if __name__ == "__main__":
    run_auto_trainer()