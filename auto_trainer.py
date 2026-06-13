# ============================================================
# auto_trainer.py (V2 - FIXED & ROBUST)
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

# Import yfinance for macro data download (not used now, but kept for flexibility)
try:
    import yfinance as yf
except ImportError:
    yf = None

# Ensure scout_core is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from scout_core import *  # noqa: F401,F403

MODEL_DIR = os.path.join(BASE_DIR, "models")
STATUS_FILE = os.path.join(MODEL_DIR, "auto_trainer_status.json")
DONE_FLAG = os.path.join(MODEL_DIR, "auto_trainer.done")

# --- Universe definitions (identical to your original) ---
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
    "DIS","CMCSA","RBLX","U","TTWO","EA",
    "DAL","UAL","AAL","LUV","FDX","UPS","XPO","ODFL",
    "DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
]))

SECTOR_MAP = {
    "הכול (כל השוק האמריקאי)": SCAN_UNIVERSE,
    "צמיחה וטכנולוגיה (Growth)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
        "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
        "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
        "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
        "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA"
    ],
    "ערך ומדד (Value/Index)": [
        "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
        "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
        "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
        "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
        "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
        "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
        "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
        "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
    ],
    "סחורות ואנרגיה (Commodities)": [
        "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
        "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
        "GLD", "SLV"
    ]
}

TRAINING_UNIVERSE = {
    "Growth (צמיחה)": SECTOR_MAP["צמיחה וטכנולוגיה (Growth)"],
    "Value/Index (ערך/מדד)": SECTOR_MAP["ערך ומדד (Value/Index)"],
    "Commodities (סחורות)": SECTOR_MAP["סחורות ואנרגיה (Commodities)"]
}

# ============================================================
# NEW FEATURE NAMES – must match FactorEngine.compute() output
# ============================================================
REQUIRED_NEW_FEATURES = [
    "f_macro_spy_bull",    # Regime_Filter
    "f_macro_vix_zscore",  # VIX_ZScore
    "f_macro_rel_str"      # Relative_Strength
]


def save_model_to_disk(slot_name, model, metadata, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = clean_filename(slot_name)
    file_path = os.path.join(MODEL_DIR, f"model_{safe_name}.pkl")
    save_data = {"model": model, "metadata": metadata, "phase_encoder": encoder}
    with open(file_path, "wb") as f:
        pickle.dump(save_data, f)
    return file_path


def write_status(state, message="", progress=0, current_slot="N/A", started_at=None, finished_at=None, error=None):
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

    # No manual macro download – run_wyckoff_anchored_backtest (via get_data)
    # already populates spy_close and vix_close in the bt_df.

    for ticker in tickers:
        time.sleep(0.2)
        print(f"  [{slot}] Processing {ticker}...")
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
                print(f"    -> No trades for {ticker}")
                continue

            df = bt_df.copy()
            # The df already contains spy_close and vix_close from get_data,
            # so FactorEngine.compute will create the macro features.

            for _, trade in audit_df.iterrows():
                entry_dt = pd.Timestamp(trade["entry_date"])
                if entry_dt not in df.index:
                    continue
                window_df = df.loc[:entry_dt].iloc[-200:] if len(df.loc[:entry_dt]) > 200 else df.loc[:entry_dt]
                factors = engine.compute(window_df)
                if len(factors) == 0:
                    continue
                feature_row = factors.iloc[-1].to_dict()
                feature_row["phase"] = df.loc[entry_dt]["wyckoff_phase"]
                feature_row["label"] = 1 if trade["win"] else 0
                feature_row["ticker"] = ticker
                feature_row["entry_date"] = trade["entry_date"]
                features_list.append(feature_row)
                added_trades += 1

            print(f"    -> OK, total trades now: {added_trades}")
        except Exception as e:
            errors += 1
            print(f"    -> ERROR in {ticker}: {e}")
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
            print(f"\n===== Sector {sector_idx}/{total_sectors}: {slot} =====")
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
                print(f"  -> No trades for slot {slot}, skipping.")
                write_status(
                    state="running",
                    message=f"אין עסקאות לסקטור {slot}, מדלג",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot,
                    started_at=started_at
                )
                continue

            # --- VERIFY NEW FEATURES PRESENT ---
            missing_features = [f for f in REQUIRED_NEW_FEATURES if f not in combined_df.columns]
            if missing_features:
                error_msg = (
                    f"Missing new features for slot {slot}: {missing_features}. "
                    "Available columns: " + ", ".join(combined_df.columns)
                )
                print(f"  -> {error_msg}")
                raise ValueError(error_msg)

            print(f"  -> Combined data shape: {combined_df.shape}, features OK")
            combined_df.to_csv(history_path, index=False)

            if combined_df["label"].nunique() < 2:
                print(f"  -> Not enough label diversity, skipping training.")
                write_status(
                    state="running",
                    message=f"לא מספיק מגוון תוויות בסקטור {slot}, מדלג על אימון",
                    progress=int((sector_idx / total_sectors) * 100),
                    current_slot=slot,
                    started_at=started_at
                )
                continue

            # Prepare training data
            y = combined_df["label"].values
            le = LabelEncoder()
            phase_encoded = le.fit_transform(combined_df["phase"].fillna("לא בתהליך איסוף"))
            phase_dummies = pd.get_dummies(phase_encoded, prefix="phase").astype(int)

            drop_cols = ["phase", "label", "ticker", "entry_date"]
            tech_factors = combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns]).select_dtypes(include=[np.number])
            X = pd.concat([tech_factors.reset_index(drop=True), phase_dummies.reset_index(drop=True)], axis=1).fillna(0)

            # Check for any remaining NaN (should be none after fillna, but just in case)
            if X.isnull().any().any():
                print("  -> Warning: NaN found in training features after fillna, filling with 0 again.")
                X = X.fillna(0)

            print(f"  -> Training RandomForest on {X.shape[0]} samples, {X.shape[1]} features...")
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
            print(f"  -> Model saved. OOB accuracy: {train_acc*100:.1f}%, recommended threshold: {optimal_th}")

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