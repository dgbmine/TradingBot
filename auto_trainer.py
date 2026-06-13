import argparse
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scout_core import (
    run_wyckoff_anchored_backtest, FactorEngine, BacktestConfig, 
    clean_filename, save_model_to_disk, calculate_optimal_threshold
)

# מגדיר את ה-Universe המלא (ניתן לערוך ולהוסיף ככל שיידרש)
TRAINING_UNIVERSE = {
    "Growth (צמיחה)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","CRM",
        "NFLX","AMD","ADBE","CSCO","TXN","QCOM","INTC","INTU","ADI",
        "PANW","CRWD","FTNT","ZS","DDOG","SNOW","MDB","NET","PLTR",
        "UBER","ABNB","COIN","SOFI","UPST","ONTO","KLAC","LRCX",
        "AMAT","MRVL","SMCI","DELL","HPQ","RBLX","U","TTWO","EA"
    ],
    "Value/Index (ערך/מדד)": [
        "BRK-B","JPM","JNJ","V","UNH","PG","MA","HD","MRK","ABBV",
        "PEP","KO","COST","WMT","LLY","TMO","MCD","ACN","BAC","ABT",
        "DHR","RTX","HON","NKE","AMGN","PM","IBM","SBUX","GS","CAT",
        "BA","GE","SPGI","AXP","BLK","DE","ISRG","MDLZ","GILD",
        "REGN","SYK","ZTS","MMC","AON","TJX","SCHW","CB","USB","WFC",
        "C","MS","CVS","CI","AMT","PLD","CCI","EQIX","SPG","O",
        "WELL","DLR","DIS","CMCSA","DAL","UAL","AAL","LUV","FDX",
        "UPS","XPO","ODFL","DKNG","MGM","CZR","RCL","CCL","MAR","HLT"
    ],
    "Commodities (סחורות)": [
        "XOM","CVX","SLB","EOG","OXY","COP","PSX","VLO",
        "FCX","NEM","GOLD","AEM","WPM","FNV","PAAS","AG",
        "GLD", "SLV"
    ]
}

def main():
    parser = argparse.ArgumentParser(description="Auto-Trainer for Institutional Scout Pro")
    parser.add_argument("--mode", choices=["full", "quarterly"], required=True, help="Full training or quarterly update")
    args = parser.parse_args()
    mode = args.mode

    end_date_dt = datetime.today()
    if mode == "full":
        start_date_dt = end_date_dt - timedelta(days=6*365)
    else:
        start_date_dt = end_date_dt - timedelta(days=90)

    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    print(f"[*] Starting Auto-Trainer in '{mode}' mode.")
    print(f"[*] Time Range: {start_date} -> {end_date}")
    
    results_summary = {}
    base_threshold = 50

    start_time = time.time()

    for slot, tickers in TRAINING_UNIVERSE.items():
        print(f"\n=============================================")
        print(f"[*] Processing Sector: {slot} ({len(tickers)} tickers)")
        print(f"=============================================")
        
        safe_slot_name = clean_filename(slot)
        history_path = f"models/training_data_{safe_slot_name}.csv"
        os.makedirs("models", exist_ok=True)
        
        features_list = []
        errors = 0
        added_trades = 0
        
        for ticker in tickers:
            time.sleep(0.5) # Rate limiting
            try:
                bt_df, audit_df = run_wyckoff_anchored_backtest(
                    ticker, use_ai=False, threshold=base_threshold, period=None,
                    start=start_date, end=end_date, risk_profile="Aggressive"
                )
                
                if audit_df is None or audit_df.empty:
                    continue
                    
                engine = FactorEngine(BacktestConfig())
                df = bt_df.copy()
                
                for _, trade in audit_df.iterrows():
                    entry_dt = pd.Timestamp(trade['entry_date'])
                    if entry_dt in df.index:
                        window_df = df.loc[:entry_dt].iloc[-200:] if len(df.loc[:entry_dt]) > 200 else df.loc[:entry_dt]
                        factors = engine.compute(window_df)
                        if len(factors) > 0:
                            feature_row = factors.iloc[-1].to_dict()
                            feature_row['phase'] = df.loc[entry_dt]['wyckoff_phase']
                            feature_row['label'] = 1 if trade['win'] else 0
                            feature_row['ticker'] = ticker
                            feature_row['entry_date'] = trade['entry_date']
                            features_list.append(feature_row)
                            added_trades += 1
                            
            except Exception as e:
                print(f"[!] ERROR processing {ticker}: {e}")
                errors += 1
                continue
                
        # לאחר עיבוד כל מניות הסקטור, נשמור את הנתונים ונתחיל אימון
        if not features_list and not os.path.exists(history_path):
            print(f"[!] No valid trades found for sector {slot}. Skipping.")
            continue
            
        new_df = pd.DataFrame(features_list) if features_list else pd.DataFrame()
        
        if os.path.exists(history_path):
            hist_df = pd.read_csv(history_path)
            combined_df = pd.concat([hist_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['ticker', 'entry_date'], keep='last')
        else:
            combined_df = new_df
            
        if combined_df.empty:
            continue
            
        combined_df.to_csv(history_path, index=False)
        print(f"[*] {slot}: {added_trades} new trades added | Total in Library: {len(combined_df)} | Errors: {errors}")
        
        # --- תהליך האימון ---
        print(f"[*] Training Model for {slot}...")
        y = combined_df['label'].values
        le = LabelEncoder()
        phase_encoded = le.fit_transform(combined_df['phase'].fillna("לא בתהליך איסוף"))
        phase_dummies = pd.get_dummies(phase_encoded, prefix='phase').astype(int)
        
        drop_cols = ['phase', 'label', 'ticker', 'entry_date']
        tech_factors = combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns]).select_dtypes(include=[np.number])
        X = pd.concat([tech_factors.reset_index(drop=True), phase_dummies.reset_index(drop=True)], axis=1).fillna(0)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, oob_score=True, random_state=42, n_jobs=-1)
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
            "slot": slot,
            "model_type": "Wyckoff-Anchored",
            "num_trades": len(combined_df),
            "recommended_threshold": optimal_th
        }
        
        save_model_to_disk(slot, model, meta, le)
        results_summary[slot] = {"tickers": len(tickers), "oob": train_acc * 100, "th": optimal_th}
        
        time.sleep(3) # מנוחה למניעת Rate Limiting לפני סקטור חדש

    # הדפסת סיכום סופי
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    print("\n╔══════════════════════════════════════════╗")
    print("║ AUTO-TRAINER COMPLETE                    ║")
    for slot, data in results_summary.items():
        s_name = slot.split()[0] # מדפיס שם קצר ליופי
        print(f"║ {s_name.ljust(15)}: {str(data['tickers']).rjust(3)} מניות | OOB: {data['oob']:.1f}% | TH: {data['th']}  ║")
    print(f"║ זמן ריצה כולל: {elapsed}                  ║")
    print("╚══════════════════════════════════════════╝\n")

if __name__ == "__main__":
    main()
