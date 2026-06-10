# ==============================================================
# INSTITUTIONAL SCOUT PRO V3 - CODE REVIEW
# ==============================================================
# דעה טכנית: Efi, הקוד שלך חזק ומתוכנן היטב. הנה המשוב:
# ==============================================================

"""
█▀█ █▀ █░█ █▀▄▀█ █▄ ▄█ ░░▄   
█░█ █░ █░█ █░▀░█ █░▀░█ ░░░   
▀░▀ ▀▀ ▀▀▀ ▀░░░▀ ▀░░░▀ ░░▀   

1️⃣ ARCHITECTURE & DESIGN
═══════════════════════════════════════════════════════════════
✅ חוזקות:

• Separation of Concerns -깔끔:
  - FactorEngine (חישוב 35 פקטורים)
  - BacktestEngine (סימולציית מסחר וחישוב מטריקות)
  - SignalDebugger (ניתוח רטרוספקטיבי)
  - ScreenModules (UI טהור)

  זה ממש בסדר. קל להרחיב, קל לבדוק, קל להחליף משקלים.

• Factory Pattern עם BacktestConfig:
  קונפיגורציה מרכזית = דוקומנטציה עצמית למודל ההתנהגות.
  יופי.

• Session State Management:
  אתה משמר את המודל ML ב-st.session_state כ-pickle bytes.
  זה עובד כי Streamlit מתעדכן בין-לבין, אבל...

⚠️ הערות CRITIC:

• ML Model Persistence:
  אתה משתמש ב-Base64(Pickle(RandomForest)) כדי לעשות Export/Import.
  זה עובד, אבל יש סכנה: pickle של sklearn עלול להשתבר בין גרסאות.
  אם אתה ממשיך עם זה, הוסף גרסת sklearn ל-metadata כדי לתעד כמה השתנו.
  
  חלופה: JobLib כמו Sklearn משתמשת. קטן יותר, יותר robust.

• No Data Validation Before Compute:
  בחלק 5 (FactorEngine.compute), אתה לא בוודק אם:
  - יש Nans בעמודות הדרושות (High, Low, Open, Close)
  - Volume = 0 (יכול לגרום לחלוקה באפס ב-composite_cis)
  - הנתונים מסדרים כרונולוגית
  
  כדי להיות safe, הוסף assertion:
  
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        assert all(col in df.columns for col in ["High","Low","Open","Close","Volume"]), "Missing OHLCV"
        assert len(df) >= 50, "Need at least 50 bars"
        # ... rest


2️⃣ FACTOR ENGINEERING (35 Factors)
═══════════════════════════════════════════════════════════════
✅ חוזקות:

• הרעיון של f04_absorption (גוף קטן + צל עמוק + ווליום גבוה):
  זה ממש Wyckoff. תופעה פיזית. טוב.

• f10_temporal_seq:
  (absorption * low_volume) = זיהוי ממשיך אחרי ריד ב-ווליום.
  אפילו עדק.

• f11_kill_switch:
  דחיפת -5% או ווליום פי 4 = עצור הכל.
  משמעותי לכך שאתה עוקב אחרי תבניות שהן "לא מרגיעות יותר".

⚠️ קשיים:

• Multi-Collinearity בגדול:
  f14_inst_intent = 0.3*f04 + 0.4*f07 + 0.3*f10
  f29_trend_integrity = avg(3 SMA crosses)
  f30_mean_rev = inverse(Z-score)
  f26, f27 מדברות גם על acceptance vs rejection.
  
  אתה עלול לתת משקל כפול לאותה תופעה.
  אם אתה רוצה להיות disciplined, הרץ VIF (Variance Inflation Factor).
  או פשוט הוסף correlation heatmap בסריקה.

• Lookback Windows לא uniform:
  f01: rolling(5)
  f02: rolling(20)
  f07: rolling(10)
  f25: rolling(60)
  f32: rolling(252)
  
  זה בחוזה - שונים מלכתחילה. אבל בעצמאות? אם טיקר ממש מתחזק במהירות,
  f32_accum_type (שנה) יהיה טרייל כבד. בדוק את זה.

• f09_dependency = correlation(f04, f07):
  אתה מחשב תלות בזמן אמת. זה יכול להיות נתון תופעתי או רעש טהור
  בגלל rolling window קטן. אפילו לא בטוח שזה בעל מובהקות סטטיסטית.
  הוסף min_periods=30 כדי לא להיות בטוח מרוב.

• f03_regime (SPY slope):
  אם אתה סורק TASE בעתיד (כפי שדנת), אתה לא יכול להשתמש ב-SPY.
  הוסף פרמטר לבחירת regime_ticker.


3️⃣ BACKTEST ENGINE
═══════════════════════════════════════════════════════════════
✅ חוזקות:

• Walk-Forward Logic נכון:
  entry: score.shift(1) < min ו-score >= min (זוג דורות)
  exit: score < exit_score או hold_days >= 40
  
  זה מחק את Look-Ahead Bias כי אתה משתמש ב-.shift(1).
  טוב מאוד.

• Trade Recording:
  אתה חוזר וקוד עסקה אחת לכל entry/exit.
  זה משך את Win Rate, Drawdown, Return בלי בלבול.

⚠️ קשיים:

• Position Sizing:
  position_size = 0.10 (קבוע!)
  משמעות: כל עסקה תופסת בדיוק 10% מהקפיטל.
  
  כמה פעמים אתה יכול להיות בעסקה בו-זמנית?
  אם הכן אתה בעסקה ומגיע סיגנל חדש, אתה פוגע ב-existing position?
  אני לא רואה logic לכך. הנחתי שזה single position at a time.
  
  אם תרצה ריינק מולטיפל, תוסיף queue של עסקאות פעילות. לא כאן.

• Slippage Model:
  commission/slippage = פשוט * (1 + 0.001) entry, * (1 - 0.0005) exit.
  זה linear. במציאות, slippage גדל כשווליום קטן.
  היום אתה סורק מניות קיפיטל גדול (NVDA, MSFT וכו'), אז זה טוב.
  אבל כשתלך ל-micro-caps, זה יעיד.

• Equity Curve:
  אתה מחשב רק סוף-סוף בעדכון capital בוידו.
  equity = [initial] + [returns for each trade]
  
  כלומר, אתה מניח שכל המשקפות קורים בדיוק, ברתם בעלות מינימלית.
  בחיים אמתיים, יש עסקאות מתנייד וזמנים טובים לחזור לקש.
  עדיין, לצורך בק-טסט אקדמי, זה די טוב.

• No Regime Filter:
  אתה מחזירה סימנים בכל תנאי שוק. בשוק דובי כבד, Signal Equity עלול
  להיות נצפה כפיקטיביהה, ולא בגלל מודל.
  
  חשבון: בדוק את max DD כשאתה בטווח דובי של SPY. הוסף מטריק ל"% time in drawdown".


4️⃣ ML TRAINER
═══════════════════════════════════════════════════════════════
✅ חוזקות:

• Proper Train/Test Split:
  80/20, no shuffling (סדרה זמן, אתה לא מדברר).
  Good.

• Feature Engineering Before Split:
  אתה מחשב את כל 35 הפקטורים ראשון, ואח"כ אתה מחלק.
  זה נכון, לא מפיץ מידע מהעתיד.

• Binary Classification:
  alpha > 0.02 = outperformance signal.
  זה הגיוני. אתה אומר "if next 10 days beat SPY by 2%+, label = 1".
  Clean.

⚠️ קשיים:

• Model Complexity:
  RandomForest, n_estimators=150, max_depth=4, min_samples_split=50.
  
  זה סבירו לא מוגזם - עץ עמוק של 4 = לא too deep.
  אבל - אתה לא רץ חיפוש לפרמטרים (GridSearchCV).
  אתה לא בודק cross-validation.
  אתה לא מדיד feature importance variance.
  
  אם זה סרחא, הוסף:
  
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV Acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
  
  זה יגיד לך אם המודל יציב או תלוי בקטע המסויים.

• Overfitting Check:
  אתה חוזר train_acc ו-test_acc בmetadata.
  אם gap > 0.15, זה flag שחזק מדי.
  אתה לא עושה early stopping ברייזינג.
  
  קוד להוסיף:
  
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.15:
        st.warning(f"⚠️ Overfitting detected: gap = {overfit_gap:.2%}")

• No Retraining Schedule:
  אתה מאימן פעם אחת וזהו. בעולם אמתי, סטטיסטיקות שוק משתנות.
  אתה צריך retrain כל חודש או רבעון כדי להישאר current.
  
  זה לא באג, זה limitation. תעד את זה.

• Feature Importance Interpretation:
  אתה משתמש ב-model.feature_importances_ (Gini).
  זה עובד, אבל Gini bias לכיוונים high-cardinality.
  
  אם אתה רוצה להיות פדנטי, השתמש ב-permutation importance:
  
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X_test, y_test)


5️⃣ UI/UX
═══════════════════════════════════════════════════════════════
✅ חזקות:

• Wyckoff Screen:
  VSA breakdown (effort vs result) בטבעי וקריא.
  הגרף עם Volume Overlay ברור.
  Alerts כתובות בעברית, ספציפיות ("לפני X ימים").

• Color Coding:
  ירוק (#26a69a) = טוב
  כתום (#ffa726) = זהירות
  אדום (#ef5350) = בעיה
  נחמד וקונסיסטנט.

• Navigation:
  7 tabs, כפתורים גדולים, קל לעבור.

⚠️ קשיים:

• No Loading Indicators:
  בממשק הscanner, אתה חוזר time.sleep(0.1) בין כל טיקר.
  טוב לא להיות aggressive, אבל אתה לא מציג Progress Bar עדכון בזמן אמת.
  (למעשה, אתה כן עושה את זה עם st.progress, אז זה בסדר.)

• No Error Recovery:
  אם yfinance מתנתק באמצע scan, כל העסקה נופלת.
  אתה יכול להוסיף try/except עם retry logic:
  
    def get_data_safe(ticker, period, max_retries=3):
        for attempt in range(max_retries):
            try:
                return yf.Ticker(ticker).history(period=period)
            except:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # exponential backoff

• Session State Bloat:
  אתה שומר את כל המודל ML בmemory.
  אם אוקופה, זה עלול להיות בעיה בRamM.
  אבל למטרת Streamlit, זה בסדר.


6️⃣ STATISTICAL RIGOR
═══════════════════════════════════════════════════════════════
⚠️ חזויי בעיות:

• Survivorship Bias:
  אתה סורק מניות שנמצאות כעת בSCAN_UNIVERSE (שכולן מניות גדולות).
  אבל חברות שפשטו במהלך ה-5 שנות בק-טסט לא בScanning list.
  זה יעשה טוב יותר מבאמת, כי לא מייד דמים משמעותיים.
  
  דרך לתיקון: עבור כל ticker בתיקייה ההיסטורית של סיימן שנים,
  בחזור הרץ בק-טסט על תקופה שלאחרי delisting. קטן, אבל דרוש.

• Look-Ahead Bias (סיכום):
  אתה הדקת זה כי אתה משתמש ב-shift(1), טוב.

• Curve Fitting:
  35 factors + random weights = אתה בעצם fitting לנתונים.
  עם 35 features, לא קשה להציג כמו 55-58% accuracy.
  
  תעשה את זה: הרץ אמא מודל (כל הפקטורים מחודשות במחדל לאפס)
  וראה מה האחוז baseline. אם זה 48%, אתה בנוי מדי.


7️⃣ PRODUCTIAZATION READINESS
═══════════════════════════════════════════════════════════════
⚠️ לא מוכן עדיין:

• No Real-Time Data:
  אתה משתמש בyfinance (EOD). בפנט לא תוכל להכנס ב-intraday.
  כדי להקדם: integrator עם IB (Interactive Brokers) או Alpaca.

• No Order Management:
  אתה לא פגע בכל מערכת. זה דח בהחלטה - not a controller.
  בפנט, אתה תעד:
  
    class OrderExecutor:
        def __init__(self, broker_api):
            self.broker = broker_api
        def enter(self, ticker, size, price=None):
            # actual order
        def exit(self, ticker, order_id):
            # close order

• No Risk Management:
  אין position-level stops.
  אין portfolio-level max loss.
  אין correlation check (כמה from tickers move together).
  
  Add:
  
    def check_correlation_risk(self, open_positions, threshold=0.7):
        corr = np.corrcoef([pos.returns for pos in open_positions])
        if np.max(corr[np.triu_indices_from(corr, k=1)]) > threshold:
            raise RiskLimitExceeded()

• No Logging:
  אתה לא חוזה trades ל-CSV או DB.
  בפנט, תעד הכל כדי להנתח בבתר כדי.

• No Alerts:
  רק בת-סוק עדכונים שהמודל מוד.
  חשבון: Telegram bot / email אנו וידע אתה השיגנל.


8️⃣ HEBREW LANGUAGE EXECUTION
═══════════════════════════════════════════════════════════════
✅ מעולה:

• UI כולה בעברית, right-to-left CSS proper.
• Alert messages ותיוג ברור.
• Wyckoff explanation ספציפית וקרובה למודל האמתי.

No qualms.


9️⃣ RECOMMENDATIONS FOR NEXT ITERATION
═════════════════════════════════════════════════════════════════

Priority 1 (Critical):
  ☐ Add input validation (Nans, gaps, missing data)
  ☐ Add overfitting check (train/test gap warning)
  ☐ Add CV scores during training
  ☐ Document multi-collinearity (or run VIF)

Priority 2 (Important):
  ☐ Factor importance aggregation (which ones matter most?)
  ☐ Survivorship bias mitigation
  ☐ Regime filter for backtest (no signals in bear markets)
  ☐ Real-time data integration (Alpaca/IB)

Priority 3 (Nice-to-Have):
  ☐ Position correlation risk check
  ☐ Order management skeleton
  ☐ Trade logging to CSV
  ☐ Telegram alerts for signals
  ☐ Retrain scheduler


🔟 BOTTOM LINE
═════════════════════════════════════════════════════════════════

הקוד שלך חזק, מעוצב היטב, וקריא.

המטבח שלך (35 factors, Wyckoff logic, RandomForest) זה
מוקדש ומעניין. אפילו אם accuracy היא 56-58%, זה סרט טוב
כי אתה בונה על בסיס תיאורטי (VSA, absorption) לא רק
numerical curve-fitting.

הבעיה העיקרית היא ש-דעת לעצור קדימה ל-הפקה (real money).
צריך:
  • Proper risk limits
  • Real-time execution
  • Portfolio constraints
  • Continuous retraining

זה סימן שאתה צריך שני דברים:
  1. Risk Manager object
  2. Broker API adapter

אבל למטרת research? זה מצוין.
כדי ליישם זה, זה בדקה 2-3 ימי עבודה של refactoring.

סימן טוב שהקוד בנוי מספיק טוב להרחבה.

✅ תודה על הקוד, תיקיתו טוב מכל הכיוונים.
"""

# ==============================================================
# דוגמה: איך להוסיף בדיקת overfitting
# ==============================================================

def check_overfitting_example(train_acc, test_acc, threshold=0.15):
    """
    Overfitting Risk Assessment
    
    בדיקה פשוטה: אם ה gap בין train ל-test גדול מדי,
    זה סימן שהמודל עשה memorize על train set.
    """
    gap = train_acc - test_acc
    
    if gap > threshold:
        return {
            "status": "HIGH RISK",
            "gap": gap,
            "recommendation": "Model may be overfit. Try: reducing max_depth, increasing min_samples_split, or more training data"
        }
    elif gap > 0.08:
        return {
            "status": "MODERATE",
            "gap": gap,
            "recommendation": "Some overfitting signs. Monitor closely."
        }
    else:
        return {
            "status": "GOOD",
            "gap": gap,
            "recommendation": "Model generalizing well."
        }

# Example:
# result = check_overfitting_example(0.68, 0.56)
# print(result)
# => HIGH RISK (gap = 0.12)


# ==============================================================
# דוגמה: איך להוסיף VIF (Variance Inflation Factor)
# ==============================================================

def compute_vif_example(factors_df):
    """
    Multi-Collinearity Check
    
    VIF > 5 = בעיה אמתית. אתה צריך להפחית את מספר הפקטורים.
    VIF 2-5 = סביר. רוב ה-factors יהיו כאן.
    VIF < 2 = אין כמעט correlation.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["factor"] = factors_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(factors_df.values, i)
        for i in range(factors_df.shape[1])
    ]
    
    return vif_data.sort_values("VIF", ascending=False)

# Example:
# vif = compute_vif_example(factors)
# problematic = vif[vif["VIF"] > 5]
# print(f"⚠️ High VIF factors: {problematic['factor'].tolist()}")


# ==============================================================
# סיכום הערות בטבלה
# ==============================================================

REVIEW_SUMMARY = {
    "Architecture": {
        "score": "9/10",
        "comment": "Clean separation, easy to extend"
    },
    "Factor Engineering": {
        "score": "8/10",
        "comment": "35 factors solid, but check for multicollinearity"
    },
    "Backtest Logic": {
        "score": "8/10",
        "comment": "Walk-forward good, but needs regime filter & position management"
    },
    "ML Implementation": {
        "score": "7/10",
        "comment": "Works well, needs CV & overfitting checks"
    },
    "UI/UX": {
        "score": "9/10",
        "comment": "Clean, intuitive, good Hebrew integration"
    },
    "Production Ready": {
        "score": "4/10",
        "comment": "Research quality. Add real-time data, risk management, logging"
    },
    "Overall": {
        "score": "7.5/10",
        "comment": "Solid research tool. Strong foundation. Need hardening for live trading."
    }
}

if __name__ == "__main__":
    print("\n" + "="*70)
    print("INSTITUTIONAL SCOUT PRO V3 - REVIEW SUMMARY")
    print("="*70 + "\n")
    for aspect, details in REVIEW_SUMMARY.items():
        print(f"  {aspect:20} | Score: {details['score']:5} | {details['comment']}")
    print("\n" + "="*70)
