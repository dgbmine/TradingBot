def analyze_wyckoff_deep(df):
    """
    ניתוח וייקוף עמוק: זיהוי Climactic Action ו-Test
    """
    curr = df.iloc[-1]
    prev_vol = df["Volume"].rolling(20).mean().iloc[-1]
    vol_ratio = curr["Volume"] / prev_vol
    price_range = curr["High"] - curr["Low"]
    
    # זיהוי אירועי מפתח
    is_stopping_vol = (vol_ratio > 2.0) and (price_range < df["ATR"].iloc[-1]) # מאמץ גדול, תנועה קטנה = עצירה
    is_spring = (curr["Close"] > df["Low"].rolling(20).min().iloc[-1]) and (curr["Close"] < df["SMA50"].iloc[-1])
    
    analysis = {
        "phase": "Unknown",
        "action": "המתנה לסיגנל איכותי",
        "logic": "לא זוהה דפוס מובהק"
    }

    if is_stopping_vol:
        analysis["phase"] = "Phase A (Stopping Action)"
        analysis["action"] = "המתנה ל-Test של רמת הווליום הגבוה."
        analysis["logic"] = "הכסף החכם בולע את ההיצע/ביקוש."
    elif is_spring:
        analysis["phase"] = "Phase C (Spring/Test)"
        analysis["action"] = "אפשרות ללונג עם סטופ הדוק מתחת לנמוך של ה-Spring."
        analysis["logic"] = "ניעור ידיים חלשות מתחת לתמיכה."
    elif curr["Close"] > df["SMA50"].iloc[-1] and vol_ratio > 1.2:
        analysis["phase"] = "Phase D (Markup)"
        analysis["action"] = "קנייה בתיקונים."
        analysis["logic"] = "המגמה מאושרת בווליום."
        
    return analysis
