def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_accumulation_prob(df, support, resistance):
    last = df.iloc[-1]

    features = []

    # Feature 1: position in range
    pos = (last["Close"] - support) / (resistance - support + 1e-9)
    features.append(1 - pos)  # closer to support = better

    # Feature 2: volume spike
    vol = last["Volume"] / (df["Volume"].rolling(20).mean().iloc[-1] + 1e-9)
    features.append(min(vol / 2, 1))

    # Feature 3: compression
    compression = 1 - (last["ATR_NORM"] / (df["ATR_NORM"].rolling(50).mean().iloc[-1] + 1e-9))
    features.append(np.clip(compression, 0, 1))

    # Feature 4: trend weakness (accumulation prefers non-trending / weak downtrend)
    trend = last["Close"] / (last["SMA50"] + 1e-9)
    trend_score = 1 - min(trend - 1, 1)
    features.append(np.clip(trend_score, 0, 1))

    # weighted sum (learnable later)
    weights = np.array([0.35, 0.25, 0.25, 0.15])

    raw_score = np.dot(weights, features)

    # convert to probability space
    prob = sigmoid((raw_score - 0.5) * 6)

    return float(prob * 100)