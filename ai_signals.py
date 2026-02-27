"""
AI/ML signals: regime detection, DCF valuation, badges, coaching.

ML components:
- Random Forest classifier with TimeSeriesSplit cross-validation for trading signals
- Hidden Markov Model (HMM) for 3-state regime detection (bull/sideways/bear)
- VADER sentiment scoring as offline fallback for news headlines
- Composite signal blending RF (50%) + HMM (30%) + sentiment (20%)
"""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


def _detect_regime_sma(price_history):
    """Fallback: Detect market regime using SMA 20/50 crossover."""
    default = {"regime": "Neutral", "signal_strength": 0.0, "confidence": "Low",
               "sma20": 0.0, "sma50": 0.0, "description": "Insufficient data.",
               "model_type": "SMA"}
    if price_history.empty or "Close" not in price_history.columns:
        return default
    close = price_history["Close"]
    if len(close) < 50:
        return default

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])
    price = float(close.iloc[-1])
    spread = (s20 - s50) / s50 * 100

    if s20 > s50 and price > s20:
        regime, desc = "Bullish", f"SMA20 (${s20:.2f}) above SMA50 (${s50:.2f}), price above both — uptrend confirmed."
    elif s20 < s50 and price < s20:
        regime, desc = "Bearish", f"SMA20 (${s20:.2f}) below SMA50 (${s50:.2f}), price below both — downtrend confirmed."
    elif s20 > s50:
        regime, desc = "Bullish", "SMA20 above SMA50 but price near SMA — weakening uptrend."
    elif s20 < s50:
        regime, desc = "Bearish", "SMA20 below SMA50 but price near SMA — weakening downtrend."
    else:
        regime, desc = "Neutral", "SMAs converging — no clear trend."

    strength = min(abs(spread) / 5.0, 1.0)
    if abs(spread) > 2 and ((regime == "Bullish" and price > s20) or (regime == "Bearish" and price < s20)):
        conf = "High"
    elif abs(spread) > 1:
        conf = "Medium"
    else:
        conf = "Low"

    return {"regime": regime, "signal_strength": round(strength, 2), "confidence": conf,
            "sma20": round(s20, 2), "sma50": round(s50, 2), "description": desc,
            "model_type": "SMA"}


def _build_ml_features(close, volume=None, high=None, low=None):
    """Build expanded feature matrix for ML signal generation.

    10 features engineered from price/volume data:
    - RSI(14): Momentum oscillator (mean-reversion signal)
    - MACD histogram: Trend-following momentum
    - Bollinger %B: Volatility-relative position (0-1 range)
    - 5-day return: Short-term momentum
    - 20-day return: Medium-term momentum
    - Volume ratio: Current vs 20-day avg (unusual activity detector)
    - ATR(14): Average True Range (volatility context)
    - Price vs SMA50: Trend position relative to 50-day average
    - Price vs SMA200: Long-term trend position
    - SMA 20/50 spread: Moving average convergence/divergence
    """
    df = pd.DataFrame(index=close.index)

    # RSI(14) using exponential moving average (Wilder's method)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # Bollinger %B (clamped to avoid extreme outliers)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    df["bb_pct_b"] = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)

    # Short and medium-term momentum
    df["momentum_5"] = close.pct_change(5) * 100
    df["momentum_20"] = close.pct_change(20) * 100

    # Volume ratio: current volume / 20-day average volume
    if volume is not None and len(volume) == len(close):
        vol_avg = volume.rolling(20).mean()
        df["vol_ratio"] = np.where(vol_avg > 0, volume / vol_avg, 1.0)
    else:
        df["vol_ratio"] = 1.0

    # ATR(14): Average True Range for volatility context
    if high is not None and low is not None:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean() / close * 100  # Normalize as % of price
    else:
        # Approximate ATR from close-only data using daily range proxy
        df["atr"] = close.pct_change().abs().rolling(14).mean() * 100

    # Price relative to SMA50 and SMA200
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["price_vs_sma50"] = np.where(sma50 > 0, (close - sma50) / sma50 * 100, 0)
    df["price_vs_sma200"] = np.where(sma200 > 0, (close - sma200) / sma200 * 100, 0)

    # SMA 20/50 spread (trend convergence)
    df["sma_spread"] = np.where(sma50 > 0, (sma20 - sma50) / sma50 * 100, 0)

    return df


def _generate_labels(close, forward_days=10, bull_threshold=1.0, bear_threshold=-1.0):
    """Self-supervised labels from forward returns.

    Looks 10 days ahead to create training labels:
    - >+1% forward return → "Bullish"
    - <-1% forward return → "Bearish"
    - Otherwise → "Neutral"

    Note: shift(-forward_days) prevents look-ahead bias during training
    by only using labels that would have been knowable at prediction time
    when combined with TimeSeriesSplit cross-validation.
    """
    fwd_ret = close.pct_change(forward_days).shift(-forward_days) * 100
    labels = pd.Series("Neutral", index=close.index)
    labels[fwd_ret > bull_threshold] = "Bullish"
    labels[fwd_ret < bear_threshold] = "Bearish"
    return labels


def detect_market_regime(price_history):
    """Detect market regime using ML Random Forest with TimeSeriesSplit validation.

    Trains a Random Forest (200 trees, depth 8) on 10 technical features using
    proper time-series cross-validation (no look-ahead bias). Reports out-of-sample
    accuracy alongside the prediction.

    Falls back to SMA crossover when sklearn is unavailable or data < 100 points.
    """
    default = {"regime": "Neutral", "signal_strength": 0.0, "confidence": "Low",
               "sma20": 0.0, "sma50": 0.0, "description": "Insufficient data.",
               "model_type": "N/A"}
    if price_history.empty or "Close" not in price_history.columns:
        return default

    close = price_history["Close"]

    # Need sklearn and at least 100 data points for ML
    if not SKLEARN_AVAILABLE or len(close) < 100:
        return _detect_regime_sma(price_history)

    try:
        volume = price_history.get("Volume")
        high = price_history.get("High")
        low = price_history.get("Low")
        features_df = _build_ml_features(close, volume, high, low)
        labels = _generate_labels(close)

        # Combine and drop NaN rows
        combined = features_df.copy()
        combined["label"] = labels
        combined = combined.dropna()

        # Need at least some Bullish and Bearish labels
        label_counts = combined["label"].value_counts()
        has_bull = label_counts.get("Bullish", 0) >= 5
        has_bear = label_counts.get("Bearish", 0) >= 5

        if not has_bull or not has_bear:
            return _detect_regime_sma(price_history)

        feature_cols = [c for c in features_df.columns]
        X = combined[feature_cols]
        y = combined["label"]

        if len(X) < 50:
            return _detect_regime_sma(price_history)

        # TimeSeriesSplit cross-validation (no look-ahead bias)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            cv_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
            cv_clf.fit(X_tr, y_tr)
            cv_scores.append(cv_clf.score(X_te, y_te))
        avg_cv_accuracy = np.mean(cv_scores)

        # Train final model on all data except last point
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        clf.fit(X.iloc[:-1], y.iloc[:-1])

        # Predict on most recent data point
        X_latest = X.iloc[[-1]]
        prediction = clf.predict(X_latest)[0]
        probabilities = clf.predict_proba(X_latest)[0]
        class_labels = clf.classes_

        # Get probability for predicted class
        pred_idx = list(class_labels).index(prediction)
        prob = probabilities[pred_idx]

        # Feature importance
        importance = dict(zip(feature_cols, clf.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

        # Confidence from probability
        if prob >= 0.6:
            conf = "High"
        elif prob >= 0.4:
            conf = "Medium"
        else:
            conf = "Low"

        # SMA values for display
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])

        # Build description
        feature_labels = {
            "rsi": "RSI", "macd_hist": "MACD", "bb_pct_b": "Bollinger %B",
            "momentum_5": "5d Momentum", "momentum_20": "20d Momentum",
            "vol_ratio": "Volume Ratio", "atr": "ATR",
            "price_vs_sma50": "Price/SMA50", "price_vs_sma200": "Price/SMA200",
            "sma_spread": "SMA Spread",
        }
        top_str = ", ".join(f"{feature_labels.get(f, f)} ({v:.0%})" for f, v in top_features)
        desc = (f"ML model predicts {prediction} with {prob:.0%} probability. "
                f"CV accuracy: {avg_cv_accuracy:.0%}. Top drivers: {top_str}.")

        return {
            "regime": prediction,
            "signal_strength": round(prob, 2),
            "confidence": conf,
            "sma20": round(s20, 2),
            "sma50": round(s50, 2),
            "description": desc,
            "model_type": "ML",
            "feature_importance": importance,
            "cv_accuracy": round(avg_cv_accuracy, 3),
        }

    except Exception:
        return _detect_regime_sma(price_history)


def detect_regime_hmm(price_history):
    """Detect market regime using a 3-state Hidden Markov Model.

    Models the market as switching between 3 hidden states based on
    daily returns and rolling volatility. States are labeled by sorting
    on mean return: lowest = Bear, middle = Sideways, highest = Bull.

    HMMs capture regime persistence — if today is bull, tomorrow is
    likely bull too — which SMA crossovers cannot model.

    Returns:
        Dict with regime, probabilities per state, state history, and model details.
    """
    default = {"regime": "Neutral", "probabilities": {"Bull": 0.33, "Sideways": 0.34, "Bear": 0.33},
               "description": "Insufficient data for HMM.", "model_type": "N/A"}

    if not HMM_AVAILABLE:
        return default
    if price_history.empty or "Close" not in price_history.columns:
        return default

    close = price_history["Close"]
    if len(close) < 60:
        return default

    try:
        # Features: daily returns and 20-day rolling volatility
        returns = close.pct_change().dropna()
        volatility = returns.rolling(20).std().dropna()

        # Align series
        common_idx = returns.index.intersection(volatility.index)
        if len(common_idx) < 40:
            return default

        features = np.column_stack([
            returns.loc[common_idx].values,
            volatility.loc[common_idx].values,
        ])

        # Fit 3-state Gaussian HMM (diag covariance is more numerically stable)
        model = GaussianHMM(
            n_components=3, covariance_type="diag",
            n_iter=100, random_state=42,
        )
        model.fit(features)
        hidden_states = model.predict(features)
        state_probs = model.predict_proba(features)

        # Label states by mean return: lowest=Bear, middle=Sideways, highest=Bull
        state_means = []
        for i in range(3):
            mask = hidden_states == i
            if mask.sum() > 0:
                state_means.append(features[mask, 0].mean())
            else:
                state_means.append(0.0)
        sorted_states = np.argsort(state_means)  # ascending: Bear, Sideways, Bull
        label_map = {sorted_states[0]: "Bear", sorted_states[1]: "Sideways", sorted_states[2]: "Bull"}

        # Current state (last observation)
        current_state = hidden_states[-1]
        current_label = label_map[current_state]
        current_probs = state_probs[-1]

        # Map probabilities to labeled states
        prob_dict = {}
        for state_idx, label in label_map.items():
            prob_dict[label] = round(float(current_probs[state_idx]), 3)

        # Build state history for visualization (map to labeled regimes)
        state_history = pd.Series(
            [label_map[s] for s in hidden_states],
            index=common_idx,
        )

        desc = (f"HMM detects {current_label} regime — "
                f"Bull: {prob_dict['Bull']:.0%}, Sideways: {prob_dict['Sideways']:.0%}, "
                f"Bear: {prob_dict['Bear']:.0%}.")

        return {
            "regime": current_label,
            "probabilities": prob_dict,
            "state_history": state_history,
            "description": desc,
            "model_type": "HMM",
        }

    except Exception:
        return default


def score_sentiment_vader(headlines):
    """Score financial headlines using VADER sentiment (offline, deterministic).

    VADER is rule-based — no API key needed, zero latency, always available.
    Returns a normalized score from -1.0 (bearish) to +1.0 (bullish).

    Args:
        headlines: List of headline strings.

    Returns:
        Dict with overall score (-1 to 1), label (Bullish/Bearish/Neutral),
        and per-headline scores.
    """
    if not VADER_AVAILABLE or not headlines:
        return {"score": 0.0, "label": "Neutral", "details": []}

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    details = []
    for headline in headlines:
        vs = analyzer.polarity_scores(str(headline))
        compound = vs["compound"]  # -1 to +1
        scores.append(compound)
        details.append({
            "headline": headline,
            "score": round(compound, 3),
            "label": "Bullish" if compound > 0.05 else "Bearish" if compound < -0.05 else "Neutral",
        })

    avg_score = np.mean(scores) if scores else 0.0

    if avg_score > 0.05:
        label = "Bullish"
    elif avg_score < -0.05:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "score": round(float(avg_score), 3),
        "label": label,
        "details": details,
    }


def generate_composite_signal(price_history, headlines=None):
    """Generate a composite ML trading signal blending three components.

    Weights: 50% Random Forest signal + 30% HMM regime + 20% VADER sentiment.
    Each component is scored -1.0 (bearish) to +1.0 (bullish), then blended.

    Falls back gracefully if any component is unavailable — remaining
    components are re-weighted to sum to 1.0.

    Args:
        price_history: DataFrame with Close, Volume, High, Low columns.
        headlines: Optional list of news headline strings for sentiment.

    Returns:
        Dict with composite signal, individual component scores, and breakdown.
    """
    components = {}
    weights = {"rf": 0.5, "hmm": 0.3, "sentiment": 0.2}

    # Component 1: Random Forest ML signal
    rf_result = detect_market_regime(price_history)
    if rf_result["model_type"] == "ML":
        # Map regime + probability to a -1 to +1 score
        prob = rf_result["signal_strength"]
        if rf_result["regime"] == "Bullish":
            rf_score = prob
        elif rf_result["regime"] == "Bearish":
            rf_score = -prob
        else:
            rf_score = 0.0
        components["rf"] = {
            "score": round(rf_score, 3),
            "label": rf_result["regime"],
            "confidence": rf_result["confidence"],
            "description": rf_result["description"],
            "cv_accuracy": rf_result.get("cv_accuracy", 0),
            "feature_importance": rf_result.get("feature_importance", {}),
        }
    else:
        # SMA fallback — lower weight
        weights["rf"] = 0.2
        if rf_result["regime"] == "Bullish":
            rf_score = rf_result["signal_strength"]
        elif rf_result["regime"] == "Bearish":
            rf_score = -rf_result["signal_strength"]
        else:
            rf_score = 0.0
        components["rf"] = {
            "score": round(rf_score, 3),
            "label": rf_result["regime"],
            "confidence": rf_result["confidence"],
            "description": rf_result["description"] + " (SMA fallback)",
        }

    # Component 2: HMM regime detection
    hmm_result = detect_regime_hmm(price_history)
    if hmm_result["model_type"] == "HMM":
        probs = hmm_result["probabilities"]
        # Score = P(Bull) - P(Bear), range -1 to +1
        hmm_score = probs.get("Bull", 0) - probs.get("Bear", 0)
        components["hmm"] = {
            "score": round(hmm_score, 3),
            "label": hmm_result["regime"],
            "probabilities": probs,
            "description": hmm_result["description"],
        }
    else:
        weights["hmm"] = 0.0
        components["hmm"] = {"score": 0.0, "label": "N/A", "description": "HMM unavailable."}

    # Component 3: VADER sentiment
    if headlines:
        vader_result = score_sentiment_vader(headlines)
        # VADER compound score is already -1 to +1
        components["sentiment"] = {
            "score": vader_result["score"],
            "label": vader_result["label"],
            "num_headlines": len(headlines),
            "description": f"VADER: {vader_result['label']} ({vader_result['score']:+.2f}) from {len(headlines)} headlines.",
        }
    else:
        weights["sentiment"] = 0.0
        components["sentiment"] = {"score": 0.0, "label": "N/A", "description": "No headlines available."}

    # Normalize weights for available components
    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        norm_weights = {"rf": 1.0, "hmm": 0.0, "sentiment": 0.0}

    # Compute weighted composite score
    composite_score = sum(
        norm_weights[k] * components[k]["score"] for k in weights
    )

    # Map composite score to signal label
    if composite_score > 0.15:
        signal = "Bullish"
    elif composite_score < -0.15:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Confidence from score magnitude
    abs_score = abs(composite_score)
    if abs_score > 0.4:
        confidence = "High"
    elif abs_score > 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "signal": signal,
        "score": round(composite_score, 3),
        "confidence": confidence,
        "components": components,
        "weights": {k: round(v, 2) for k, v in norm_weights.items()},
        "description": (
            f"Composite signal: {signal} ({composite_score:+.2f}). "
            f"RF: {components['rf']['score']:+.2f} ({norm_weights['rf']:.0%}), "
            f"HMM: {components['hmm']['score']:+.2f} ({norm_weights['hmm']:.0%}), "
            f"Sentiment: {components['sentiment']['score']:+.2f} ({norm_weights['sentiment']:.0%})."
        ),
    }


def _calculate_beta(price_history, benchmark_ticker="SPY"):
    """Calculate stock beta vs benchmark from overlapping daily returns.

    Uses 1-year of daily returns when available (minimum 60 days).
    Beta = Cov(stock, market) / Var(market).

    Args:
        price_history: DataFrame with 'Close' column for the stock.
        benchmark_ticker: Benchmark symbol (default SPY).

    Returns:
        Float beta value, or 1.0 if calculation fails.
    """
    try:
        import yfinance as yf
        if price_history is None or len(price_history) < 60:
            return 1.0
        spy = yf.download(benchmark_ticker, period="1y", progress=False)
        if spy.empty or len(spy) < 60:
            return 1.0
        # Handle multi-level columns from yfinance
        if hasattr(spy.columns, 'levels') and len(spy.columns.levels) > 1:
            spy.columns = spy.columns.droplevel(1)
        stock_close = price_history["Close"].copy()
        spy_close = spy["Close"].copy()
        # Align on dates (tz-naive)
        stock_close.index = pd.to_datetime(stock_close.index).tz_localize(None)
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
        merged = pd.DataFrame({"stock": stock_close, "spy": spy_close}).dropna()
        if len(merged) < 60:
            return 1.0
        stock_ret = merged["stock"].pct_change().dropna()
        spy_ret = merged["spy"].pct_change().dropna()
        cov = np.cov(stock_ret, spy_ret)
        if cov[1, 1] == 0:
            return 1.0
        beta = cov[0, 1] / cov[1, 1]
        return round(max(0.1, min(3.0, beta)), 2)  # Cap at [0.1, 3.0]
    except Exception:
        return 1.0


def _compute_custom_dcf(financial_data, stock_price, shares_outstanding,
                        price_history=None):
    """Compute intrinsic value using a discounted cash flow model.

    Steps:
        1. Extract historical FCF from cash flow statements
        2. Estimate growth rate from revenue + FCF trends (capped 2-20%)
        3. Calculate beta from stock vs SPY returns (CAPM)
        4. Get effective tax rate from income statement
        5. Calculate WACC from CAPM (cost of equity) + cost of debt
        6. Project FCF for 5 years, compute terminal value (perpetuity growth)
        7. Discount to present, subtract net debt, divide by shares

    Returns:
        Dict with dcf_per_share, wacc, growth_rate, beta, etc., or None on failure.
    """
    try:
        cashflows = financial_data.get("cashflow", [])
        income = financial_data.get("income", [])
        balance = financial_data.get("balance", [])

        if not cashflows or len(cashflows) < 2:
            return None

        # Extract historical FCF (operating cash flow - capex)
        fcf_list = []
        for cf in cashflows:
            ocf = cf.get("operatingCashFlow", 0) or 0
            capex = abs(cf.get("capitalExpenditure", 0) or 0)
            fcf = ocf - capex
            if fcf != 0:
                fcf_list.append(fcf)

        if len(fcf_list) < 2 or fcf_list[0] <= 0:
            return None

        # Estimate growth rate from FCF + revenue trends
        fcf_growth_rates = []
        for i in range(len(fcf_list) - 1):
            if fcf_list[i + 1] > 0:
                fcf_growth_rates.append((fcf_list[i] - fcf_list[i + 1]) / abs(fcf_list[i + 1]))

        rev_growth_rates = []
        for i in range(len(income) - 1):
            r0 = income[i].get("revenue", 0) or 0
            r1 = income[i + 1].get("revenue", 0) or 0
            if r1 > 0:
                rev_growth_rates.append((r0 - r1) / r1)

        # Blend FCF and revenue growth, cap between 2-20%
        all_rates = fcf_growth_rates + rev_growth_rates
        if all_rates:
            avg_growth = np.median(all_rates)
        else:
            avg_growth = 0.05
        growth_rate = max(0.02, min(0.20, avg_growth))

        # ── WACC estimation (Weighted Average Cost of Capital) ──
        # WACC = E/(E+D) × Ke + D/(E+D) × Kd × (1 − T)
        # where Ke = cost of equity (CAPM), Kd = cost of debt, T = tax rate

        # Cost of equity via CAPM: Ke = Rf + β × (Rm − Rf)
        # Rf = risk-free rate (10Y Treasury), Rm-Rf = equity risk premium
        risk_free = 0.043  # ~10Y Treasury yield (updated periodically)
        market_premium = 0.06  # Long-run equity risk premium (Damodaran estimate)
        beta = _calculate_beta(price_history)  # Computed from stock vs SPY covariance

        cost_of_equity = risk_free + beta * market_premium

        # Cost of debt: Kd = interest expense / total debt (implied yield)
        cost_of_debt = 0.05  # Default if data unavailable
        total_debt = 0
        if balance and balance[0]:
            total_debt = (balance[0].get("totalDebt", 0) or 0)
            interest = abs(income[0].get("interestExpense", 0) or 0) if income else 0
            if total_debt > 0 and interest > 0:
                cost_of_debt = min(interest / total_debt, 0.15)  # Cap at 15%

        # Effective tax rate: T = income tax expense / pre-tax income
        # Fallback: 21% US statutory corporate rate
        tax_rate = 0.21
        if income and income[0]:
            pretax = income[0].get("incomeBeforeTax", 0) or 0
            tax_expense = income[0].get("incomeTaxExpense", 0) or 0
            if pretax > 0 and tax_expense > 0:
                effective_tax = tax_expense / pretax
                tax_rate = max(0.0, min(0.40, effective_tax))  # Cap [0%, 40%]

        # Capital structure weights: E/(E+D) and D/(E+D)
        equity_value = stock_price * shares_outstanding if shares_outstanding > 0 else 0
        debt_value = total_debt
        total_capital = equity_value + debt_value

        if total_capital > 0:
            equity_weight = equity_value / total_capital
            debt_weight = debt_value / total_capital
        else:
            equity_weight, debt_weight = 0.8, 0.2  # Assume 80/20 if data missing

        # Final WACC with tax shield on debt
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
        wacc = max(0.06, min(0.20, wacc))  # Sanity bounds to avoid extreme valuations

        # ── Stage 1: Project FCF for 5 years ──
        # FCF_t = base_FCF × (1 + g)^t
        base_fcf = fcf_list[0]
        projected_fcf = []
        for yr in range(1, 6):
            projected_fcf.append(base_fcf * (1 + growth_rate) ** yr)

        # ── Stage 2: Terminal value (Gordon Growth Model) ──
        # TV = FCF_5 × (1 + g_terminal) / (WACC − g_terminal)
        # Assumes company grows at GDP-like rate (2.5%) in perpetuity after year 5
        terminal_growth = 0.025
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)

        # ── Stage 3: Discount all cash flows to present value ──
        # PV = Σ FCF_t / (1 + WACC)^t  +  TV / (1 + WACC)^5
        pv_fcf = sum(fcf / (1 + wacc) ** yr for yr, fcf in enumerate(projected_fcf, 1))
        pv_terminal = terminal_value / (1 + wacc) ** 5

        enterprise_value = pv_fcf + pv_terminal

        # ── Stage 4: Enterprise → Equity value ──
        # Equity = Enterprise Value − Net Debt
        net_debt = debt_value - (balance[0].get("cashAndCashEquivalents", 0) or 0) if balance else 0
        equity_value_dcf = enterprise_value - net_debt

        if shares_outstanding > 0:
            dcf_per_share = equity_value_dcf / shares_outstanding
        else:
            return None

        return {
            "dcf_per_share": round(dcf_per_share, 2),
            "wacc": round(wacc * 100, 1),
            "growth_rate": round(growth_rate * 100, 1),
            "base_fcf": round(base_fcf / 1e6, 1),
            "terminal_growth": 2.5,
            "beta": beta,
            "tax_rate": round(tax_rate * 100, 1),
            "cost_of_equity": round(cost_of_equity * 100, 1),
            "cost_of_debt": round(cost_of_debt * 100, 1),
        }

    except Exception:
        return None


def compute_dcf_sensitivity(dcf_result, shares_outstanding, net_debt):
    """Compute DCF per share across a grid of growth rates and discount rates.

    Takes the base DCF result and re-runs the projection for 7 growth rate values
    and 7 WACC values centered around the base-case parameters.

    Args:
        dcf_result: Dict from _compute_custom_dcf() with base_fcf, growth_rate, wacc.
        shares_outstanding: Number of shares outstanding.
        net_debt: Net debt (total debt - cash).

    Returns:
        Dict with growth_rates, discount_rates, and values (7x7 matrix), or None.
    """
    if not dcf_result or shares_outstanding <= 0:
        return None
    try:
        base_fcf = dcf_result["base_fcf"] * 1e6  # Convert back from millions
        base_growth = dcf_result["growth_rate"] / 100
        base_wacc = dcf_result["wacc"] / 100

        # Build 7 growth rates centered on base (2% steps)
        growth_rates = sorted(set(
            max(0.01, min(0.25, base_growth + delta))
            for delta in [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]
        ))
        # Build 7 discount rates centered on base (1% steps)
        discount_rates = sorted(set(
            max(0.05, min(0.20, base_wacc + delta))
            for delta in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
        ))

        terminal_growth = 0.025
        values = []
        for g in growth_rates:
            row = []
            for d in discount_rates:
                projected = [base_fcf * (1 + g) ** yr for yr in range(1, 6)]
                if d <= terminal_growth:
                    row.append(0)
                    continue
                tv = projected[-1] * (1 + terminal_growth) / (d - terminal_growth)
                pv = sum(f / (1 + d) ** yr for yr, f in enumerate(projected, 1))
                pv_tv = tv / (1 + d) ** 5
                equity_val = pv + pv_tv - net_debt
                row.append(round(equity_val / shares_outstanding, 2))
            values.append(row)

        return {
            "growth_rates": [round(g * 100, 1) for g in growth_rates],
            "discount_rates": [round(d * 100, 1) for d in discount_rates],
            "values": values,
        }
    except Exception:
        return None


def analyze_valuation(analyst_data, financial_data=None, stock_price=None,
                      shares_outstanding=None, price_history=None):
    """Analyze stock valuation. Tries custom DCF first, falls back to FMP.

    Args:
        analyst_data: Dict from get_analyst_data() (FMP estimates + DCF).
        financial_data: Dict from get_financial_data() (for custom DCF).
        stock_price: Current stock price.
        shares_outstanding: Number of shares outstanding.
        price_history: DataFrame with Close prices (for beta calculation).
    """
    default = {"signal": "N/A", "dcf": 0, "stock_price": 0,
               "margin_of_safety": 0, "revenue_growth": 0,
               "description": "No analyst data.", "model_type": "N/A"}

    # Try custom DCF first
    if financial_data and stock_price and shares_outstanding:
        custom = _compute_custom_dcf(financial_data, stock_price,
                                     shares_outstanding, price_history)
        if custom and custom["dcf_per_share"] > 0:
            dcf = custom["dcf_per_share"]
            margin = (dcf - stock_price) / stock_price * 100

            if margin > 15:
                signal = "Undervalued"
                desc = f"Custom DCF ${dcf:,.0f} is {margin:+.1f}% above price ${stock_price:,.2f} — potential upside."
            elif margin < -15:
                signal = "Overvalued"
                desc = f"Custom DCF ${dcf:,.0f} is {margin:+.1f}% below price ${stock_price:,.2f} — priced above intrinsic."
            else:
                signal = "Fair Value"
                desc = f"Custom DCF ${dcf:,.0f} within 15% of price ${stock_price:,.2f} — reasonably valued."

            desc += (f" WACC: {custom['wacc']}%, Growth: {custom['growth_rate']}%,"
                     f" Beta: {custom['beta']}, Tax: {custom['tax_rate']}%.")

            # Compute sensitivity matrix
            balance = financial_data.get("balance", [])
            debt_val = (balance[0].get("totalDebt", 0) or 0) if balance else 0
            cash_val = (balance[0].get("cashAndCashEquivalents", 0) or 0) if balance else 0
            net_debt = debt_val - cash_val
            sensitivity = compute_dcf_sensitivity(custom, shares_outstanding, net_debt)

            return {
                "signal": signal,
                "dcf": round(dcf, 2),
                "stock_price": round(stock_price, 2),
                "margin_of_safety": round(margin, 1),
                "revenue_growth": round(custom["growth_rate"], 1),
                "description": desc,
                "model_type": "Custom DCF",
                "wacc": custom["wacc"],
                "growth_rate": custom["growth_rate"],
                "beta": custom["beta"],
                "tax_rate": custom["tax_rate"],
                "cost_of_equity": custom["cost_of_equity"],
                "cost_of_debt": custom["cost_of_debt"],
                "sensitivity": sensitivity,
            }

    # Fallback to FMP pre-computed DCF
    if not analyst_data:
        return default
    dcf = analyst_data.get("dcf", 0)
    sp = analyst_data.get("stock_price", 0) or (stock_price or 0)
    estimates = analyst_data.get("estimates", [])
    if not dcf or not sp:
        return default

    margin = (dcf - sp) / sp * 100
    revenue_growth = 0
    if len(estimates) >= 2:
        r0, r1 = estimates[0].get("revenueAvg", 0), estimates[1].get("revenueAvg", 0)
        if r0 and r1:
            revenue_growth = (r0 - r1) / r1 * 100

    if margin > 15:
        signal, desc = "Undervalued", f"DCF ${dcf:,.0f} is {margin:+.1f}% above price ${sp:,.2f} — potential upside."
    elif margin < -15:
        signal, desc = "Overvalued", f"DCF ${dcf:,.0f} is {margin:+.1f}% below price ${sp:,.2f} — priced above intrinsic."
    else:
        signal, desc = "Fair Value", f"DCF ${dcf:,.0f} within 15% of price ${sp:,.2f} — reasonably valued."
    if revenue_growth:
        desc += f" Revenue growth: {revenue_growth:+.1f}%."

    return {"signal": signal, "dcf": round(dcf, 2), "stock_price": round(sp, 2),
            "margin_of_safety": round(margin, 1), "revenue_growth": round(revenue_growth, 1),
            "description": desc, "model_type": "FMP API"}


def compute_rsi(close, period=14):
    """Relative Strength Index (0-100). >70 overbought, <30 oversold."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close, period=20, std_dev=2):
    """Bollinger Bands: middle, upper, lower."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def compute_option_payoff(spot, strike, premium, opt_type, action, contracts=1):
    """Compute option P/L at expiration across a price range.

    Returns:
        prices (array), pnl (array), breakeven (float), max_profit, max_loss
    """
    prices = np.linspace(spot * 0.5, spot * 1.5, 200)
    multiplier = 100 * contracts

    if opt_type == "call":
        intrinsic = np.maximum(prices - strike, 0)
    else:
        intrinsic = np.maximum(strike - prices, 0)

    if "Buy" in action:
        pnl = (intrinsic - premium) * multiplier
        if opt_type == "call":
            breakeven = strike + premium
        else:
            breakeven = strike - premium
        max_loss = -premium * multiplier
        max_profit = float('inf') if opt_type == "call" else (strike - premium) * multiplier
    else:
        pnl = (premium - intrinsic) * multiplier
        if opt_type == "call":
            breakeven = strike + premium
        else:
            breakeven = strike - premium
        max_profit = premium * multiplier
        max_loss = float('-inf') if opt_type == "call" else -(strike - premium) * multiplier

    return prices, pnl, breakeven, max_profit, max_loss


BADGE_META = {
    "First Trade":    {"icon": "🏁", "desc": "Completed your first trade", "hint": "Place any buy or sell order"},
    "Diversifier":    {"icon": "🌐", "desc": "Holding 5+ different stocks", "hint": "Buy shares in 5 different tickers"},
    "Options Trader": {"icon": "📊", "desc": "Entered the options market", "hint": "Place any options trade"},
    "Risk Manager":   {"icon": "🛡️", "desc": "Used a stop-loss order", "hint": "Place a trade with stop-loss order type"},
    "Consistent":     {"icon": "📅", "desc": "Traded on 5+ different days", "hint": "Make trades across 5 separate days"},
    "Six-Figure Club":{"icon": "💰", "desc": "Portfolio value hit $110K+", "hint": "Grow your account above $110,000"},
    "Watchful":       {"icon": "👁️", "desc": "Tracking 5+ stocks", "hint": "Add 5 stocks to your watchlist"},
}

BADGE_DEFS = [
    ("First Trade", lambda d, **kw: len(d["journal"]) >= 1),
    ("Diversifier", lambda d, **kw: len(set(t["ticker"] for t in d["portfolio"])) >= 5),
    ("Options Trader", lambda d, **kw: len(d["options"]) >= 1),
    ("Risk Manager", lambda d, **kw: any(t.get("order_type") == "stop-loss" for t in d["portfolio"])),
    ("Consistent", lambda d, **kw: len(set(t["date"] for t in d["journal"])) >= 5),
    ("Six-Figure Club", lambda d, **kw: kw.get("portfolio_value", 0) >= 110000),
    ("Watchful", lambda d, **kw: len(d["watchlist"]) >= 5),
]


def check_badges(data, portfolio_value=0):
    """Check which badges the user has earned.

    Args:
        data: Account data dict.
        portfolio_value: Current total portfolio value (cash + stocks + options).

    Returns:
        List of earned badge names.
    """
    earned = []
    for name, check_fn in BADGE_DEFS:
        try:
            if check_fn(data, portfolio_value=portfolio_value):
                earned.append(name)
        except Exception:
            pass
    return earned


def _gather_coaching_context(data, holdings, regime=None, risk_metrics=None,
                             composite_signal=None):
    """Build structured portfolio summary for LLM coaching.

    Args:
        data: Session data dict with cash, journal, badges.
        holdings: Dict of {ticker: {shares, cost}}.
        regime: Optional dict from detect_market_regime() with regime, confidence.
        risk_metrics: Optional dict with var, sharpe, max_drawdown.
        composite_signal: Optional dict from generate_composite_signal().
    """
    total_invested = sum(v["cost"] for v in holdings.values())
    cash = data["cash"]
    cash_pct = cash / (cash + total_invested) * 100 if total_invested > 0 else 100
    journal = data.get("journal", [])
    trade_days = len(set(t["date"] for t in journal))

    # Win/loss stats
    buys, wins, losses = {}, 0, 0
    for t in journal:
        if t["action"] == "buy":
            buys.setdefault(t["ticker"], []).append(t["price"])
        elif t["action"] == "sell" and t["ticker"] in buys and buys[t["ticker"]]:
            bp = buys[t["ticker"]].pop(0)
            if t["price"] >= bp:
                wins += 1
            else:
                losses += 1

    badges = data.get("badges", [])
    tickers = list(holdings.keys())

    lines = [
        f"Holdings: {', '.join(tickers)} ({len(tickers)} positions)",
        f"Cash: ${cash:,.0f} ({cash_pct:.0f}% of account)",
        f"Total trades: {len(journal)} across {trade_days} days",
        f"Wins: {wins}, Losses: {losses}",
        f"Badges earned: {', '.join(badges) if badges else 'None'}",
    ]

    # Enrich with ML regime detection
    if regime and regime.get("regime"):
        conf = regime.get("confidence", "N/A")
        lines.append(f"Market regime: {regime['regime']} (model: {regime.get('model_type', 'N/A')}, confidence: {conf})")

    # Enrich with risk metrics
    if risk_metrics:
        if "var" in risk_metrics:
            var = risk_metrics["var"]
            if isinstance(var, dict):
                lines.append(f"Value at Risk (95%): ${var.get('dollar', 0):,.0f} ({var.get('percent', 0):.1f}%)")
            else:
                lines.append(f"Value at Risk: {var}")
        if "sharpe" in risk_metrics:
            lines.append(f"Sharpe ratio: {risk_metrics['sharpe']:.2f}")
        if "max_drawdown" in risk_metrics:
            lines.append(f"Max drawdown: {risk_metrics['max_drawdown']:.1f}%")

    # Enrich with composite signal
    if composite_signal and composite_signal.get("signal"):
        lines.append(f"Composite ML signal: {composite_signal['signal']} (score: {composite_signal.get('score', 0):+.2f})")

    return "\n".join(lines)


def _get_llm_coaching(context):
    """Get coaching tips from Groq (primary) or Gemini (fallback).

    Returns list of 3 tip strings, or None if both fail.
    """
    import os
    try:
        import streamlit as st
    except ImportError:
        st = None

    def _get_key(name):
        if st:
            try:
                return st.secrets[name]
            except Exception:
                pass
        key = os.environ.get(name)
        if key:
            return key
        try:
            import tomllib
            app_dir = os.path.dirname(os.path.abspath(__file__))
            secrets_path = os.path.join(app_dir, ".streamlit", "secrets.toml")
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
                return secrets.get(name)
        except Exception:
            return None

    prompt = (
        "You are an AI trading coach for a stock portfolio simulator. "
        "Based on the portfolio summary below, provide exactly 3 concise, actionable coaching tips. "
        "Each tip should be 1-2 sentences. Focus on portfolio construction, risk management, and trading discipline. "
        "Return ONLY a JSON array of 3 strings, e.g. [\"tip1\", \"tip2\", \"tip3\"]. No markdown.\n\n"
        f"PORTFOLIO:\n{context}"
    )

    def _parse_tips(text):
        """Parse JSON tips array from LLM response text."""
        import json
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        tips = json.loads(text)
        if isinstance(tips, list) and len(tips) >= 2:
            return tips[:3]
        return None

    # Try OpenAI first (reliable, cheap)
    openai_key = _get_key("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            result = _parse_tips(response.choices[0].message.content)
            if result:
                return result
        except Exception:
            pass

    # Fallback to Groq (free)
    groq_key = _get_key("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            result = _parse_tips(response.choices[0].message.content)
            if result:
                return result
        except Exception:
            pass

    # Last resort: Gemini
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            result = _parse_tips(response.text)
            if result:
                return result
        except Exception:
            pass

    return None


def _rule_based_tips(data, holdings):
    """Fallback: rule-based coaching tips."""
    tips = []
    if not holdings:
        tips.append("Start by making a few trades to build your portfolio.")
        return tips

    if len(holdings) == 1:
        tips.append("Your portfolio has only 1 stock. Consider diversifying across sectors.")
    elif len(holdings) == 2:
        tips.append("Consider adding more positions to reduce concentration risk.")

    total_invested = sum(v["cost"] for v in holdings.values())
    cash = data["cash"]
    if total_invested > 0:
        cash_pct = cash / (cash + total_invested) * 100
        if cash_pct > 80:
            tips.append(f"You have {cash_pct:.0f}% in cash. Consider deploying more capital.")
        elif cash_pct < 5:
            tips.append(f"Only {cash_pct:.0f}% cash remaining. Keep some reserve for opportunities.")

    journal = data.get("journal", [])
    trade_days = len(set(t["date"] for t in journal))
    if len(journal) > 20 and trade_days < 3:
        tips.append("Many trades in few days — avoid overtrading. Quality over quantity.")

    if len(journal) >= 6:
        buys = {}
        losses_streak = 0
        for t in journal:
            if t["action"] == "buy":
                buys.setdefault(t["ticker"], []).append(t["price"])
            elif t["action"] == "sell" and t["ticker"] in buys and buys[t["ticker"]]:
                bp = buys[t["ticker"]].pop(0)
                if t["price"] < bp:
                    losses_streak += 1
                else:
                    losses_streak = 0
        if losses_streak >= 3:
            tips.append("Recent losing streak detected. Review your entry criteria before next trade.")

    if not tips:
        tips.append("Portfolio looks balanced. Keep monitoring your positions.")
    return tips


def generate_coaching_tips(data, holdings, regime=None, risk_metrics=None,
                           composite_signal=None):
    """Generate coaching tips using LLM with rule-based fallback.

    Tries OpenAI first, then Groq, then Gemini, then falls back to static rules.
    Enriches LLM context with regime detection, risk metrics, and composite signal
    when available.

    Returns:
        Tuple of (tips_list, source_str) where source is "LLM" or "Rules".
    """
    if not holdings:
        return (["Start by making a few trades to build your portfolio."], "Rules")

    context = _gather_coaching_context(data, holdings, regime=regime,
                                       risk_metrics=risk_metrics,
                                       composite_signal=composite_signal)
    llm_tips = _get_llm_coaching(context)
    if llm_tips:
        return (llm_tips, "LLM")

    return (_rule_based_tips(data, holdings), "Rules")
