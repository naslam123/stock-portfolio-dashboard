"""
AI/ML signals: regime detection, DCF valuation, badges, coaching.
Uses scikit-learn Random Forest for regime detection with SMA fallback.
"""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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


def _build_ml_features(close, volume=None):
    """Build feature matrix for ML regime detection.

    Features: RSI(14), MACD histogram, Bollinger %B, 10-day momentum,
    20-day volume change, SMA20/50 spread.
    """
    df = pd.DataFrame(index=close.index)

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # Bollinger %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)

    # 10-day momentum (% change)
    df["momentum_10"] = close.pct_change(10) * 100

    # 20-day volume change
    if volume is not None and len(volume) == len(close):
        df["vol_change_20"] = volume.pct_change(20) * 100
    else:
        df["vol_change_20"] = 0.0

    # SMA 20/50 spread
    sma50 = close.rolling(50).mean()
    df["sma_spread"] = (sma20 - sma50) / sma50 * 100

    return df


def _generate_labels(close, forward_days=10, bull_threshold=1.0, bear_threshold=-1.0):
    """Self-supervised labels from forward returns."""
    fwd_ret = close.pct_change(forward_days).shift(-forward_days) * 100
    labels = pd.Series("Neutral", index=close.index)
    labels[fwd_ret > bull_threshold] = "Bullish"
    labels[fwd_ret < bear_threshold] = "Bearish"
    return labels


def detect_market_regime(price_history):
    """Detect market regime using ML Random Forest (falls back to SMA).

    Trains a self-supervised Random Forest on technical features and
    10-day forward returns. Falls back to SMA crossover when sklearn
    is unavailable or when there are fewer than 100 data points.
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
        volume = price_history["Volume"] if "Volume" in price_history.columns else None
        features_df = _build_ml_features(close, volume)
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

        feature_cols = ["rsi", "macd_hist", "bb_pct_b", "momentum_10", "vol_change_20", "sma_spread"]
        X_train = combined[feature_cols].iloc[:-10]
        y_train = combined["label"].iloc[:-10]

        if len(X_train) < 30:
            return _detect_regime_sma(price_history)

        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on most recent data point
        X_latest = combined[feature_cols].iloc[[-1]]
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
        feature_names = {"rsi": "RSI", "macd_hist": "MACD", "bb_pct_b": "Bollinger %B",
                         "momentum_10": "Momentum", "vol_change_20": "Volume", "sma_spread": "SMA Spread"}
        top_str = ", ".join(f"{feature_names.get(f, f)} ({v:.0%})" for f, v in top_features)
        desc = f"ML model predicts {prediction} with {prob:.0%} probability. Top drivers: {top_str}."

        return {
            "regime": prediction,
            "signal_strength": round(prob, 2),
            "confidence": conf,
            "sma20": round(s20, 2),
            "sma50": round(s50, 2),
            "description": desc,
            "model_type": "ML",
            "feature_importance": importance,
        }

    except Exception:
        return _detect_regime_sma(price_history)


def _compute_custom_dcf(financial_data, stock_price, shares_outstanding):
    """Compute intrinsic value using a discounted cash flow model.

    Steps:
        1. Extract historical FCF from cash flow statements
        2. Estimate growth rate from revenue + FCF trends (capped 2-20%)
        3. Calculate WACC from CAPM (cost of equity) + cost of debt
        4. Project FCF for 5 years, compute terminal value (perpetuity growth)
        5. Discount to present, subtract net debt, divide by shares

    Returns:
        Dict with dcf_per_share, wacc, growth_rate, or None on failure.
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

        # WACC estimation via CAPM
        risk_free = 0.043  # ~10Y Treasury yield
        market_premium = 0.06  # Equity risk premium
        beta = 1.0  # Default beta

        # Cost of equity = Rf + Beta * Market Premium
        cost_of_equity = risk_free + beta * market_premium

        # Cost of debt from interest expense / total debt
        cost_of_debt = 0.05  # Default
        if balance and balance[0]:
            total_debt = (balance[0].get("totalDebt", 0) or 0)
            interest = abs(income[0].get("interestExpense", 0) or 0) if income else 0
            if total_debt > 0 and interest > 0:
                cost_of_debt = min(interest / total_debt, 0.15)

        # Capital structure weights
        equity_value = stock_price * shares_outstanding if shares_outstanding > 0 else 0
        debt_value = (balance[0].get("totalDebt", 0) or 0) if balance else 0
        total_capital = equity_value + debt_value

        if total_capital > 0:
            equity_weight = equity_value / total_capital
            debt_weight = debt_value / total_capital
        else:
            equity_weight, debt_weight = 0.8, 0.2

        tax_rate = 0.21  # US corporate tax
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
        wacc = max(0.06, min(0.20, wacc))  # Sanity bounds

        # Project FCF for 5 years
        base_fcf = fcf_list[0]
        projected_fcf = []
        for yr in range(1, 6):
            projected_fcf.append(base_fcf * (1 + growth_rate) ** yr)

        # Terminal value (perpetuity growth model, 2.5% terminal growth)
        terminal_growth = 0.025
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)

        # Discount to present
        pv_fcf = sum(fcf / (1 + wacc) ** yr for yr, fcf in enumerate(projected_fcf, 1))
        pv_terminal = terminal_value / (1 + wacc) ** 5

        enterprise_value = pv_fcf + pv_terminal

        # Subtract net debt
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
        }

    except Exception:
        return None


def analyze_valuation(analyst_data, financial_data=None, stock_price=None, shares_outstanding=None):
    """Analyze stock valuation. Tries custom DCF first, falls back to FMP.

    Args:
        analyst_data: Dict from get_analyst_data() (FMP estimates + DCF).
        financial_data: Dict from get_financial_data() (for custom DCF).
        stock_price: Current stock price.
        shares_outstanding: Number of shares outstanding.
    """
    default = {"signal": "N/A", "dcf": 0, "stock_price": 0,
               "margin_of_safety": 0, "revenue_growth": 0,
               "description": "No analyst data.", "model_type": "N/A"}

    # Try custom DCF first
    if financial_data and stock_price and shares_outstanding:
        custom = _compute_custom_dcf(financial_data, stock_price, shares_outstanding)
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

            desc += f" WACC: {custom['wacc']}%, Growth: {custom['growth_rate']}%."

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


def _gather_coaching_context(data, holdings):
    """Build structured portfolio summary for LLM coaching."""
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

    return (
        f"Holdings: {', '.join(tickers)} ({len(tickers)} positions)\n"
        f"Cash: ${cash:,.0f} ({cash_pct:.0f}% of account)\n"
        f"Total trades: {len(journal)} across {trade_days} days\n"
        f"Wins: {wins}, Losses: {losses}\n"
        f"Badges earned: {', '.join(badges) if badges else 'None'}"
    )


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

    # Try Groq first
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
            import json
            text = response.choices[0].message.content.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            tips = json.loads(text)
            if isinstance(tips, list) and len(tips) >= 2:
                return tips[:3]
        except Exception:
            pass

    # Try Gemini fallback
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            import json
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            tips = json.loads(text)
            if isinstance(tips, list) and len(tips) >= 2:
                return tips[:3]
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


def generate_coaching_tips(data, holdings):
    """Generate coaching tips using LLM (Groq/Gemini) with rule-based fallback.

    Tries Groq first, then Gemini, then falls back to static rules.
    Returns:
        Tuple of (tips_list, source_str) where source is "LLM" or "Rules".
    """
    if not holdings:
        return (["Start by making a few trades to build your portfolio."], "Rules")

    context = _gather_coaching_context(data, holdings)
    llm_tips = _get_llm_coaching(context)
    if llm_tips:
        return (llm_tips, "LLM")

    return (_rule_based_tips(data, holdings), "Rules")
