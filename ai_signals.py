"""
AI/ML signal generation: market regime detection and sentiment aggregation.
Pure numpy/pandas — no Streamlit dependency.
"""

import numpy as np
import pandas as pd


def detect_market_regime(price_history: pd.DataFrame) -> dict:
    """Detect market regime using SMA 20/50 crossover classification.

    Compares the 20-day and 50-day simple moving averages to classify
    the current regime as Bullish, Bearish, or Neutral.

    Args:
        price_history: DataFrame with a "Close" column and
            DatetimeIndex (at least 50 rows for meaningful results).

    Returns:
        Dict with keys:
            regime: "Bullish", "Bearish", or "Neutral"
            signal_strength: float 0-1 indicating strength of signal
            confidence: "High", "Medium", or "Low"
            sma20: current SMA20 value
            sma50: current SMA50 value
            description: human-readable explanation
    """
    default = {
        "regime": "Neutral",
        "signal_strength": 0.0,
        "confidence": "Low",
        "sma20": 0.0,
        "sma50": 0.0,
        "description": "Insufficient data for regime detection.",
    }

    if price_history.empty or "Close" not in price_history.columns:
        return default

    close = price_history["Close"]
    if len(close) < 50:
        return default

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    current_sma20 = float(sma20.iloc[-1])
    current_sma50 = float(sma50.iloc[-1])
    current_price = float(close.iloc[-1])

    # Percentage spread between SMAs
    spread = (current_sma20 - current_sma50) / current_sma50 * 100

    # Determine regime
    if current_sma20 > current_sma50 and current_price > current_sma20:
        regime = "Bullish"
        description = (
            f"SMA20 (${current_sma20:.2f}) above SMA50 (${current_sma50:.2f}), "
            f"price above both — uptrend confirmed."
        )
    elif current_sma20 < current_sma50 and current_price < current_sma20:
        regime = "Bearish"
        description = (
            f"SMA20 (${current_sma20:.2f}) below SMA50 (${current_sma50:.2f}), "
            f"price below both — downtrend confirmed."
        )
    elif current_sma20 > current_sma50:
        regime = "Bullish"
        description = (
            f"SMA20 (${current_sma20:.2f}) above SMA50 (${current_sma50:.2f}) "
            f"but price near SMA — weakening uptrend."
        )
    elif current_sma20 < current_sma50:
        regime = "Bearish"
        description = (
            f"SMA20 (${current_sma20:.2f}) below SMA50 (${current_sma50:.2f}) "
            f"but price near SMA — weakening downtrend."
        )
    else:
        regime = "Neutral"
        description = "SMAs converging — no clear trend direction."

    # Signal strength: based on spread magnitude (capped at 5%)
    signal_strength = min(abs(spread) / 5.0, 1.0)

    # Confidence: based on spread and price alignment
    if abs(spread) > 2 and (
        (regime == "Bullish" and current_price > current_sma20)
        or (regime == "Bearish" and current_price < current_sma20)
    ):
        confidence = "High"
    elif abs(spread) > 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "regime": regime,
        "signal_strength": round(signal_strength, 2),
        "confidence": confidence,
        "sma20": round(current_sma20, 2),
        "sma50": round(current_sma50, 2),
        "description": description,
    }


def analyze_valuation(analyst_data: dict) -> dict:
    """Analyze stock valuation using analyst estimates and DCF from FMP.

    Compares DCF intrinsic value to current price and summarizes
    analyst revenue/EPS growth expectations.

    Args:
        analyst_data: Dict from market_data.get_analyst_data() with keys
            "estimates" (list), "dcf" (float), "stock_price" (float).

    Returns:
        Dict with keys:
            signal: "Undervalued", "Overvalued", or "Fair Value"
            dcf: DCF intrinsic value
            stock_price: current stock price
            margin_of_safety: percentage difference (DCF vs price)
            revenue_growth: estimated next-year revenue growth %
            description: human-readable summary
    """
    default = {
        "signal": "N/A",
        "dcf": 0,
        "stock_price": 0,
        "margin_of_safety": 0,
        "revenue_growth": 0,
        "description": "No analyst data available.",
    }

    if not analyst_data:
        return default

    dcf = analyst_data.get("dcf", 0)
    stock_price = analyst_data.get("stock_price", 0)
    estimates = analyst_data.get("estimates", [])

    if not dcf or not stock_price:
        return default

    # Margin of safety: how much DCF differs from price
    margin = (dcf - stock_price) / stock_price * 100

    # Revenue growth from estimates
    revenue_growth = 0
    if len(estimates) >= 2:
        rev_current = estimates[0].get("revenueAvg", 0)
        rev_next = estimates[1].get("revenueAvg", 0) if len(estimates) > 1 else 0
        if rev_current and rev_next:
            revenue_growth = (rev_current - rev_next) / rev_next * 100

    # Signal classification
    if margin > 15:
        signal = "Undervalued"
        description = (
            f"DCF value ${dcf:,.0f} is {margin:+.1f}% above current price "
            f"${stock_price:,.2f} — potential upside."
        )
    elif margin < -15:
        signal = "Overvalued"
        description = (
            f"DCF value ${dcf:,.0f} is {margin:+.1f}% below current price "
            f"${stock_price:,.2f} — priced above intrinsic value."
        )
    else:
        signal = "Fair Value"
        description = (
            f"DCF value ${dcf:,.0f} within 15% of price "
            f"${stock_price:,.2f} — reasonably valued."
        )

    if revenue_growth:
        description += f" Analyst avg revenue growth: {revenue_growth:+.1f}%."

    return {
        "signal": signal,
        "dcf": round(dcf, 2),
        "stock_price": round(stock_price, 2),
        "margin_of_safety": round(margin, 1),
        "revenue_growth": round(revenue_growth, 1),
        "description": description,
    }
