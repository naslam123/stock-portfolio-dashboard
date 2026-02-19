"""
AI/ML signals: regime detection, DCF valuation, badges, coaching.
Pure numpy/pandas — no Streamlit dependency.
"""

import numpy as np
import pandas as pd


def detect_market_regime(price_history):
    """Detect market regime using SMA 20/50 crossover."""
    default = {"regime": "Neutral", "signal_strength": 0.0, "confidence": "Low",
               "sma20": 0.0, "sma50": 0.0, "description": "Insufficient data."}
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
        regime, desc = "Bullish", f"SMA20 above SMA50 but price near SMA — weakening uptrend."
    elif s20 < s50:
        regime, desc = "Bearish", f"SMA20 below SMA50 but price near SMA — weakening downtrend."
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
            "sma20": round(s20, 2), "sma50": round(s50, 2), "description": desc}


def analyze_valuation(analyst_data):
    """Analyze stock valuation using analyst estimates and DCF from FMP."""
    default = {"signal": "N/A", "dcf": 0, "stock_price": 0,
               "margin_of_safety": 0, "revenue_growth": 0, "description": "No analyst data."}
    if not analyst_data:
        return default
    dcf = analyst_data.get("dcf", 0)
    stock_price = analyst_data.get("stock_price", 0)
    estimates = analyst_data.get("estimates", [])
    if not dcf or not stock_price:
        return default

    margin = (dcf - stock_price) / stock_price * 100
    revenue_growth = 0
    if len(estimates) >= 2:
        r0, r1 = estimates[0].get("revenueAvg", 0), estimates[1].get("revenueAvg", 0)
        if r0 and r1:
            revenue_growth = (r0 - r1) / r1 * 100

    if margin > 15:
        signal, desc = "Undervalued", f"DCF ${dcf:,.0f} is {margin:+.1f}% above price ${stock_price:,.2f} — potential upside."
    elif margin < -15:
        signal, desc = "Overvalued", f"DCF ${dcf:,.0f} is {margin:+.1f}% below price ${stock_price:,.2f} — priced above intrinsic."
    else:
        signal, desc = "Fair Value", f"DCF ${dcf:,.0f} within 15% of price ${stock_price:,.2f} — reasonably valued."
    if revenue_growth:
        desc += f" Revenue growth: {revenue_growth:+.1f}%."

    return {"signal": signal, "dcf": round(dcf, 2), "stock_price": round(stock_price, 2),
            "margin_of_safety": round(margin, 1), "revenue_growth": round(revenue_growth, 1), "description": desc}


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
    ("First Trade", lambda d: len(d["journal"]) >= 1),
    ("Diversifier", lambda d: len(set(t["ticker"] for t in d["portfolio"])) >= 5),
    ("Options Trader", lambda d: len(d["options"]) >= 1),
    ("Risk Manager", lambda d: any(t.get("order_type") == "stop-loss" for t in d["portfolio"])),
    ("Consistent", lambda d: len(set(t["date"] for t in d["journal"])) >= 5),
    ("Six-Figure Club", lambda d: d["cash"] + sum(t["shares"] * t["price"] for t in d["portfolio"] if t["type"] == "buy") >= 110000),
    ("Watchful", lambda d: len(d["watchlist"]) >= 5),
]


def check_badges(data):
    """Check which badges the user has earned. Returns list of badge names."""
    earned = []
    for name, check_fn in BADGE_DEFS:
        try:
            if check_fn(data):
                earned.append(name)
        except Exception:
            pass
    return earned


def generate_coaching_tips(data, holdings):
    """Rule-based AI coaching tips based on trading behavior."""
    tips = []
    if not holdings:
        tips.append("Start by making a few trades to build your portfolio.")
        return tips

    # Concentration risk
    if len(holdings) == 1:
        tips.append("Your portfolio has only 1 stock. Consider diversifying across sectors.")
    elif len(holdings) == 2:
        tips.append("Consider adding more positions to reduce concentration risk.")

    # Cash allocation
    total_invested = sum(v["cost"] for v in holdings.values())
    cash = data["cash"]
    if total_invested > 0:
        cash_pct = cash / (cash + total_invested) * 100
        if cash_pct > 80:
            tips.append(f"You have {cash_pct:.0f}% in cash. Consider deploying more capital.")
        elif cash_pct < 5:
            tips.append(f"Only {cash_pct:.0f}% cash remaining. Keep some reserve for opportunities.")

    # Trading frequency
    journal = data.get("journal", [])
    trade_days = len(set(t["date"] for t in journal))
    if len(journal) > 20 and trade_days < 3:
        tips.append("Many trades in few days — avoid overtrading. Quality over quantity.")

    # Win/loss pattern
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
