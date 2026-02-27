"""
Dashboard helpers for the Trading Simulator home page.
Provides market overview, top movers, and alert checking.
"""

MARKET_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
}


def get_market_overview(get_price_fn) -> list[dict]:
    """Fetch current prices and changes for major market indices.

    Returns:
        List of {symbol, name, price, change, change_pct}
    """
    results = []
    for symbol, name in MARKET_INDICES.items():
        try:
            price, change, pct = get_price_fn(symbol)
            results.append({
                "symbol": symbol,
                "name": name,
                "price": price,
                "change": change,
                "change_pct": pct,
            })
        except Exception:
            results.append({
                "symbol": symbol,
                "name": name,
                "price": 0,
                "change": 0,
                "change_pct": 0,
            })
    return results


def get_top_movers(holdings: dict, get_price_fn, n: int = 3) -> tuple[list, list]:
    """Find top gainers and losers among current holdings.

    Args:
        holdings: {ticker: {shares, cost}}
        get_price_fn: function returning (price, change, pct)
        n: number of top movers to return per side

    Returns:
        (gainers, losers) — each a list of {ticker, price, change_pct, value}
    """
    movers = []
    for ticker, h in holdings.items():
        try:
            price, change, pct = get_price_fn(ticker)
            movers.append({
                "ticker": ticker,
                "price": price,
                "change_pct": pct,
                "value": h["shares"] * price,
            })
        except Exception:
            continue

    gainers = sorted([m for m in movers if m["change_pct"] > 0],
                     key=lambda x: x["change_pct"], reverse=True)[:n]
    losers = sorted([m for m in movers if m["change_pct"] < 0],
                    key=lambda x: x["change_pct"])[:n]
    return gainers, losers


def get_triggered_alerts(watchlist: list, price_alerts: dict, get_price_fn) -> list[dict]:
    """Check which watchlist price alerts have been triggered.

    Supports both upside and downside alerts. If the alert price is above
    the stock's last close, it triggers when price >= alert. If below,
    it triggers when price <= alert.

    Returns:
        List of {ticker, current_price, alert_price, direction}
    """
    triggered = []
    for ticker in watchlist:
        if ticker not in price_alerts:
            continue
        try:
            price = get_price_fn(ticker)[0]
            alert_price = float(price_alerts[ticker])
            if price >= alert_price:
                triggered.append({
                    "ticker": ticker,
                    "current_price": price,
                    "alert_price": alert_price,
                    "direction": "above",
                })
            elif price <= alert_price:
                triggered.append({
                    "ticker": ticker,
                    "current_price": price,
                    "alert_price": alert_price,
                    "direction": "below",
                })
        except Exception:
            continue
    return triggered
