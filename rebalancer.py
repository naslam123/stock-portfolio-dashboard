"""
Portfolio rebalancing engine for the Trading Simulator.
Computes current weights, target allocations, and exact trades needed.
"""


def compute_current_weights(holdings: dict, get_price_fn) -> dict:
    """Calculate current portfolio weights for each holding.

    Args:
        holdings: {ticker: {shares, cost}} from get_holdings()
        get_price_fn: function returning (price, change, pct)

    Returns:
        {ticker: {shares, price, value, weight_pct}, "_total": float}
    """
    result = {}
    total_value = 0.0

    for ticker, h in holdings.items():
        price = get_price_fn(ticker)[0]
        value = h["shares"] * price
        result[ticker] = {
            "shares": h["shares"],
            "price": price,
            "value": value,
            "weight_pct": 0.0,
        }
        total_value += value

    if total_value > 0:
        for ticker in result:
            result[ticker]["weight_pct"] = round(
                result[ticker]["value"] / total_value * 100, 2
            )

    result["_total"] = total_value
    return result


def generate_equal_weight_targets(tickers: list[str]) -> dict:
    """Generate equal-weight target allocation.

    Returns:
        {ticker: target_weight_pct} all equal, summing to 100.
    """
    if not tickers:
        return {}
    weight = round(100.0 / len(tickers), 2)
    targets = {t: weight for t in tickers}
    # Adjust last ticker to ensure exact 100
    remainder = 100.0 - weight * len(tickers)
    if tickers:
        targets[tickers[-1]] = round(targets[tickers[-1]] + remainder, 2)
    return targets


def generate_custom_targets(ticker_weights: dict) -> dict:
    """Validate and normalize custom target weights to sum to 100.

    Args:
        ticker_weights: {ticker: weight_pct} from user input

    Returns:
        Normalized {ticker: weight_pct} summing to 100.
    """
    total = sum(ticker_weights.values())
    if total <= 0:
        return generate_equal_weight_targets(list(ticker_weights.keys()))
    return {t: round(w / total * 100, 2) for t, w in ticker_weights.items()}


def compute_rebalance_trades(
    holdings: dict,
    target_weights: dict,
    cash: float,
    get_price_fn,
    commission_per_trade: float = 0.0,
    deploy_cash: bool = True,
) -> list[dict]:
    """Calculate exact trades needed to rebalance to target weights.

    Args:
        holdings: {ticker: {shares, cost}}
        target_weights: {ticker: target_pct}
        cash: current cash balance
        get_price_fn: price function
        commission_per_trade: per-trade commission
        deploy_cash: if True, target weights apply to total account value (invested + cash)

    Returns:
        List of {ticker, action, shares, price, current_weight, target_weight, value_change}
    """
    current = compute_current_weights(holdings, get_price_fn)
    invested_value = current.pop("_total", 0)
    total_value = (invested_value + cash) if deploy_cash else invested_value

    if total_value <= 0:
        return []

    trades = []
    all_tickers = set(list(current.keys()) + list(target_weights.keys()))

    for ticker in sorted(all_tickers):
        target_pct = target_weights.get(ticker, 0.0)
        target_value = total_value * target_pct / 100.0

        cur = current.get(ticker, {"shares": 0, "price": 0, "value": 0, "weight_pct": 0})
        price = cur["price"] if cur["price"] > 0 else get_price_fn(ticker)[0]

        if price <= 0:
            continue

        current_value = cur["value"]
        diff_value = target_value - current_value
        diff_shares = abs(diff_value) / price

        # Skip de minimis trades
        if diff_shares < 0.01:
            continue

        action = "buy" if diff_value > 0 else "sell"
        trades.append({
            "ticker": ticker,
            "action": action,
            "shares": round(diff_shares, 4),
            "price": price,
            "current_weight": cur["weight_pct"],
            "target_weight": target_pct,
            "value_change": round(diff_value, 2),
        })

    return trades


def estimate_rebalance_cost(trades: list[dict], commission: float) -> dict:
    """Estimate total cost of executing all rebalance trades.

    Returns:
        {total_buys, total_sells, net_cash_needed, total_commissions, num_trades}
    """
    total_buys = sum(t["shares"] * t["price"] for t in trades if t["action"] == "buy")
    total_sells = sum(t["shares"] * t["price"] for t in trades if t["action"] == "sell")
    num_trades = len(trades)
    total_commissions = num_trades * commission

    return {
        "total_buys": round(total_buys, 2),
        "total_sells": round(total_sells, 2),
        "net_cash_needed": round(total_buys - total_sells + total_commissions, 2),
        "total_commissions": round(total_commissions, 2),
        "num_trades": num_trades,
    }
