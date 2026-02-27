"""
Portfolio calculations: holdings, P/L, portfolio value, trade stats.
"""

import streamlit as st
from market_data import get_price, get_option_chain_data


def get_holdings() -> dict:
    """Calculate net stock positions from all portfolio trades.

    Uses average cost basis method. Rounds cost to 2 decimal places
    after each sell to prevent floating-point drift over many trades.
    """
    holdings = {}
    for t in st.session_state.data["portfolio"]:
        tk = t["ticker"]
        if tk not in holdings:
            holdings[tk] = {"shares": 0.0, "cost": 0.0}
        if t["type"] == "buy":
            holdings[tk]["shares"] += t["shares"]
            holdings[tk]["cost"] += round(t["shares"] * t["price"], 2)
        else:
            # Reduce cost proportionally using average cost basis
            h = holdings[tk]
            sell_shares = min(t["shares"], h["shares"])
            if h["shares"] > 0:
                avg_cost = h["cost"] / h["shares"]
                h["cost"] = round(h["cost"] - sell_shares * avg_cost, 2)
            h["shares"] = round(h["shares"] - sell_shares, 6)
            # Clean up near-zero positions
            if h["shares"] < 0.001:
                h["shares"] = 0.0
                h["cost"] = 0.0
    return {k: v for k, v in holdings.items() if v["shares"] > 0.001}


def get_options_holdings() -> dict:
    """Calculate net options positions from all options trades."""
    holdings = {}
    for opt in st.session_state.data["options"]:
        key = f"{opt['ticker']}_{opt['type']}_{opt['strike']}_{opt['expiration']}"
        if key not in holdings:
            holdings[key] = {
                "ticker": opt["ticker"], "type": opt["type"],
                "strike": opt["strike"], "expiration": opt["expiration"],
                "contracts": 0, "total_cost": 0,
            }
        if "Buy" in opt["action"]:
            holdings[key]["contracts"] += opt["contracts"]
            holdings[key]["total_cost"] += opt["total"]
        else:
            holdings[key]["contracts"] -= opt["contracts"]
    return {k: v for k, v in holdings.items() if v["contracts"] > 0}


def get_option_current_price(ticker, expiration, strike, opt_type):
    """Get current mid-price for a specific option contract."""
    try:
        df = get_option_chain_data(ticker, expiration, opt_type.capitalize())
        if not df.empty:
            row = df[df["strike"] == strike]
            if not row.empty:
                bid, ask = row.iloc[0]["bid"], row.iloc[0]["ask"]
                return (bid + ask) / 2 if bid > 0 and ask > 0 else row.iloc[0]["lastPrice"]
    except Exception:
        pass
    return 0


def options_portfolio_value():
    """Calculate total current value of all options positions."""
    total = 0
    for key, opt in get_options_holdings().items():
        price = get_option_current_price(opt["ticker"], opt["expiration"], opt["strike"], opt["type"])
        total += price * 100 * opt["contracts"]
    return total


def portfolio_value():
    """Calculate total account value: cash + stocks + options."""
    h = get_holdings()
    stock_value = sum(get_price(t)[0] * v["shares"] for t, v in h.items())
    return st.session_state.data["cash"] + stock_value + options_portfolio_value()


def get_trade_stats(journal: list) -> dict:
    """Compute trade journal analytics: win rate, avg win/loss, profit factor.

    Pairs buy/sell trades by ticker to determine wins/losses.
    """
    if not journal:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0, "profit_factor": 0}

    # Track buy prices per ticker
    buys = {}
    results = []
    for t in journal:
        tk = t["ticker"]
        if t["action"] == "buy":
            buys.setdefault(tk, []).append(t["price"])
        elif t["action"] == "sell" and tk in buys and buys[tk]:
            buy_price = buys[tk].pop(0)
            pnl_pct = (t["price"] - buy_price) / buy_price * 100
            results.append(pnl_pct)

    if not results:
        return {"total": len(journal), "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0, "profit_factor": 0}

    wins = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        "total": len(journal),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(results) * 100 if results else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best": max(results) if results else 0,
        "worst": min(results) if results else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0,
    }
