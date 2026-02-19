"""
Portfolio calculation functions for holdings, P/L, and portfolio value.
Accesses st.session_state.data directly (pragmatic for Streamlit project).
"""

import streamlit as st
from market_data import get_price, get_option_chain_data


def get_holdings() -> dict:
    """Calculate net stock positions from all portfolio trades.

    Returns:
        Dict mapping ticker -> {"shares": float, "cost": float}
        for positions with shares > 0.001.
    """
    holdings = {}
    for t in st.session_state.data["portfolio"]:
        tk = t["ticker"]
        if tk not in holdings:
            holdings[tk] = {"shares": 0, "cost": 0}
        if t["type"] == "buy":
            holdings[tk]["shares"] += t["shares"]
            holdings[tk]["cost"] += t["shares"] * t["price"]
        else:
            holdings[tk]["shares"] -= t["shares"]
    return {k: v for k, v in holdings.items() if v["shares"] > 0.001}


def get_options_holdings() -> dict:
    """Calculate net options positions from all options trades.

    Returns:
        Dict mapping contract key -> {"ticker", "type", "strike",
        "expiration", "contracts", "total_cost"} for positions
        with contracts > 0.
    """
    holdings = {}
    for opt in st.session_state.data["options"]:
        key = f"{opt['ticker']}_{opt['type']}_{opt['strike']}_{opt['expiration']}"
        if key not in holdings:
            holdings[key] = {
                "ticker": opt["ticker"],
                "type": opt["type"],
                "strike": opt["strike"],
                "expiration": opt["expiration"],
                "contracts": 0,
                "total_cost": 0,
            }
        if "Buy" in opt["action"]:
            holdings[key]["contracts"] += opt["contracts"]
            holdings[key]["total_cost"] += opt["total"]
        else:  # Sell to Close
            holdings[key]["contracts"] -= opt["contracts"]
    return {k: v for k, v in holdings.items() if v["contracts"] > 0}


def get_option_current_price(
    ticker: str, expiration: str, strike: float, opt_type: str
) -> float:
    """Get current mid-price for a specific option contract.

    Args:
        ticker: Underlying stock symbol.
        expiration: Expiration date string.
        strike: Strike price.
        opt_type: "call" or "put".

    Returns:
        Mid-price between bid/ask, or lastPrice, or 0 on failure.
    """
    try:
        df = get_option_chain_data(ticker, expiration, opt_type.capitalize())
        if not df.empty:
            row = df[df["strike"] == strike]
            if not row.empty:
                bid = row.iloc[0]["bid"]
                ask = row.iloc[0]["ask"]
                return (bid + ask) / 2 if bid > 0 and ask > 0 else row.iloc[0]["lastPrice"]
    except Exception:
        pass
    return 0


def options_portfolio_value() -> float:
    """Calculate total current value of all options positions.

    Returns:
        Dollar value of all open options contracts.
    """
    total = 0
    for key, opt in get_options_holdings().items():
        price = get_option_current_price(
            opt["ticker"], opt["expiration"], opt["strike"], opt["type"]
        )
        total += price * 100 * opt["contracts"]
    return total


def portfolio_value() -> float:
    """Calculate total account value: cash + stocks + options.

    Returns:
        Total portfolio dollar value.
    """
    h = get_holdings()
    stock_value = sum(get_price(t)[0] * v["shares"] for t, v in h.items())
    opts_value = options_portfolio_value()
    return st.session_state.data["cash"] + stock_value + opts_value
