"""
Risk metrics: VaR, Sharpe ratio, max drawdown, and portfolio daily returns.
Pure numpy/pandas — no Streamlit dependency.
"""

import numpy as np
import pandas as pd


def compute_sharpe_ratio(
    daily_returns: pd.Series, risk_free_rate: float = 0.05
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        daily_returns: Series of daily portfolio returns (as decimals).
        risk_free_rate: Annualized risk-free rate (default 5%).

    Returns:
        Annualized Sharpe ratio, or 0.0 if returns are empty or
        standard deviation is zero.
    """
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Compute maximum drawdown (peak-to-trough percentage).

    Args:
        cumulative_returns: Series of cumulative return values
            (e.g. portfolio value over time, or 1+cumulative %).

    Returns:
        Max drawdown as a negative percentage (e.g. -0.15 for -15%).
        Returns 0.0 if series is empty or has no drawdown.
    """
    if cumulative_returns.empty:
        return 0.0
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())


def compute_var_historical(
    daily_returns: pd.Series,
    confidence: float = 0.95,
    portfolio_val: float = 100000.0,
) -> float:
    """Compute historical Value at Risk (dollar amount).

    Args:
        daily_returns: Series of daily portfolio returns (as decimals).
        confidence: Confidence level (default 95%).
        portfolio_val: Current portfolio dollar value.

    Returns:
        Dollar VaR (positive number representing potential loss).
        Returns 0.0 if returns are empty.
    """
    if daily_returns.empty:
        return 0.0
    percentile = np.percentile(daily_returns, (1 - confidence) * 100)
    return float(abs(percentile) * portfolio_val)


def build_portfolio_daily_returns(
    holdings: dict,
    get_history_fn,
    starting_balance: float,
) -> pd.Series:
    """Build weighted daily returns for the portfolio.

    Args:
        holdings: Dict mapping ticker -> {"shares": float, "cost": float}.
        get_history_fn: Callable(ticker, period) -> DataFrame with "Close" column.
        starting_balance: Account starting balance for weight calculation.

    Returns:
        Series of daily portfolio returns (as decimals). Empty Series
        if no price data is available.
    """
    if not holdings:
        return pd.Series(dtype=float)

    all_returns = {}
    weights = {}
    total_invested = sum(v["cost"] for v in holdings.values())

    if total_invested == 0:
        return pd.Series(dtype=float)

    for ticker, h in holdings.items():
        hist = get_history_fn(ticker, "6mo")
        if hist.empty or "Close" not in hist.columns:
            continue
        returns = hist["Close"].pct_change().dropna()
        all_returns[ticker] = returns
        weights[ticker] = h["cost"] / total_invested

    if not all_returns:
        return pd.Series(dtype=float)

    # Align all return series to common dates
    df = pd.DataFrame(all_returns)
    df = df.dropna()

    if df.empty:
        return pd.Series(dtype=float)

    # Weighted portfolio returns
    portfolio_returns = pd.Series(0.0, index=df.index)
    for ticker in df.columns:
        portfolio_returns += df[ticker] * weights.get(ticker, 0)

    return portfolio_returns
