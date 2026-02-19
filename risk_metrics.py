"""
Risk metrics: VaR, Sharpe, max drawdown, Monte Carlo, correlation.
Pure numpy/pandas — no Streamlit dependency.
"""

import numpy as np
import pandas as pd


def compute_sharpe_ratio(daily_returns, risk_free_rate=0.05):
    """Annualized Sharpe ratio."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_max_drawdown(cumulative_returns):
    """Maximum drawdown (peak-to-trough %)."""
    if cumulative_returns.empty:
        return 0.0
    peak = cumulative_returns.expanding().max()
    return float(((cumulative_returns - peak) / peak).min())


def compute_var_historical(daily_returns, confidence=0.95, portfolio_val=100000.0):
    """Historical Value at Risk (dollar amount)."""
    if daily_returns.empty:
        return 0.0
    return float(abs(np.percentile(daily_returns, (1 - confidence) * 100)) * portfolio_val)


def build_portfolio_daily_returns(holdings, get_history_fn, starting_balance):
    """Build weighted daily returns for the portfolio."""
    if not holdings:
        return pd.Series(dtype=float)
    total_invested = sum(v["cost"] for v in holdings.values())
    if total_invested == 0:
        return pd.Series(dtype=float)

    all_returns, weights = {}, {}
    for ticker, h in holdings.items():
        hist = get_history_fn(ticker, "6mo")
        if hist.empty or "Close" not in hist.columns:
            continue
        close = hist["Close"]
        if close.index.tz is not None:
            close = close.tz_localize(None)
        all_returns[ticker] = close.pct_change().dropna()
        weights[ticker] = h["cost"] / total_invested

    if not all_returns:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_returns).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    portfolio_returns = pd.Series(0.0, index=df.index)
    for ticker in df.columns:
        portfolio_returns += df[ticker] * weights.get(ticker, 0)
    return portfolio_returns


def compute_monte_carlo(daily_returns, portfolio_val, days=252, sims=500):
    """Run Monte Carlo simulation. Returns dict with percentile bands.

    Returns:
        Dict with keys: p5, p25, p50, p75, p95 — each a list of values over `days`.
    """
    if daily_returns.empty:
        return {}
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    results = np.zeros((sims, days))
    for i in range(sims):
        daily = np.random.normal(mean_ret, std_ret, days)
        results[i] = portfolio_val * np.cumprod(1 + daily)
    return {
        "p5": np.percentile(results, 5, axis=0).tolist(),
        "p25": np.percentile(results, 25, axis=0).tolist(),
        "p50": np.percentile(results, 50, axis=0).tolist(),
        "p75": np.percentile(results, 75, axis=0).tolist(),
        "p95": np.percentile(results, 95, axis=0).tolist(),
    }


def compute_correlation_matrix(holdings, get_history_fn):
    """Compute pairwise correlation matrix for portfolio holdings.

    Returns:
        DataFrame of correlations, or empty DataFrame.
    """
    if len(holdings) < 2:
        return pd.DataFrame()
    closes = {}
    for ticker in holdings:
        hist = get_history_fn(ticker, "6mo")
        if not hist.empty and "Close" in hist.columns:
            close = hist["Close"]
            if close.index.tz is not None:
                close = close.tz_localize(None)
            closes[ticker] = close
    if len(closes) < 2:
        return pd.DataFrame()
    return pd.DataFrame(closes).pct_change().dropna().corr()
