"""Tests for risk_metrics.py — Sharpe, VaR, Monte Carlo."""

import numpy as np
import pandas as pd

from risk_metrics import (
    compute_sharpe_ratio,
    compute_var_historical,
    compute_monte_carlo,
    build_portfolio_daily_returns,
)


class TestSharpeRatio:
    def test_sharpe_ratio_positive(self):
        """Uptrending returns should produce positive Sharpe."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_empty(self):
        """Empty series should return 0."""
        sharpe = compute_sharpe_ratio(pd.Series(dtype=float))
        assert sharpe == 0.0


class TestVaR:
    def test_var_historical(self):
        """VaR should return a positive dollar value."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0, 0.02, 252))
        var = compute_var_historical(returns, confidence=0.95,
                                     portfolio_val=100000)
        assert var > 0


class TestMonteCarlo:
    def test_monte_carlo_shape(self):
        """Monte Carlo should return dict with all percentile keys."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        result = compute_monte_carlo(returns, portfolio_val=100000,
                                     days=30, sims=100)
        assert isinstance(result, dict)
        for key in ("p5", "p25", "p50", "p75", "p95"):
            assert key in result
            assert len(result[key]) == 30


class TestBuildPortfolio:
    def test_build_portfolio_accepts_price_fn(self):
        """build_portfolio_daily_returns should accept get_price_fn."""
        holdings = {"AAPL": {"shares": 10, "cost": 1500}}

        def mock_history(ticker, period):
            np.random.seed(42)
            dates = pd.date_range("2024-01-01", periods=60, freq="B")
            prices = 150 + np.cumsum(np.random.randn(60) * 0.5)
            return pd.DataFrame({"Close": prices}, index=dates)

        def mock_price(ticker):
            return (155.0, 3.0, "N/A")

        result = build_portfolio_daily_returns(
            holdings, mock_history, 100000, get_price_fn=mock_price
        )
        assert isinstance(result, pd.Series)
        assert len(result) > 0
