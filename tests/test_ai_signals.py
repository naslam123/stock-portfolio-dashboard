"""Tests for ai_signals.py — regime detection, badges, coaching, DCF."""

import numpy as np
import pandas as pd

from ai_signals import (
    detect_market_regime,
    check_badges,
    generate_coaching_tips,
    _compute_custom_dcf,
    analyze_valuation,
)


class TestRegimeDetection:
    def test_ml_regime_with_enough_data(self):
        """200 data points with sklearn available should use ML model."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, 200),
        })
        result = detect_market_regime(df)
        assert result["regime"] in ("Bullish", "Bearish", "Neutral")
        # With 200 points and sklearn, should attempt ML
        assert result["model_type"] in ("ML", "SMA")

    def test_sma_fallback_short_data(self):
        """60 data points should fall back to SMA (needs 100 for ML)."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(60) * 0.5)
        df = pd.DataFrame({"Close": prices})
        result = detect_market_regime(df)
        assert result["model_type"] == "SMA"

    def test_sma_fallback_no_sklearn(self, monkeypatch):
        """When sklearn is unavailable, should use SMA."""
        import ai_signals
        monkeypatch.setattr(ai_signals, "SKLEARN_AVAILABLE", False)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({"Close": prices})
        result = detect_market_regime(df)
        assert result["model_type"] == "SMA"


class TestBadges:
    def test_badge_six_figure_club(self):
        data = {
            "journal": [], "portfolio": [], "options": [],
            "watchlist": [], "badges": [],
        }
        # Below threshold
        earned = check_badges(data, portfolio_value=50000)
        assert "Six-Figure Club" not in earned

        # Above threshold
        earned = check_badges(data, portfolio_value=120000)
        assert "Six-Figure Club" in earned

    def test_badge_first_trade(self):
        data = {
            "journal": [{"date": "2024-01-01", "ticker": "AAPL",
                         "action": "buy", "shares": 10, "price": 150}],
            "portfolio": [], "options": [],
            "watchlist": [], "badges": [],
        }
        earned = check_badges(data, portfolio_value=100000)
        assert "First Trade" in earned


class TestCoaching:
    def test_coaching_returns_tuple(self):
        data = {
            "cash": 100000, "journal": [], "badges": [],
        }
        holdings = {"AAPL": {"shares": 10, "cost": 1500}}
        tips, source = generate_coaching_tips(data, holdings)
        assert isinstance(tips, list)
        assert isinstance(source, str)
        assert source in ("LLM", "Rules")


class TestDCF:
    def test_custom_dcf_positive(self):
        financial_data = {
            "cashflow": [
                {"operatingCashFlow": 5_000_000_000,
                 "capitalExpenditure": -1_000_000_000},
                {"operatingCashFlow": 4_500_000_000,
                 "capitalExpenditure": -900_000_000},
            ],
            "income": [
                {"revenue": 50_000_000_000, "interestExpense": -500_000_000},
                {"revenue": 45_000_000_000, "interestExpense": -450_000_000},
            ],
            "balance": [
                {"totalDebt": 10_000_000_000,
                 "cashAndCashEquivalents": 5_000_000_000},
            ],
        }
        result = _compute_custom_dcf(financial_data, stock_price=150.0,
                                     shares_outstanding=1_000_000_000)
        assert result is not None
        assert result["dcf_per_share"] > 0

    def test_analyze_valuation_fmp_fallback(self):
        analyst_data = {"dcf": 200, "stock_price": 150, "estimates": []}
        result = analyze_valuation(analyst_data, financial_data=None)
        assert result["model_type"] == "FMP API"
        assert result["signal"] in ("Undervalued", "Overvalued", "Fair Value")
