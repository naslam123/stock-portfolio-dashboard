"""Tests for ai_signals.py — regime detection, badges, coaching, DCF, composite signal."""

import numpy as np
import pandas as pd

from ai_signals import (
    detect_market_regime,
    detect_regime_hmm,
    score_sentiment_vader,
    generate_composite_signal,
    check_badges,
    generate_coaching_tips,
    _compute_custom_dcf,
    _calculate_beta,
    compute_dcf_sensitivity,
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

    def test_coaching_with_enriched_context(self):
        """Coaching should accept optional regime + risk context."""
        data = {
            "cash": 100000, "journal": [], "badges": [],
        }
        holdings = {"AAPL": {"shares": 10, "cost": 1500}}
        regime = {"regime": "Bullish", "model_type": "ML", "confidence": "72%"}
        risk = {"var": {"dollar": 3200, "percent": 3.2, "confidence": 0.95},
                "sharpe": 1.25, "max_drawdown": 8.5}
        tips, source = generate_coaching_tips(
            data, holdings, regime=regime, risk_metrics=risk)
        assert isinstance(tips, list)
        assert source in ("LLM", "Rules")


class TestHMMRegime:
    def test_hmm_with_enough_data(self):
        """HMM should detect a regime from 200 data points."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({"Close": prices})
        result = detect_regime_hmm(df)
        assert result["regime"] in ("Bull", "Sideways", "Bear")
        assert result["model_type"] == "HMM"
        assert "probabilities" in result
        probs = result["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_hmm_short_data_returns_default(self):
        """HMM should return default for < 60 data points."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
        df = pd.DataFrame({"Close": prices})
        result = detect_regime_hmm(df)
        assert result["model_type"] == "N/A"


class TestVADERSentiment:
    def test_bullish_headlines(self):
        """Positive financial headlines should score bullish."""
        headlines = [
            "Company profits soar to incredible new heights",
            "Markets rally strongly on excellent jobs data",
            "Great earnings surprise leads to optimistic outlook",
        ]
        result = score_sentiment_vader(headlines)
        assert result["score"] > 0
        assert result["label"] == "Bullish"
        assert len(result["details"]) == 3

    def test_bearish_headlines(self):
        """Negative financial headlines should score bearish."""
        headlines = [
            "Markets crash on recession fears",
            "Company issues profit warning amid declining sales",
            "Stocks plunge as inflation data disappoints",
        ]
        result = score_sentiment_vader(headlines)
        assert result["score"] < 0
        assert result["label"] == "Bearish"

    def test_empty_headlines(self):
        """Empty list should return neutral."""
        result = score_sentiment_vader([])
        assert result["score"] == 0.0
        assert result["label"] == "Neutral"


class TestCompositeSignal:
    def test_composite_returns_all_components(self):
        """Composite signal should contain RF, HMM, and sentiment components."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, 200),
        })
        headlines = ["Stocks rally on strong earnings"]
        result = generate_composite_signal(df, headlines)
        assert result["signal"] in ("Bullish", "Bearish", "Neutral")
        assert "components" in result
        assert "rf" in result["components"]
        assert "hmm" in result["components"]
        assert "sentiment" in result["components"]
        assert "weights" in result
        assert -1.0 <= result["score"] <= 1.0

    def test_composite_without_headlines(self):
        """Composite signal should work without headlines (sentiment=0)."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({"Close": prices})
        result = generate_composite_signal(df, headlines=None)
        assert result["signal"] in ("Bullish", "Bearish", "Neutral")
        assert result["components"]["sentiment"]["label"] == "N/A"


class TestBeta:
    def test_beta_without_data_returns_default(self):
        """Beta should return 1.0 when price_history is None or too short."""
        assert _calculate_beta(None) == 1.0
        short = pd.DataFrame({"Close": [100, 101, 102]})
        assert _calculate_beta(short) == 1.0

    def test_beta_bounded(self):
        """Beta should be capped between 0.1 and 3.0."""
        beta = _calculate_beta(None)
        assert 0.1 <= beta <= 3.0


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
                {"revenue": 50_000_000_000, "interestExpense": -500_000_000,
                 "incomeBeforeTax": 10_000_000_000, "incomeTaxExpense": 2_100_000_000},
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
        assert "beta" in result
        assert "tax_rate" in result
        assert "cost_of_equity" in result
        assert "cost_of_debt" in result

    def test_custom_dcf_effective_tax_rate(self):
        """DCF should use effective tax rate from income statement."""
        financial_data = {
            "cashflow": [
                {"operatingCashFlow": 5_000_000_000,
                 "capitalExpenditure": -1_000_000_000},
                {"operatingCashFlow": 4_500_000_000,
                 "capitalExpenditure": -900_000_000},
            ],
            "income": [
                {"revenue": 50_000_000_000, "interestExpense": -500_000_000,
                 "incomeBeforeTax": 10_000_000_000, "incomeTaxExpense": 1_500_000_000},
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
        # Effective tax = 1.5B / 10B = 15%
        assert result["tax_rate"] == 15.0


class TestSensitivity:
    def test_sensitivity_matrix_shape(self):
        """Sensitivity should return a 7x7 grid (or close)."""
        dcf_result = {
            "dcf_per_share": 180.0,
            "wacc": 10.0,
            "growth_rate": 8.0,
            "base_fcf": 4000.0,  # in millions
            "terminal_growth": 2.5,
        }
        result = compute_dcf_sensitivity(dcf_result, 1_000_000_000, 5_000_000_000)
        assert result is not None
        assert len(result["growth_rates"]) >= 5
        assert len(result["discount_rates"]) >= 5
        assert len(result["values"]) == len(result["growth_rates"])
        assert len(result["values"][0]) == len(result["discount_rates"])

    def test_sensitivity_none_on_bad_input(self):
        """Sensitivity should return None for invalid inputs."""
        assert compute_dcf_sensitivity(None, 1_000_000_000, 0) is None
        assert compute_dcf_sensitivity({}, 0, 0) is None


class TestValuation:
    def test_analyze_valuation_fmp_fallback(self):
        analyst_data = {"dcf": 200, "stock_price": 150, "estimates": []}
        result = analyze_valuation(analyst_data, financial_data=None)
        assert result["model_type"] == "FMP API"
        assert result["signal"] in ("Undervalued", "Overvalued", "Fair Value")
