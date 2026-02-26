"""Tests for portfolio.py — holdings calculation and trade stats."""

import sys
import types
import pytest


def _make_session_state(data):
    """Set up the fake st.session_state.data for portfolio imports."""
    sys.modules["streamlit"].session_state.data = data


# We need to mock market_data before importing portfolio
if "market_data" not in sys.modules:
    md = types.ModuleType("market_data")
    md.get_price = lambda ticker: (100.0, 0.0, "N/A")
    md.get_option_chain_data = lambda *a, **kw: None
    sys.modules["market_data"] = md

from portfolio import get_holdings, get_trade_stats


class TestGetHoldings:
    def test_buy_increases_shares_and_cost(self):
        _make_session_state({
            "portfolio": [
                {"ticker": "AAPL", "type": "buy", "shares": 100, "price": 10.0}
            ],
            "options": [],
        })
        h = get_holdings()
        assert h["AAPL"]["shares"] == 100
        assert h["AAPL"]["cost"] == 1000.0

    def test_sell_reduces_cost_proportionally(self):
        _make_session_state({
            "portfolio": [
                {"ticker": "AAPL", "type": "buy", "shares": 100, "price": 10.0},
                {"ticker": "AAPL", "type": "sell", "shares": 50, "price": 15.0},
            ],
            "options": [],
        })
        h = get_holdings()
        assert h["AAPL"]["shares"] == 50
        assert h["AAPL"]["cost"] == pytest.approx(500.0)

    def test_multiple_buys_then_sell(self):
        _make_session_state({
            "portfolio": [
                {"ticker": "MSFT", "type": "buy", "shares": 50, "price": 10.0},
                {"ticker": "MSFT", "type": "buy", "shares": 50, "price": 20.0},
                {"ticker": "MSFT", "type": "sell", "shares": 50, "price": 25.0},
            ],
            "options": [],
        })
        h = get_holdings()
        # avg cost = (500+1000)/100 = 15, sell 50 removes 50*15=750
        assert h["MSFT"]["shares"] == 50
        assert h["MSFT"]["cost"] == pytest.approx(750.0)

    def test_empty_portfolio(self):
        _make_session_state({"portfolio": [], "options": []})
        h = get_holdings()
        assert h == {}

    def test_sell_all_removes_holding(self):
        _make_session_state({
            "portfolio": [
                {"ticker": "TSLA", "type": "buy", "shares": 100, "price": 10.0},
                {"ticker": "TSLA", "type": "sell", "shares": 100, "price": 15.0},
            ],
            "options": [],
        })
        h = get_holdings()
        assert "TSLA" not in h


class TestTradeStats:
    def test_empty_journal(self):
        stats = get_trade_stats([])
        assert stats["total"] == 0
        assert stats["win_rate"] == 0

    def test_one_winning_trade(self):
        journal = [
            {"ticker": "AAPL", "action": "buy", "price": 100, "shares": 10,
             "date": "2024-01-01"},
            {"ticker": "AAPL", "action": "sell", "price": 120, "shares": 10,
             "date": "2024-01-02"},
        ]
        stats = get_trade_stats(journal)
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["win_rate"] == 100.0
