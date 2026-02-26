"""Tests for data_manager.py — SQLite persistence and JSON migration."""

import json
import os
import sys
import types

# Mock config module before importing data_manager
if "config" not in sys.modules:
    config_mod = types.ModuleType("config")
    config_mod.DATA_FILE = "trading_data.json"
    config_mod.DB_FILE = "trading_data.db"
    # Mock sp500_tickers dependency of config
    if "sp500_tickers" not in sys.modules:
        sp = types.ModuleType("sp500_tickers")
        sp.SP500 = {"AAPL": "Apple Inc."}
        sys.modules["sp500_tickers"] = sp
    sys.modules["config"] = config_mod

from data_manager import default_data, load_data, save_data
import data_manager


class TestDefaultData:
    def test_default_data_schema(self):
        d = default_data()
        required_keys = [
            "cash", "starting_balance", "portfolio", "watchlist",
            "options", "pending_orders", "journal", "theme",
            "commission_enabled", "commission_stock", "commission_option",
            "portfolio_history", "badges", "price_alerts", "colorblind",
            "rebalance_targets",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"
        assert d["cash"] == 100000.0


class TestSQLiteRoundtrip:
    def test_sqlite_roundtrip(self, tmp_db, monkeypatch):
        monkeypatch.setattr(data_manager, "DB_FILE", tmp_db)
        monkeypatch.setattr(data_manager, "DATA_FILE", tmp_db + ".json")

        data = default_data()
        data["cash"] = 75000.0
        data["portfolio"] = [
            {"ticker": "AAPL", "type": "buy", "order_type": "market",
             "shares": 10, "price": 150.0, "commission": 0,
             "short": False, "timestamp": "2024-01-01"},
        ]
        data["journal"] = [
            {"date": "2024-01-01", "ticker": "AAPL", "action": "buy",
             "shares": 10, "price": 150.0, "notes": "test"},
        ]
        data["watchlist"] = ["MSFT", "GOOG"]
        data["badges"] = ["First Trade"]

        save_data(data)
        loaded = load_data()

        assert loaded["cash"] == 75000.0
        assert len(loaded["portfolio"]) == 1
        assert loaded["portfolio"][0]["ticker"] == "AAPL"
        assert loaded["watchlist"] == ["MSFT", "GOOG"]
        assert loaded["badges"] == ["First Trade"]
        assert len(loaded["journal"]) == 1


class TestJSONMigration:
    def test_json_migration(self, tmp_path, monkeypatch):
        json_path = str(tmp_path / "trading_data.json")
        db_path = str(tmp_path / "trading_data.db")

        monkeypatch.setattr(data_manager, "DATA_FILE", json_path)
        monkeypatch.setattr(data_manager, "DB_FILE", db_path)

        # Create a JSON file to migrate
        data = default_data()
        data["cash"] = 88000.0
        data["watchlist"] = ["TSLA"]
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Load should trigger migration
        loaded = load_data()
        assert loaded["cash"] == 88000.0
        assert loaded["watchlist"] == ["TSLA"]

        # JSON should be renamed to .bak
        assert not os.path.exists(json_path)
        assert os.path.exists(json_path + ".bak")
        # DB should exist
        assert os.path.exists(db_path)


class TestLoadEmptyDB:
    def test_load_empty_db(self, tmp_path, monkeypatch):
        json_path = str(tmp_path / "no_exist.json")
        db_path = str(tmp_path / "fresh.db")

        monkeypatch.setattr(data_manager, "DATA_FILE", json_path)
        monkeypatch.setattr(data_manager, "DB_FILE", db_path)

        loaded = load_data()
        assert loaded["cash"] == 100000.0
        assert loaded["portfolio"] == []
