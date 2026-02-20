"""
JSON persistence layer for the Trading Simulator.
"""

import json
import os
from config import DATA_FILE


def default_data() -> dict:
    """Return a fresh default account structure."""
    return {
        "cash": 100000.0,
        "starting_balance": 100000.0,
        "portfolio": [],
        "watchlist": [],
        "options": [],
        "pending_orders": [],
        "journal": [],
        "theme": "dark",
        "commission_enabled": True,
        "commission_stock": 0.0,
        "commission_option": 0.65,
        "portfolio_history": [],
        "badges": [],
        "price_alerts": {},
        "colorblind": False,
        "rebalance_targets": {},
    }


def load_data() -> dict:
    """Load account data from JSON, merging missing keys from defaults."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for key, val in default_data().items():
                    if key not in data:
                        data[key] = val
                return data
        except Exception:
            return default_data()
    return default_data()


def save_data(data: dict) -> None:
    """Persist account data to JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
