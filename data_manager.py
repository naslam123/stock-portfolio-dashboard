"""
JSON persistence layer for the Trading Simulator.
Handles loading, saving, and default account data.
"""

import json
import os
from config import DATA_FILE


def default_data() -> dict:
    """Return a fresh default account structure.

    Returns:
        Dict with cash, starting_balance, portfolio, watchlist, options,
        pending_orders, journal, theme, and commission settings.
    """
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
    }


def load_data() -> dict:
    """Load account data from JSON file, merging any missing keys from defaults.

    Returns:
        Account data dict. Falls back to default_data() if file is
        missing or corrupt.
    """
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
    """Persist account data to JSON file.

    Args:
        data: The account data dict to save.
    """
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
