"""
SQLite persistence layer for the Trading Simulator.

Replaces the original JSON file storage with normalized SQLite tables while
keeping the same dict interface (load_data / save_data / default_data) so
that no other module needs to change.

Auto-migrates from trading_data.json on first run.
"""

import json
import os
import sqlite3
from config import DATA_FILE, DB_FILE


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


# --------------- Schema ---------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL DEFAULT 100000.0,
    starting_balance REAL NOT NULL DEFAULT 100000.0,
    theme TEXT NOT NULL DEFAULT 'dark',
    commission_enabled INTEGER NOT NULL DEFAULT 1,
    commission_stock REAL NOT NULL DEFAULT 0.0,
    commission_option REAL NOT NULL DEFAULT 0.65,
    colorblind INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    type TEXT NOT NULL,
    order_type TEXT DEFAULT 'market',
    shares REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0.0,
    short INTEGER DEFAULT 0,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS options_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    type TEXT NOT NULL,
    strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    premium REAL,
    action TEXT,
    total REAL,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS price_alerts (
    ticker TEXT PRIMARY KEY,
    target_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS badges (
    name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS pending_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rebalance_targets (
    ticker TEXT PRIMARY KEY,
    weight REAL NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure schema exists."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    # Ensure account row exists
    cur = conn.execute("SELECT COUNT(*) FROM account")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO account (id, cash, starting_balance) VALUES (1, 100000.0, 100000.0)"
        )
        conn.commit()
    return conn


# --------------- Migration from JSON ---------------

def _migrate_from_json():
    """One-time migration: import trading_data.json into SQLite, rename to .bak."""
    if not os.path.exists(DATA_FILE):
        return
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return

    conn = _get_conn()
    try:
        # Account
        conn.execute(
            "UPDATE account SET cash=?, starting_balance=?, theme=?, "
            "commission_enabled=?, commission_stock=?, commission_option=?, colorblind=? "
            "WHERE id=1",
            (
                data.get("cash", 100000.0),
                data.get("starting_balance", 100000.0),
                data.get("theme", "dark"),
                1 if data.get("commission_enabled", True) else 0,
                data.get("commission_stock", 0.0),
                data.get("commission_option", 0.65),
                1 if data.get("colorblind", False) else 0,
            ),
        )

        # Trades
        for t in data.get("portfolio", []):
            conn.execute(
                "INSERT INTO trades (ticker, type, order_type, shares, price, commission, short, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (t["ticker"], t["type"], t.get("order_type", "market"),
                 t["shares"], t["price"], t.get("commission", 0),
                 1 if t.get("short", False) else 0, t.get("timestamp", "")),
            )

        # Options
        for o in data.get("options", []):
            conn.execute(
                "INSERT INTO options_trades (ticker, type, strike, expiration, contracts, premium, action, total, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (o["ticker"], o["type"], o["strike"], o["expiration"],
                 o["contracts"], o.get("premium", 0), o.get("action", ""),
                 o.get("total", 0), o.get("timestamp", "")),
            )

        # Journal
        for j in data.get("journal", []):
            conn.execute(
                "INSERT INTO journal (date, ticker, action, shares, price, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (j["date"], j["ticker"], j["action"], j["shares"], j["price"],
                 j.get("notes", "")),
            )

        # Watchlist
        for tk in data.get("watchlist", []):
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (tk,)
            )

        # Price alerts
        for tk, price in data.get("price_alerts", {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO price_alerts (ticker, target_price) VALUES (?, ?)",
                (tk, price),
            )

        # Portfolio history
        for ph in data.get("portfolio_history", []):
            conn.execute(
                "INSERT INTO portfolio_history (date, value) VALUES (?, ?)",
                (ph["date"], ph["value"]),
            )

        # Badges
        for b in data.get("badges", []):
            conn.execute("INSERT OR IGNORE INTO badges (name) VALUES (?)", (b,))

        # Pending orders (store as JSON blobs)
        for po in data.get("pending_orders", []):
            conn.execute(
                "INSERT INTO pending_orders (data) VALUES (?)", (json.dumps(po),)
            )

        # Rebalance targets
        for tk, w in data.get("rebalance_targets", {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO rebalance_targets (ticker, weight) VALUES (?, ?)",
                (tk, w),
            )

        conn.commit()

        # Rename JSON to .bak
        bak = DATA_FILE + ".bak"
        if os.path.exists(bak):
            os.remove(bak)
        os.rename(DATA_FILE, bak)

    except Exception:
        conn.rollback()
    finally:
        conn.close()


# --------------- Read from SQLite → dict ---------------

def _load_from_db() -> dict:
    """Read all tables into the standard dict format."""
    conn = _get_conn()
    data = default_data()

    try:
        # Account
        row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row:
            data["cash"] = row[1]
            data["starting_balance"] = row[2]
            data["theme"] = row[3]
            data["commission_enabled"] = bool(row[4])
            data["commission_stock"] = row[5]
            data["commission_option"] = row[6]
            data["colorblind"] = bool(row[7])

        # Trades → portfolio
        data["portfolio"] = [
            {"ticker": r[1], "type": r[2], "order_type": r[3], "shares": r[4],
             "price": r[5], "commission": r[6], "short": bool(r[7]), "timestamp": r[8]}
            for r in conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
        ]

        # Options
        data["options"] = [
            {"ticker": r[1], "type": r[2], "strike": r[3], "expiration": r[4],
             "contracts": r[5], "premium": r[6], "action": r[7], "total": r[8],
             "timestamp": r[9]}
            for r in conn.execute("SELECT * FROM options_trades ORDER BY id").fetchall()
        ]

        # Journal
        data["journal"] = [
            {"date": r[1], "ticker": r[2], "action": r[3], "shares": r[4],
             "price": r[5], "notes": r[6]}
            for r in conn.execute("SELECT * FROM journal ORDER BY id").fetchall()
        ]

        # Watchlist
        data["watchlist"] = [
            r[0] for r in conn.execute("SELECT ticker FROM watchlist ORDER BY id").fetchall()
        ]

        # Price alerts
        data["price_alerts"] = {
            r[0]: r[1]
            for r in conn.execute("SELECT ticker, target_price FROM price_alerts").fetchall()
        }

        # Portfolio history
        data["portfolio_history"] = [
            {"date": r[0], "value": r[1]}
            for r in conn.execute("SELECT date, value FROM portfolio_history ORDER BY id").fetchall()
        ]

        # Badges
        data["badges"] = [
            r[0] for r in conn.execute("SELECT name FROM badges").fetchall()
        ]

        # Pending orders
        data["pending_orders"] = [
            json.loads(r[0])
            for r in conn.execute("SELECT data FROM pending_orders ORDER BY id").fetchall()
        ]

        # Rebalance targets
        data["rebalance_targets"] = {
            r[0]: r[1]
            for r in conn.execute("SELECT ticker, weight FROM rebalance_targets").fetchall()
        }

    finally:
        conn.close()

    return data


# --------------- Write dict → SQLite ---------------

def _save_to_db(data: dict) -> None:
    """Sync the full dict back into SQLite (delete + reinsert for list tables)."""
    conn = _get_conn()
    try:
        # Account
        conn.execute(
            "UPDATE account SET cash=?, starting_balance=?, theme=?, "
            "commission_enabled=?, commission_stock=?, commission_option=?, colorblind=? "
            "WHERE id=1",
            (
                data.get("cash", 100000.0),
                data.get("starting_balance", 100000.0),
                data.get("theme", "dark"),
                1 if data.get("commission_enabled", True) else 0,
                data.get("commission_stock", 0.0),
                data.get("commission_option", 0.65),
                1 if data.get("colorblind", False) else 0,
            ),
        )

        # Trades
        conn.execute("DELETE FROM trades")
        for t in data.get("portfolio", []):
            conn.execute(
                "INSERT INTO trades (ticker, type, order_type, shares, price, commission, short, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (t["ticker"], t["type"], t.get("order_type", "market"),
                 t["shares"], t["price"], t.get("commission", 0),
                 1 if t.get("short", False) else 0, t.get("timestamp", "")),
            )

        # Options
        conn.execute("DELETE FROM options_trades")
        for o in data.get("options", []):
            conn.execute(
                "INSERT INTO options_trades (ticker, type, strike, expiration, contracts, premium, action, total, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (o["ticker"], o["type"], o["strike"], o["expiration"],
                 o["contracts"], o.get("premium", 0), o.get("action", ""),
                 o.get("total", 0), o.get("timestamp", "")),
            )

        # Journal
        conn.execute("DELETE FROM journal")
        for j in data.get("journal", []):
            conn.execute(
                "INSERT INTO journal (date, ticker, action, shares, price, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (j["date"], j["ticker"], j["action"], j["shares"], j["price"],
                 j.get("notes", "")),
            )

        # Watchlist
        conn.execute("DELETE FROM watchlist")
        for tk in data.get("watchlist", []):
            conn.execute("INSERT INTO watchlist (ticker) VALUES (?)", (tk,))

        # Price alerts
        conn.execute("DELETE FROM price_alerts")
        for tk, price in data.get("price_alerts", {}).items():
            conn.execute(
                "INSERT INTO price_alerts (ticker, target_price) VALUES (?, ?)",
                (tk, price),
            )

        # Portfolio history
        conn.execute("DELETE FROM portfolio_history")
        for ph in data.get("portfolio_history", []):
            conn.execute(
                "INSERT INTO portfolio_history (date, value) VALUES (?, ?)",
                (ph["date"], ph["value"]),
            )

        # Badges
        conn.execute("DELETE FROM badges")
        for b in data.get("badges", []):
            conn.execute("INSERT INTO badges (name) VALUES (?)", (b,))

        # Pending orders
        conn.execute("DELETE FROM pending_orders")
        for po in data.get("pending_orders", []):
            conn.execute(
                "INSERT INTO pending_orders (data) VALUES (?)", (json.dumps(po),)
            )

        # Rebalance targets
        conn.execute("DELETE FROM rebalance_targets")
        for tk, w in data.get("rebalance_targets", {}).items():
            conn.execute(
                "INSERT INTO rebalance_targets (ticker, weight) VALUES (?, ?)",
                (tk, w),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------- Public API (same interface as original) ---------------

def load_data() -> dict:
    """Load account data from SQLite. Auto-migrates from JSON on first run."""
    # Migrate JSON → SQLite if JSON still exists
    if os.path.exists(DATA_FILE):
        _migrate_from_json()

    return _load_from_db()


def save_data(data: dict) -> None:
    """Persist account data to SQLite."""
    _save_to_db(data)
