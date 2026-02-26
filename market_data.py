"""
Market data layer with FMP primary + yfinance fallback.

- get_price() and get_history(): FMP first, yfinance fallback
- get_info(): yfinance only (FMP profile is paid tier)
- Options functions: yfinance only (FMP options is premium)
- get_news_sentiment(): FMP free endpoint

Works fully without an FMP API key (silent yfinance fallback).
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import requests

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"


def _get_fmp_key() -> str | None:
    """Read FMP API key from Streamlit secrets or environment variable.

    Returns:
        API key string, or None if not configured.
    """
    try:
        return st.secrets["FMP_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    # Last resort: read from secrets.toml directly
    try:
        import tomllib
        app_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(app_dir, ".streamlit", "secrets.toml")
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
            return secrets.get("FMP_API_KEY")
    except Exception:
        pass
    return None


# --------------- Price ---------------

@st.cache_data(ttl=60)
def get_price(ticker: str) -> tuple[float, float, float]:
    """Fetch current price, dollar change, and percent change.

    Tries FMP real-time quote first; falls back to yfinance.

    Args:
        ticker: Stock symbol (e.g. "AAPL").

    Returns:
        (price, dollar_change, pct_change). Returns (0, 0, 0) on failure.
    """
    key = _get_fmp_key()
    if key:
        try:
            url = f"{FMP_STABLE_URL}/quote?symbol={ticker}&apikey={key}"
            resp = requests.get(url, timeout=5)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    price = item.get("price", 0)
                    if price:
                        prev = item.get("previousClose", price)
                        chg = price - prev
                        pct = (chg / prev * 100) if prev else 0
                        return price, chg, pct
        except Exception:
            pass  # Fall through to yfinance

    # yfinance info fallback
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if price:
            prev = info.get("previousClose", price)
            return price, price - prev, ((price - prev) / prev * 100) if prev else 0
    except Exception:
        pass

    # yfinance history fallback (last close price)
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                chg = price - prev
                pct = (chg / prev * 100) if prev else 0
                return price, chg, pct
            return price, 0, 0
    except Exception:
        pass

    return 0, 0, 0


# --------------- Info ---------------

@st.cache_data(ttl=60)
def get_info(ticker: str) -> dict:
    """Fetch detailed stock info from yfinance.

    Args:
        ticker: Stock symbol.

    Returns:
        Info dict from yfinance, or empty dict on failure.
    """
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


# --------------- History ---------------

@st.cache_data(ttl=300)
def get_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch historical price data.

    Tries FMP daily history first; falls back to yfinance.

    Args:
        ticker: Stock symbol.
        period: Time period string (e.g. "6mo", "1y"). Used by yfinance
                fallback; FMP always fetches last 180 days.

    Returns:
        DataFrame with OHLCV columns. Empty DataFrame on failure.
    """
    key = _get_fmp_key()
    if key:
        try:
            url = (
                f"{FMP_STABLE_URL}/historical-price-eod/full"
                f"?symbol={ticker}&apikey={key}"
            )
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                raise ValueError(f"FMP returned {resp.status_code}")
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                })
                if "Open" in df.columns:
                    return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass  # Fall through to yfinance

    # yfinance fallback
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


# --------------- Options (yfinance only) ---------------

def get_option_dates(ticker: str) -> list[str]:
    """Get option expiration dates for a ticker.

    Args:
        ticker: Stock symbol.

    Returns:
        List of expiration date strings, or empty list on failure.
    """
    try:
        t = yf.Ticker(ticker)
        return list(t.options) if t.options else []
    except Exception:
        return []


def get_option_chain_data(ticker: str, exp_date: str, opt_type: str) -> pd.DataFrame:
    """Get option chain as DataFrame (not cached to avoid serialization issues).

    Args:
        ticker: Stock symbol.
        exp_date: Expiration date string.
        opt_type: "Call" or "Put".

    Returns:
        DataFrame with option chain data, or empty DataFrame on failure.
    """
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(exp_date)
        df = chain.calls if opt_type == "Call" else chain.puts
        return df
    except Exception:
        return pd.DataFrame()


# --------------- Financial Statements (FMP stable) ---------------

@st.cache_data(ttl=600)
def get_financial_data(ticker: str) -> dict:
    """Fetch income statement, cash flow, and balance sheet from FMP.

    Returns:
        Dict with keys: income (list), cashflow (list), balance (list).
        Returns {"error": "reason"} on failure.
    """
    key = _get_fmp_key()
    if not key:
        return {"error": "no_key"}
    result = {}
    endpoints = {
        "income": f"{FMP_STABLE_URL}/income-statement?symbol={ticker}&period=annual&apikey={key}",
        "cashflow": f"{FMP_STABLE_URL}/cash-flow-statement?symbol={ticker}&period=annual&apikey={key}",
        "balance": f"{FMP_STABLE_URL}/balance-sheet-statement?symbol={ticker}&period=annual&apikey={key}",
    }
    for name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                result[name] = []
                continue
            data = resp.json()
            if isinstance(data, list) and data:
                result[name] = data[:5]  # Last 5 years
            else:
                result[name] = []
        except Exception:
            result[name] = []
    if not any(result.get(k) for k in ["income", "cashflow", "balance"]):
        result["error"] = "No financial data available"
    return result


# --------------- Analyst Estimates & DCF (FMP stable) ---------------

@st.cache_data(ttl=300)
def get_analyst_data(ticker: str) -> dict:
    """Fetch analyst estimates and DCF valuation from FMP stable API.

    Args:
        ticker: Stock symbol.

    Returns:
        Dict with keys: estimates (list), dcf (float), stock_price (float).
        Returns {"error": "reason"} if key missing or API fails.
    """
    key = _get_fmp_key()
    if not key:
        return {"error": "no_key"}
    result = {}
    try:
        url = f"{FMP_STABLE_URL}/analyst-estimates?symbol={ticker}&period=annual&apikey={key}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and data:
                result["estimates"] = data[:3]
            elif isinstance(data, dict) and "error" in str(data).lower():
                result["error"] = f"FMP: {data}"
        else:
            result["error"] = "FMP API rate-limited"
    except Exception as e:
        result["error"] = f"estimates: {e}"
    try:
        url = f"{FMP_STABLE_URL}/discounted-cash-flow?symbol={ticker}&apikey={key}"
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            raise ValueError(f"FMP returned {resp.status_code}")
        data = resp.json()
        if isinstance(data, list) and data:
            result["dcf"] = data[0].get("dcf", 0)
            result["stock_price"] = data[0].get("Stock Price", 0)
    except Exception as e:
        if "error" not in result:
            result["error"] = f"dcf: {e}"
    return result
