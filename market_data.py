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
        key = st.secrets.get("FMP_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("FMP_API_KEY")


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

    # yfinance fallback
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev = info.get("previousClose", price)
        return price, price - prev, ((price - prev) / prev * 100) if prev else 0
    except Exception:
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


# --------------- Analyst Estimates & DCF (FMP stable) ---------------

@st.cache_data(ttl=300)
def get_analyst_data(ticker: str) -> dict:
    """Fetch analyst estimates and DCF valuation from FMP stable API.

    Args:
        ticker: Stock symbol.

    Returns:
        Dict with keys: estimates (list), dcf (float), stock_price (float).
        Empty dict if FMP key is missing or on failure.
    """
    key = _get_fmp_key()
    if not key:
        return {}
    result = {}
    try:
        # Analyst estimates
        url = f"{FMP_STABLE_URL}/analyst-estimates?symbol={ticker}&period=annual&apikey={key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if isinstance(data, list) and data:
            result["estimates"] = data[:3]  # Next 3 fiscal years
    except Exception:
        pass
    try:
        # DCF valuation
        url = f"{FMP_STABLE_URL}/discounted-cash-flow?symbol={ticker}&apikey={key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if isinstance(data, list) and data:
            result["dcf"] = data[0].get("dcf", 0)
            result["stock_price"] = data[0].get("Stock Price", 0)
    except Exception:
        pass
    return result
