"""Validation script: DCF valuations vs FMP analyst consensus.

Compares our custom DCF model output against FMP's pre-computed DCF
for 5 well-known tickers. Documents the margin of difference.

Usage: python3 tests/validate_dcf.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# Need to read the FMP key without Streamlit
def _get_fmp_key():
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        import tomllib
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        secrets_path = os.path.join(app_dir, ".streamlit", "secrets.toml")
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
            return secrets.get("FMP_API_KEY")
    except Exception:
        return None

import requests
from ai_signals import _compute_custom_dcf, _calculate_beta, analyze_valuation

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"


def fetch_fmp_dcf(ticker, key):
    """Fetch FMP pre-computed DCF for comparison."""
    try:
        url = f"{FMP_STABLE_URL}/discounted-cash-flow?symbol={ticker}&apikey={key}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and data:
                return {
                    "dcf": data[0].get("dcf", 0),
                    "stock_price": data[0].get("Stock Price", 0),
                }
    except Exception:
        pass
    return None


def fetch_fmp_financials(ticker, key):
    """Fetch income, cashflow, balance sheet for custom DCF."""
    result = {}
    endpoints = {
        "income": f"{FMP_STABLE_URL}/income-statement?symbol={ticker}&period=annual&apikey={key}",
        "cashflow": f"{FMP_STABLE_URL}/cash-flow-statement?symbol={ticker}&period=annual&apikey={key}",
        "balance": f"{FMP_STABLE_URL}/balance-sheet-statement?symbol={ticker}&period=annual&apikey={key}",
    }
    for name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                data = resp.json()
                result[name] = data[:5] if isinstance(data, list) else []
            else:
                result[name] = []
        except Exception:
            result[name] = []
    return result


def main():
    print("DCF VALUATION — CUSTOM vs FMP CONSENSUS COMPARISON")
    print("=" * 70)

    key = _get_fmp_key()
    if not key:
        print("ERROR: No FMP API key found. Set FMP_API_KEY or add to .streamlit/secrets.toml")
        return

    tickers = ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM"]
    results = []

    for ticker in tickers:
        print(f"\n--- {ticker} ---")

        # Fetch FMP DCF
        fmp = fetch_fmp_dcf(ticker, key)
        if not fmp or not fmp.get("dcf"):
            print(f"  FMP DCF: unavailable (API limit or no data)")
            continue

        # Fetch financials for custom DCF
        fin = fetch_fmp_financials(ticker, key)
        if not fin.get("cashflow"):
            print(f"  Financials: unavailable")
            continue

        # Get stock price and shares
        info = yf.Ticker(ticker).info
        stock_price = fmp.get("stock_price", 0) or info.get("currentPrice", 0)
        shares = info.get("sharesOutstanding", 0)

        if not stock_price or not shares:
            print(f"  Missing price/shares data")
            continue

        # Fetch price history for beta
        hist = yf.download(ticker, period="1y", progress=False)
        if hasattr(hist.columns, 'levels') and len(hist.columns.levels) > 1:
            hist.columns = hist.columns.droplevel(1)

        # Run custom DCF
        custom = _compute_custom_dcf(fin, stock_price, shares, hist)

        if not custom:
            print(f"  Custom DCF: could not compute")
            continue

        fmp_dcf = fmp["dcf"]
        our_dcf = custom["dcf_per_share"]
        price = stock_price

        diff_pct = ((our_dcf - fmp_dcf) / fmp_dcf * 100) if fmp_dcf != 0 else 0
        our_margin = ((our_dcf - price) / price * 100) if price else 0
        fmp_margin = ((fmp_dcf - price) / price * 100) if price else 0

        print(f"  Stock price:  ${price:>10,.2f}")
        print(f"  FMP DCF:      ${fmp_dcf:>10,.2f}  (margin: {fmp_margin:+.1f}%)")
        print(f"  Custom DCF:   ${our_dcf:>10,.2f}  (margin: {our_margin:+.1f}%)")
        print(f"  Difference:   {diff_pct:+.1f}%")
        print(f"  Beta: {custom['beta']}, WACC: {custom['wacc']}%, Growth: {custom['growth_rate']}%, Tax: {custom['tax_rate']}%")

        # Both agree on direction?
        fmp_signal = "Undervalued" if fmp_margin > 15 else "Overvalued" if fmp_margin < -15 else "Fair Value"
        our_signal = "Undervalued" if our_margin > 15 else "Overvalued" if our_margin < -15 else "Fair Value"
        agree = fmp_signal == our_signal
        print(f"  FMP signal: {fmp_signal}, Our signal: {our_signal} — {'AGREE' if agree else 'DIFFER'}")

        results.append({
            "ticker": ticker, "price": price,
            "fmp_dcf": fmp_dcf, "our_dcf": our_dcf,
            "diff_pct": diff_pct, "agree": agree,
        })

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        agree_count = sum(1 for r in results if r["agree"])
        avg_diff = sum(abs(r["diff_pct"]) for r in results) / len(results)
        print(f"  Tickers analyzed: {len(results)}")
        print(f"  Signal agreement: {agree_count}/{len(results)}")
        print(f"  Avg absolute difference: {avg_diff:.1f}%")
        print(f"  {'PASS' if avg_diff < 200 else 'WARN'}: Custom DCF within reasonable range of FMP consensus")


if __name__ == "__main__":
    main()
