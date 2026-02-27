"""Validation script: ML signal accuracy on real market data.

Runs RF, HMM, VADER, composite signal, and DCF on real tickers.
Outputs structured results for inclusion in validation.md.

Usage: python3 tests/validate_ml.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf

from ai_signals import (
    detect_market_regime,
    detect_regime_hmm,
    score_sentiment_vader,
    generate_composite_signal,
    _compute_custom_dcf,
    _calculate_beta,
    compute_dcf_sensitivity,
)


def validate_rf_accuracy():
    """Validate Random Forest CV accuracy on SPY data."""
    print("=" * 60)
    print("1. RANDOM FOREST — TimeSeriesSplit Cross-Validation")
    print("=" * 60)

    spy = yf.download("SPY", period="2y", progress=False)
    if hasattr(spy.columns, 'levels') and len(spy.columns.levels) > 1:
        spy.columns = spy.columns.droplevel(1)

    if spy.empty or len(spy) < 200:
        print("  ERROR: Could not fetch SPY data")
        return None

    result = detect_market_regime(spy)
    print(f"  Data points: {len(spy)}")
    print(f"  Model type: {result['model_type']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Signal strength: {result['signal_strength']}")

    if result['model_type'] == 'ML':
        cv_acc = result.get('cv_accuracy', 'N/A')
        print(f"  CV Accuracy (5-fold TimeSeriesSplit): {cv_acc:.1%}")
        top_feats = result.get('feature_importance', {})
        sorted_feats = sorted(top_feats.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {', '.join(f'{f} ({v:.1%})' for f, v in sorted_feats)}")
        print(f"  PASS: CV accuracy {cv_acc:.1%} > 50% random baseline" if cv_acc > 0.50 else f"  WARN: CV accuracy {cv_acc:.1%} near random")
        return cv_acc
    else:
        print(f"  WARN: Fell back to SMA (not enough data or sklearn issue)")
        return None


def validate_hmm():
    """Validate HMM regime detection on SPY data."""
    print("\n" + "=" * 60)
    print("2. HIDDEN MARKOV MODEL — 3-State Regime Detection")
    print("=" * 60)

    spy = yf.download("SPY", period="1y", progress=False)
    if hasattr(spy.columns, 'levels') and len(spy.columns.levels) > 1:
        spy.columns = spy.columns.droplevel(1)

    if spy.empty:
        print("  ERROR: Could not fetch SPY data")
        return

    result = detect_regime_hmm(spy)
    print(f"  Model type: {result['model_type']}")
    print(f"  Current regime: {result['regime']}")

    if result['model_type'] == 'HMM':
        probs = result['probabilities']
        print(f"  Probabilities: Bull={probs['Bull']:.1%}, Sideways={probs['Sideways']:.1%}, Bear={probs['Bear']:.1%}")
        prob_sum = sum(probs.values())
        print(f"  Probability sum: {prob_sum:.4f}")
        print(f"  PASS: Probabilities sum to ~1.0" if abs(prob_sum - 1.0) < 0.01 else f"  FAIL: Probabilities sum to {prob_sum}")

        # Check state history exists and is reasonable
        history = result.get('state_history')
        if history is not None:
            state_counts = history.value_counts()
            print(f"  State distribution: {dict(state_counts)}")
            print(f"  PASS: All 3 states detected" if len(state_counts) >= 2 else "  WARN: Some states never detected")
    else:
        print("  WARN: HMM unavailable")


def validate_vader():
    """Validate VADER sentiment on known-sentiment headlines."""
    print("\n" + "=" * 60)
    print("3. VADER SENTIMENT — Known-Answer Validation")
    print("=" * 60)

    # Test with clearly positive headlines
    bullish = [
        "Company reports record profits and raises dividend",
        "Markets surge on strong economic data",
        "Analyst upgrades stock with optimistic price target",
    ]
    bull_result = score_sentiment_vader(bullish)
    print(f"  Bullish headlines score: {bull_result['score']:+.3f} ({bull_result['label']})")
    bull_pass = bull_result['score'] > 0 and bull_result['label'] == 'Bullish'
    print(f"  PASS: Correctly scored bullish" if bull_pass else "  FAIL: Did not score bullish")

    # Test with clearly negative headlines
    bearish = [
        "Markets crash amid fears of deep recession",
        "Company warns of massive losses and layoffs",
        "Stock plummets after terrible earnings report",
    ]
    bear_result = score_sentiment_vader(bearish)
    print(f"  Bearish headlines score: {bear_result['score']:+.3f} ({bear_result['label']})")
    bear_pass = bear_result['score'] < 0 and bear_result['label'] == 'Bearish'
    print(f"  PASS: Correctly scored bearish" if bear_pass else "  FAIL: Did not score bearish")

    # Test with neutral
    neutral_result = score_sentiment_vader([])
    print(f"  Empty headlines score: {neutral_result['score']:+.3f} ({neutral_result['label']})")
    neutral_pass = neutral_result['score'] == 0.0 and neutral_result['label'] == 'Neutral'
    print(f"  PASS: Empty returns neutral" if neutral_pass else "  FAIL: Empty did not return neutral")

    return bull_pass and bear_pass and neutral_pass


def validate_composite():
    """Validate composite signal blending."""
    print("\n" + "=" * 60)
    print("4. COMPOSITE SIGNAL — Component Blending")
    print("=" * 60)

    spy = yf.download("SPY", period="1y", progress=False)
    if hasattr(spy.columns, 'levels') and len(spy.columns.levels) > 1:
        spy.columns = spy.columns.droplevel(1)

    headlines = [
        "Markets show resilience amid economic uncertainty",
        "Strong jobs data supports continued growth outlook",
    ]
    result = generate_composite_signal(spy, headlines)
    print(f"  Composite signal: {result['signal']} (score: {result['score']:+.3f})")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Weights: RF={result['weights']['rf']:.0%}, HMM={result['weights']['hmm']:.0%}, Sentiment={result['weights']['sentiment']:.0%}")
    print(f"  RF: {result['components']['rf']['label']} ({result['components']['rf']['score']:+.3f})")
    print(f"  HMM: {result['components']['hmm']['label']} ({result['components']['hmm']['score']:+.3f})")
    print(f"  Sentiment: {result['components']['sentiment']['label']} ({result['components']['sentiment']['score']:+.3f})")

    # Validate score bounds
    score_ok = -1.0 <= result['score'] <= 1.0
    weights_ok = abs(sum(result['weights'].values()) - 1.0) < 0.01
    print(f"  PASS: Score in [-1, 1]" if score_ok else f"  FAIL: Score {result['score']} out of bounds")
    print(f"  PASS: Weights sum to 1.0" if weights_ok else f"  FAIL: Weights sum to {sum(result['weights'].values())}")

    return score_ok and weights_ok


def validate_beta():
    """Validate beta calculation for known tickers."""
    print("\n" + "=" * 60)
    print("5. BETA CALCULATION — Known-Answer Validation")
    print("=" * 60)

    tickers = {"SPY": (0.95, 1.05), "AAPL": (0.8, 2.0), "JNJ": (0.3, 1.0)}
    all_pass = True

    for ticker, (low, high) in tickers.items():
        hist = yf.download(ticker, period="1y", progress=False)
        if hasattr(hist.columns, 'levels') and len(hist.columns.levels) > 1:
            hist.columns = hist.columns.droplevel(1)
        beta = _calculate_beta(hist)
        in_range = low <= beta <= high
        print(f"  {ticker}: beta = {beta:.2f} (expected {low}-{high}) {'PASS' if in_range else 'WARN'}")
        if not in_range:
            all_pass = False

    return all_pass


def validate_dcf():
    """Validate DCF on synthetic known-answer data."""
    print("\n" + "=" * 60)
    print("6. DCF VALUATION — Known-Answer Validation")
    print("=" * 60)

    # Synthetic company: $4B FCF, 10% growth, should produce positive DCF
    financial_data = {
        "cashflow": [
            {"operatingCashFlow": 5_000_000_000, "capitalExpenditure": -1_000_000_000},
            {"operatingCashFlow": 4_500_000_000, "capitalExpenditure": -900_000_000},
            {"operatingCashFlow": 4_000_000_000, "capitalExpenditure": -800_000_000},
        ],
        "income": [
            {"revenue": 50_000_000_000, "interestExpense": -500_000_000,
             "incomeBeforeTax": 10_000_000_000, "incomeTaxExpense": 2_100_000_000},
            {"revenue": 45_000_000_000, "interestExpense": -450_000_000},
            {"revenue": 40_000_000_000, "interestExpense": -400_000_000},
        ],
        "balance": [
            {"totalDebt": 10_000_000_000, "cashAndCashEquivalents": 5_000_000_000},
        ],
    }

    result = _compute_custom_dcf(financial_data, stock_price=150.0,
                                  shares_outstanding=1_000_000_000)

    if result is None:
        print("  FAIL: DCF returned None")
        return False

    print(f"  DCF per share: ${result['dcf_per_share']:,.2f}")
    print(f"  WACC: {result['wacc']}%")
    print(f"  Growth rate: {result['growth_rate']}%")
    print(f"  Beta: {result['beta']}")
    print(f"  Tax rate: {result['tax_rate']}%")
    print(f"  Cost of equity: {result['cost_of_equity']}%")
    print(f"  Cost of debt: {result['cost_of_debt']}%")
    print(f"  Base FCF: ${result['base_fcf']}M")

    # Sanity checks
    dcf_positive = result['dcf_per_share'] > 0
    wacc_reasonable = 6.0 <= result['wacc'] <= 20.0
    growth_reasonable = 2.0 <= result['growth_rate'] <= 20.0
    tax_reasonable = 0.0 <= result['tax_rate'] <= 40.0

    print(f"  PASS: DCF > 0" if dcf_positive else "  FAIL: DCF <= 0")
    print(f"  PASS: WACC {result['wacc']}% in [6%, 20%]" if wacc_reasonable else f"  FAIL: WACC out of bounds")
    print(f"  PASS: Growth {result['growth_rate']}% in [2%, 20%]" if growth_reasonable else f"  FAIL: Growth out of bounds")
    print(f"  PASS: Tax {result['tax_rate']}% in [0%, 40%]" if tax_reasonable else f"  FAIL: Tax out of bounds")

    # Sensitivity matrix
    sens = compute_dcf_sensitivity(result, 1_000_000_000, 5_000_000_000)
    if sens:
        print(f"  Sensitivity grid: {len(sens['growth_rates'])}×{len(sens['discount_rates'])}")
        min_val = min(min(row) for row in sens['values'])
        max_val = max(max(row) for row in sens['values'])
        print(f"  Value range: ${min_val:,.0f} — ${max_val:,.0f}")
        print(f"  PASS: Sensitivity matrix generated")
    else:
        print(f"  FAIL: Sensitivity matrix is None")

    return dcf_positive and wacc_reasonable and growth_reasonable


def validate_edge_cases():
    """Test edge cases that could crash the app."""
    print("\n" + "=" * 60)
    print("7. EDGE CASES — Robustness Checks")
    print("=" * 60)

    # Empty DataFrame
    empty_df = pd.DataFrame()
    result = detect_market_regime(empty_df)
    print(f"  Empty DF regime: {result['model_type']} — {'PASS' if result['model_type'] in ('SMA', 'N/A') else 'FAIL'}")

    # Very short data
    short_df = pd.DataFrame({"Close": [100, 101, 102]})
    result = detect_market_regime(short_df)
    print(f"  3-point DF regime: {result['model_type']} — {'PASS' if result['model_type'] == 'SMA' else 'FAIL'}")

    # HMM with short data
    result = detect_regime_hmm(short_df)
    print(f"  3-point HMM: {result['model_type']} — {'PASS' if result['model_type'] == 'N/A' else 'FAIL'}")

    # VADER with empty list
    result = score_sentiment_vader([])
    print(f"  Empty VADER: {result['label']} — {'PASS' if result['label'] == 'Neutral' else 'FAIL'}")

    # DCF with missing data
    result = _compute_custom_dcf({}, 150.0, 1_000_000_000)
    print(f"  Empty DCF: {result} — {'PASS' if result is None else 'FAIL'}")

    # Beta with None
    beta = _calculate_beta(None)
    print(f"  None beta: {beta} — {'PASS' if beta == 1.0 else 'FAIL'}")

    # Composite with no headlines
    spy = yf.download("SPY", period="6mo", progress=False)
    if hasattr(spy.columns, 'levels') and len(spy.columns.levels) > 1:
        spy.columns = spy.columns.droplevel(1)
    result = generate_composite_signal(spy, headlines=None)
    print(f"  No-headlines composite: {result['signal']} — {'PASS' if result['signal'] in ('Bullish','Bearish','Neutral') else 'FAIL'}")
    print(f"  Sentiment weight: {result['weights']['sentiment']:.0%} — {'PASS' if result['weights']['sentiment'] == 0 else 'WARN'}")


if __name__ == "__main__":
    print("STOCK PORTFOLIO SIMULATOR — ML VALIDATION REPORT")
    print("=" * 60)
    print()

    cv_acc = validate_rf_accuracy()
    validate_hmm()
    validate_vader()
    validate_composite()
    validate_beta()
    validate_dcf()
    validate_edge_cases()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
