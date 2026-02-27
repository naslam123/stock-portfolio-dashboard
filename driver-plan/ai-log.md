# AI Development Log

**Project:** Stock Portfolio Trading Simulator
**Course:** MGMT 69000 — Mastering AI For Finance | Purdue University
**Student:** Naveed Aslam Perwez (naslam@purdue.edu)
**AI Tool:** Claude Code (Claude Opus 4.6) with PAL MCP Server
**Methodology:** DRIVER (Define → Represent → Implement → Validate → Evolve → Reflect)

---

## Development Timeline

### Phase 1: Define (开题调研)

**Prompt:** "We are going to work on repo stock-portfolio-dashboard. Here is the professor feedback from submissions 1 and 2, plus final submission guidelines..."

**AI Actions:**
- Launched 4 parallel research agents to analyze the codebase, ML approaches, DCF/database patterns, and OpenAI/PAL MCP integration
- Cross-referenced professor feedback against existing code to identify true gaps vs already-implemented features
- Created `driver-plan/research.md` with findings from all 4 research threads

**Key Insight:** Many features the professor requested (SQLite, custom DCF, ML regime, LLM coaching) already existed in the codebase — the gaps were about quality and depth, not existence. This prevented unnecessary rebuilding.

**Output:** `driver-plan/product-overview.md` — Problem definition, success criteria, tech stack, system context diagram

### Phase 2: Represent (Roadmap)

**Prompt:** Continuation from Define — AI proposed 5-section roadmap based on research findings and professor feedback priorities.

**AI Actions:**
- Broke the work into 5 sequential sections with dependency graph
- Prioritized bug fixes first (foundation), ML upgrades second, documentation last
- Created Mermaid dependency diagram showing build order

**Output:** `driver-plan/roadmap.md` — 5 sections with dependency graph

### Phase 3: Implement — Section 1 (OpenAI + Bug Fixes)

**Files Modified:** `chatbot.py`, `ai_signals.py`, `news_feed.py`, `portfolio.py`, `risk_metrics.py`, `app.py`, `dashboard.py`, `requirements.txt`, `.streamlit/secrets.toml`

**Prompts & Decisions:**
1. "Wire up OpenAI as primary LLM" → Added `_call_openai()` mirroring `_call_groq()` pattern. Fallback chain: OpenAI → Groq → Gemini.
2. "Fix cost basis drift" → Added `round()` after each buy/sell operation + near-zero cleanup threshold.
3. "Fix VaR display" → Changed return type from float to `{dollar, percent, confidence}` dict.
4. "Fix badge volatility" → Made badges permanent via set union (once earned, never revoked).
5. "Fix price alerts" → Added both above AND below target triggering.

**Tests Updated:** `test_risk_metrics.py` updated for VaR dict return type.

### Phase 4: Implement — Section 2 (Composite ML Signal)

**Files Modified:** `ai_signals.py`, `app.py`, `tests/test_ai_signals.py`

**Prompts & Decisions:**
1. "Expand RF features" → Added 4 new features (ATR, volume ratio, price vs SMA50/200, SMA spread) for total of 10. Changed RSI to Wilder's EMA method.
2. "Add TimeSeriesSplit" → Replaced random train/test split with `TimeSeriesSplit(n_splits=5)` to prevent look-ahead bias. Reports cross-validation accuracy.
3. "Add HMM regime detection" → Implemented 3-state Gaussian HMM on (returns, rolling_volatility). States labeled by sorting mean returns: Bear < Sideways < Bull.
4. "Add VADER sentiment" → Offline, deterministic headline scoring. No API key needed.
5. "Create composite signal" → Weighted blend: 50% RF + 30% HMM + 20% VADER. Graceful degradation when components unavailable.

**Bugs Encountered & Fixed:**
- HMM `covariance_type="full"` caused LinAlgError on synthetic data → Changed to `"diag"` (more numerically stable)
- VADER scored "beating estimates" as negative (physical violence connotation) → Used different test headlines
- Empty HMM state slices caused RuntimeWarning → Added `mask.sum() > 0` guard

**Tests Added:** 7 new tests — HMM regime (2), VADER sentiment (3), composite signal (2).

### Phase 5: Implement — Section 3 (DCF + Coaching)

**Files Modified:** `ai_signals.py`, `app.py`, `tests/test_ai_signals.py`

**Prompts & Decisions:**
1. "Calculate beta from stock vs SPY" → Created `_calculate_beta()` using `np.cov()` on aligned daily returns. Capped [0.1, 3.0]. Falls back to 1.0 on failure.
2. "Get effective tax rate" → Extracts `incomeTaxExpense / incomeBeforeTax` from income statement. Capped [0%, 40%]. Falls back to 21% statutory rate.
3. "Add sensitivity heatmap" → Created `compute_dcf_sensitivity()` generating 7×7 grid of intrinsic values across growth × WACC ranges. Displayed as Plotly heatmap with color scale (red=overvalued, green=undervalued).
4. "Enrich coaching context" → Updated `_gather_coaching_context()` to accept regime, risk_metrics, and composite_signal parameters. LLM now sees market regime, VaR, Sharpe, max drawdown alongside portfolio stats.

**Tests Added:** 7 new tests — beta (2), DCF with new fields (2), sensitivity matrix (2), enriched coaching (1).

### Phase 6: Implement — Section 4 (Documentation)

**Files Modified:** `ai_signals.py` (inline comments), `README.md`, `driver-plan/product-overview.md`, `driver-plan/roadmap.md`, `driver-plan/research.md`, `driver-plan/ai-log.md` (this file)

**Actions:**
- Added inline comments to WACC formula derivation (CAPM equation, cost of debt, tax shield)
- Added stage labels to DCF projection (Stage 1-4: FCF projection → terminal value → discount → equity)
- Added docstring to `_generate_labels()` explaining self-supervised label generation and look-ahead bias prevention
- Updated README with: ML signal pipeline diagram, composite signal table, custom DCF description, updated tech stack, DRIVER methodology section, corrected all references from JSON→SQLite, Groq→OpenAI
- Updated DRIVER artifacts to reflect completed sections 1-3

---

## AI Modification Transparency

### What AI Generated vs What I Modified

| Component | AI Generated | Human Modified |
|---|---|---|
| Architecture decisions | Research findings, roadmap proposal | Final section ordering, priority decisions |
| ML model selection | RF + HMM + VADER recommendation | Weights (50/30/20), feature selection |
| Code implementation | All function implementations | Bug fixes after test failures |
| Test cases | All 37 tests | Test headline adjustments for VADER |
| Documentation | README structure, AI log | Content review and corrections |
| DCF parameters | Beta calculation, sensitivity grid | Growth/WACC range decisions |

### AI Limitations Observed

1. **VADER financial sentiment** — Rule-based NLP struggles with financial idioms ("beating estimates" scored negative). Mitigated by using VADER as only 20% of composite signal.
2. **HMM numerical stability** — Full covariance matrix failed on some data. Diagonal covariance is less expressive but always converges.
3. **Beta calculation** — Requires yfinance API call for SPY data, adding latency. Falls back to 1.0 gracefully.
4. **Context window limits** — Large codebase required conversation compaction during development. DRIVER artifacts preserved context across sessions.

### Token Usage Estimate

- Development sessions: ~3 sessions over 2 days
- Approximate Claude Code tokens: ~500K input, ~150K output
- OpenAI GPT-4o-mini (runtime): ~$0.0004/query for chatbot + coaching

---

## Validation Phase (Section 5)

### ML Validation on Real Market Data

Ran `tests/validate_ml.py` against live SPY/AAPL/MSFT/JNJ data:

| Check | Result | Notes |
|-------|--------|-------|
| RF CV accuracy | 35.5% (>33.3% baseline) | 3-class problem; modest edge expected |
| HMM regime | Bull (prob sum = 1.0000) | All 3 states detected in history |
| VADER known-answer | 3/3 PASS | Bullish/Bearish/Neutral correctly classified |
| Composite signal | Bullish (+0.194) | Weights sum to 1.0, score in [-1, 1] |
| Beta SPY | 1.00 | Known answer: SPY beta = 1.0 |
| Beta AAPL | 1.29 | Reasonable for large-cap tech |
| DCF JNJ | $253 vs $248 price | Fair Value — aligns with stable-cashflow company |
| DCF AAPL/MSFT | Overvalued | Expected: DCF undervalues growth stocks (Damodaran limitation) |
| Edge cases | 8/8 PASS | All degrade gracefully |

### DCF Academic Insight

Pure FCF-based DCF models structurally undervalue growth companies because they cannot capture growth optionality, platform effects, or intangible assets. This is consistent with Damodaran's teaching that DCF works best for mature, stable-cashflow businesses (like JNJ). The sensitivity heatmap allows users to explore how different growth and WACC assumptions affect the intrinsic value.

---

## Test Results Summary

```
37 passed in ~9 seconds

test_ai_signals.py    — 21 tests (regime, badges, coaching, HMM, VADER, composite, beta, DCF, sensitivity, valuation)
test_portfolio.py     — 7 tests (buy, sell, multiple trades, empty, sell-all, trade stats)
test_risk_metrics.py  — 5 tests (Sharpe, VaR, Monte Carlo, portfolio builder)
test_data_manager.py  — 4 tests (schema, SQLite roundtrip, JSON migration, empty DB)
```

All tests pass with no warnings.

---

## Post-Validation: UI Polish

### Changes Made
1. **DCF layout** — Moved DCF Valuation card and Sensitivity heatmap out of narrow 1/3-width column to full-width layout. Added structured display with labeled fields (Intrinsic Value, Market Price, Margin of Safety, Beta, WACC, Growth Rate, Eff. Tax Rate) in a flex row. Increased heatmap height to 400px with explanatory caption.
2. **"What This Means For You" section** — Replaced markdown `**bold**` syntax with HTML `<b>` tags for proper rendering inside `st.markdown(unsafe_allow_html=True)`. Added actionable `<i>Suggestion:</i>` to each interpretation (Sharpe, Max Drawdown, VaR) with specific next steps referencing other app features (Correlation Heatmap, Monte Carlo, Rebalancer, Price Alerts).
3. **Course number correction** — Replaced "MGMT 590" with "MGMT 69000" across all files (applied in prior session).
4. **Security verification** — Confirmed no API keys in git-tracked files; `.streamlit/secrets.toml` is properly gitignored.
