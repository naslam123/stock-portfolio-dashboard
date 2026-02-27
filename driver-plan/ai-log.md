# AI Development Log

**Project:** Stock Portfolio Trading Simulator
**Course:** MGMT 69000 — Mastering AI For Finance | Purdue University
**Student:** Naveed Aslam Perwez (naslam@purdue.edu)
**AI Tool:** Claude Code (Claude Opus 4.6) with PAL MCP Server
**Methodology:** DRIVER (Define → Represent → Implement → Validate → Evolve → Reflect)

---

## How AI Was Used Systematically (Not Ad Hoc)

This log documents how Claude Code was used **within the DRIVER framework** — meaning AI assistance followed a structured process where research preceded implementation, plans preceded code, and validation checked results. Each section below shows the prompt, what the AI did, and critically, **what decision changed** because of the AI's output.

---

## Phase 1: Define (开题调研) — Research Before Building

**Prompt:** "We are going to work on repo stock-portfolio-dashboard. Here is the professor feedback from submissions 1 and 2, plus final submission guidelines..."

**AI Actions:**
- Launched **4 parallel research agents** to investigate different aspects simultaneously:
  1. Codebase audit — deep read of all 12 modules (4,874 lines) to identify what already existed
  2. ML approaches — evaluated LSTM, FinBERT, XGBoost, Random Forest + HMM + VADER
  3. DCF/database patterns — identified 5 hardcoded assumptions in existing DCF
  4. OpenAI/PAL MCP integration — assessed cost, reliability, and code reuse patterns
- Cross-referenced professor feedback against existing code to separate "doesn't exist" from "exists but is shallow"
- Persisted all findings to `driver-plan/research.md` **before** any implementation began

**Key Decision Pivot:** The codebase audit revealed that most features the professor requested (SQLite, custom DCF, ML regime, LLM coaching) **already existed** in some form. This completely changed the approach: instead of building from scratch, we focused on strengthening and validating what was there. Without this research step, we would have spent days rebuilding existing functionality.

**What the AI Got Wrong:** Initially assumed the codebase had no DCF model — the deep read corrected this, finding a full 5-year FCF projection already in `ai_signals.py:207-326`. The research process caught the AI's own assumption.

**Output:** `driver-plan/research.md` (findings), `driver-plan/product-overview.md` (problem definition, success criteria)

---

## Phase 2: Represent (Roadmap) — Plan Before Code

**Prompt:** Continuation from Define — "Break this into buildable sections based on the research findings."

**AI Actions:**
- Proposed 5 sections with explicit dependency reasoning:
  - Section 1 (Bug Fixes + OpenAI) must come first because VaR bug would corrupt Section 5 validation
  - Sections 2-3 can overlap (different functions in ai_signals.py)
  - Section 4 (Documentation) follows 2-3 because you can't document unwritten code
  - Section 5 (Validation) must be last to reflect final state
- Created Mermaid dependency diagram showing build order
- Architecture decisions anchored to research: "Research found hmmlearn is 5MB vs TensorFlow 500MB, so HMM instead of LSTM"

**What Changed Because of This Stage:** The original instinct was to start with the "exciting" ML work (Section 2). The roadmap forced bug fixes first (Section 1), which turned out to be critical — the VaR return type change (`float` → `{dollar, percent, confidence}` dict) required updating test_risk_metrics.py. If we'd done validation (Section 5) before fixing this, all VaR tests would have failed on the old return type.

**Output:** `driver-plan/roadmap.md` — 5 sections with dependency graph, created before Section 1 began

---

## Phase 3: Implement — Research Findings → Code

Each section below shows the **research finding that drove the implementation**, not just what was built.

### Section 1: OpenAI + Bug Fixes

**Files Modified:** `chatbot.py`, `ai_signals.py`, `news_feed.py`, `portfolio.py`, `risk_metrics.py`, `app.py`, `dashboard.py`, `requirements.txt`

| Research Finding | Prompt to AI | Implementation | Decision |
|-----------------|-------------|----------------|----------|
| `_call_groq()` mirrors OpenAI SDK pattern | "Wire up OpenAI as primary LLM" | `_call_openai()` in 3 modules | Fallback chain: OpenAI → Groq → Gemini |
| Cost basis drift at `portfolio.py:20-26` | "Fix cost basis drift" | `round()` after each operation | Near-zero cleanup threshold added |
| VaR returns float, inconsistent display | "Fix VaR display" | Return `{dollar, percent, confidence}` dict | Required updating test_risk_metrics.py |
| Badge earned/lost on intraday swings | "Fix badge volatility" | Set union (once earned, never revoked) | Permanent badges via `\|=` operator |
| Price alerts only trigger upward | "Fix price alerts" | Both above AND below target | Bidirectional triggering |

### Section 2: Composite ML Signal

**Files Modified:** `ai_signals.py`, `app.py`, `tests/test_ai_signals.py`

| Research Finding | Prompt to AI | Implementation | What Changed During Implementation |
|-----------------|-------------|----------------|-------------------------------------|
| RF needs TimeSeriesSplit (critical for time series) | "Add TimeSeriesSplit" | `TimeSeriesSplit(n_splits=5)` with CV accuracy reporting | Worked as planned — no pivot |
| Research recommended 10 features including day-of-week | "Expand RF features" | Added ATR, volume ratio, price/SMA50, price/SMA200, SMA spread | **Dropped day-of-week** — added no predictive value |
| hmmlearn 5MB library for regime detection | "Add HMM regime detection" | 3-state Gaussian HMM on (returns, rolling_vol) | **Changed covariance_type from "full" to "diag"** — full caused LinAlgError |
| VADER for offline sentiment fallback | "Add VADER sentiment" | `analyze_sentiment_vader()` with compound score mapping | Discovered "beating estimates" misclassification — documented as limitation |
| 50/30/20 composite architecture | "Create composite signal" | Weighted blend with graceful degradation | Worked as designed in Represent stage |

**Bugs Encountered (R↔I feedback loop):**
- HMM `covariance_type="full"` caused `LinAlgError` on synthetic data → Changed to `"diag"` (more numerically stable). **Updated the plan** to reflect this.
- VADER scored "beating estimates" as negative (physical violence connotation) → Changed test headlines, documented limitation
- Empty HMM state slices caused `RuntimeWarning` → Added `mask.sum() > 0` guard

**Tests Added:** 7 new tests — HMM regime (2), VADER sentiment (3), composite signal (2)

### Section 3: DCF + Coaching

**Files Modified:** `ai_signals.py`, `app.py`, `tests/test_ai_signals.py`

| Research Finding (Hardcoded Assumption) | Prompt to AI | Implementation |
|----------------------------------------|-------------|----------------|
| Beta = 1.0 (hardcoded) | "Calculate beta from stock vs SPY" | `_calculate_beta()` using `np.cov()`, capped [0.1, 3.0] |
| Tax rate = 21% (statutory default) | "Get effective tax rate" | Extract from `incomeTaxExpense / incomeBeforeTax`, capped [0%, 40%] |
| No sensitivity analysis | "Add sensitivity heatmap" | `compute_dcf_sensitivity()` — 7×7 grid, Plotly heatmap |
| Coaching context lacks regime/risk data | "Enrich coaching context" | `_gather_coaching_context()` accepts regime, risk_metrics, composite_signal |

**Tests Added:** 7 new tests — beta (2), DCF with new fields (2), sensitivity matrix (2), enriched coaching (1)

### Section 4: Documentation

**Files Modified:** `ai_signals.py` (comments), `README.md`, all `driver-plan/` artifacts

**Actions driven by roadmap:** Added inline comments to WACC formula, DCF stages, and label generation docstring. Updated README with ML pipeline diagram and composite signal table. Created this AI log. All documentation reflects the final implementation state because Section 4 was sequenced after Sections 2-3 in the roadmap.

### Post-Implementation: UI Polish

**Driven by professor feedback and user testing:**
1. DCF layout — moved from narrow 1/3-width column to full-width with labeled metrics
2. "What This Means For You" — replaced markdown `**bold**` with HTML `<b>` tags, added actionable suggestions
3. Course number — corrected "MGMT 590" to "MGMT 69000" across all files

---

## Phase 4: Validate — Checking the Instruments

**Prompt:** "Run validation on real market data — cross-check ML accuracy, DCF vs consensus, edge cases."

**AI Actions:**
- Created `tests/validate_ml.py` to run RF, HMM, VADER, composite, beta, and DCF against live SPY/AAPL/MSFT/JNJ data
- Created `tests/validate_dcf.py` for DCF-specific comparison against analyst consensus

### What Validation Revealed (Not Just Confirmed)

| Check | Result | What We Did About It |
|-------|--------|---------------------|
| RF CV accuracy | 35.5% (>33.3% baseline) | **Confirmed composite architecture was necessary** — RF alone is barely above random. Honestly reported, no inflation. |
| HMM regime | Bull (prob sum = 1.0000) | All 3 states detected in history — model is working correctly |
| VADER known-answer | 3/3 PASS | Bullish/Bearish/Neutral correctly classified on clear headlines |
| Composite signal | Bullish (+0.194) | Weights sum to 1.0, score in [-1, 1] — architecture validated |
| Beta SPY | 1.00 | Known answer PASS — SPY beta to itself must equal 1.0 |
| Beta AAPL | 1.29 | Reasonable for large-cap tech |
| Beta JNJ | 0.10 (hit floor) | **Documented as limitation** — floor may be too aggressive for defensive stocks |
| DCF JNJ | $253 vs $248 price | Fair Value — validates DCF works for stable-cashflow companies |
| DCF AAPL/MSFT | Overvalued | **Expected limitation** — DCF undervalues growth stocks (Damodaran). Added interpretation text. |
| Edge cases | 8/8 PASS | All degrade gracefully — validated the "fallback everywhere" design |

**What Validation Changed:**
- Added honest disclosure of RF accuracy and look-ahead bias to DRIVER.md
- Added "What This Means For You" interpretations to analytics page (because raw numbers without context are misleading)
- Documented DCF growth stock limitation with academic reference (Damodaran)
- Added sensitivity heatmap caption explaining green=undervalued scenarios

**Output:** `driver-plan/validation.md` — comprehensive report with evidence

---

## AI Modification Transparency

### What AI Generated vs What I Steered

| Component | AI Generated | Human Steered |
|---|---|---|
| Research findings | 4 parallel agents produced findings | I directed what to research and prioritized findings |
| Roadmap ordering | AI proposed 5 sections | I approved dependency ordering, chose "bug fixes first" |
| ML model selection | AI recommended RF + HMM + VADER | I chose the 50/30/20 weights and feature set |
| Code implementation | All function implementations | I directed bug fixes after test failures, adjusted test headlines for VADER |
| DCF parameters | AI built beta calculation, sensitivity grid | I set growth/WACC ranges and approved parameter bounds |
| Validation | AI created and ran validation scripts | I interpreted results and decided what to disclose vs what to fix |
| Documentation | AI drafted README, DRIVER.md, this log | I reviewed for accuracy and added process-driven language |

### AI Limitations Observed

1. **VADER financial sentiment** — Rule-based NLP struggles with financial idioms ("beating estimates" scored negative). Mitigated by 20% composite weight.
2. **HMM numerical stability** — Full covariance matrix failed on some data. Research plan said "full" but implementation reality required "diag." This is the R↔I feedback loop in action.
3. **Beta calculation** — Requires yfinance API call for SPY data, adding latency. Falls back to 1.0 gracefully.
4. **Context window limits** — Large codebase required conversation compaction during development. DRIVER artifacts (persisted to files) preserved context across sessions — this is why the framework uses files, not just chat.
5. **Codebase assumption** — AI initially missed the existing DCF model during early scanning. The structured deep-read in Define stage caught this.

### Token Usage Estimate

- Development sessions: ~4 sessions over 3 days
- Approximate Claude Code tokens: ~800K input, ~250K output
- OpenAI GPT-4o-mini (runtime): ~$0.0004/query for chatbot + coaching

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

## Summary: DRIVER as Process, Not Template

The key evidence that DRIVER drove the process:

1. **`research.md` was created before any implementation code changed.** Research findings (LSTM too heavy, beta hardcoded, VADER for offline fallback) directly became implementation decisions.

2. **`roadmap.md` determined build order before Section 1 began.** Bug fixes came first because VaR return type change would have broken validation. This ordering wasn't obvious — it came from dependency analysis in Represent.

3. **Implementation pivoted when reality differed from plan.** HMM covariance changed from "full" to "diag." Day-of-week feature was dropped. These aren't failures — they're the R↔I feedback loop working correctly.

4. **Validation caught real issues that changed documentation.** RF accuracy of 35.5% confirmed the composite architecture decision. DCF growth stock bias drove the addition of interpretive text. JNJ beta floor was documented as a limitation.

5. **Each stage's output was the next stage's input.** Define → Research findings. Represent → Roadmap + architecture. Implement → Working code. Validate → Honest results. Reflect → Lessons learned. The chain is traceable.
