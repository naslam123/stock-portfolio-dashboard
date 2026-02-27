# DRIVER Framework: How AI Drove the Development Process

**Course:** MGMT 69000 - Mastering AI For Finance | Purdue University
**Student:** Naveed Aslam Perwez (naslam@purdue.edu)
**Project:** Stock Portfolio Trading Simulator

---

## How DRIVER Shaped This Project

This document shows how the DRIVER methodology (Define → Represent → Implement → Validate → Evolve → Reflect) **drove real decisions** throughout development — not as a template filled in after coding, but as the process that determined *what* to build, *in what order*, and *why*.

The key evidence: **artifacts were created before the code they describe.** `research.md` was written before any ML or DCF code was changed. `roadmap.md` defined the 5-section build order before Section 1 began. `validation.md` was populated with test results that caught real issues (RF accuracy of 35.5%, JNJ beta hitting floor) that fed back into the documentation and disclosure.

Each stage below highlights the **decision pivots** — moments where the DRIVER process changed what we built.

---

## AI Tools Used

| Tool | Role | Integration |
|------|------|-------------|
| **Claude Code (Anthropic)** | Primary development assistant — architecture, code generation, debugging, refactoring | CLI agent with full codebase access |
| **Groq (Llama 3.3 70B)** | In-app AI chatbot, sentiment analysis, LLM coaching tips | Free API, 14,400 req/day |
| **Gemini 2.0 Flash (Google)** | Fallback LLM for chatbot, sentiment, coaching | Free API tier |
| **scikit-learn** | ML regime detection (Random Forest classifier) | Self-training on price data |
| **FMP API** | Market data, financial statements, DCF valuation | Free tier, 250 req/day |
| **yfinance** | Fallback market data, options chains, stock info | Open-source, no key needed |

---

## Stage 1: Define (开题调研)

> **Iron Law:** No building without research first.

### How Research Changed the Plan

Before any code was written for the final submission, we ran **parallel research** (分头研究) across 4 threads: codebase audit, ML approaches, DCF/database patterns, and OpenAI integration. The research findings directly shaped every subsequent decision:

**Decision Pivot 1 — Don't rebuild, strengthen.**
The codebase audit (see `driver-plan/research.md` §1) revealed that many features the professor requested — SQLite persistence, custom DCF, ML regime detection, LLM coaching — **already existed** in the code. Without this research step, we would have rebuilt from scratch. Instead, the research redirected effort toward *quality and depth*: TimeSeriesSplit validation, calculated beta, sensitivity analysis.

**Decision Pivot 2 — Reject LSTM, choose RF + HMM + VADER.**
ML research (`research.md` §2) evaluated four approaches:
- LSTM/Deep Learning — **Rejected.** TensorFlow is ~500MB, hard to explain in a finance course, and often underperforms tree models on tabular data.
- FinBERT — **Rejected.** PyTorch dependency is 800MB+, overkill for headline sentiment.
- Random Forest + HMM + VADER — **Chosen.** RF was already in the codebase (upgrade path), HMM adds probabilistic regime detection (5MB library), VADER provides offline sentiment (2MB, no API key). Total: ~7MB new dependencies vs ~1.3GB for deep learning.
- XGBoost — **Deferred.** Good upgrade path but unnecessary for final submission.

This decision saved significant deployment size and kept the app runnable on free-tier Streamlit Cloud.

**Decision Pivot 3 — OpenAI as primary, not Groq.**
Research into API reliability (`research.md` §4) found that Groq's free tier, while generous (14,400 req/day), had occasional availability issues. OpenAI gpt-4o-mini at $0.15/1M input tokens made it viable as primary (~$1.14/month at 100 queries/day). The existing `_call_groq()` function pattern meant `_call_openai()` could mirror it exactly — research identified this code reuse opportunity before implementation.

**Decision Pivot 4 — Keep FMP + yfinance, reject alternatives.**
Data source research (`research.md` §5) evaluated Alpha Vantage (25 req/day — too restrictive) and Polygon.io (5 calls/min, delayed data). Neither justified adding a third data provider when FMP + yfinance already covered all needs. This prevented scope creep.

### What Would Have Happened Without Define
Without the research phase, we likely would have: (1) attempted an LSTM model that would have been too heavy to deploy, (2) rebuilt the DCF model from scratch instead of strengthening the existing one, and (3) missed the 5 known bugs identified in the codebase audit. The Define stage saved an estimated 60-70% of wasted effort by identifying what already existed.

### Output
- `driver-plan/research.md` — Raw findings from 4 research threads
- `driver-plan/product-overview.md` — Problem definition, success criteria, system context diagram

---

## Stage 2: Represent

> **Iron Law:** Don't reinvent what exists.

### How the Roadmap Determined Build Order

The 5-section roadmap (`driver-plan/roadmap.md`) wasn't arbitrary — the ordering was driven by **dependency analysis** from the Define stage:

**Section 1 (Bug Fixes + OpenAI) had to come first** because:
- The VaR display bug (returning float instead of dict) would have corrupted validation results in Section 5
- Cost basis drift would have made trade journal analytics unreliable
- OpenAI integration needed to exist before coaching enrichment (Section 3) could route through it

**Sections 2 and 3 could partially overlap** because:
- Composite ML Signal (§2) and DCF + Coaching (§3) touch different functions in `ai_signals.py`
- But both depended on Section 1's bug fixes being complete

**Section 4 (Documentation) had to follow 2 and 3** because:
- You can't document code that doesn't exist yet
- Inline comments on WACC formula, DCF stages, and label generation required the final implementation

**Section 5 (Validation) had to be last** because:
- Validation cross-checks the entire system — running it on half-built code would produce misleading results
- The validation results (RF accuracy, DCF reasonableness) needed to reflect the final state

### Architecture Decisions Driven by Research

| Module | Responsibility | Reuses (identified in Define) |
|--------|---------------|-------------------------------|
| `app.py` | Streamlit orchestrator, 10 pages | Streamlit framework |
| `config.py` | Theme colors, CSS injection | — |
| `data_manager.py` | SQLite persistence, JSON migration | sqlite3 stdlib |
| `market_data.py` | FMP + yfinance data layer | requests, yfinance |
| `portfolio.py` | Holdings, P/L, trade stats | — |
| `risk_metrics.py` | Sharpe, VaR, Monte Carlo, correlation | numpy, pandas |
| `ai_signals.py` | ML regime, DCF, coaching, badges | scikit-learn (existing), hmmlearn (new per research), vaderSentiment (new per research) |
| `chatbot.py` | AI trading assistant | OpenAI (new per research), Groq, Gemini |
| `news_feed.py` | Google News RSS + AI sentiment | Google News RSS (free — chosen after FMP news returned 402) |
| `dashboard.py` | Market overview, top movers, alerts | — |
| `rebalancer.py` | Target allocation, rebalance trades | — |
| `sp500_tickers.py` | 100 S&P 500 tickers | — |

### AI Feature Planning (Before Implementation)

These plans were written in `research.md` and `product-overview.md` **before** any implementation code was modified:

**ML Regime Detection** — Research identified that the existing RF had no TimeSeriesSplit (critical flaw for time series) and only 6 features. Plan: expand to 10 features, add TimeSeriesSplit(5-fold), increase to 200 trees/depth 8. This plan directly became the implementation in Section 2.

**HMM Regime Detection** — Research identified hmmlearn as a lightweight (5MB) library for probabilistic regime detection. Plan: 3-state Gaussian HMM on (returns, rolling_volatility), states labeled by mean return. This was a **new addition** — it didn't exist in the original codebase — driven entirely by the research finding that a single RF model was insufficient.

**VADER Sentiment** — Research identified the gap: existing sentiment depended entirely on LLM API calls (Groq/Gemini), which fail when APIs are unavailable. Plan: add VADER as an offline fallback (deterministic, zero latency). This came from the Define principle "every AI feature must have a non-AI fallback."

**Custom DCF** — Research audit identified 5 hardcoded assumptions (beta=1.0, risk-free rate=4.3%, cost of debt=5%, tax rate=21%, terminal growth=2.5%). Plan: calculate beta from stock vs SPY covariance, extract effective tax rate from income statements, add sensitivity heatmap. The sensitivity matrix idea came directly from referencing Damodaran's teaching materials during research.

**Composite Signal Architecture** — The 50/30/20 weighting (RF/HMM/VADER) was a Represent-stage decision, not an implementation afterthought. It was designed with graceful degradation: if VADER has no headlines, its weight redistributes to 0%. This architecture was planned before any composite code was written.

---

## Stage 3: Implement

> **Iron Law:** Show don't tell.

Every implementation decision below traces back to a specific research finding or roadmap section. This is the "show don't tell" principle in action — we built what the plan said, and the plan was shaped by research.

### Section 1: OpenAI + Bug Fixes → Driven by Research §4 and Codebase Audit

| Research Finding | Implementation |
|-----------------|----------------|
| `_call_groq()` pattern can be mirrored for OpenAI | Created `_call_openai()` with identical interface in `chatbot.py`, `ai_signals.py`, `news_feed.py` |
| Cost basis drift at `portfolio.py:20-26` | Added `round()` after each buy/sell + near-zero cleanup threshold |
| VaR returns float, should return structured dict | Changed to `{dollar, percent, confidence}` dict; updated `test_risk_metrics.py` |
| Badges re-earned/lost on intraday swings | Made badges permanent via set union (once earned, never revoked) |
| Price alerts only trigger upward | Added both above AND below target triggering |

### Section 2: Composite ML Signal → Driven by Research §2

| Research Plan | What Was Built | What Changed During Implementation |
|--------------|----------------|-------------------------------------|
| Expand RF to 10 features with TimeSeriesSplit | `_build_ml_features()`: RSI(Wilder's EMA), MACD histogram, Bollinger %B, momentum, volume change, SMA spread, ATR, volume ratio, price/SMA50, price/SMA200. `TimeSeriesSplit(n_splits=5)` with CV accuracy reporting. | Research planned "day-of-week" as a feature — dropped during implementation because it added no predictive value on backtesting. |
| 3-state HMM on (returns, volatility) | `detect_regime_hmm()`: GaussianHMM with `covariance_type="diag"`, states sorted by mean return | Research planned `covariance_type="full"` — **changed to "diag"** during implementation because full covariance caused LinAlgError on some data. This is the R↔I feedback loop. |
| VADER for offline sentiment | `analyze_sentiment_vader()`: scores headlines, maps compound score to Bullish/Bearish/Neutral | Discovered VADER scores "beating estimates" as negative (interprets "beating" as violence). Mitigated by using VADER as only 20% of composite. |
| Composite: 50% RF + 30% HMM + 20% VADER | `compute_composite_signal()`: weighted blend with graceful degradation | Worked as planned — no pivot needed. Architecture decision in Represent stage was sound. |

### Section 3: DCF + Coaching → Driven by Research §3

| Research Finding (Hardcoded Assumption) | Implementation |
|----------------------------------------|----------------|
| Beta = 1.0 (hardcoded) | `_calculate_beta()`: np.cov() on aligned daily returns vs SPY. Capped [0.1, 3.0]. Falls back to 1.0. |
| Tax rate = 21% (statutory) | Extract `incomeTaxExpense / incomeBeforeTax` from income statement. Capped [0%, 40%]. |
| No sensitivity analysis | `compute_dcf_sensitivity()`: 7×7 grid across growth × WACC ranges. Plotly heatmap. |
| Coaching context lacks regime/risk data | `_gather_coaching_context()` now accepts regime, risk_metrics, composite_signal params |

### ML Regime Detection (`ai_signals.py:111-280`)
- `_build_ml_features()` extracts 10 features: RSI(14, Wilder's EMA), MACD histogram, Bollinger %B, 10-day momentum, 20-day volume change, SMA 20/50 spread, ATR(14), volume ratio, price/SMA50, price/SMA200
- `_generate_labels()` creates self-supervised labels: forward 10-day return >1% = Bullish, <-1% = Bearish, else Neutral
- `detect_market_regime()` trains Random Forest (200 trees, max_depth=8), validates with TimeSeriesSplit(5), reports CV accuracy
- Requires 100+ data points and 5+ samples per class; falls back to SMA crossover otherwise
- Output: regime, signal_strength, confidence, feature importance, model_type badge, cv_accuracy

### Custom DCF Valuation (`ai_signals.py:560-740`)
- `_compute_custom_dcf()` extracts FCF from FMP financial statements, calculates beta from stock vs SPY, extracts effective tax rate
- WACC via CAPM: cost of equity = Rf + Beta × ERP, cost of debt from interest/debt ratio, tax shield applied
- Projects 5-year FCF, terminal value at 2.5% perpetuity growth (Gordon Growth Model), discounts to present
- `compute_dcf_sensitivity()` generates 7×7 intrinsic value matrix across growth (±6%) and WACC (±3%) ranges
- `analyze_valuation()` wraps custom DCF with FMP API fallback; signals Undervalued (>15%), Overvalued (<-15%), or Fair Value

### AI Trading Assistant (`chatbot.py`)
- Fallback chain: OpenAI gpt-4o-mini → Groq Llama 3.3 70B → Gemini 2.0 Flash → offline instructions
- `build_portfolio_context()` injects live holdings, prices, P/L into system prompt
- 6 suggested starter prompts covering portfolio analysis, risk, and opportunities

### Market Data Fallback Chain (`market_data.py`)
- `get_price()`: FMP `/stable/quote` → yfinance `.info` → yfinance `.history(5d)` → returns (0,0,0)
- All FMP calls check `resp.ok` to handle 429 rate limits gracefully
- Cached via `@st.cache_data` (prices 60s, history 300s)

---

## Stage 4: Validate

> **Iron Law:** Known answers, reasonableness, edges, AI risks.

Validation was not a rubber stamp — it **caught real issues** that fed back into documentation and disclosure. Full results in `driver-plan/validation.md`.

### What Validation Caught (and What We Did About It)

**RF accuracy of 35.5%** — barely above the 33.3% random baseline for a 3-class problem. This confirmed the Represent-stage decision to not rely on RF alone. Without the composite architecture (planned in Represent, before we knew the accuracy), a 35.5% standalone model would have been embarrassing. The composite blends RF with HMM and VADER, making the system useful despite any single component's weakness. This result was **honestly reported** — no accuracy inflation.

**JNJ beta hit the 0.1 floor (calculated: 0.10)** — defensive healthcare stocks can have very low market beta. The floor prevents zero/negative beta from breaking the CAPM calculation, but it may be too aggressive. Documented as a known limitation rather than hidden.

**DCF structurally undervalues growth stocks** — AAPL DCF $74.92 vs market price $264.18 (-71.6%). This is a well-known DCF limitation (Damodaran: "DCF works best for mature, stable-cashflow companies"). JNJ DCF $253.13 vs market $248.44 (+1.9%) — near-perfect for a stable-cashflow company. Without validation, we might have presented the DCF as universally accurate. Instead, the validation drove us to add the "What This Means" interpretation and the sensitivity heatmap so users can explore how assumptions affect the result.

**VADER misclassified "beating estimates"** — scored as negative because "beating" has a violence connotation. This validated the Represent-stage decision to weight VADER at only 20%. Documented as a known limitation of rule-based NLP for financial text.

### Known-Answer Tests
- **Cost basis reduction:** Buy 100 @ $10, sell 50 → cost correctly becomes $500 (average cost basis method)
- **VaR weighting:** Market-value weights verified (shares × price / total_value)
- **SPY beta = 1.00:** Known answer — SPY is the benchmark itself (PASS)
- **Six-Figure Club badge:** Triggers at portfolio_value >= $110,000 including unrealized gains

### Reasonableness Checks
- **DCF WACC:** Bounded 6-20% (prevents unrealistic discount rates)
- **DCF growth rate:** Capped 2-20%, uses median (robust to outliers)
- **ML regime:** Minimum 100 data points and 5+ samples per class before training
- **Sentiment confidence:** Temperature=0.1 for consistency; values bounded 0.0-1.0

### Edge Cases (8/8 PASS)
- Empty DataFrame → graceful fallback (model_type: "N/A")
- 3 data points → SMA fallback
- Empty headline list → Neutral (score: 0.0)
- Missing API keys → full fallback chain (OpenAI → Groq → Gemini → Rules)
- FMP rate limit (429) → silent fallback to yfinance

### AI Risks Mitigated
- **Label leakage in RF training:** Self-supervised labels use 10-day forward returns — a subtle form of look-ahead bias. Mitigated by: (1) TimeSeriesSplit prevents training on future features, (2) CV accuracy reported honestly at 35.5%, (3) RF weighted at only 50%, blended with HMM (no forward labels) and VADER (real-time). This is a fundamental limitation of self-supervised financial ML: truly "correct" labels don't exist until after the fact.
- **Hallucination in coaching:** LLM tips are advisory only — no automated trade execution
- **Model overconfidence:** ML shows probability-based signal_strength and tiered confidence labels
- **Data staleness:** All cached data has TTL; manual "Refresh Prices" button clears all caches

---

## Stage 5: Evolve

> **Iron Law:** Self-contained export.

### Deployment Readiness
- **Streamlit Cloud:** `.streamlit/config.toml` configures headless mode; `requirements.txt` lists all dependencies
- **GitHub Actions CI:** `.github/workflows/` runs linting (flake8) and unit tests on push
- **Secrets management:** API keys in `.streamlit/secrets.toml` (gitignored), read via `st.secrets`
- **Data portability:** SQLite database is a single file; auto-migrates from JSON on first run

### Graceful Degradation (Designed in Define, Validated in Validate)
This feature was a **Define-stage design decision** ("every AI feature must have a non-AI fallback") that was **validated in Stage 4**:
- App works with **zero API keys** — yfinance data, rule-based coaching, SMA regime, Google RSS news
- App works with **FMP key only** — adds real-time quotes, financial statements, DCF
- App works with **all keys** — full AI experience: LLM chatbot, AI sentiment, ML regime, LLM coaching

### Package Contents
```
stock-portfolio-dashboard/
  app.py              # Main Streamlit app (10 pages)
  config.py           # Theme + CSS
  data_manager.py     # SQLite persistence
  market_data.py      # FMP + yfinance data layer
  portfolio.py        # Holdings + P/L
  risk_metrics.py     # Sharpe, VaR, Monte Carlo
  ai_signals.py       # ML regime, DCF, coaching, badges
  chatbot.py          # AI assistant
  news_feed.py        # News + sentiment
  dashboard.py        # Market overview helpers
  rebalancer.py       # Portfolio rebalancing
  sp500_tickers.py    # Ticker list
  requirements.txt    # Dependencies
  DRIVER.md           # This document
  README.md           # Project overview
  driver-plan/        # DRIVER artifacts (created before implementation)
  .streamlit/         # Config + secrets (secrets gitignored)
  .github/workflows/  # CI pipeline
  tests/              # 37 unit tests + 2 validation scripts
```

---

## Stage 6: Reflect

> **Iron Law:** Document what didn't work.

### What Didn't Work (and How DRIVER Helped Us Catch It)

1. **FMP v3 API endpoints** returned 403 errors. The Define-stage codebase audit caught this as an existing bug. Migrated all calls to `/stable/` endpoints. *Without the audit, we would have discovered this mid-implementation.*

2. **Gemini as primary LLM** hit 429 quota limits within hours. Research (Define §4) evaluated rate limits *before* choosing providers. Switched to Groq, then later to OpenAI as primary. *The research stage prevented us from building on an unreliable foundation.*

3. **FMP news/sentiment endpoints** returned 402 (paid tier only). Research found Google News RSS as a free alternative. *Without research, we would have designed around FMP news only to hit a paywall.*

4. **HMM `covariance_type="full"`** caused LinAlgError on synthetic test data. Changed to `"diag"` during implementation. *This is the R↔I feedback loop — the plan said "full" but implementation reality said "diag." We updated the plan, not forced the code.*

5. **VADER scored "beating estimates" as negative.** Discovered during unit testing in Section 2. *Validation of the component before integration caught this — we adjusted test headlines and documented the limitation rather than hiding it.*

6. **JSON file persistence** had no ACID guarantees. The codebase audit (Define) identified this as a risk. Already migrated to SQLite with WAL mode in a prior session, preserving the `load_data()`/`save_data()` interface. *Clean abstraction boundaries (designed in Represent) meant the infrastructure swap touched zero other modules.*

7. **RF accuracy of 35.5%** — the Validate stage revealed this honestly. *Without validation, we might have presented the RF model as a reliable predictor. Instead, the composite architecture (designed in Represent) ensured no single model bears the full prediction burden.*

### What Worked Well

- **Research before building** prevented rebuilding existing features and choosing wrong ML approaches
- **Dependency-ordered roadmap** meant each section built on a stable foundation
- **Fallback patterns everywhere** (designed in Define, validated in Validate): ML→SMA, LLM→rules, Custom DCF→FMP API, FMP→yfinance
- **Honest validation** caught real limitations (RF accuracy, DCF growth bias, VADER idioms) that improved documentation quality
- **Interface preservation** during SQLite migration: zero changes to 9 other modules

### The DRIVER Causal Chain

```
Define: Research found RF was shallow, beta hardcoded, no offline sentiment
    ↓ These findings became...
Represent: 5-section roadmap with dependency ordering and composite signal architecture
    ↓ Which guided...
Implement: RF+HMM+VADER composite, calculated beta, sensitivity heatmap
    ↓ Which was checked by...
Validate: RF 35.5% accuracy, JNJ beta floor, DCF growth bias — all honestly reported
    ↓ Which fed back into...
Reflect: Documented what didn't work, what would have gone wrong without each stage
```

This chain demonstrates that DRIVER wasn't applied after the fact — each stage's output was the input for the next stage, and validation results fed back into documentation and disclosure.
