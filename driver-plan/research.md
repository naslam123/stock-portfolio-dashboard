# 分头研究 — Research Findings

## Date: 2026-02-27
## Context: Final submission for MGMT 69000 — Mastering AI For Finance

---

## 1. Current Codebase State (4,874 lines, 12 modules)

### What Already Exists
| Feature | Module | Quality | Gap |
|---------|--------|---------|-----|
| ML Regime Detection | ai_signals.py | Random Forest (100 trees, depth 5) | Shallow, no TimeSeriesSplit, self-supervised labels arbitrary |
| Custom DCF Model | ai_signals.py | 5-year FCF projection + WACC | Beta hardcoded to 1.0, no sensitivity matrix |
| SQLite Persistence | data_manager.py | Full migration from JSON | No foreign keys, migration not idempotent |
| LLM Coaching | ai_signals.py | Groq primary, Gemini fallback | Rule-based fallback is static, context lacks regime/risk data |
| Sentiment Analysis | news_feed.py | Groq/Gemini batch scoring | No offline fallback, JSON parsing brittle |
| Technical Indicators | ai_signals.py | RSI, MACD, Bollinger | RSI uses SMA not EMA (not Wilder's) |
| Options Payoff | ai_signals.py | Single-leg payoff diagrams | No spreads, no expiration check |
| Rebalancer | rebalancer.py | Equal-weight + custom targets | No slippage modeling |
| AI Chatbot | chatbot.py | Groq Llama 3.3 70B + Gemini | No OpenAI, no streaming |

### Known Bugs (from feedback + code review)
1. **Cost basis floating-point drift** — `portfolio.py:20-26` — repeated sells accumulate rounding errors
2. **VaR percent/dollar conflation** — `risk_metrics.py:30` — display inconsistency
3. **Badge unrealized gains** — "Six-Figure Club" badge re-earned/lost on intraday swings
4. **Price alerts one-direction only** — only triggers on price >= target, no downside
5. **Options expiration not checked** — stale contracts retained in portfolio

---

## 2. ML Trading Signals — Research

### Recommended: Random Forest + HMM + VADER Composite

**Random Forest (already exists, needs strengthening):**
- Increase to `n_estimators=200, max_depth=8`
- Use `TimeSeriesSplit(n_splits=5)` — CRITICAL: never random-split time series
- Add features: ATR, volume ratio, price vs SMA50/200, day-of-week
- Output: probability 0.0-1.0 → Bullish (>0.6) / Bearish (<0.4) / Neutral

**HMM Regime Detection (new — replaces SMA crossover):**
- Library: `hmmlearn` (~5MB, fast)
- 3-state Gaussian HMM on (returns, rolling volatility)
- Maps states by mean return: Bull / Sideways / Bear
- Probabilistic output (e.g., "78% bull, 15% sideways, 7% bear")
- Visualize as colored background overlay on price chart

**VADER Sentiment (new — offline fallback for news):**
- Library: `vaderSentiment` (~2MB, no API needed)
- Deterministic, zero latency
- Use alongside existing Groq/Gemini sentiment for comparison
- Financial headlines work reasonably well

**Composite Signal:**
- 50% ML signal (RF probability) + 30% regime (HMM) + 20% sentiment (VADER/LLM)
- Each component shown individually in UI for transparency
- Graceful degradation — if one fails, others still work

### Not Recommended for This Project
- **LSTM/Deep Learning** — Too heavy (TensorFlow ~500MB), hard to explain, often underperforms tree models on tabular data
- **FinBERT** — Torch dependency is 800MB+, overkill for a course project
- **XGBoost** — Good upgrade path but not necessary for final submission

### New Dependencies
```
pip install hmmlearn vaderSentiment
```
(scikit-learn already installed)

---

## 3. DCF Valuation — Improvements Needed

### Current Implementation (ai_signals.py lines 207-326)
Already has: FCF extraction, growth rate blending, WACC via CAPM, 5-year projection, terminal value

### Hardcoded Assumptions to Fix
| Parameter | Current | Should Be |
|-----------|---------|-----------|
| Beta | 1.0 (hardcoded) | Calculate from stock vs SPY returns (60-day rolling) |
| Risk-free rate | 4.3% | Fetch from FMP or use 10Y Treasury |
| Cost of debt | 5% default | Calculate from interest expense / total debt |
| Tax rate | 21% | Get from effective tax rate in income statement |
| Terminal growth | 2.5% | Make configurable (1.5-3.0% range) |

### Add: Sensitivity Analysis Matrix
- Growth rate (rows) × Discount rate (columns)
- Plotly heatmap showing intrinsic value at each combination
- Highlight current assumptions cell
- Reference: Damodaran's sensitivity tables

### Add: numpy-financial for NPV
```
pip install numpy-financial
```
Use `npf.npv()` instead of manual discounting loop — more robust and academically standard.

---

## 4. OpenAI GPT Integration

### PAL MCP Server State
- Located at `/home/naslam/pal-mcp-server/`
- Registered in `~/.claude/mcp_servers.json`
- **OPENAI_API_KEY is placeholder** — needs real key
- Supports GPT-5, GPT-5-mini, GPT-5-nano, o3, o4-mini, gpt-4.1
- Enabled tools: chat, thinkdeep, planner, consensus, codereview, precommit, debug, challenge

### For Streamlit App (chatbot.py + ai_signals.py)
- Add `openai>=1.0.0` to requirements.txt
- Add `OPENAI_API_KEY` to `.streamlit/secrets.toml`
- `_call_openai()` mirrors `_call_groq()` exactly (both use OpenAI-compatible SDK)
- **Model: gpt-4o-mini** — $0.15/1M input, $0.60/1M output (3.3x cheaper than gpt-3.5-turbo)
- Estimated cost: ~$0.00038/query, ~$1.14/month at 100 queries/day

### Fallback Chain (recommended)
1. **OpenAI gpt-4o-mini** — primary (reliable, cheap, high quality)
2. **Groq Llama 3.3 70B** — fallback (free, fast)
3. **Gemini 2.0 Flash** — last resort (quota issues)

### Coaching Tips Enhancement
- Already routes through LLM — just enrich context with:
  - Market regime state (from HMM)
  - Risk metrics (Sharpe, VaR, max drawdown)
  - Recent trade outcomes (last 5 trades)
  - Sector concentration

---

## 5. Data Source Assessment

### FMP Stable API (Primary — Already Working)
- Free tier: 250 req/day
- Working endpoints: `/quote`, `/historical-price-eod`, `/income-statement`, `/cash-flow-statement`, `/balance-sheet-statement`, `/analyst-estimates`, `/discounted-cash-flow`
- **Not working (402):** news/sentiment endpoints (paid)

### yfinance (Fallback — Already Working)
- Free, no key, unreliable for production
- Known issues: rate limiting, occasional data gaps
- Keep as fallback but document limitations

### Alpha Vantage (Alternative — Not Implemented)
- Free: 25 req/day (too restrictive)
- Premium: $49.99/month
- **Not recommended** for free tier due to severe rate limits

### Polygon.io (Alternative — Not Implemented)
- Free: 5 API calls/min, delayed data
- Good for historical data
- **Not worth adding** given FMP already works

### Recommendation
Keep FMP primary + yfinance fallback. Add rate-limit tracking to warn users approaching 250/day limit.

---

## 6. Final Submission Requirements Checklist

| Component | Status | Action Needed |
|-----------|--------|---------------|
| Repository + README | ✅ Done | Updated with ML pipeline, DRIVER section, all new features |
| SubStack Post | Not started | Write after Section 5 |
| Demo (working app) | ✅ Done | All bugs fixed, ML + DCF + coaching added |
| Slides (5-10) | Not started | Create after Section 5 |
| Validation | Not started | Section 5: cross-check ML accuracy, DCF vs consensus |
| AI Log | ✅ Done | `driver-plan/ai-log.md` |
| Code documentation | ✅ Done | Inline comments on WACC, DCF stages, label generation |
| Test suite | ✅ Done | 37 tests across 4 modules, all passing |
