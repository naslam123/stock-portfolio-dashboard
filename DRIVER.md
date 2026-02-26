# DRIVER Framework: AI Tool Usage Documentation

**Course:** MGMT 590 - Mastering AI For Finance | Purdue University
**Student:** Naveed Aslam Perwez (naslam@purdue.edu)
**Project:** Stock Portfolio Trading Simulator

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

## Stage 1: Define

> **Iron Law:** No building without research first.

### Vision
Build a full-featured stock portfolio trading simulator for MGMT 590 that integrates genuine AI/ML models — not just static rules or API passthroughs — while remaining fully functional on free-tier API keys.

### Research & Discovery
- **Market data APIs:** Evaluated FMP (structured, rate-limited at 250/day) vs yfinance (free, less reliable). Decision: FMP primary with yfinance fallback chain.
- **LLM providers:** Evaluated Groq (free, 14,400 req/day, fast inference) vs Gemini (free but quota issues) vs OpenAI (paid). Decision: Groq primary, Gemini fallback.
- **ML for regime detection:** Researched SMA crossover (too simple, not real ML) vs Random Forest (interpretable, works with small datasets, self-supervised labeling from forward returns). Decision: Random Forest with SMA fallback.
- **Valuation models:** Researched FMP pre-computed DCF (black box) vs custom DCF with WACC/FCF projection (transparent, educational). Decision: Custom DCF with FMP fallback.
- **Persistence:** Researched JSON file storage (simple but fragile) vs SQLite (ACID, WAL mode, structured). Decision: SQLite with auto-migration from JSON.

### Key Design Decisions
1. Every AI feature must have a non-AI fallback (ML to SMA, LLM to rules, Custom DCF to FMP API)
2. Modular architecture: 10 separate Python modules with clean interfaces
3. Theme system via CSS injection (dark/light/colorblind modes)
4. Free-tier only — no paid API keys required

---

## Stage 2: Represent

> **Iron Law:** Don't reinvent what exists.

### Architecture Plan (10 Modules)

| Module | Responsibility | Reuses |
|--------|---------------|--------|
| `app.py` | Streamlit orchestrator, 10 pages | Streamlit framework |
| `config.py` | Theme colors, CSS injection | — |
| `data_manager.py` | SQLite persistence, JSON migration | sqlite3 stdlib |
| `market_data.py` | FMP + yfinance data layer | requests, yfinance |
| `portfolio.py` | Holdings, P/L, trade stats | — |
| `risk_metrics.py` | Sharpe, VaR, Monte Carlo, correlation | numpy, pandas |
| `ai_signals.py` | ML regime, DCF, coaching, badges | scikit-learn, Groq, Gemini |
| `chatbot.py` | AI trading assistant | Groq, Gemini |
| `news_feed.py` | Google News RSS + AI sentiment | Google News RSS (free), Groq/Gemini |
| `dashboard.py` | Market overview, top movers, alerts | — |
| `rebalancer.py` | Target allocation, rebalance trades | — |
| `sp500_tickers.py` | 100 S&P 500 tickers | — |

### AI Feature Planning

**ML Regime Detection** — Planned as a Random Forest classifier with 6 technical features (RSI, MACD histogram, Bollinger %B, momentum, volume change, SMA spread). Self-supervised labels from 10-day forward returns avoid need for external labeled data. Leverages scikit-learn (existing library, not reinventing).

**Custom DCF Valuation** — Planned as a 5-year FCF projection with WACC via CAPM. Reuses standard finance formulas (cost of equity = Rf + Beta x ERP, terminal value with Gordon Growth). Leverages FMP financial statements API for input data.

**LLM Coaching** — Planned as a context-aware prompt to Groq/Gemini requesting 3 JSON-formatted tips. Reuses the same LLM infrastructure as the chatbot.

**AI Sentiment Analysis** — Planned as batch headline classification via Groq/Gemini. Reuses Google News RSS (free, no API key) for headline sourcing.

**AI Trading Assistant** — Planned as a portfolio-aware chatbot. Reuses Groq/Gemini with a system prompt injecting live holdings, prices, and P/L.

---

## Stage 3: Implement

> **Iron Law:** Show don't tell.

### ML Regime Detection (`ai_signals.py:111-204`)
- `_build_ml_features()` extracts 6 features: RSI(14), MACD histogram, Bollinger %B, 10-day momentum, 20-day volume change, SMA 20/50 spread
- `_generate_labels()` creates self-supervised labels: forward 10-day return >1% = Bullish, <-1% = Bearish, else Neutral
- `detect_market_regime()` trains a Random Forest (100 trees, max_depth=5) on the labeled features, predicts current regime with probability-based confidence
- Requires 100+ data points and 5+ samples per class; falls back to `_detect_regime_sma()` otherwise
- Output includes regime, signal_strength, confidence level, feature importance dict, and model_type badge ("ML" or "SMA")

### Custom DCF Valuation (`ai_signals.py:207-400`)
- `_compute_custom_dcf()` extracts FCF (operating cash flow minus capex) from FMP financial statements
- Growth rate: median of blended FCF + revenue YoY growth rates, capped 2-20%
- WACC via CAPM: risk-free rate 4.3%, equity risk premium 6%, beta 1.0 default, cost of debt from interest/debt ratio; final WACC capped 6-20%
- Projects 5-year FCF, terminal value at 2.5% perpetuity growth, discounts to present
- `analyze_valuation()` wraps custom DCF with FMP API fallback; signals Undervalued (>15% margin), Overvalued (<-15%), or Fair Value

### LLM Coaching Tips (`ai_signals.py:538-681`)
- `_gather_coaching_context()` summarizes portfolio: holdings count, cash %, trade count, win/loss ratio, badges earned
- `_get_llm_coaching()` sends context to Groq Llama 3.3 70B (primary) or Gemini 2.0 Flash (fallback), requesting 3 JSON-formatted tips
- `_rule_based_tips()` provides static fallback: cash allocation advice, diversification warnings, overtrading detection, losing streak alerts
- UI displays "LLM Powered" vs "Rules" badge in Analytics page

### AI Trading Assistant (`chatbot.py`)
- `build_portfolio_context()` injects live holdings, current prices, daily changes, position P/L, watchlist, and trade count into system prompt
- `get_ai_response()` sends last 10 chat messages + user query to Groq (primary) or Gemini (fallback)
- Falls back to instructions for adding API keys if both LLMs unavailable
- 6 suggested starter prompts covering portfolio analysis, market outlook, risk metrics, trade review, opportunities, and app tutorial

### AI Sentiment Analysis (`news_feed.py`)
- `get_stock_news()` fetches headlines from Google News RSS (free, no API key)
- `analyze_sentiment_batch()` sends headlines to Groq/Gemini with a structured prompt requesting JSON array of {sentiment, confidence} per headline
- Falls back to Neutral (confidence 0.0) if both LLMs fail
- Results cached 300 seconds via `@st.cache_data`

### SQLite Persistence (`data_manager.py`)
- 9 normalized tables: account, trades, options_trades, journal, watchlist, price_alerts, portfolio_history, badges, pending_orders
- `_migrate_from_json()` auto-migrates from `trading_data.json` on first run, renames to `.bak`
- WAL journal mode for concurrent read safety
- Public API (`load_data()` / `save_data()` / `default_data()`) unchanged — zero changes needed in other modules

### Market Data Fallback Chain (`market_data.py`)
- `get_price()`: FMP `/stable/quote` -> yfinance `.info` -> yfinance `.history(5d)` -> returns (0,0,0)
- `get_history()`: FMP `/stable/historical-price-eod/full` -> yfinance `.history(period)` -> empty DataFrame
- All FMP calls check `resp.ok` before JSON parsing to handle 429 rate limits gracefully
- Cached via `@st.cache_data` (prices 60s, history 300s)

---

## Stage 4: Validate

> **Iron Law:** Known answers, reasonableness, edges, AI risks.

### Known-Answer Tests
- **Cost basis reduction:** Buy 100 shares @ $10 ($1,000 cost), sell 50 shares -> cost correctly becomes $500 (average cost basis method in `portfolio.py:19-26`)
- **VaR weighting:** Verified market-value weights (shares x current_price / total_value) used when price function available; cost-basis weights as fallback (`risk_metrics.py:46-74`)
- **Six-Figure Club badge:** Triggers at portfolio_value >= $110,000 including unrealized gains, not just cash (`ai_signals.py:481`)

### Reasonableness Checks
- **DCF WACC:** Bounded 6-20% (sanity check against unrealistic discount rates)
- **DCF growth rate:** Capped 2-20%, uses median (robust to outliers) not mean
- **DCF terminal growth:** Set at 2.5% (below long-term GDP growth, conservative)
- **ML regime:** Requires minimum 100 data points and 5+ samples per class before training; prevents overfitting on sparse data
- **Sentiment confidence:** Groq prompted with temperature=0.1 for consistency; confidence values bounded 0.0-1.0

### Edge Cases Handled
- **FMP rate limit (429):** All endpoints check `resp.ok` before parsing; fall through to yfinance silently
- **yfinance empty info:** `get_price()` has 3-tier fallback (info -> history -> zero); Research tab renders chart even when `get_info()` returns `{}`
- **Empty portfolio:** All analytics pages show helpful onboarding messages ("Add positions to see analytics")
- **Insufficient ML data:** Falls back to SMA crossover with clear "SMA" badge in UI
- **LLM unavailable:** Coaching falls back to rule-based tips; chatbot shows API key setup instructions; sentiment defaults to Neutral

### AI Risks Mitigated
- **Hallucination in coaching:** LLM tips are advisory only; no automated trade execution from LLM output
- **Sentiment bias:** Confidence scores displayed alongside sentiment labels so users can assess reliability
- **Model overconfidence:** ML regime detection shows probability-based signal_strength (0-1) and tiered confidence labels (High/Medium/Low)
- **Data staleness:** All cached data has TTL (prices 60s, history 300s, financials 600s); manual "Refresh Prices" button clears all caches

---

## Stage 5: Evolve

> **Iron Law:** Self-contained export.

### Deployment Readiness
- **Streamlit Cloud:** `.streamlit/config.toml` configures headless mode; `requirements.txt` lists all dependencies
- **GitHub Actions CI:** `.github/workflows/` runs linting (flake8) and unit tests on push
- **Secrets management:** API keys stored in `.streamlit/secrets.toml` (gitignored), read via `st.secrets` with fallbacks to env vars and direct file parsing
- **Data portability:** SQLite database (`trading_data.db`) is a single file; auto-migrates from JSON on first run

### Self-Contained Features
- App works with **zero API keys** — yfinance provides market data, rule-based tips replace LLM coaching, SMA replaces ML regime, news still fetches via Google RSS
- App works with **FMP key only** — adds real-time quotes, financial statements, DCF valuation
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
  .streamlit/         # Config + secrets (secrets gitignored)
  .github/workflows/  # CI pipeline
  tests/              # Unit tests
```

---

## Stage 6: Reflect

> **Iron Law:** Document what didn't work.

### What Didn't Work
1. **FMP v3 API endpoints** returned 403 errors. Had to migrate all calls to `/stable/` endpoints. Lesson: Always check API version compatibility before building on an endpoint.
2. **Gemini as primary LLM** hit 429 quota limits within hours. Switched to Groq as primary (14,400 req/day free tier is much more generous). Lesson: Evaluate rate limits before choosing a primary provider.
3. **FMP news/sentiment endpoints** returned 402 (paid tier only). Replaced with Google News RSS (free, no key) + LLM-based sentiment scoring. Lesson: Free-tier APIs may not cover all endpoints — always have a plan B.
4. **yfinance `.info` returning empty dict** silently broke the Research tab — the guard `if info and not df.empty` skipped all rendering when `info` was falsy, even though price history (`df`) was available. Fixed by relaxing the guard to `if not df.empty` and wrapping fundamentals in an inner `if info:` check. Lesson: Falsy checks on dicts (`{}` is falsy) can silently gate too much code.
5. **Timezone mismatch** between tz-aware (yfinance) and tz-naive (pandas) datetimes caused crashes in `risk_metrics.py`. Fixed with `tz_localize(None)`. Lesson: Always normalize timezones at the data boundary.
6. **Hardcoded dark theme in `config.toml`** conflicted with the CSS-based light/dark toggle. Streamlit's native theme overrode CSS for internal components (dropdowns, Plotly chart text), making light mode unreadable. Fixed by removing the `[theme]` section entirely and letting CSS handle both modes. Lesson: Don't fight the framework's theme system — work with it or fully replace it, not both.
7. **JSON file persistence** had no ACID guarantees and could corrupt on concurrent writes. Migrated to SQLite with WAL mode, preserving the same `load_data()`/`save_data()` interface so no other modules needed changes. Lesson: Design clean abstraction boundaries so infrastructure swaps don't cascade.

### What Worked Well
- **Fallback patterns everywhere:** ML->SMA, LLM->rules, Custom DCF->FMP API, FMP->yfinance->history. The app never fully breaks.
- **Self-supervised labeling** for the Random Forest avoids external labeled datasets — the model generates training labels from forward price returns.
- **Interface preservation** during the SQLite migration: zero changes to 9 other modules.
- **Claude Code as development partner:** Plan mode for architecture decisions, codebase-aware implementation, systematic debugging of timezone and API issues.
