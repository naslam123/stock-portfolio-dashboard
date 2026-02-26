# DRIVER Framework: AI Tool Usage Documentation

**Course:** MGMT 590 - Mastering AI For Finance | Purdue University
**Student:** Naveed Aslam Perwez (naslam@purdue.edu)
**Project:** Stock Portfolio Trading Simulator

---

## AI Tools Used

| Tool | Role | Integration |
|------|------|-------------|
| **Claude Code (Anthropic)** | Primary development assistant — architecture, code generation, debugging, refactoring | CLI agent with full codebase access |
| **PAL MCP Server** | Model Context Protocol server for structured AI interactions | Registered in `~/.claude/mcp_servers.json` |
| **Groq (Llama 3.3 70B)** | In-app AI chatbot + LLM coaching tips | Free API, 14,400 req/day |
| **Gemini (Google)** | Fallback LLM for chatbot + coaching | Free API tier |
| **scikit-learn** | ML regime detection (Random Forest classifier) | Self-training on price data |

---

## DRIVER Process for Each Major Feature

### 1. ML Regime Detection (Random Forest)

**Describe:** The original market regime detection used a simple SMA 20/50 crossover which is not genuine AI/ML. Professor feedback required replacing it with a real machine learning model.

**Request:** Asked Claude Code to design a self-training Random Forest classifier that:
- Extracts 6 technical features (RSI, MACD histogram, Bollinger %B, momentum, volume change, SMA spread)
- Generates self-supervised labels from 10-day forward returns
- Falls back to SMA when sklearn unavailable or insufficient data (<100 points)

**Inspect:** Reviewed the generated `_build_ml_features()`, `_generate_labels()`, and updated `detect_market_regime()` functions. Verified feature engineering aligns with standard quantitative finance practices.

**Verify:** Tested with SPY and AAPL price histories. Confirmed:
- ML model trains on 100+ data points with both Bullish and Bearish labels
- Prediction includes probability-based confidence scores
- Feature importance dict identifies top signal drivers
- Falls back gracefully to SMA on insufficient data

**Evaluate:** The ML model provides more nuanced regime detection than SMA crossover. Probability outputs give users interpretable confidence levels. Self-supervised labeling avoids need for external labeled data.

**Refine:** Added `model_type` field to distinguish "ML Model" vs "SMA" in the UI. Added feature importance display. Capped minimum training samples at 30.

---

### 2. Custom DCF Valuation Model

**Describe:** The original DCF was a passthrough from FMP's pre-computed endpoint — no local computation. Professor required a custom valuation model.

**Request:** Asked Claude Code to implement a full DCF model that:
- Extracts FCF from cash flow statements (operating cash flow - capex)
- Estimates growth rate from blended FCF + revenue trends (capped 2-20%)
- Calculates WACC via CAPM (cost of equity) + cost of debt from financials
- Projects 5-year FCF with perpetuity terminal value
- Discounts to present, subtracts net debt, divides by shares outstanding

**Inspect:** Reviewed `_compute_custom_dcf()` for financial modeling correctness:
- WACC bounded between 6-20% (sanity check)
- Terminal growth rate set at 2.5% (below long-term GDP growth)
- Growth rate uses median (robust to outliers) not mean

**Verify:** Tested with several tickers. Confirmed:
- Custom DCF fires when FMP financial data available
- Falls back to FMP pre-computed DCF when data unavailable
- WACC and growth rate displayed alongside valuation signal

**Evaluate:** Custom DCF gives students transparency into valuation assumptions (WACC, growth rate, base FCF) rather than a black-box number. Educational value significantly higher.

**Refine:** Added `model_type` badge ("Custom DCF" vs "FMP API") in the Research page UI. Added WACC and growth rate to the description string.

---

### 3. LLM-Powered Coaching Tips

**Describe:** Original coaching tips were static if-else rules. Professor flagged this as not utilizing AI. Needed to route through an actual LLM.

**Request:** Asked Claude Code to:
- Build a `_gather_coaching_context()` function that summarizes portfolio state
- Send context to Groq (primary) / Gemini (fallback) requesting 3 JSON-formatted tips
- Keep rule-based logic as `_rule_based_tips()` fallback

**Inspect:** Reviewed the prompt engineering: system role instructs the LLM to be a trading coach, provide actionable advice, and return strict JSON array format.

**Verify:** Tested with Groq API key present — confirmed LLM returns personalized tips referencing actual portfolio state. Tested without API key — confirmed graceful fallback to rule-based tips.

**Evaluate:** LLM tips are contextual, varied, and reference the user's specific holdings and trading patterns. Significantly better than static rules.

**Refine:** Added "LLM Powered" vs "Rules" badge in Analytics UI. Added markdown code fence stripping for robust JSON parsing.

---

### 4. AI Trading Assistant (Chatbot)

**Describe:** Needed a full-page AI assistant that knows the user's portfolio, real-time prices, and can help navigate the app.

**Request:** Asked Claude Code to implement:
- Groq (Llama 3.3 70B) as primary, Gemini as fallback
- Portfolio-aware system prompt with live holdings, prices, and P/L
- Chat history management with suggested starter prompts

**Inspect:** Reviewed `_build_system_prompt()` for comprehensive context injection, including portfolio state, real-time prices, and full app navigation guide.

**Verify:** Tested multi-turn conversations. Confirmed the assistant references actual portfolio data, gives navigation guidance, and maintains conversation context.

**Evaluate:** The chatbot provides a natural language interface to the entire simulator. Students can ask about their portfolio, get explanations of risk metrics, and receive trading guidance.

**Refine:** Added 6 suggested starter prompts covering portfolio analysis, market outlook, risk metrics, trade review, opportunities, and app tutorial.

---

### 5. SQLite Persistence (Infrastructure)

**Describe:** Professor flagged JSON file storage as a limitation. Needed proper database persistence.

**Request:** Asked Claude Code to:
- Design normalized SQLite schema (10 tables)
- Auto-migrate from `trading_data.json` on first run
- Keep identical `load_data()` / `save_data()` / `default_data()` interface
- Zero changes needed in app.py or other modules

**Inspect:** Reviewed schema normalization, migration logic, and dict serialization/deserialization. Confirmed the public API is unchanged.

**Verify:** Tested migration: created sample JSON data, ran the app, confirmed data appeared correctly in SQLite. Confirmed JSON renamed to `.bak`. Confirmed subsequent runs use SQLite directly.

**Evaluate:** SQLite provides ACID guarantees, WAL mode for concurrent reads, and structured queries. Major improvement over JSON serialization.

**Refine:** Added WAL journal mode for better performance. Added `.bak` and `.db` to `.gitignore`.

---

### 6. Bug Fixes (Cost Basis, VaR Weights, Badge)

**Describe:** Professor identified 3 bugs:
1. Sells don't reduce cost basis in `get_holdings()`
2. VaR uses cost-basis weights instead of market-value weights
3. Six-Figure Club badge ignores unrealized gains

**Request:** Asked Claude Code to fix each bug with minimal code changes.

**Inspect:** Reviewed each fix:
1. Cost basis: `avg_cost = cost / shares; cost -= sell_shares * avg_cost`
2. VaR: Added `get_price_fn` parameter for market-value weights
3. Badge: Added `portfolio_value` kwarg using actual `portfolio_value()` result

**Verify:** Tested cost basis: buy 100@$10 ($1000 cost), sell 50 → cost correctly becomes $500. Tested badge: triggers on unrealized gains when portfolio value exceeds $110K.

**Evaluate:** All three bugs fixed with surgical changes. No regressions in dependent code.

**Refine:** Guarded sell shares with `min(t["shares"], h["shares"])` to prevent negative shares.

---

## Development Workflow

1. **Planning:** Used Claude Code in Plan mode to design implementation strategy, identify file dependencies, and order tasks.
2. **Implementation:** Claude Code generated code with real-time codebase awareness, following existing patterns and conventions.
3. **Testing:** Manual testing via `streamlit run app.py` after each phase. Verified each feature independently.
4. **Version Control:** All changes committed via `git` with descriptive messages.

## Lessons Learned

- **Fallback patterns are essential:** Every AI feature has a non-AI fallback (ML→SMA, LLM→rules, Custom DCF→FMP API) ensuring the app never breaks due to API limits or missing dependencies.
- **Self-supervised learning** avoids the need for external labeled datasets — the model generates its own training labels from price data.
- **Interface preservation** during the SQLite migration meant zero changes to 9 other modules — a clean abstraction boundary.
- **DRIVER process** forced deliberate inspection of AI-generated code rather than blind acceptance, catching edge cases and improving robustness.
