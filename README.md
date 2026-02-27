# Stock Portfolio Trading Simulator

A professional-grade stock portfolio dashboard and trading simulator built with Streamlit for **MGMT 69000 - Mastering AI For Finance** at Purdue University. Features 10 interactive pages, a composite ML trading signal (Random Forest + HMM + VADER), custom DCF valuation with sensitivity analysis, an AI trading assistant (OpenAI GPT-4o-mini), live news with sentiment analysis, portfolio rebalancing, quantitative risk analytics, Monte Carlo simulations, and options payoff modeling — built following the **DRIVER methodology** using **Claude Code** with the **PAL MCP Server**.

## Table of Contents

- [Architecture](#architecture)
- [Pages & Features](#pages--features)
- [AI/ML Components](#aiml-components)
- [Risk Metrics](#risk-metrics)
- [Data Sources](#data-sources)
- [Development Tools — Claude Code & PAL MCP Server](#development-tools--claude-code--pal-mcp-server)
- [Installation](#installation)
- [Requirements](#requirements)
- [Tech Stack](#tech-stack)

---

## Architecture

```
stock-portfolio-dashboard/
├── app.py              # Streamlit orchestrator — 10-page routing & UI
├── config.py           # Constants, theme colors (dark/light/colorblind), CSS
├── data_manager.py     # SQLite persistence layer (load/save/default schema)
├── market_data.py      # FMP primary + yfinance fallback, @st.cache_data caching
├── portfolio.py        # Holdings, P/L, portfolio value, trade stats (FIFO matching)
├── risk_metrics.py     # Sharpe, max drawdown, VaR (dollar+%), Monte Carlo, correlation
├── ai_signals.py       # ML regime (RF+HMM), composite signal, custom DCF w/ sensitivity,
│                       # VADER sentiment, RSI, MACD, Bollinger, badges, coaching
├── chatbot.py          # AI Trading Assistant (OpenAI primary, Groq/Gemini fallback)
├── news_feed.py        # Google News RSS fetching + AI sentiment scoring
├── rebalancer.py       # Portfolio rebalancing engine (weights, trades, cost)
├── dashboard.py        # Dashboard helpers (market overview, top movers, alerts)
├── sp500_tickers.py    # S&P 500 ticker dictionary (100 tickers)
├── requirements.txt    # Python dependencies
├── trading_data.db     # SQLite database (auto-created)
├── tests/              # pytest test suite (37 tests)
│   ├── test_ai_signals.py
│   ├── test_portfolio.py
│   ├── test_risk_metrics.py
│   └── test_data_manager.py
├── driver-plan/        # DRIVER methodology artifacts
│   ├── product-overview.md
│   ├── roadmap.md
│   ├── research.md
│   └── ai-log.md
├── .streamlit/
│   └── secrets.toml    # API keys (gitignored)
└── .gitignore
```

### Module Interaction

```
┌──────────────────────────────────────────────────────────┐
│                     app.py (Orchestrator)                 │
│  10 pages: Dashboard, Portfolio, Trade, Options,         │
│  Watchlist, Research, Analytics, Rebalance,              │
│  AI Assistant, Settings                                  │
├──────────────┬──────────────┬────────────────────────────┤
│  config.py   │ data_manager │    market_data.py          │
│  Theme/CSS   │  SQLite I/O  │  FMP + yfinance + cache    │
├──────────────┼──────────────┼────────────────────────────┤
│ portfolio.py │ risk_metrics │    ai_signals.py           │
│ Holdings/P&L │ Sharpe/VaR/MC│  RF+HMM+VADER/DCF/TA      │
├──────────────┼──────────────┼────────────────────────────┤
│  chatbot.py  │  news_feed   │    rebalancer.py           │
│ OpenAI/Groq  │ RSS+Sentiment│  Weights/Trades/Cost       │
├──────────────┼──────────────┼────────────────────────────┤
│ dashboard.py │sp500_tickers │                            │
│ Indices/Movers│ Ticker dict │                            │
└──────────────┴──────────────┴────────────────────────────┘
```

### ML Signal Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Random Forest│     │   HMM (3    │     │   VADER     │
│ (200 trees)  │     │   states)   │     │ (offline)   │
│ 10 features  │     │ returns +   │     │ headline    │
│ TimeSeriesSplit│   │ volatility  │     │ sentiment   │
│   50% weight │     │  30% weight │     │  20% weight │
└──────┬───────┘     └──────┬──────┘     └──────┬──────┘
       │                    │                    │
       └────────────┬───────┘                    │
                    │  Composite Signal           │
                    ├─────────────────────────────┘
                    ▼
           Bullish / Neutral / Bearish
           (score: -1.0 to +1.0)
```

**Data Flow:**
1. `st.session_state.data` loaded once at startup from `trading_data.db` (SQLite)
2. User actions validated via `portfolio.py` constraints
3. Trades appended to portfolio/journal, cash updated
4. `save_data()` persists to SQLite after every mutation
5. Market data fetched via `market_data.py` with automatic FMP -> yfinance failover
6. ML signals and analytics computed on-the-fly by `ai_signals.py` and `risk_metrics.py`

---

## Pages & Features

### 1. Dashboard (Home Page)
The default landing page providing a market-wide and portfolio-level overview.

- **Market Overview** — Live S&P 500 (SPY), Nasdaq 100 (QQQ), and Dow Jones (DIA) prices with daily change %
- **Portfolio Snapshot** — Total value, P/L, cash, invested amount, and a mini allocation donut chart
- **Market Sentiment** — S&P 500 regime detection (Bullish/Bearish/Neutral) via ML Random Forest
- **Top Movers** — Best and worst performing holdings today, color-coded with green/red borders
- **Active Alerts** — Triggered price alerts from watchlist with current vs target prices
- **Market News** — 5 live headlines from Google News RSS with AI-powered sentiment badges (Bullish/Bearish/Neutral)
- **Quick Actions** — One-click navigation buttons to Trade, Research, Watchlist, Analytics, and AI Assistant

### 2. Portfolio
- View all stock holdings with live prices, P/L, and return %
- Equity curve chart with starting balance reference line
- Portfolio allocation pie chart (stocks, cash, options)
- Sector allocation pie chart (auto-detected via yfinance)
- Options positions table with mid-price valuation
- Quick sell buttons for rapid position exits
- Achievement badges display with unlock progress
- Transaction history (stocks and options) with CSV export

### 3. Trade
- Search any of 100 S&P 500 stocks with live price display
- **Order types**: Market, Limit, Stop-Loss
- **Actions**: Buy, Sell, Short Sell, Cover Short
- Order preview with commission breakdown
- Real-time validation (cash/shares checks)
- Confirmation dialog before execution

### 4. Options Trading
- Real-time options chain data (calls and puts) via yfinance
- Multiple expiration date support
- Buy to Open / Sell to Close
- **P/L payoff diagrams** — visual breakeven, max profit, max loss at expiration
- Per-contract commission tracking

### 5. Watchlist
- Add/remove S&P 500 stocks
- Real-time prices with daily change % and 52-week high/low
- **Price alerts** — set target prices with notification status
- **Quick buy buttons** for one-click order preview

### 6. Research
- Multi-timeframe charts: 1W, 1M, 3M, 6M, 1Y, 5Y
- Chart types: Line and Candlestick with SMA 20/50
- **Toggleable technical indicators**: RSI (14), MACD (12/26/9), Bollinger Bands (20, 2σ)
- Volume subplot with color-coded bars
- **Live News with AI Sentiment** — per-stock news headlines from Google News with Bullish/Bearish/Neutral badges scored by Groq/Gemini
- **Composite ML Signal** — 3-component signal (RF + HMM + VADER) with per-component breakdown
- **DCF Valuation** — Custom DCF with calculated beta, WACC, sensitivity heatmap

### 7. Analytics & Risk
- Portfolio vs S&P 500 benchmark overlay chart
- **Risk metrics**: Sharpe ratio, max drawdown, VaR (95%)
- **Monte Carlo fan chart** — 500-simulation 1-year projection with percentile bands (p5–p95)
- **Correlation heatmap** for diversification analysis
- **Trade journal analytics** — total trades, win rate, avg win/loss, profit factor, best/worst trade
- **AI Trading Coach** — LLM-powered tips enriched with regime, VaR, Sharpe, and composite signal
- Position weight bar chart

### 8. Portfolio Rebalancer
- **Current allocation** table showing each holding's shares, price, value, and weight %
- **Strategy selector**: Equal Weight or Custom Weights
- **Current vs Target** grouped bar chart (Plotly)
- **Suggested trades** table — exact buy/sell actions with shares, price, and value
- **Cost summary** — total buys, sells, net cash needed, commissions
- **One-click Execute** — rebalances entire portfolio in a single action
- Cash deployment toggle (include/exclude idle cash)

### 9. AI Trading Assistant
- Full-page conversational AI with chat bubbles and avatars
- **Portfolio-aware** — injects holdings, cash, P/L, watchlist, and live prices into every response
- 6 suggested prompt buttons (portfolio analysis, market outlook, opportunities, risk explanation, trade review, app tour)
- Powered by **OpenAI GPT-4o-mini** with **Groq** and **Google Gemini** fallback
- Markdown-formatted responses with conversation history

### 10. Settings
- Dark/Light theme toggle
- **Colorblind mode** (blue/orange instead of green/red)
- Commission configuration (stocks per-trade and options per-contract)
- Account reset with custom starting balance

---

## AI/ML Components

### Composite ML Trading Signal (`ai_signals.py`)

Three independent ML/NLP models blended into a single trading signal:

| Component | Model | Weight | Input | Output |
|---|---|---|---|---|
| **Random Forest** | 200 trees, depth 8, TimeSeriesSplit CV (5 folds) | 50% | 10 technical features | Bullish/Bearish/Neutral + probability |
| **Hidden Markov Model** | 3-state Gaussian HMM (hmmlearn) | 30% | Daily returns + rolling volatility | Bull/Sideways/Bear + state probabilities |
| **VADER Sentiment** | Rule-based NLP (vaderSentiment) | 20% | News headlines | Score -1.0 to +1.0 |

**RF Features (10):** RSI(14) via Wilder's EMA, MACD histogram, Bollinger %B, 5d/20d momentum, volume ratio, ATR(14), price vs SMA50, price vs SMA200, SMA 20/50 spread.

**Labels:** Self-supervised from 10-day forward returns (>+1% = Bullish, <-1% = Bearish). TimeSeriesSplit prevents look-ahead bias.

**Fallback:** When sklearn is unavailable or data < 100 points, falls back to SMA crossover with reduced weight. Unavailable components are re-weighted so weights always sum to 1.0.

### Custom DCF Valuation (`ai_signals.py`)

Full discounted cash flow model (not a passthrough from FMP):

- **WACC via CAPM**: `Ke = Rf + β(Rm − Rf)` where β is calculated from stock vs SPY covariance
- **Calculated beta**: 60+ days of daily returns vs SPY (capped [0.1, 3.0])
- **Effective tax rate**: From income statement (`tax_expense / pre_tax_income`), fallback 21%
- **Cost of debt**: `interest_expense / total_debt` (implied yield)
- **5-year FCF projection** + Gordon Growth Model terminal value (2.5% perpetuity growth)
- **Sensitivity heatmap**: 7x7 grid of intrinsic values across growth rate × WACC combinations
- **Signal**: Undervalued (>15% margin), Fair Value, Overvalued (<-15% margin)

### AI Trading Assistant (`chatbot.py`)

A portfolio-aware conversational AI:

- **Real-time context**: Injects current holdings, cash balance, P/L, watchlist prices, and trade history into every response
- **App navigation**: Knows all 10 pages and guides users to the right feature
- **Multi-provider**: OpenAI GPT-4o-mini primary (~$0.0004/query), Groq fallback (Llama 3.3 70B), Gemini last resort
- **Full markdown rendering**: Tables, bold, lists, code blocks in responses

### AI Trading Coach (`ai_signals.py`)

LLM-powered coaching enriched with ML context:

- **Context-aware**: Portfolio holdings, win/loss stats, cash allocation, badges
- **ML-enriched**: Market regime (RF), risk metrics (VaR, Sharpe, max drawdown), composite signal
- **Multi-provider**: OpenAI → Groq → Gemini → rule-based fallback
- Generates 3 concise, actionable tips per request

### News Sentiment Analysis (`news_feed.py`)

Live news headlines scored for stock market sentiment:

- **Source**: Google News RSS (free, no API key required)
- **LLM scoring**: Batch classification via OpenAI/Groq/Gemini — each headline scored as Bullish, Bearish, or Neutral
- **VADER fallback**: Offline deterministic sentiment when LLM APIs unavailable
- **Caching**: 5-minute TTL to avoid excessive API calls

### Technical Indicators (`ai_signals.py`)

| Indicator | Description |
|---|---|
| **RSI (14)** | Relative Strength Index via Wilder's EMA — overbought (>70) / oversold (<30) |
| **MACD (12/26/9)** | Moving Average Convergence Divergence — trend momentum |
| **Bollinger Bands (20, 2σ)** | Volatility bands around 20-period SMA |

### Achievement System (`ai_signals.py`)

7 gamification badges (permanent — once earned, never revoked):

| Badge | Criteria |
|---|---|
| First Trade | Complete 1 trade |
| Diversifier | Hold 5+ different stocks |
| Options Trader | Place 1 options trade |
| Risk Manager | Use a stop-loss order |
| Consistent | Trade on 5+ different days |
| Six-Figure Club | Portfolio value hits $110K+ |
| Watchful | Track 5+ stocks on watchlist |

---

## Risk Metrics (`risk_metrics.py`)

All calculations use pure numpy/pandas:

| Metric | Method | Description |
|---|---|---|
| **Sharpe Ratio** | `compute_sharpe_ratio()` | Annualized excess return per unit of risk (rf: 5%) |
| **Max Drawdown** | `compute_max_drawdown()` | Largest peak-to-trough decline |
| **Value at Risk** | `compute_var_historical()` | 95th percentile daily loss (dollar + percent) |
| **Monte Carlo** | `compute_monte_carlo()` | 500-sim 1-year projection with p5–p95 bands |
| **Correlation Matrix** | `compute_correlation_matrix()` | Pairwise correlation heatmap |

---

## Data Sources

Dual data-source architecture with automatic failover:

| Data Type | Primary Source | Fallback |
|---|---|---|
| Stock prices | Financial Modeling Prep (stable API) | Yahoo Finance |
| Historical OHLCV | Financial Modeling Prep (stable API) | Yahoo Finance |
| Stock fundamentals | Yahoo Finance | — |
| Options chains | Yahoo Finance | — |
| Analyst estimates & DCF | Financial Modeling Prep | — |
| News headlines | Google News RSS | — |
| News sentiment | OpenAI (GPT-4o-mini) | Groq / Gemini / VADER |
| AI Chat | OpenAI (GPT-4o-mini) | Groq / Gemini |
| AI Coaching | OpenAI (GPT-4o-mini) | Groq / Gemini / Rules |
| Offline sentiment | VADER (vaderSentiment) | — |
| Beta calculation | yfinance (SPY benchmark) | Default 1.0 |

All market data cached with `@st.cache_data` (60s prices, 300s history/news/sentiment).

---

## Development Tools — Claude Code & PAL MCP Server

This project was developed using **Claude Code** (Anthropic's AI coding assistant CLI) with the **PAL MCP Server** (Model Context Protocol) as an AI-powered development plugin.

### Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is Anthropic's official CLI tool for AI-assisted software development. It provides:

- **Agentic coding** — Claude reads, writes, and edits files autonomously while following project patterns
- **Context-aware** — understands the full codebase structure, dependencies, and conventions
- **Multi-tool execution** — runs shell commands, searches code, manages git operations
- **Session memory** — persists project knowledge across conversations for consistent development

Claude Code was used for the entire development lifecycle of this project:
- Initial architecture design and modular refactoring
- Feature implementation across all 10 pages
- API integration (FMP, yfinance, Groq, Gemini, Google News RSS)
- Code review, debugging, and iterative refinement
- Git operations and deployment

### PAL MCP Server (DRIVER Plugin)

The **PAL MCP Server** is a Model Context Protocol server that extends Claude Code with additional AI-powered development tools. It connects as a plugin via the MCP protocol, enabling Claude to delegate specialized tasks to external AI models (Gemini, OpenAI, etc.).

**Configuration** (`~/.claude/mcp_servers.json`):
```json
{
  "pal": {
    "command": "/path/to/pal-mcp-server/.pal_venv/bin/python",
    "args": ["/path/to/pal-mcp-server/server.py"]
  }
}
```

**Environment** (`pal-mcp-server/.env`):
```env
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key      # Optional
OPENROUTER_API_KEY=your_key  # Optional
```

#### PAL MCP Tools Used

| Tool | Purpose | How It Was Used |
|---|---|---|
| **`chat`** | Interactive AI brainstorming | Ideating feature designs, exploring UI patterns |
| **`thinkdeep`** | Step-by-step deep analysis | Architecting the modular refactor, data flow design |
| **`planner`** | Multi-step task planning | Breaking down the 3-feature implementation (Dashboard, News+Sentiment, Rebalancer) |
| **`codereview`** | Comprehensive code review | Reviewing app.py, validating theme consistency, checking error handling |
| **`analyze`** | File and code analysis | Understanding existing module patterns before adding new features |
| **`debug`** | Root cause analysis | Diagnosing FMP API endpoint restrictions, fixing import issues |
| **`consensus`** | Multi-model consensus | Getting perspectives from multiple AI models on architecture decisions |
| **`listmodels`** | List available models | Selecting optimal models for different tasks |
| **`version`** | Server version info | Verifying MCP server connectivity |

#### How MCP Integration Works

```
┌─────────────┐    MCP Protocol (stdio)    ┌──────────────────┐
│ Claude Code  │ ◄──────────────────────► │  PAL MCP Server  │
│  (CLI Agent) │    JSON-RPC messages      │   (Python)       │
│              │                           │                  │
│ - Reads code │    Tool calls:            │ - Gemini API     │
│ - Writes code│    chat, thinkdeep,       │ - OpenAI API     │
│ - Runs tests │    planner, codereview,   │ - OpenRouter     │
│ - Git ops    │    analyze, debug...      │ - Custom models  │
└─────────────┘                           └──────────────────┘
```

1. **Claude Code** launches the PAL MCP server as a subprocess
2. Server advertises available tools via MCP protocol handshake
3. Claude calls tools (e.g., `planner` to design features, `codereview` to validate code)
4. PAL routes requests to configured AI providers (Gemini, OpenAI, etc.)
5. Results flow back to Claude, which incorporates them into its development workflow

#### Development Workflow with MCP

The typical development workflow used for this project:

1. **Plan** — Use PAL `planner` and `thinkdeep` tools to architect features
2. **Explore** — Claude Code reads the codebase, PAL `analyze` examines specific modules
3. **Implement** — Claude Code writes code following established patterns
4. **Review** — PAL `codereview` validates the implementation
5. **Debug** — PAL `debug` tool helps diagnose issues (e.g., FMP API restrictions)
6. **Iterate** — Refine based on testing and user feedback

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/naslam123/stock-portfolio-dashboard.git
cd stock-portfolio-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add API keys** to `.streamlit/secrets.toml`:
```toml
# Primary LLM for chatbot, coaching, and sentiment (~$0.0004/query)
OPENAI_API_KEY = "your_key"  # https://platform.openai.com

# Required for DCF valuation and price data (free: 250 req/day)
FMP_API_KEY = "your_key"     # https://financialmodelingprep.com/developer

# Fallback LLM (free: 14,400 req/day)
GROQ_API_KEY = "your_key"    # https://console.groq.com

# Second fallback for AI features
GEMINI_API_KEY = "your_key"  # https://aistudio.google.com/apikey
```

> **Note**: News headlines work without any API key (Google News RSS). VADER sentiment works offline with no API key. The ML composite signal (RF + HMM) needs no API key. OpenAI key is recommended for best chatbot/coaching quality.

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open** `http://localhost:8501`

### Optional: PAL MCP Server Setup (for development)

If you want to replicate the AI-assisted development workflow:

1. Clone and set up the PAL MCP server:
```bash
git clone <pal-mcp-server-repo>
cd pal-mcp-server
python -m venv .pal_venv
source .pal_venv/bin/activate
pip install -r requirements.txt
```

2. Configure API keys in `pal-mcp-server/.env`:
```env
GEMINI_API_KEY=your_gemini_key
```

3. Register with Claude Code:
```bash
claude mcp add pal /path/to/.pal_venv/bin/python /path/to/server.py
```

4. Restart Claude Code to activate the MCP connection.

---

## Requirements

- Python 3.10+
- streamlit >= 1.54.0
- yfinance >= 0.2.36
- pandas >= 2.0.0
- plotly >= 5.18.0
- numpy >= 1.26.0
- requests >= 2.31.0
- openai >= 1.0.0
- groq >= 0.9.0
- google-generativeai >= 0.5.0
- scikit-learn >= 1.4.0
- hmmlearn >= 0.3.0
- vaderSentiment >= 3.3.2
- numpy-financial >= 1.0.0

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (wide layout, custom CSS theming) |
| **AI Chat** | OpenAI GPT-4o-mini / Groq (Llama 3.3 70B) / Gemini |
| **ML Signals** | scikit-learn (Random Forest), hmmlearn (HMM), VADER |
| **Sentiment Analysis** | OpenAI/Groq/Gemini + VADER offline fallback |
| **News Feed** | Google News RSS (free, no API key) |
| **Primary Market Data** | Financial Modeling Prep (stable API) |
| **Fallback Market Data** | Yahoo Finance (via yfinance) |
| **Charts** | Plotly (candlestick, line, bar, pie, heatmap, fan chart) |
| **Risk Engine** | NumPy + Pandas (Monte Carlo, Sharpe, VaR, correlation) |
| **DCF Engine** | NumPy + yfinance (CAPM beta, WACC, sensitivity analysis) |
| **Storage** | SQLite (`trading_data.db`) |
| **Testing** | pytest (37 tests across 4 test modules) |
| **Development** | Claude Code + PAL MCP Server |

---

## License

MIT License

## DRIVER Methodology

This project was developed following the **DRIVER** framework for AI-assisted finance tool development:

| Stage | Artifact | Description |
|---|---|---|
| **D**efine | `driver-plan/product-overview.md` | Problem definition, success criteria, research |
| **R**epresent | `driver-plan/roadmap.md` | 5-section buildable roadmap with dependency graph |
| **I**mplement | Source code + `tests/` | Iterative build with show-don't-tell approach |
| **V**alidate | `driver-plan/validation.md` | Cross-check vs known answers, edge cases, AI risks |
| **E**volve | Final deliverable | Export package with slides, SubStack, demo |
| **R**eflect | `driver-plan/ai-log.md` | AI development log and lessons learned |

## Author

Naveed Aslam Perwez (naslam@purdue.edu)
MGMT 69000 - Mastering AI For Finance | Purdue University
