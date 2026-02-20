# Stock Portfolio Trading Simulator

A professional-grade stock portfolio dashboard and trading simulator built with Streamlit for **MGMT 590 - Mastering AI For Finance** at Purdue University. Features 10 interactive pages, an AI trading assistant, live news with sentiment analysis, portfolio rebalancing, technical indicators, quantitative risk analytics, Monte Carlo simulations, and options payoff modeling — all developed using **Claude Code** with the **PAL MCP Server** as an AI-powered development assistant.

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
├── app.py              # Streamlit orchestrator — 10-page routing & UI (~1700 lines)
├── config.py           # Constants, theme colors (dark/light/colorblind), CSS
├── data_manager.py     # JSON persistence layer (load/save/default schema)
├── market_data.py      # FMP primary + yfinance fallback, @st.cache_data caching
├── portfolio.py        # Holdings, P/L, portfolio value, trade stats (FIFO matching)
├── risk_metrics.py     # Sharpe, max drawdown, VaR, Monte Carlo, correlation matrix
├── ai_signals.py       # Regime detection, DCF valuation, RSI, MACD, Bollinger,
│                       # options payoff, badges, coaching tips
├── chatbot.py          # AI Trading Assistant (Groq/Gemini, portfolio-aware)
├── news_feed.py        # Google News RSS fetching + AI sentiment scoring
├── rebalancer.py       # Portfolio rebalancing engine (weights, trades, cost)
├── dashboard.py        # Dashboard helpers (market overview, top movers, alerts)
├── sp500_tickers.py    # S&P 500 ticker dictionary (100 tickers)
├── requirements.txt    # Python dependencies
├── trading_data.json   # User data persistence (auto-created)
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
│  Theme/CSS   │  JSON I/O    │  FMP + yfinance + cache    │
├──────────────┼──────────────┼────────────────────────────┤
│ portfolio.py │ risk_metrics │    ai_signals.py           │
│ Holdings/P&L │ Sharpe/VaR/MC│  Regime/DCF/TA/Badges      │
├──────────────┼──────────────┼────────────────────────────┤
│  chatbot.py  │  news_feed   │    rebalancer.py           │
│ Groq/Gemini  │ RSS+Sentiment│  Weights/Trades/Cost       │
├──────────────┼──────────────┼────────────────────────────┤
│ dashboard.py │sp500_tickers │                            │
│ Indices/Movers│ Ticker dict │                            │
└──────────────┴──────────────┴────────────────────────────┘
```

**Data Flow:**
1. `st.session_state.data` loaded once at startup from `trading_data.json`
2. User actions validated via `portfolio.py` constraints
3. Trades appended to portfolio/journal, cash updated
4. `save_data()` persists to JSON after every mutation
5. Market data fetched via `market_data.py` with automatic FMP -> yfinance failover
6. Analytics computed on-the-fly by `risk_metrics.py` and `ai_signals.py`

---

## Pages & Features

### 1. Dashboard (Home Page)
The default landing page providing a market-wide and portfolio-level overview.

- **Market Overview** — Live S&P 500 (SPY), Nasdaq 100 (QQQ), and Dow Jones (DIA) prices with daily change %
- **Portfolio Snapshot** — Total value, P/L, cash, invested amount, and a mini allocation donut chart
- **Market Sentiment** — S&P 500 regime detection (Bullish/Bearish/Neutral) via SMA 20/50 crossover
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
- **AI Signals** — Market regime detection + DCF valuation analysis

### 7. Analytics & Risk
- Portfolio vs S&P 500 benchmark overlay chart
- **Risk metrics**: Sharpe ratio, max drawdown, VaR (95%)
- **Monte Carlo fan chart** — 500-simulation 1-year projection with percentile bands (p5–p95)
- **Correlation heatmap** for diversification analysis
- **Trade journal analytics** — total trades, win rate, avg win/loss, profit factor, best/worst trade
- **AI Trading Coach** — rule-based tips based on portfolio behavior
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
- Powered by **Groq (Llama 3.3 70B)** with **Google Gemini** fallback
- Markdown-formatted responses with conversation history

### 10. Settings
- Dark/Light theme toggle
- **Colorblind mode** (blue/orange instead of green/red)
- Commission configuration (stocks per-trade and options per-contract)
- Account reset with custom starting balance

---

## AI/ML Components

### AI Trading Assistant (`chatbot.py`)

A portfolio-aware conversational AI:

- **Real-time context**: Injects current holdings, cash balance, P/L, watchlist prices, and trade history into every conversation
- **App navigation**: Knows all 10 pages and guides users to the right feature
- **Multi-provider**: Groq primary (Llama 3.3 70B, 14,400 free req/day), Gemini fallback
- **Full markdown rendering**: Tables, bold, lists, code blocks in responses

### News Sentiment Analysis (`news_feed.py`)

Live news headlines scored for stock market sentiment:

- **Source**: Google News RSS (free, no API key required)
- **Sentiment scoring**: Batch classification via Groq/Gemini — each headline scored as Bullish, Bearish, or Neutral with a confidence score (0-1)
- **Display**: Color-coded sentiment pills on Dashboard and Research pages
- **Caching**: 5-minute TTL to avoid excessive API calls

### Market Regime Detection (`ai_signals.py`)

SMA 20/50 crossover classification:

- **Bullish**: SMA20 > SMA50 with price confirmation
- **Bearish**: SMA20 < SMA50 with price confirmation
- **Neutral**: SMAs converging, no clear trend

Each signal includes confidence level (High/Medium/Low) and signal strength (0-100%).

### DCF Valuation Analysis (`ai_signals.py`)

Aggregates analyst consensus from Financial Modeling Prep:

- **Undervalued**: DCF intrinsic value > 15% above current price
- **Overvalued**: DCF intrinsic value > 15% below current price
- **Fair Value**: DCF within 15% of current price

### Technical Indicators (`ai_signals.py`)

| Indicator | Description |
|---|---|
| **RSI (14)** | Relative Strength Index — overbought (>70) / oversold (<30) |
| **MACD (12/26/9)** | Moving Average Convergence Divergence — trend momentum |
| **Bollinger Bands (20, 2σ)** | Volatility bands around 20-period SMA |

### AI Trading Coach (`ai_signals.py`)

Rule-based coaching analyzing trading behavior: concentration risk, cash allocation, overtrading, and losing streak alerts.

### Achievement System (`ai_signals.py`)

7 gamification badges with toast notifications:

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
| **Value at Risk** | `compute_var_historical()` | 95th percentile daily loss in dollars |
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
| News sentiment | Groq (Llama 3.3 70B) | Google Gemini |
| AI Chat | Groq (Llama 3.3 70B) | Google Gemini |

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
# Required for DCF valuation and price data (free: 250 req/day)
FMP_API_KEY = "your_key"     # https://financialmodelingprep.com/developer

# Required for AI Assistant + Sentiment Analysis (free: 14,400 req/day)
GROQ_API_KEY = "your_key"    # https://console.groq.com

# Optional fallback for AI features
GEMINI_API_KEY = "your_key"  # https://aistudio.google.com/apikey
```

> **Note**: News headlines work without any API key (Google News RSS). The AI sentiment scoring requires either a Groq or Gemini key. FMP key is only needed for DCF valuation and primary price data.

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
- streamlit >= 1.31.0
- yfinance >= 0.2.36
- pandas >= 2.0.0
- plotly >= 5.18.0
- numpy >= 1.26.0
- requests >= 2.31.0
- groq >= 0.9.0
- google-generativeai >= 0.5.0

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (wide layout, custom CSS theming) |
| **AI Chat** | Groq (Llama 3.3 70B) / Google Gemini (2.0 Flash) |
| **Sentiment Analysis** | Groq/Gemini batch classification |
| **News Feed** | Google News RSS (free, no API key) |
| **Primary Market Data** | Financial Modeling Prep (stable API) |
| **Fallback Market Data** | Yahoo Finance (via yfinance) |
| **Charts** | Plotly (candlestick, line, bar, pie, heatmap, fan chart) |
| **Risk Engine** | NumPy + Pandas (Monte Carlo, Sharpe, VaR, correlation) |
| **Storage** | JSON file persistence (`trading_data.json`) |
| **Development** | Claude Code + PAL MCP Server |

---

## License

MIT License

## Author

MGMT 590 - Mastering AI For Finance | Purdue University
