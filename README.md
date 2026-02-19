# Stock Portfolio Trading Simulator

A professional-grade stock portfolio dashboard and trading simulator built with Streamlit for MGMT 590 - Mastering AI For Finance at Purdue University. Features an AI trading assistant, technical indicators, quantitative risk analytics, Monte Carlo simulations, options payoff modeling, and a dual data-source architecture.

## Architecture

```
stock-portfolio-dashboard/
├── app.py              # Streamlit orchestrator — page routing & UI rendering
├── config.py           # Constants, theme colors (with colorblind mode), CSS
├── data_manager.py     # JSON persistence layer (load/save/default)
├── market_data.py      # FMP primary + yfinance fallback, caching
├── portfolio.py        # Holdings, P/L, portfolio value, trade stats
├── risk_metrics.py     # Sharpe, max drawdown, VaR, Monte Carlo, correlation
├── ai_signals.py       # Regime detection, DCF valuation, RSI, MACD, Bollinger,
│                       # options payoff, badges, coaching
├── chatbot.py          # AI Trading Assistant (Groq/Gemini, portfolio-aware)
├── sp500_tickers.py    # S&P 500 ticker dictionary
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # API keys (gitignored)
└── .gitignore
```

## AI/ML Components

### AI Trading Assistant (`chatbot.py`)

A portfolio-aware conversational AI powered by Groq (Llama 3.3 70B, free tier) with Gemini fallback:

- **Real-time context**: Injects current holdings, cash balance, P/L, watchlist prices, and trade history into every conversation
- **App navigation**: Knows all 8 pages and guides users to the right feature
- **Suggested prompts**: One-click starters for portfolio analysis, market outlook, risk explanations, and trade reviews
- **Full markdown rendering**: Tables, bold, lists, code blocks in responses
- **Multi-provider**: Groq primary (14,400 free req/day), Gemini fallback

### Market Regime Detection (`ai_signals.py`)

SMA 20/50 crossover classification for any stock:

- **Bullish**: SMA20 > SMA50 with price confirmation above both averages
- **Bearish**: SMA20 < SMA50 with price confirmation below both averages
- **Neutral**: SMAs converging, no clear directional trend

Each signal includes **confidence level** (High/Medium/Low) and **signal strength** (0-100%).

### DCF Valuation Analysis (`ai_signals.py`)

Aggregates analyst consensus from Financial Modeling Prep:

- **Undervalued**: DCF intrinsic value > 15% above current price
- **Overvalued**: DCF intrinsic value > 15% below current price
- **Fair Value**: DCF within 15% of current price

Includes margin of safety percentage and revenue growth estimates.

### Technical Indicators (`ai_signals.py`)

Toggleable overlays on the Research chart:

| Indicator | Description |
|---|---|
| **RSI (14)** | Relative Strength Index — overbought (>70) / oversold (<30) levels |
| **MACD (12/26/9)** | Moving Average Convergence Divergence — trend momentum with signal line and histogram |
| **Bollinger Bands (20, 2σ)** | Volatility bands around 20-period SMA |

### AI Trading Coach (`ai_signals.py`)

Rule-based coaching engine that analyzes trading behavior:

- Concentration risk warnings (single-stock portfolios)
- Cash allocation advice (over-allocated or under-reserved)
- Overtrading detection (many trades in few days)
- Losing streak alerts with entry criteria review suggestions

### Achievement System (`ai_signals.py`)

7 gamification badges with unlock criteria, displayed as cards on the Portfolio page. Toast notifications fire when new badges are earned.

| Badge | Criteria |
|---|---|
| First Trade | Complete 1 trade |
| Diversifier | Hold 5+ different stocks |
| Options Trader | Place 1 options trade |
| Risk Manager | Use a stop-loss order |
| Consistent | Trade on 5+ different days |
| Six-Figure Club | Portfolio value hits $110K+ |
| Watchful | Track 5+ stocks on watchlist |

## Risk Metrics (`risk_metrics.py`)

All calculations use pure numpy/pandas with no Streamlit dependency:

| Metric | Method | Description |
|---|---|---|
| **Sharpe Ratio** | `compute_sharpe_ratio()` | Annualized excess return per unit of risk (rf: 5%) |
| **Max Drawdown** | `compute_max_drawdown()` | Largest peak-to-trough decline |
| **Value at Risk** | `compute_var_historical()` | 95th percentile daily loss |
| **Monte Carlo** | `compute_monte_carlo()` | 500-simulation 1-year projection with percentile bands (p5–p95) |
| **Correlation Matrix** | `compute_correlation_matrix()` | Pairwise correlation heatmap for diversification analysis |

## Data Sources

Dual data-source architecture with automatic failover:

| Data Type | Primary Source | Fallback |
|---|---|---|
| Stock prices | Financial Modeling Prep (stable API) | Yahoo Finance |
| Historical OHLCV | Financial Modeling Prep (stable API) | Yahoo Finance |
| Stock info/fundamentals | Yahoo Finance | — |
| Options chains | Yahoo Finance | — |
| Analyst estimates & DCF | Financial Modeling Prep | — |
| AI Chat | Groq (Llama 3.3 70B) | Google Gemini |

All market data cached with `@st.cache_data` (60s prices, 300s history/analyst).

## Features

### Portfolio Management
- Virtual cash account ($100,000 starting balance)
- Buy/sell/short sell stocks with live prices
- Equity curve tracking with starting balance reference line
- Sector allocation pie chart (auto-detected via yfinance)
- Quick sell buttons for rapid position exits
- Options positions with mid-price valuation
- Achievement badges with unlock progress and toast notifications
- Transaction history, trade journal, CSV export

### Trading
- Market, Limit, and Stop-Loss order types
- **Short selling** and cover short positions
- Order confirmation dialogs with commission breakdown
- Trade validation (cash/shares checks)
- Configurable commission settings

### Options Trading
- Real-time options chain data (calls and puts)
- Multiple expiration dates
- Buy to Open / Sell to Close
- **P/L payoff diagrams** — visual breakeven, max profit, max loss at expiration
- Per-contract commission tracking

### Research & AI Signals
- Multi-timeframe charts: 1W, 1M, 3M, 6M, 1Y, 5Y
- Interactive price charts (Line and Candlestick)
- **Technical indicators**: RSI, MACD, Bollinger Bands (toggleable)
- **Volume subplot** with color-coded bars
- Market regime detection (Bullish/Bearish/Neutral)
- DCF valuation signal (Undervalued/Overvalued/Fair Value)
- Links to Yahoo Finance, WSJ, and Google News

### Analytics & Risk
- Portfolio vs S&P 500 benchmark overlay chart
- Sharpe ratio, max drawdown, VaR metric cards
- **Monte Carlo fan chart** — 1-year projection with 5th–95th percentile bands
- **Correlation heatmap** for diversification analysis
- **Trade journal analytics** — win rate, avg win/loss, profit factor, best/worst trade
- **AI Trading Coach** — rule-based tips based on trading behavior
- Position weight analysis

### AI Assistant
- Full-page conversational AI with chat bubbles and avatars
- 6 suggested prompt buttons for common queries
- Portfolio-aware with real-time price injection
- Markdown-formatted responses
- Conversation history with clear chat option

### Watchlist
- Add/remove S&P 500 stocks
- Real-time prices with daily change
- **Price alerts** — set target prices with notification status
- **Quick buy buttons** for one-click order preview

### Settings & Accessibility
- Dark/Light theme toggle
- **Colorblind mode** (blue/orange instead of green/red)
- Commission configuration (stocks and options)
- Account reset with custom starting balance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/naslam123/stock-portfolio-dashboard.git
cd stock-portfolio-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add API keys to `.streamlit/secrets.toml`:
```toml
# Required for DCF valuation (free: 250 req/day)
FMP_API_KEY = "your_key"     # https://financialmodelingprep.com/developer

# Required for AI Assistant (free: 14,400 req/day)
GROQ_API_KEY = "your_key"   # https://console.groq.com

# Optional fallback for AI Assistant
GEMINI_API_KEY = "your_key"  # https://aistudio.google.com/apikey
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open `http://localhost:8501`

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

## Tech Stack

- **Frontend**: Streamlit
- **AI Chat**: Groq (Llama 3.3 70B) / Google Gemini
- **Primary Data**: Financial Modeling Prep API (stable endpoints)
- **Fallback Data**: Yahoo Finance (via yfinance)
- **Charts**: Plotly (candlestick, line, bar, pie, heatmap, fan chart)
- **Risk Engine**: NumPy + Pandas
- **Storage**: JSON file persistence

## License

MIT License

## Author

MGMT 590 - Mastering AI For Finance | Purdue University
