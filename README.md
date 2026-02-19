# Stock Portfolio Trading Simulator

A modular stock portfolio dashboard and trading simulator built with Streamlit for MGMT 590 - Mastering AI For Finance at Purdue University. Features AI-driven market signals, quantitative risk metrics, and dual data-source architecture.

## Architecture

The application follows a modular design, split across purpose-specific files:

```
stock-portfolio-dashboard/
├── app.py              # Streamlit orchestrator — page routing & UI rendering
├── config.py           # Constants, theme colors, CSS styling
├── data_manager.py     # JSON persistence layer (load/save/default)
├── market_data.py      # FMP primary + yfinance fallback, caching
├── portfolio.py        # Holdings, P/L, portfolio value calculations
├── risk_metrics.py     # Sharpe ratio, max drawdown, Value at Risk
├── ai_signals.py       # Market regime detection, DCF valuation analysis
├── sp500_tickers.py    # S&P 500 ticker dictionary and search utilities
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # FMP API key (gitignored)
└── .gitignore
```

## AI/ML Components

### Market Regime Detection (`ai_signals.py`)

Uses SMA 20/50 crossover classification to detect the current market regime for any stock:

- **Bullish**: SMA20 > SMA50 with price confirmation above both averages
- **Bearish**: SMA20 < SMA50 with price confirmation below both averages
- **Neutral**: SMAs converging, no clear directional trend

Each signal includes a **confidence level** (High/Medium/Low) based on SMA spread magnitude and price alignment, and a **signal strength** (0-100%) based on the percentage spread between the two moving averages.

### DCF Valuation Analysis (`ai_signals.py`)

Aggregates analyst consensus data from Financial Modeling Prep to produce a valuation signal:

- **Undervalued**: DCF intrinsic value > 15% above current price
- **Overvalued**: DCF intrinsic value > 15% below current price
- **Fair Value**: DCF within 15% of current price

Includes margin of safety percentage and analyst revenue growth estimates.

## Risk Metrics (`risk_metrics.py`)

All risk calculations use pure numpy/pandas with no Streamlit dependency:

| Metric | Method | Description |
|---|---|---|
| **Sharpe Ratio** | `compute_sharpe_ratio()` | Annualized excess return per unit of risk (risk-free rate: 5%) |
| **Max Drawdown** | `compute_max_drawdown()` | Largest peak-to-trough decline in portfolio value |
| **Value at Risk** | `compute_var_historical()` | 95th percentile historical VaR — maximum expected daily loss |

Portfolio daily returns are built using cost-weighted positions via `build_portfolio_daily_returns()`.

## Data Sources

The application uses a **dual data-source architecture** with automatic failover:

| Data Type | Primary Source | Fallback |
|---|---|---|
| Stock prices | Financial Modeling Prep (stable API) | Yahoo Finance |
| Historical OHLCV | Financial Modeling Prep (stable API) | Yahoo Finance |
| Stock info/fundamentals | Yahoo Finance | — |
| Options chains | Yahoo Finance | — |
| Analyst estimates & DCF | Financial Modeling Prep | — |

All market data is cached using `@st.cache_data` (60s for prices, 300s for history/analyst data). The app works fully without an FMP API key — it silently falls back to Yahoo Finance for prices and history, and shows an info message for AI signals.

## Features

### Portfolio Management
- Virtual cash account ($100,000 starting balance)
- Buy/sell stocks with live prices
- Track holdings, P/L, and portfolio allocation
- Options positions tracking with mid-price valuation
- Transaction history and trade journal
- CSV export

### Trading
- Market, Limit, and Stop-Loss order types
- Order confirmation dialogs
- Trade validation (checks cash/shares before execution)
- Configurable commission settings

### Options Trading
- Real-time options chain data (calls and puts)
- Multiple expiration dates
- Buy to Open / Sell to Close
- Per-contract commission tracking

### Research & AI Signals
- Stock research with key metrics (P/E, Market Cap, 52W High/Low)
- Interactive price charts (Line and Candlestick)
- SMA 20/50 technical indicators
- **Market regime detection** (Bullish/Bearish/Neutral with confidence)
- **DCF valuation signal** (Undervalued/Overvalued/Fair Value)
- Links to Yahoo Finance, WSJ, and Google News

### Analytics & Risk
- Portfolio performance vs S&P 500 benchmark
- **Sharpe ratio, max drawdown, and VaR metric cards**
- Position weight analysis
- Trade journal review

### Additional Features
- Watchlist management
- Dark/Light theme toggle
- Auto-suggest stock search across 100 S&P 500 tickers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/stock-portfolio-dashboard.git
cd stock-portfolio-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Add your free FMP API key for enhanced data:
```bash
# Get a free key at https://financialmodelingprep.com/developer
# Edit .streamlit/secrets.toml:
FMP_API_KEY = "your_key_here"
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## Requirements

- Python 3.10+
- streamlit >= 1.31.0
- yfinance >= 0.2.36
- pandas >= 2.0.0
- plotly >= 5.18.0
- numpy >= 1.26.0
- requests >= 2.31.0

## Tech Stack

- **Frontend**: Streamlit
- **Primary Data**: Financial Modeling Prep API (stable endpoints)
- **Fallback Data**: Yahoo Finance (via yfinance)
- **Charts**: Plotly (candlestick, line, bar, pie)
- **Risk Engine**: NumPy + Pandas
- **Storage**: JSON file persistence

## License

MIT License

## Author

MGMT 590 - Mastering AI For Finance | Purdue University
