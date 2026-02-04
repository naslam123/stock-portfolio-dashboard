# Stock Portfolio Trading Simulator

A real-time stock portfolio dashboard and trading simulator built with Streamlit for MGMT 590 - Mastering AI For Finance at Purdue University.

## Features

### Portfolio Management
- Virtual cash account ($100,000 starting balance)
- Buy/sell stocks with live Yahoo Finance prices (15-min delayed)
- Track holdings, P/L, and portfolio allocation
- Transaction history and trade journal

### Trading
- Market, Limit, and Stop-Loss order types
- Order confirmation dialogs
- Trade validation (checks cash/shares before execution)
- Configurable commission settings

### Options Trading
- Real-time options chain data
- Call and Put options
- Multiple expiration dates
- Buy to Open / Sell to Close

### Research & Analytics
- Stock research with key metrics (P/E, Market Cap, 52W High/Low)
- Interactive price charts (Line and Candlestick)
- SMA 20/50 technical indicators
- Portfolio performance vs S&P 500 benchmark
- Position weight analysis

### Additional Features
- Watchlist management
- Dark/Light theme toggle
- Auto-suggest stock search
- CSV export for portfolio data

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

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Requirements

- Python 3.8+
- streamlit >= 1.31.0
- yfinance >= 0.2.36
- pandas >= 2.0.0
- plotly >= 5.18.0

## Tech Stack

- **Frontend**: Streamlit
- **Data**: Yahoo Finance API (via yfinance)
- **Charts**: Plotly
- **Storage**: JSON file persistence

## Screenshots

The dashboard features a professional Bloomberg/TradingView-inspired dark theme with:
- Real-time price display with color-coded changes
- Interactive allocation pie charts
- Performance bar charts
- Candlestick and line charts with technical indicators

## License

MIT License

## Author

MGMT 590 - Mastering AI For Finance | Purdue University
