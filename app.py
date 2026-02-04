import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os

st.set_page_config(page_title="Trading Simulator", page_icon="📈", layout="wide")

# S&P 500 tickers
SP500 = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "BRK.B": "Berkshire Hathaway", "UNH": "UnitedHealth Group",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase", "V": "Visa Inc.",
    "PG": "Procter & Gamble", "XOM": "Exxon Mobil", "MA": "Mastercard Inc.",
    "HD": "Home Depot", "CVX": "Chevron Corp", "MRK": "Merck & Co.",
    "ABBV": "AbbVie Inc.", "LLY": "Eli Lilly", "PEP": "PepsiCo Inc.",
    "KO": "Coca-Cola", "COST": "Costco", "AVGO": "Broadcom Inc.",
    "WMT": "Walmart Inc.", "MCD": "McDonald's", "CSCO": "Cisco Systems",
    "TMO": "Thermo Fisher", "ACN": "Accenture", "ABT": "Abbott Labs",
    "ADBE": "Adobe Inc.", "NKE": "Nike Inc.", "ORCL": "Oracle Corp",
    "CRM": "Salesforce", "AMD": "AMD", "INTC": "Intel Corp",
    "DIS": "Walt Disney", "VZ": "Verizon", "CMCSA": "Comcast",
    "TXN": "Texas Instruments", "PM": "Philip Morris", "WFC": "Wells Fargo",
    "RTX": "RTX Corp", "QCOM": "Qualcomm", "HON": "Honeywell",
    "UNP": "Union Pacific", "IBM": "IBM", "CAT": "Caterpillar",
    "BA": "Boeing", "GE": "General Electric", "AMGN": "Amgen",
    "LOW": "Lowe's", "DE": "Deere & Co", "INTU": "Intuit",
    "GS": "Goldman Sachs", "AXP": "American Express", "BKNG": "Booking Holdings",
    "BLK": "BlackRock", "SBUX": "Starbucks", "GILD": "Gilead Sciences",
    "PYPL": "PayPal", "MU": "Micron", "SQ": "Block Inc.",
}

# Pre-built list for searchable dropdowns
TICKER_OPTIONS = [""] + [f"{t} - {n}" for t, n in SP500.items()]

# Data file
DATA_FILE = "trading_data.json"

def default_data():
    return {
        "cash": 100000.0,
        "starting_balance": 100000.0,
        "portfolio": [],
        "watchlist": [],
        "options": [],
        "pending_orders": [],
        "journal": [],
        "theme": "dark",
        "commission_enabled": True,
        "commission_stock": 0.0,
        "commission_option": 0.65,
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for key, val in default_data().items():
                    if key not in data:
                        data[key] = val
                return data
        except:
            return default_data()
    return default_data()

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump(st.session_state.data, f, indent=2, default=str)

# Load data ONCE
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Theme from saved data
theme = st.session_state.data.get("theme", "dark")
dark = theme == "dark"

# Simple color scheme
if dark:
    BG = "#0d1117"
    BG2 = "#161b22"
    TEXT = "#ffffff"
    TEXT2 = "#8b949e"
    BORDER = "#30363d"
    GREEN = "#3fb950"
    RED = "#f85149"
    BLUE = "#58a6ff"
    YELLOW = "#d29922"
else:
    BG = "#ffffff"
    BG2 = "#f6f8fa"
    TEXT = "#24292f"
    TEXT2 = "#57606a"
    BORDER = "#d0d7de"
    GREEN = "#1a7f37"
    RED = "#cf222e"
    BLUE = "#0969da"
    YELLOW = "#9a6700"

# CSS
st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: {BG} !important;
}}
section[data-testid="stSidebar"] > div {{
    background: {BG2} !important;
}}
h1, h2, h3 {{
    color: {TEXT} !important;
}}
p, span, label {{
    color: {TEXT} !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT2} !important;
}}
.stTextInput input, .stNumberInput input {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
}}
.stSelectbox > div > div {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
}}
.stButton > button {{
    background: {BLUE} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}}
hr {{
    border-color: {BORDER} !important;
}}
.stDataFrame {{
    border: 1px solid {BORDER} !important;
}}
.stCheckbox > label {{
    color: {TEXT} !important;
}}
.stRadio > div > label {{
    color: {TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)

# Cache functions - NOT caching option chain to avoid serialization issues
@st.cache_data(ttl=60)
def get_price(ticker):
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev = info.get("previousClose", price)
        return price, price - prev, ((price - prev) / prev * 100) if prev else 0
    except:
        return 0, 0, 0

@st.cache_data(ttl=60)
def get_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

@st.cache_data(ttl=300)
def get_history(ticker, period="6mo"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except:
        return pd.DataFrame()

def get_option_dates(ticker):
    """Get option expiration dates - not cached"""
    try:
        t = yf.Ticker(ticker)
        return list(t.options) if t.options else []
    except:
        return []

def get_option_chain_data(ticker, exp_date, opt_type):
    """Get option chain as DataFrame - not cached to avoid serialization issues"""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(exp_date)
        df = chain.calls if opt_type == "Call" else chain.puts
        return df
    except:
        return pd.DataFrame()

def get_holdings():
    holdings = {}
    for t in st.session_state.data["portfolio"]:
        tk = t["ticker"]
        if tk not in holdings:
            holdings[tk] = {"shares": 0, "cost": 0}
        if t["type"] == "buy":
            holdings[tk]["shares"] += t["shares"]
            holdings[tk]["cost"] += t["shares"] * t["price"]
        else:
            holdings[tk]["shares"] -= t["shares"]
    return {k: v for k, v in holdings.items() if v["shares"] > 0.001}

def portfolio_value():
    h = get_holdings()
    return st.session_state.data["cash"] + sum(get_price(t)[0] * v["shares"] for t, v in h.items())

# Sidebar
with st.sidebar:
    st.markdown(f"<h2 style='text-align:center; color:{TEXT};'>Trading Simulator</h2>", unsafe_allow_html=True)
    st.caption("MGMT 590 | Purdue University")

    st.divider()

    # Dark mode toggle - properly aligned
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding:8px 0;">
        <span style="color:{TEXT}; font-size:14px;">Dark Mode</span>
    </div>
    """, unsafe_allow_html=True)

    new_dark = st.checkbox("Enable", value=dark, key="dark_toggle", label_visibility="collapsed")

    # Only update if changed AND different from current
    if new_dark and st.session_state.data["theme"] != "dark":
        st.session_state.data["theme"] = "dark"
        save_data()
        st.rerun()
    elif not new_dark and st.session_state.data["theme"] != "light":
        st.session_state.data["theme"] = "light"
        save_data()
        st.rerun()

    st.divider()

    # Account summary
    total = portfolio_value()
    cash = st.session_state.data["cash"]
    start = st.session_state.data["starting_balance"]
    pl = total - start

    st.metric("Account Value", f"${total:,.2f}", f"{pl/start*100:+.2f}%")

    col1, col2 = st.columns(2)
    col1.metric("Cash", f"${cash:,.0f}")
    col2.metric("Invested", f"${total-cash:,.0f}")

    st.divider()

    page = st.radio("Navigation", ["Portfolio", "Trade", "Options", "Watchlist", "Research", "Analytics", "Settings"], label_visibility="collapsed")

# Dialog state
if "show_confirm" not in st.session_state:
    st.session_state.show_confirm = False
if "pending_trade" not in st.session_state:
    st.session_state.pending_trade = None

def show_trade_dialog():
    if st.session_state.show_confirm and st.session_state.pending_trade:
        t = st.session_state.pending_trade
        commission = st.session_state.data["commission_stock"] if st.session_state.data["commission_enabled"] else 0
        total_cost = t["shares"] * t["price"] + commission

        st.markdown(f"""
        <div style="background:{BG2}; border:2px solid {YELLOW}; border-radius:12px; padding:24px; margin:20px 0;">
            <h3 style="color:{YELLOW}; margin-bottom:16px;">⚠️ Confirm Order</h3>
            <p style="color:{TEXT}; font-size:16px;"><strong>{t['action'].upper()}</strong> {t['shares']:.2f} shares of <strong>{t['ticker']}</strong></p>
            <p style="color:{TEXT};">Price: ${t['price']:.2f}</p>
            <p style="color:{TEXT};">Commission: ${commission:.2f}</p>
            <p style="color:{TEXT}; font-size:18px; margin-top:12px;"><strong>Total: ${total_cost:,.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ CONFIRM", type="primary", use_container_width=True):
                execute_trade(t)
                st.session_state.show_confirm = False
                st.session_state.pending_trade = None
                st.rerun()
        with col2:
            if st.button("✕ CANCEL", use_container_width=True):
                st.session_state.show_confirm = False
                st.session_state.pending_trade = None
                st.rerun()
        return True
    return False

def execute_trade(t):
    commission = st.session_state.data["commission_stock"] if st.session_state.data["commission_enabled"] else 0
    total = t["shares"] * t["price"]

    if t["action"] == "buy":
        st.session_state.data["cash"] -= (total + commission)
    else:
        st.session_state.data["cash"] += (total - commission)

    st.session_state.data["portfolio"].append({
        "ticker": t["ticker"],
        "type": t["action"],
        "order_type": t.get("order_type", "market"),
        "shares": t["shares"],
        "price": t["price"],
        "commission": commission,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    st.session_state.data["journal"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": t["ticker"],
        "action": t["action"],
        "shares": t["shares"],
        "price": t["price"],
        "notes": t.get("notes", "")
    })

    save_data()

# ==================== PORTFOLIO ====================
if page == "Portfolio":
    st.header("Portfolio")

    if show_trade_dialog():
        st.stop()

    holdings = get_holdings()

    if not holdings:
        st.info("No holdings. Go to Trade to place your first order.")
    else:
        data = []
        total_val = 0
        total_cost = 0

        for tk, h in holdings.items():
            price, chg, pct = get_price(tk)
            val = h["shares"] * price
            cost = h["cost"]
            pl = val - cost
            pl_pct = (pl / cost * 100) if cost > 0 else 0

            data.append({
                "Ticker": tk,
                "Shares": h["shares"],
                "Avg Cost": cost / h["shares"] if h["shares"] > 0 else 0,
                "Price": price,
                "Value": val,
                "P/L": pl,
                "P/L %": pl_pct
            })
            total_val += val
            total_cost += cost

        total_account = st.session_state.data["cash"] + total_val
        total_pl = total_account - st.session_state.data["starting_balance"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Account", f"${total_account:,.0f}")
        c2.metric("Cash", f"${st.session_state.data['cash']:,.0f}")
        c3.metric("Holdings", f"${total_val:,.0f}")
        c4.metric("Total P/L", f"${total_pl:,.0f}", f"{total_pl/st.session_state.data['starting_balance']*100:+.1f}%")
        c5.metric("Positions", len(holdings))

        st.divider()

        df = pd.DataFrame(data)
        df_show = df.copy()
        df_show["Shares"] = df_show["Shares"].apply(lambda x: f"{x:.2f}")
        df_show["Avg Cost"] = df_show["Avg Cost"].apply(lambda x: f"${x:.2f}")
        df_show["Price"] = df_show["Price"].apply(lambda x: f"${x:.2f}")
        df_show["Value"] = df_show["Value"].apply(lambda x: f"${x:,.0f}")
        df_show["P/L"] = df_show["P/L"].apply(lambda x: f"${x:+,.0f}")
        df_show["P/L %"] = df_show["P/L %"].apply(lambda x: f"{x:+.1f}%")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Holdings")
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Allocation")
            labels = list(df["Ticker"]) + ["Cash"]
            values = list(df["Value"]) + [st.session_state.data["cash"]]

            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.5,
                marker=dict(colors=[BLUE, GREEN, "#a855f7", "#06b6d4", "#f59e0b", "#6b7280"][:len(labels)]),
                textinfo="label+percent"
            )])
            fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance")
        colors = [GREEN if x >= 0 else RED for x in df["P/L %"]]
        fig = go.Figure(data=[go.Bar(x=df["Ticker"], y=df["P/L %"], marker_color=colors,
                                     text=df["P/L %"].apply(lambda x: f"{x:+.1f}%"), textposition="outside")])
        fig.update_layout(height=250, margin=dict(t=20,b=40,l=40,r=20),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER),
                         xaxis=dict(showgrid=False), font=dict(color=TEXT2))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button("Export CSV", df_show.to_csv(index=False), "portfolio.csv", use_container_width=True)

        with st.expander("Transaction History"):
            if st.session_state.data["portfolio"]:
                st.dataframe(pd.DataFrame(st.session_state.data["portfolio"]), use_container_width=True, hide_index=True)

# ==================== TRADE ====================
elif page == "Trade":
    st.header("Trade")

    if show_trade_dialog():
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Single searchable selectbox - type to filter
        sel = st.selectbox(
            "Search Stock",
            TICKER_OPTIONS,
            index=0,
            placeholder="Type to search ticker or company...",
            help="Start typing to filter stocks"
        )

        ticker = None
        if sel:
            ticker = sel.split(" - ")[0]

            price, chg, pct = get_price(ticker)
            chg_color = GREEN if chg >= 0 else RED

            st.markdown(f"""
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:20px; margin:16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:1.6rem; font-weight:bold; color:{TEXT};">{ticker}</div>
                        <div style="color:{TEXT2};">{SP500.get(ticker, '')}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.6rem; font-weight:bold; color:{TEXT};">${price:.2f}</div>
                        <div style="color:{chg_color};">{chg:+.2f} ({pct:+.2f}%)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if ticker:
            price, _, _ = get_price(ticker)

            st.subheader("Order")

            action = st.selectbox("Action", ["Buy", "Sell"])
            order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop-Loss"])
            shares = st.number_input("Shares", min_value=0.01, value=1.0, step=1.0)

            if order_type == "Market":
                exec_price = price
                st.text_input("Price", f"${price:.2f}", disabled=True)
            elif order_type == "Limit":
                exec_price = st.number_input("Limit Price", min_value=0.01, value=price, step=0.01)
            else:
                exec_price = st.number_input("Stop Price", min_value=0.01, value=price * 0.95, step=0.01)

            commission = st.session_state.data["commission_stock"] if st.session_state.data["commission_enabled"] else 0
            total = shares * exec_price + commission

            can_trade = True
            if action == "Buy" and total > st.session_state.data["cash"]:
                st.error(f"Insufficient funds. Need ${total:,.2f}, have ${st.session_state.data['cash']:,.2f}")
                can_trade = False
            elif action == "Sell":
                h = get_holdings()
                owned = h.get(ticker, {}).get("shares", 0)
                if shares > owned:
                    st.error(f"Insufficient shares. Own {owned:.2f}")
                    can_trade = False

            st.markdown(f"""
            <div style="background:{BG2}; border-radius:8px; padding:16px; margin:16px 0;">
                <div style="display:flex; justify-content:space-between; color:{TEXT};">
                    <span>Subtotal</span>
                    <span>${shares * exec_price:,.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; color:{TEXT2}; margin-top:8px;">
                    <span>Commission</span>
                    <span>${commission:.2f}</span>
                </div>
                <hr style="border-color:{BORDER}; margin:12px 0;">
                <div style="display:flex; justify-content:space-between; font-weight:bold; color:{TEXT};">
                    <span>Total</span>
                    <span style="color:{YELLOW};">${total:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            notes = st.text_input("Trade Notes (optional)", placeholder="Why this trade?")

            if can_trade:
                if st.button(f"Preview {action.upper()} Order", type="primary", use_container_width=True):
                    st.session_state.pending_trade = {
                        "ticker": ticker,
                        "action": action.lower(),
                        "order_type": order_type.lower(),
                        "shares": shares,
                        "price": exec_price,
                        "notes": notes
                    }
                    st.session_state.show_confirm = True
                    st.rerun()
        else:
            st.info("Search for a stock to trade")

# ==================== OPTIONS ====================
elif page == "Options":
    st.header("Options Trading")

    # Single searchable selectbox
    sel = st.selectbox(
        "Search Underlying",
        TICKER_OPTIONS,
        index=0,
        placeholder="Type to search ticker...",
        help="Start typing to filter stocks"
    )

    if sel:
        ticker = sel.split(" - ")[0]

        price, chg, pct = get_price(ticker)
        chg_color = GREEN if chg >= 0 else RED

        st.markdown(f"""
        <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:16px; margin:16px 0;">
            <span style="font-size:1.3rem; font-weight:bold; color:{TEXT};">{ticker}</span>
            <span style="color:{TEXT2}; margin-left:12px;">{SP500.get(ticker, '')}</span>
            <span style="float:right; font-size:1.2rem; font-weight:bold; color:{TEXT};">${price:.2f}</span>
            <span style="float:right; color:{chg_color}; margin-right:12px;">{pct:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Get option dates without caching
        dates = get_option_dates(ticker)

        if dates:
            col1, col2 = st.columns(2)
            with col1:
                exp = st.selectbox("Expiration", dates)
            with col2:
                opt_type = st.radio("Type", ["Call", "Put"], horizontal=True)

            # Get option chain without caching
            with st.spinner("Loading options..."):
                df = get_option_chain_data(ticker, exp, opt_type)

            if not df.empty:
                df_show = df[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].copy()
                df_show.columns = ["Strike", "Last", "Bid", "Ask", "Volume", "OI", "IV"]
                df_show["IV"] = (df_show["IV"] * 100).round(1).astype(str) + "%"

                st.subheader(f"{opt_type} Options Chain")
                st.dataframe(df_show.head(15), use_container_width=True, hide_index=True)

                st.subheader("Place Order")
                c1, c2, c3 = st.columns(3)
                with c1:
                    strike = st.selectbox("Strike", df["strike"].tolist()[:15])
                with c2:
                    contracts = st.number_input("Contracts", min_value=1, value=1)
                with c3:
                    opt_action = st.selectbox("Action", ["Buy to Open", "Sell to Close"])

                row = df[df["strike"] == strike].iloc[0]
                premium = row["ask"] if "Buy" in opt_action else row["bid"]
                commission = st.session_state.data["commission_option"] * contracts if st.session_state.data["commission_enabled"] else 0
                total = premium * 100 * contracts + commission

                st.markdown(f"""
                <div style="background:{BG2}; border-radius:8px; padding:16px; margin:16px 0;">
                    <div style="color:{TEXT};">Premium: ${premium:.2f} × 100 × {contracts} = ${premium * 100 * contracts:,.2f}</div>
                    <div style="color:{TEXT2}; margin-top:8px;">Commission: ${commission:.2f}</div>
                    <div style="font-weight:bold; margin-top:12px; color:{TEXT};">Total: <span style="color:{YELLOW};">${total:,.2f}</span></div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Execute Options Order", type="primary", use_container_width=True):
                    if total <= st.session_state.data["cash"]:
                        st.session_state.data["cash"] -= total
                        st.session_state.data["options"].append({
                            "ticker": ticker,
                            "type": opt_type.lower(),
                            "strike": strike,
                            "expiration": exp,
                            "contracts": contracts,
                            "premium": premium,
                            "action": opt_action,
                            "total": total,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_data()
                        st.success(f"Executed: {opt_action} {contracts}x {ticker} ${strike} {opt_type}")
                    else:
                        st.error("Insufficient funds")
            else:
                st.warning("No option data available")
        else:
            st.warning("No options available for this ticker")

# ==================== WATCHLIST ====================
elif page == "Watchlist":
    st.header("Watchlist")

    col1, col2 = st.columns([4, 1])
    with col1:
        sel = st.selectbox(
            "Add Stock",
            TICKER_OPTIONS,
            index=0,
            placeholder="Type to search...",
            label_visibility="collapsed"
        )
    with col2:
        st.write("")  # Spacing
        if sel and st.button("Add", type="primary", use_container_width=True):
            tk = sel.split(" - ")[0]
            if tk not in st.session_state.data["watchlist"]:
                st.session_state.data["watchlist"].append(tk)
                save_data()
                st.rerun()

    st.divider()

    wl = st.session_state.data["watchlist"]
    if not wl:
        st.info("Watchlist empty")
    else:
        data = []
        for t in wl:
            info = get_info(t)
            price, chg, pct = get_price(t)
            data.append({
                "Ticker": t,
                "Price": f"${price:.2f}",
                "Change": f"{pct:+.2f}%",
                "52W High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
                "52W Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            rem = st.selectbox("Remove", ["--"] + wl)
        with col2:
            st.write("")  # Spacing
            if rem != "--" and st.button("Remove", use_container_width=True):
                st.session_state.data["watchlist"].remove(rem)
                save_data()
                st.rerun()

# ==================== RESEARCH ====================
elif page == "Research":
    st.header("Research")

    sel = st.selectbox(
        "Search Stock",
        TICKER_OPTIONS,
        index=0,
        placeholder="Type to search ticker or company...",
        help="Start typing to filter stocks"
    )

    if sel:
        ticker = sel.split(" - ")[0]

        info = get_info(ticker)
        df = get_history(ticker)
        price, chg, pct = get_price(ticker)

        if info and not df.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"{ticker} - {info.get('shortName', '')}")
            with col2:
                st.metric("Price", f"${price:.2f}", f"{pct:+.2f}%")

            st.divider()

            c1, c2, c3, c4 = st.columns(4)
            cap = info.get('marketCap', 0)
            c1.metric("Market Cap", f"${cap/1e9:.1f}B" if cap else "N/A")
            c2.metric("P/E", f"{info.get('trailingPE', 0):.1f}" if info.get('trailingPE') else "N/A")
            c3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
            c4.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

            st.divider()

            chart = st.radio("Chart", ["Line", "Candle"], horizontal=True, label_visibility="collapsed")

            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()

            if chart == "Line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color=BLUE)))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color=YELLOW, dash="dash")))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color=GREEN, dash="dash")))
            else:
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                                    low=df["Low"], close=df["Close"],
                                                    increasing_line_color=GREEN, decreasing_line_color=RED)])

            fig.update_layout(height=400, margin=dict(t=20,b=40,l=40,r=20),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
                             yaxis=dict(showgrid=True, gridcolor=BORDER),
                             font=dict(color=TEXT2), hovermode="x unified",
                             legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

# ==================== ANALYTICS ====================
elif page == "Analytics":
    st.header("Portfolio Analytics")

    holdings = get_holdings()

    if not holdings:
        st.info("Add positions to see analytics")
    else:
        total = portfolio_value()
        start = st.session_state.data["starting_balance"]
        ret = (total - start) / start * 100

        spy = get_history("SPY")
        spy_ret = ((spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0] * 100) if not spy.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{ret:+.2f}%")
        c2.metric("vs S&P 500", f"{ret - spy_ret:+.2f}%", "Beat" if ret > spy_ret else "Trail")
        c3.metric("Positions", len(holdings))
        c4.metric("Cash %", f"{st.session_state.data['cash']/total*100:.1f}%")

        st.divider()

        st.subheader("Position Weights")
        data = []
        for tk, h in holdings.items():
            price, _, _ = get_price(tk)
            val = h["shares"] * price
            data.append({"Ticker": tk, "Value": val, "Weight": val / total * 100})

        df = pd.DataFrame(data)
        fig = go.Figure(data=[go.Bar(x=df["Ticker"], y=df["Weight"], marker_color=BLUE)])
        fig.update_layout(height=250, margin=dict(t=20,b=40,l=40,r=20),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(title="Weight %", showgrid=True, gridcolor=BORDER),
                         font=dict(color=TEXT2))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Trade Journal")

        if st.session_state.data["journal"]:
            st.dataframe(pd.DataFrame(st.session_state.data["journal"]), use_container_width=True, hide_index=True)
        else:
            st.info("No trades logged yet")

# ==================== SETTINGS ====================
elif page == "Settings":
    st.header("Settings")

    st.subheader("Account")

    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Balance", f"${st.session_state.data['starting_balance']:,.0f}")
    c2.metric("Current Cash", f"${st.session_state.data['cash']:,.0f}")
    c3.metric("Transactions", len(st.session_state.data['portfolio']))

    st.divider()

    st.subheader("Commission Settings")

    comm_enabled = st.checkbox("Enable Commissions", value=st.session_state.data["commission_enabled"])
    if comm_enabled != st.session_state.data["commission_enabled"]:
        st.session_state.data["commission_enabled"] = comm_enabled
        save_data()

    if comm_enabled:
        c1, c2 = st.columns(2)
        with c1:
            stock_comm = st.number_input("Stock Commission ($)", value=st.session_state.data["commission_stock"], step=0.01)
            if stock_comm != st.session_state.data["commission_stock"]:
                st.session_state.data["commission_stock"] = stock_comm
                save_data()
        with c2:
            opt_comm = st.number_input("Option Commission ($/contract)", value=st.session_state.data["commission_option"], step=0.01)
            if opt_comm != st.session_state.data["commission_option"]:
                st.session_state.data["commission_option"] = opt_comm
                save_data()

    st.divider()

    st.subheader("Reset Account")
    new_bal = st.number_input("New Starting Balance", min_value=1000.0, value=100000.0, step=10000.0)

    if st.button("Reset Account", type="primary"):
        st.session_state.data = default_data()
        st.session_state.data["starting_balance"] = new_bal
        st.session_state.data["cash"] = new_bal
        save_data()
        st.success(f"Account reset with ${new_bal:,.0f}")
        st.rerun()

# Footer
st.divider()
st.caption("Yahoo Finance (15-min delay) | MGMT 590 | Purdue")
