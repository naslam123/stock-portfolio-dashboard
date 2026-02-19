import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from sp500_tickers import SP500
from config import TICKER_OPTIONS, get_theme_colors, render_css
from data_manager import default_data, load_data, save_data
from market_data import (
    get_price, get_info, get_history,
    get_option_dates, get_option_chain_data, get_analyst_data,
)
from portfolio import (
    get_holdings, get_options_holdings, get_option_current_price,
    options_portfolio_value, portfolio_value,
)
from risk_metrics import (
    compute_sharpe_ratio, compute_max_drawdown,
    compute_var_historical, build_portfolio_daily_returns,
)
from ai_signals import detect_market_regime, analyze_valuation

st.set_page_config(page_title="Trading Simulator", page_icon="📈", layout="wide")

# Load data ONCE
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Theme
theme = st.session_state.data.get("theme", "dark")
dark = theme == "dark"
colors = get_theme_colors(dark)
BG, BG2 = colors["BG"], colors["BG2"]
TEXT, TEXT2 = colors["TEXT"], colors["TEXT2"]
BORDER = colors["BORDER"]
GREEN, RED, BLUE, YELLOW = colors["GREEN"], colors["RED"], colors["BLUE"], colors["YELLOW"]

st.markdown(render_css(colors), unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"<h2 style='text-align:center; color:{TEXT};'>Trading Simulator</h2>", unsafe_allow_html=True)
    st.caption("MGMT 590 | Purdue University")

    st.divider()

    new_dark = st.toggle("Dark Mode", value=dark, key="dark_toggle")

    if new_dark and st.session_state.data["theme"] != "dark":
        st.session_state.data["theme"] = "dark"
        save_data(st.session_state.data)
        st.rerun()
    elif not new_dark and st.session_state.data["theme"] != "light":
        st.session_state.data["theme"] = "light"
        save_data(st.session_state.data)
        st.rerun()

    st.divider()

    if st.button("🔄 Refresh Prices", use_container_width=True, key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

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

        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.markdown(f"""
            <div style="background:{BG}; border:1px solid {BORDER}; border-radius:12px; padding:24px; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
                <div style="color:{TEXT}; font-size:16px; font-weight:bold; margin-bottom:16px; text-align:center;">Confirm Order</div>
                <div style="color:{TEXT}; font-size:14px; margin-bottom:12px; text-align:center;">
                    <strong>{t['action'].upper()}</strong> {t['shares']:.2f} × <strong>{t['ticker']}</strong>
                </div>
                <div style="color:{TEXT2}; font-size:13px; text-align:center;">@ ${t['price']:.2f}</div>
                <hr style="border-color:{BORDER}; margin:16px 0;">
                <div style="display:flex; justify-content:space-between; color:{TEXT2}; font-size:13px; margin-bottom:8px;">
                    <span>Commission</span><span>${commission:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; color:{TEXT}; font-size:15px; font-weight:bold;">
                    <span>Total</span><span>${total_cost:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✓ Confirm", use_container_width=True, key="confirm_trade"):
                    execute_trade(t)
                    st.session_state.show_confirm = False
                    st.session_state.pending_trade = None
                    st.rerun()
            with btn_col2:
                if st.button("✕ Cancel", use_container_width=True, key="cancel_trade"):
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

    save_data(st.session_state.data)

# ==================== PORTFOLIO ====================
if page == "Portfolio":
    st.header("Portfolio")

    if show_trade_dialog():
        st.stop()

    holdings = get_holdings()
    options_holdings = get_options_holdings()

    if not holdings and not options_holdings:
        st.info("No holdings. Go to Trade or Options to place your first order.")
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

        opts_val = options_portfolio_value()
        total_account = st.session_state.data["cash"] + total_val + opts_val
        total_pl = total_account - st.session_state.data["starting_balance"]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Account", f"${total_account:,.0f}")
        c2.metric("Cash", f"${st.session_state.data['cash']:,.0f}")
        c3.metric("Stocks", f"${total_val:,.0f}")
        c4.metric("Options", f"${opts_val:,.0f}")
        c5.metric("Total P/L", f"${total_pl:,.0f}", f"{total_pl/st.session_state.data['starting_balance']*100:+.1f}%")
        c6.metric("Positions", f"{len(holdings)} / {len(options_holdings)}")

        st.divider()

        if holdings:
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
                st.subheader("Stock Holdings")
                st.dataframe(df_show, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Allocation")
                labels = list(df["Ticker"]) + ["Cash"]
                values = list(df["Value"]) + [st.session_state.data["cash"]]

                if opts_val > 0:
                    labels.append("Options")
                    values.append(opts_val)

                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=0.4,
                    marker=dict(
                        colors=["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"][:len(labels)],
                        line=dict(color='#ffffff', width=2)
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=12, color="white"),
                    pull=[0.05] * len(labels),
                    rotation=45,
                    direction="clockwise"
                )])
                fig.update_layout(
                    height=320,
                    margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT2),
                    annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=14, font_color=TEXT, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)

        if options_holdings:
            st.subheader("Options Positions")
            opts_data = []
            for key, opt in options_holdings.items():
                current_price = get_option_current_price(opt["ticker"], opt["expiration"], opt["strike"], opt["type"])
                current_value = current_price * 100 * opt["contracts"]
                avg_cost = opt["total_cost"] / opt["contracts"] / 100 if opt["contracts"] > 0 else 0
                pl = current_value - opt["total_cost"]
                pl_pct = (pl / opt["total_cost"] * 100) if opt["total_cost"] > 0 else 0

                opts_data.append({
                    "Ticker": opt["ticker"],
                    "Type": opt["type"].capitalize(),
                    "Strike": f"${opt['strike']:.2f}",
                    "Expiration": opt["expiration"],
                    "Contracts": opt["contracts"],
                    "Avg Cost": f"${avg_cost:.2f}",
                    "Current": f"${current_price:.2f}",
                    "Value": f"${current_value:,.2f}",
                    "P/L": f"${pl:+,.2f}",
                    "P/L %": f"{pl_pct:+.1f}%"
                })

            st.dataframe(pd.DataFrame(opts_data), use_container_width=True, hide_index=True)

        if holdings:
            st.subheader("Performance")
            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                blues = ["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
                fig1 = go.Figure(data=[go.Bar(
                    x=df["Ticker"], y=df["P/L %"],
                    marker=dict(
                        color=blues[:len(df)],
                        line=dict(width=1, color='rgba(255,255,255,0.3)')
                    ),
                    text=df["P/L %"].apply(lambda x: f"{x:+.1f}%"),
                    textposition="outside",
                    textfont=dict(size=11, color=TEXT)
                )])
                fig1.update_layout(
                    title=dict(text="Return %", font=dict(size=14, color=TEXT)),
                    height=280,
                    margin=dict(t=40,b=40,l=40,r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True, zerolinecolor=TEXT2, zerolinewidth=2),
                    xaxis=dict(showgrid=False),
                    font=dict(color=TEXT2),
                    bargap=0.3
                )
                st.plotly_chart(fig1, use_container_width=True)

            with perf_col2:
                st.markdown(f"""
                <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:20px; height:260px;">
                    <div style="color:{TEXT}; font-size:14px; font-weight:bold; margin-bottom:16px;">Portfolio Summary</div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="color:{TEXT2};">Total Invested</span>
                        <span style="color:{TEXT};">${total_cost:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="color:{TEXT2};">Current Value</span>
                        <span style="color:{TEXT};">${total_val:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="color:{TEXT2};">Total P/L</span>
                        <span style="color:{GREEN if (total_val - total_cost) >= 0 else RED};">${total_val - total_cost:+,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="color:{TEXT2};">Return</span>
                        <span style="color:{GREEN if (total_val - total_cost) >= 0 else RED};">{((total_val - total_cost) / total_cost * 100) if total_cost > 0 else 0:+.2f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{TEXT2};">Positions</span>
                        <span style="color:{TEXT};">{len(holdings)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.download_button("Export CSV", df_show.to_csv(index=False), "portfolio.csv", use_container_width=True)

        with st.expander("Stock Transaction History"):
            if st.session_state.data["portfolio"]:
                st.dataframe(pd.DataFrame(st.session_state.data["portfolio"]), use_container_width=True, hide_index=True)
            else:
                st.info("No stock transactions yet")

        with st.expander("Options Transaction History"):
            if st.session_state.data["options"]:
                st.dataframe(pd.DataFrame(st.session_state.data["options"]), use_container_width=True, hide_index=True)
            else:
                st.info("No options transactions yet")

# ==================== TRADE ====================
elif page == "Trade":
    st.header("Trade")

    if show_trade_dialog():
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
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

# ==================== OPTIONS ====================
elif page == "Options":
    st.header("Options Trading")

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

        dates = get_option_dates(ticker)

        if dates:
            col1, col2 = st.columns(2)
            with col1:
                exp = st.selectbox("Expiration", dates)
            with col2:
                opt_type = st.radio("Type", ["Call", "Put"], horizontal=True)

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
                        save_data(st.session_state.data)
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
        st.write("")
        if sel and st.button("Add", type="primary", use_container_width=True):
            tk = sel.split(" - ")[0]
            if tk not in st.session_state.data["watchlist"]:
                st.session_state.data["watchlist"].append(tk)
                save_data(st.session_state.data)
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
            st.write("")
            if rem != "--" and st.button("Remove", use_container_width=True):
                st.session_state.data["watchlist"].remove(rem)
                save_data(st.session_state.data)
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

            # Latest News
            st.divider()
            st.subheader("Latest News")

            company_name = info.get('shortName', ticker).replace(' ', '+')
            st.markdown(f"""
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                <a href="https://finance.yahoo.com/quote/{ticker}/news" target="_blank" style="color:{BLUE}; font-size:14px; text-decoration:none; font-weight:500;">
                    📰 {ticker} News on Yahoo Finance
                </a>
                <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">Latest news and updates</div>
            </div>
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                <a href="https://www.wsj.com/search?query={ticker}" target="_blank" style="color:{BLUE}; font-size:14px; text-decoration:none; font-weight:500;">
                    📰 {ticker} on Wall Street Journal
                </a>
                <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">WSJ coverage and analysis</div>
            </div>
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                <a href="https://www.google.com/search?q={ticker}+{company_name}+stock+news&tbm=nws" target="_blank" style="color:{BLUE}; font-size:14px; text-decoration:none; font-weight:500;">
                    📰 {ticker} on Google News
                </a>
                <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">All recent news articles</div>
            </div>
            """, unsafe_allow_html=True)

            # AI Signals Section
            st.divider()
            st.subheader("AI Signals")

            sig_col1, sig_col2 = st.columns(2)

            with sig_col1:
                st.markdown(f"<div style='color:{TEXT}; font-weight:bold; margin-bottom:8px;'>Market Regime</div>", unsafe_allow_html=True)
                regime = detect_market_regime(df)
                regime_color = GREEN if regime["regime"] == "Bullish" else RED if regime["regime"] == "Bearish" else YELLOW
                st.markdown(f"""
                <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:16px;">
                    <div style="font-size:1.4rem; font-weight:bold; color:{regime_color};">{regime['regime']}</div>
                    <div style="color:{TEXT2}; margin-top:4px;">Confidence: {regime['confidence']} | Strength: {regime['signal_strength']:.0%}</div>
                    <div style="color:{TEXT2}; font-size:12px; margin-top:8px;">{regime['description']}</div>
                </div>
                """, unsafe_allow_html=True)

            with sig_col2:
                st.markdown(f"<div style='color:{TEXT}; font-weight:bold; margin-bottom:8px;'>DCF Valuation</div>", unsafe_allow_html=True)
                analyst_data = get_analyst_data(ticker)
                if analyst_data:
                    valuation = analyze_valuation(analyst_data)
                    val_color = GREEN if valuation["signal"] == "Undervalued" else RED if valuation["signal"] == "Overvalued" else YELLOW
                    st.markdown(f"""
                    <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:16px;">
                        <div style="font-size:1.4rem; font-weight:bold; color:{val_color};">{valuation['signal']}</div>
                        <div style="color:{TEXT2}; margin-top:4px;">DCF: ${valuation['dcf']:,.2f} | Margin: {valuation['margin_of_safety']:+.1f}%</div>
                        <div style="color:{TEXT2}; font-size:12px; margin-top:8px;">{valuation['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Add FMP_API_KEY in .streamlit/secrets.toml to enable DCF valuation.")

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

        # Risk Metrics Section
        st.subheader("Risk Metrics")

        portfolio_returns = build_portfolio_daily_returns(
            holdings, get_history, st.session_state.data["starting_balance"]
        )

        if not portfolio_returns.empty:
            sharpe = compute_sharpe_ratio(portfolio_returns)
            cumulative = (1 + portfolio_returns).cumprod()
            max_dd = compute_max_drawdown(cumulative)
            var_95 = compute_var_historical(portfolio_returns, 0.95, total)

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            rc2.metric("Max Drawdown", f"{max_dd:.1%}")
            rc3.metric("VaR (95%)", f"${var_95:,.0f}")
        else:
            st.info("Need at least 50 days of history for risk metrics.")

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
        save_data(st.session_state.data)

    if comm_enabled:
        c1, c2 = st.columns(2)
        with c1:
            stock_comm = st.number_input("Stock Commission ($)", value=st.session_state.data["commission_stock"], step=0.01)
            if stock_comm != st.session_state.data["commission_stock"]:
                st.session_state.data["commission_stock"] = stock_comm
                save_data(st.session_state.data)
        with c2:
            opt_comm = st.number_input("Option Commission ($/contract)", value=st.session_state.data["commission_option"], step=0.01)
            if opt_comm != st.session_state.data["commission_option"]:
                st.session_state.data["commission_option"] = opt_comm
                save_data(st.session_state.data)

    st.divider()

    st.subheader("Reset Account")
    new_bal = st.number_input("New Starting Balance", min_value=1000.0, value=100000.0, step=10000.0)

    if st.button("Reset Account", type="primary"):
        st.session_state.data = default_data()
        st.session_state.data["starting_balance"] = new_bal
        st.session_state.data["cash"] = new_bal
        save_data(st.session_state.data)
        st.success(f"Account reset with ${new_bal:,.0f}")
        st.rerun()

# Footer
st.divider()
st.caption("Financial Modeling Prep + Yahoo Finance (fallback) | MGMT 590 | Purdue")
