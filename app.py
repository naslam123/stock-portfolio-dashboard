import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    options_portfolio_value, portfolio_value, get_trade_stats,
)
from risk_metrics import (
    compute_sharpe_ratio, compute_max_drawdown,
    compute_var_historical, build_portfolio_daily_returns,
    compute_monte_carlo, compute_correlation_matrix,
)
from ai_signals import (
    detect_market_regime, analyze_valuation,
    check_badges, generate_coaching_tips,
    compute_rsi, compute_macd, compute_bollinger,
    compute_option_payoff, BADGE_META, BADGE_DEFS,
)
from chatbot import build_portfolio_context, get_ai_response
from news_feed import get_stock_news, get_general_news, analyze_sentiment_batch, get_sentiment_color, format_time_ago
from rebalancer import compute_current_weights, generate_equal_weight_targets, generate_custom_targets, compute_rebalance_trades, estimate_rebalance_cost
from dashboard import get_market_overview, get_top_movers, get_triggered_alerts

st.set_page_config(page_title="Trading Simulator", page_icon="📈", layout="wide")

# Load data ONCE
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Theme (with colorblind support)
theme = st.session_state.data.get("theme", "dark")
dark = theme == "dark"
colorblind = st.session_state.data.get("colorblind", False)
colors = get_theme_colors(dark, colorblind)
BG, BG2 = colors["BG"], colors["BG2"]
TEXT, TEXT2 = colors["TEXT"], colors["TEXT2"]
BORDER = colors["BORDER"]
GREEN, RED, BLUE, YELLOW = colors["GREEN"], colors["RED"], colors["BLUE"], colors["YELLOW"]

st.markdown(render_css(colors), unsafe_allow_html=True)

# Snapshot portfolio value for equity curve (once per session per day)
def _snapshot_portfolio():
    today = datetime.now().strftime("%Y-%m-%d")
    hist = st.session_state.data.get("portfolio_history", [])
    if not hist or hist[-1].get("date") != today:
        try:
            val = portfolio_value()
            st.session_state.data["portfolio_history"].append({"date": today, "value": round(val, 2)})
            save_data(st.session_state.data)
        except Exception:
            pass

if st.session_state.data.get("portfolio") or st.session_state.data.get("options"):
    _snapshot_portfolio()

# Badge toast notifications for newly earned badges
current_badges = check_badges(st.session_state.data)
prev_badges = st.session_state.data.get("badges", [])
new_badges = [b for b in current_badges if b not in prev_badges]
if new_badges:
    st.session_state.data["badges"] = current_badges
    save_data(st.session_state.data)
    for b in new_badges:
        meta = BADGE_META.get(b, {})
        st.toast(f"{meta.get('icon', '🏅')} Badge Unlocked: **{b}** — {meta.get('desc', '')}")

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

    if st.button("🔄 Refresh Prices", width="stretch", key="refresh_btn"):
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

    # Badges
    badges = check_badges(st.session_state.data)
    if badges:
        st.divider()
        st.markdown(f"<div style='color:{TEXT}; font-weight:bold; font-size:13px;'>Badges Earned</div>", unsafe_allow_html=True)
        badge_icons = {
            "First Trade": "🏁", "Diversifier": "🌐", "Options Trader": "📊",
            "Risk Manager": "🛡️", "Consistent": "📅", "Six-Figure Club": "💰",
            "Watchful": "👁️",
        }
        badge_text = " ".join(f"{badge_icons.get(b, '🏅')}" for b in badges)
        st.markdown(f"<div style='font-size:20px; margin:4px 0;'>{badge_text}</div>", unsafe_allow_html=True)
        st.caption(", ".join(badges))

    st.divider()

    nav_options = ["Dashboard", "Portfolio", "Trade", "Options", "Watchlist", "Research", "Analytics", "Rebalance", "AI Assistant", "Settings"]
    default_idx = 0
    if "_nav_override" in st.session_state:
        try:
            default_idx = nav_options.index(st.session_state.pop("_nav_override"))
        except ValueError:
            default_idx = 0
    page = st.radio("Navigation", nav_options, index=default_idx, label_visibility="collapsed")

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
                if st.button("✓ Confirm", width="stretch", key="confirm_trade"):
                    execute_trade(t)
                    st.session_state.show_confirm = False
                    st.session_state.pending_trade = None
                    st.rerun()
            with btn_col2:
                if st.button("✕ Cancel", width="stretch", key="cancel_trade"):
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
    elif t["action"] == "sell":
        st.session_state.data["cash"] += (total - commission)
    elif t["action"] == "short":
        st.session_state.data["cash"] += (total - commission)
    elif t["action"] == "cover":
        st.session_state.data["cash"] -= (total + commission)

    trade_type = "buy" if t["action"] in ("buy", "cover") else "sell"
    st.session_state.data["portfolio"].append({
        "ticker": t["ticker"],
        "type": trade_type,
        "order_type": t.get("order_type", "market"),
        "shares": t["shares"],
        "price": t["price"],
        "commission": commission,
        "short": t["action"] in ("short", "cover"),
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

# ==================== DASHBOARD ====================
if page == "Dashboard":
    st.header("Dashboard")

    # Row 1: Market Overview
    market = get_market_overview(get_price)
    idx_cols = st.columns(len(market))
    for i, idx in enumerate(market):
        with idx_cols[i]:
            chg_color = GREEN if idx["change_pct"] >= 0 else RED
            st.markdown(f"""
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:16px; text-align:center;">
                <div style="color:{TEXT2}; font-size:12px;">{idx['name']}</div>
                <div style="color:{TEXT}; font-size:1.4rem; font-weight:bold;">${idx['price']:,.2f}</div>
                <div style="color:{chg_color}; font-size:13px;">{idx['change_pct']:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Row 2: Portfolio Snapshot + Market Sentiment
    snap_col, sent_col = st.columns([2, 1])

    with snap_col:
        st.subheader("Portfolio Snapshot")
        total = portfolio_value()
        cash = st.session_state.data["cash"]
        start = st.session_state.data["starting_balance"]
        pl = total - start

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Value", f"${total:,.0f}")
        pl_color = GREEN if pl >= 0 else RED
        sc2.metric("Total P/L", f"${pl:+,.0f}", f"{pl/start*100:+.1f}%")
        sc3.metric("Cash", f"${cash:,.0f}")
        sc4.metric("Invested", f"${total - cash:,.0f}")

        # Mini allocation donut
        holdings = get_holdings()
        if holdings:
            labels = list(holdings.keys()) + ["Cash"]
            values = [h["shares"] * get_price(tk)[0] for tk, h in holdings.items()] + [cash]
            fig_mini = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.5,
                marker=dict(colors=["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"][:len(labels)]),
                textinfo="label+percent", textfont=dict(size=10, color="white"),
            )])
            fig_mini.update_layout(
                height=220, margin=dict(t=5, b=5, l=5, r=5), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text="Allocation", x=0.5, y=0.5, font_size=11, font_color=TEXT, showarrow=False)],
            )
            st.plotly_chart(fig_mini, width="stretch")

    with sent_col:
        st.subheader("Market Sentiment")
        spy_hist = get_history("SPY", "6mo")
        if not spy_hist.empty:
            regime = detect_market_regime(spy_hist)
            regime_color = GREEN if regime["regime"] == "Bullish" else RED if regime["regime"] == "Bearish" else YELLOW
            st.markdown(f"""
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:20px;">
                <div style="font-size:1.5rem; font-weight:bold; color:{regime_color};">{regime['regime']}</div>
                <div style="color:{TEXT2}; margin-top:4px;">S&P 500 Regime</div>
                <div style="color:{TEXT2}; font-size:12px; margin-top:8px;">Confidence: {regime['confidence']}</div>
                <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">{regime['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Unable to fetch market data.")

    st.divider()

    # Row 3: Top Movers + Active Alerts
    mov_col, alert_col = st.columns(2)

    with mov_col:
        st.subheader("Top Movers")
        if holdings:
            gainers, losers = get_top_movers(holdings, get_price)
            if gainers:
                for g in gainers:
                    st.markdown(f"""
                    <div style="background:{BG2}; border-left:3px solid {GREEN}; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:6px;">
                        <span style="color:{TEXT}; font-weight:bold;">{g['ticker']}</span>
                        <span style="color:{GREEN}; float:right;">{g['change_pct']:+.2f}%</span>
                        <div style="color:{TEXT2}; font-size:12px;">${g['price']:.2f} | ${g['value']:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if losers:
                for l in losers:
                    st.markdown(f"""
                    <div style="background:{BG2}; border-left:3px solid {RED}; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:6px;">
                        <span style="color:{TEXT}; font-weight:bold;">{l['ticker']}</span>
                        <span style="color:{RED}; float:right;">{l['change_pct']:+.2f}%</span>
                        <div style="color:{TEXT2}; font-size:12px;">${l['price']:.2f} | ${l['value']:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not gainers and not losers:
                st.caption("No significant movers today.")
        else:
            st.caption("Add holdings to see top movers.")

    with alert_col:
        st.subheader("Active Alerts")
        triggered = get_triggered_alerts(
            st.session_state.data.get("watchlist", []),
            st.session_state.data.get("price_alerts", {}),
            get_price,
        )
        if triggered:
            for a in triggered:
                st.markdown(f"""
                <div style="background:{BG2}; border-left:3px solid {YELLOW}; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:6px;">
                    <span style="color:{TEXT}; font-weight:bold;">{a['ticker']}</span>
                    <span style="color:{YELLOW}; float:right;">Alert!</span>
                    <div style="color:{TEXT2}; font-size:12px;">Current: ${a['current_price']:.2f} | Target: ${a['alert_price']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No triggered alerts. Set alerts on the Watchlist page.")

    st.divider()

    # Row 4: Market News
    st.subheader("Market News")
    general_news = get_general_news(limit=5)
    if general_news:
        headlines_tuple = tuple(a["title"] for a in general_news)
        sentiments = analyze_sentiment_batch(headlines_tuple)

        for i, article in enumerate(general_news):
            sent = sentiments[i] if i < len(sentiments) else {"sentiment": "Neutral", "confidence": 0.0}
            sent_color = get_sentiment_color(sent["sentiment"], colors)
            time_str = format_time_ago(article.get("publishedDate", ""))
            source = article.get("source", "")

            st.markdown(f"""
            <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <a href="{article['url']}" target="_blank" style="color:{TEXT}; font-size:14px; text-decoration:none; font-weight:500;">
                            {article['title']}
                        </a>
                        <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">{source}{(' | ' + time_str) if time_str else ''}</div>
                    </div>
                    <div style="background:{sent_color}20; color:{sent_color}; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap; margin-left:12px;">
                        {sent['sentiment']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Unable to fetch news. Check your internet connection.")

    st.divider()

    # Row 5: Quick Actions
    st.subheader("Quick Actions")
    qa_cols = st.columns(5)
    qa_pages = [("Trade", "Trade"), ("Research", "Research"), ("Watchlist", "Watchlist"), ("Analytics", "Analytics"), ("AI Assistant", "AI Assistant")]
    for i, (label, target) in enumerate(qa_pages):
        with qa_cols[i]:
            if st.button(label, key=f"qa_{label}", width="stretch"):
                st.session_state["_nav_override"] = target
                st.rerun()

# ==================== PORTFOLIO ====================
elif page == "Portfolio":
    st.header("Portfolio")

    if show_trade_dialog():
        st.stop()

    holdings = get_holdings()
    options_holdings = get_options_holdings()

    if not holdings and not options_holdings:
        st.info("👋 Welcome! Go to **Trade** to place your first order, or add stocks to your **Watchlist** to start tracking.")
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

        # Equity Curve
        ph = st.session_state.data.get("portfolio_history", [])
        if len(ph) >= 2:
            eq_df = pd.DataFrame(ph)
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["value"], name="Portfolio",
                line=dict(color=BLUE, width=2), fill="tozeroy",
                fillcolor=f"rgba(37,99,235,0.1)"
            ))
            fig_eq.add_hline(y=st.session_state.data["starting_balance"],
                            line_dash="dash", line_color=TEXT2, annotation_text="Starting Balance")
            fig_eq.update_layout(
                title=dict(text="Equity Curve", font=dict(size=14, color=TEXT)),
                height=250, margin=dict(t=40, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor=BORDER, tickprefix="$"),
                xaxis=dict(showgrid=False),
                font=dict(color=TEXT2), showlegend=False,
            )
            st.plotly_chart(fig_eq, width="stretch")

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
                st.dataframe(df_show, width="stretch", hide_index=True)

                # Quick trade buttons
                qt_cols = st.columns(min(len(holdings), 6))
                for i, tk in enumerate(list(holdings.keys())[:6]):
                    with qt_cols[i]:
                        if st.button(f"Sell {tk}", key=f"qt_sell_{tk}", width="stretch"):
                            price, _, _ = get_price(tk)
                            st.session_state.pending_trade = {
                                "ticker": tk, "action": "sell", "order_type": "market",
                                "shares": holdings[tk]["shares"], "price": price, "notes": "Quick sell"
                            }
                            st.session_state.show_confirm = True
                            st.rerun()

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
                    rotation=45, direction="clockwise"
                )])
                fig.update_layout(
                    height=320, margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT2),
                    annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=14, font_color=TEXT, showarrow=False)]
                )
                st.plotly_chart(fig, width="stretch")

            # Sector Allocation
            sectors = {}
            for tk, h in holdings.items():
                info = get_info(tk)
                sector = info.get("sector", "Unknown")
                price, _, _ = get_price(tk)
                sectors[sector] = sectors.get(sector, 0) + h["shares"] * price

            if sectors:
                st.subheader("Sector Allocation")
                sec_labels = list(sectors.keys())
                sec_values = list(sectors.values())
                sector_colors = ["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
                                 "#bfdbfe", "#0ea5e9", "#0284c7", "#0369a1", "#075985"]
                fig_sec = go.Figure(data=[go.Pie(
                    labels=sec_labels, values=sec_values, hole=0.35,
                    marker=dict(colors=sector_colors[:len(sec_labels)]),
                    textinfo="label+percent",
                    textfont=dict(size=11, color="white"),
                )])
                fig_sec.update_layout(
                    height=280, margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT2),
                )
                st.plotly_chart(fig_sec, width="stretch")

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

            st.dataframe(pd.DataFrame(opts_data), width="stretch", hide_index=True)

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
                    height=280, margin=dict(t=40,b=40,l=40,r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True, zerolinecolor=TEXT2, zerolinewidth=2),
                    xaxis=dict(showgrid=False),
                    font=dict(color=TEXT2), bargap=0.3
                )
                st.plotly_chart(fig1, width="stretch")

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
                st.download_button("Export CSV", df_show.to_csv(index=False), "portfolio.csv", width="stretch")

        # Achievements
        st.divider()
        st.subheader("Achievements")
        all_badge_names = [name for name, _ in BADGE_DEFS]
        earned = check_badges(st.session_state.data)
        badge_cols = st.columns(len(all_badge_names))
        for i, name in enumerate(all_badge_names):
            meta = BADGE_META.get(name, {})
            is_earned = name in earned
            with badge_cols[i]:
                if is_earned:
                    st.markdown(f"""
                    <div style="text-align:center; background:{BG2}; border:2px solid {BLUE}; border-radius:12px; padding:12px 6px;">
                        <div style="font-size:28px;">{meta.get('icon','🏅')}</div>
                        <div style="color:{TEXT}; font-size:11px; font-weight:bold; margin-top:4px;">{name}</div>
                        <div style="color:{TEXT2}; font-size:10px; margin-top:2px;">{meta.get('desc','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align:center; background:{BG2}; border:1px solid {BORDER}; border-radius:12px; padding:12px 6px; opacity:0.4;">
                        <div style="font-size:28px;">🔒</div>
                        <div style="color:{TEXT2}; font-size:11px; font-weight:bold; margin-top:4px;">{name}</div>
                        <div style="color:{TEXT2}; font-size:10px; margin-top:2px;">{meta.get('hint','')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.caption(f"{len(earned)} / {len(all_badge_names)} badges earned")

        with st.expander("Stock Transaction History"):
            if st.session_state.data["portfolio"]:
                st.dataframe(pd.DataFrame(st.session_state.data["portfolio"]), width="stretch", hide_index=True)
            else:
                st.info("No stock transactions yet")

        with st.expander("Options Transaction History"):
            if st.session_state.data["options"]:
                st.dataframe(pd.DataFrame(st.session_state.data["options"]), width="stretch", hide_index=True)
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
        else:
            st.info("👋 Search for a stock above to start trading. You can buy, sell, or short sell any S&P 500 stock.")

    with col2:
        if ticker:
            price, _, _ = get_price(ticker)

            st.subheader("Order")

            action = st.selectbox("Action", ["Buy", "Sell", "Short Sell", "Cover Short"])
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
            elif action == "Cover Short":
                if total > st.session_state.data["cash"]:
                    st.error(f"Insufficient funds to cover. Need ${total:,.2f}")
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

            action_map = {"Buy": "buy", "Sell": "sell", "Short Sell": "short", "Cover Short": "cover"}

            if can_trade:
                if st.button(f"Preview {action.upper()} Order", type="primary", width="stretch"):
                    st.session_state.pending_trade = {
                        "ticker": ticker,
                        "action": action_map[action],
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

    if not sel:
        st.info("👋 Select an underlying stock to view options chains and place options trades.")

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
                st.dataframe(df_show.head(15), width="stretch", hide_index=True)

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

                if st.button("Execute Options Order", type="primary", width="stretch"):
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

                # P/L Payoff Diagram
                st.divider()
                st.subheader("P/L at Expiration")
                prices, pnl, breakeven, max_p, max_l = compute_option_payoff(
                    price, strike, premium, opt_type.lower(), opt_action, contracts
                )
                fig_pl = go.Figure()
                pnl_colors = ["rgba(37,99,235,0.8)" if v >= 0 else "rgba(248,81,73,0.8)" for v in pnl]
                fig_pl.add_trace(go.Scatter(
                    x=prices, y=pnl, name="P/L", fill="tozeroy",
                    line=dict(color=BLUE, width=2),
                    fillcolor="rgba(37,99,235,0.1)"
                ))
                fig_pl.add_vline(x=strike, line_dash="dash", line_color=TEXT2,
                                annotation_text=f"Strike ${strike:.0f}")
                fig_pl.add_vline(x=breakeven, line_dash="dot", line_color=YELLOW,
                                annotation_text=f"BE ${breakeven:.2f}")
                fig_pl.add_vline(x=price, line_dash="solid", line_color=GREEN,
                                annotation_text=f"Spot ${price:.2f}", annotation_position="bottom right")
                fig_pl.add_hline(y=0, line_color=TEXT2, line_width=1)

                max_p_str = f"${max_p:,.0f}" if max_p != float('inf') else "Unlimited"
                max_l_str = f"${max_l:,.0f}" if max_l != float('-inf') else "Unlimited"

                fig_pl.update_layout(
                    height=300, margin=dict(t=10, b=40, l=50, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Stock Price at Expiration", showgrid=False, tickprefix="$"),
                    yaxis=dict(title="Profit / Loss", showgrid=True, gridcolor=BORDER, tickprefix="$"),
                    font=dict(color=TEXT2), hovermode="x unified", showlegend=False,
                )
                st.plotly_chart(fig_pl, width="stretch")

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Breakeven", f"${breakeven:.2f}")
                mc2.metric("Max Profit", max_p_str)
                mc3.metric("Max Loss", max_l_str)
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
        if sel and st.button("Add", type="primary", width="stretch"):
            tk = sel.split(" - ")[0]
            if tk not in st.session_state.data["watchlist"]:
                st.session_state.data["watchlist"].append(tk)
                save_data(st.session_state.data)
                st.rerun()

    st.divider()

    wl = st.session_state.data["watchlist"]
    if not wl:
        st.info("👋 Your watchlist is empty. Search above to add stocks you want to track.")
    else:
        data = []
        alerts = st.session_state.data.get("price_alerts", {})
        for t in wl:
            info = get_info(t)
            price, chg, pct = get_price(t)
            alert_price = alerts.get(t)
            alert_status = ""
            if alert_price:
                if price >= alert_price:
                    alert_status = f"🔔 Above ${alert_price:.2f}"
                else:
                    alert_status = f"Target: ${alert_price:.2f}"
            data.append({
                "Ticker": t,
                "Price": f"${price:.2f}",
                "Change": f"{pct:+.2f}%",
                "52W High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
                "52W Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
                "Alert": alert_status,
            })
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)

        # Quick Buy buttons
        st.subheader("Quick Buy")
        qb_cols = st.columns(min(len(wl), 6))
        for i, tk in enumerate(wl[:6]):
            with qb_cols[i]:
                if st.button(f"Buy {tk}", key=f"wb_{tk}", width="stretch"):
                    price, _, _ = get_price(tk)
                    st.session_state.pending_trade = {
                        "ticker": tk, "action": "buy", "order_type": "market",
                        "shares": 1.0, "price": price, "notes": "Quick buy from watchlist"
                    }
                    st.session_state.show_confirm = True
                    st.rerun()

        st.divider()

        # Price Alerts
        st.subheader("Price Alerts")
        al_col1, al_col2, al_col3 = st.columns([2, 1, 1])
        with al_col1:
            alert_ticker = st.selectbox("Ticker", ["--"] + wl, key="alert_tk")
        with al_col2:
            alert_target = st.number_input("Target Price ($)", min_value=0.01, value=100.0, step=1.0, key="alert_price")
        with al_col3:
            st.write("")
            if alert_ticker != "--" and st.button("Set Alert", width="stretch"):
                st.session_state.data.setdefault("price_alerts", {})[alert_ticker] = alert_target
                save_data(st.session_state.data)
                st.success(f"Alert set for {alert_ticker} at ${alert_target:.2f}")
                st.rerun()

        st.divider()

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            rem = st.selectbox("Remove", ["--"] + wl)
        with col2:
            st.write("")
            if rem != "--" and st.button("Remove", width="stretch"):
                st.session_state.data["watchlist"].remove(rem)
                st.session_state.data.get("price_alerts", {}).pop(rem, None)
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

    if not sel:
        st.info("👋 Search for any S&P 500 stock to view charts, fundamentals, and AI-powered signals.")

    if sel:
        ticker = sel.split(" - ")[0]

        info = get_info(ticker)
        price, chg, pct = get_price(ticker)

        # Multi-timeframe selector
        timeframe = st.radio("Timeframe", ["1W", "1M", "3M", "6M", "1Y", "5Y"], horizontal=True, index=3)
        period_map = {"1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
        df = get_history(ticker, period_map[timeframe])

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

            ch_col1, ch_col2 = st.columns([1, 3])
            with ch_col1:
                chart = st.radio("Chart", ["Line", "Candle"], horizontal=True, label_visibility="collapsed")
            with ch_col2:
                ind_opts = st.multiselect("Indicators", ["Bollinger Bands", "RSI", "MACD"], default=[], key="indicators")

            show_bb = "Bollinger Bands" in ind_opts
            show_rsi = "RSI" in ind_opts
            show_macd = "MACD" in ind_opts

            # Determine subplot layout
            extra_rows = sum([True, show_rsi, show_macd])  # volume always shown
            row_heights = [0.55]
            subplot_titles = [""]
            if show_rsi:
                row_heights.append(0.15)
                subplot_titles.append("RSI")
            if show_macd:
                row_heights.append(0.15)
                subplot_titles.append("MACD")
            row_heights.append(0.15)  # volume
            subplot_titles.append("Volume")
            total_rows = len(row_heights)

            fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True,
                               vertical_spacing=0.03, row_heights=row_heights)

            # Price chart (row 1)
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()

            if chart == "Line":
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color=BLUE)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color=YELLOW, dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color=GREEN, dash="dash")), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                            low=df["Low"], close=df["Close"],
                                            increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)

            # Bollinger Bands overlay on price
            if show_bb:
                bb_mid, bb_upper, bb_lower = compute_bollinger(df["Close"])
                fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper",
                                        line=dict(color=TEXT2, width=1, dash="dot"), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower",
                                        line=dict(color=TEXT2, width=1, dash="dot"), fill="tonexty",
                                        fillcolor="rgba(88,166,255,0.08)", showlegend=False), row=1, col=1)

            current_row = 2

            # RSI subplot
            if show_rsi:
                rsi = compute_rsi(df["Close"])
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color=BLUE, width=1.5)), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color=RED, line_width=0.8, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=GREEN, line_width=0.8, row=current_row, col=1)
                fig.update_yaxes(range=[0, 100], title_text="RSI", row=current_row, col=1)
                current_row += 1

            # MACD subplot
            if show_macd:
                macd_line, signal_line, histogram = compute_macd(df["Close"])
                hist_colors = [GREEN if v >= 0 else RED for v in histogram]
                fig.add_trace(go.Bar(x=df.index, y=histogram, name="MACD Hist",
                                    marker_color=hist_colors, opacity=0.5), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                                        line=dict(color=BLUE, width=1.5)), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal",
                                        line=dict(color=YELLOW, width=1.5, dash="dash")), row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                current_row += 1

            # Volume (always last row)
            if "Volume" in df.columns:
                vol_colors = [GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                                    marker_color=vol_colors, opacity=0.5), row=current_row, col=1)
                fig.update_yaxes(title_text="Vol", row=current_row, col=1)

            chart_height = 450 + (show_rsi * 120) + (show_macd * 120)
            fig.update_layout(height=chart_height, margin=dict(t=20,b=40,l=40,r=20),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
                             yaxis=dict(showgrid=True, gridcolor=BORDER),
                             font=dict(color=TEXT2), hovermode="x unified",
                             legend=dict(orientation="h", y=1.08),
                             showlegend=True)
            # Hide grid on all sub-axes
            for r in range(2, total_rows + 1):
                fig.update_xaxes(showgrid=False, row=r, col=1)
                fig.update_yaxes(showgrid=True, gridcolor=BORDER, row=r, col=1)
            st.plotly_chart(fig, width="stretch")

            # Latest News with Sentiment
            st.divider()
            st.subheader("Latest News")

            news_articles = get_stock_news(ticker, limit=8)
            if news_articles:
                headlines_tuple = tuple(a["title"] for a in news_articles)
                sentiments = analyze_sentiment_batch(headlines_tuple)

                for i, article in enumerate(news_articles):
                    sent = sentiments[i] if i < len(sentiments) else {"sentiment": "Neutral", "confidence": 0.0}
                    sent_color = get_sentiment_color(sent["sentiment"], colors)
                    time_str = format_time_ago(article.get("publishedDate", ""))
                    source = article.get("source", "")

                    st.markdown(f"""
                    <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div style="flex:1;">
                                <a href="{article['url']}" target="_blank" style="color:{TEXT}; font-size:14px; text-decoration:none; font-weight:500;">
                                    {article['title']}
                                </a>
                                <div style="color:{TEXT2}; font-size:12px; margin-top:4px;">{source} {(' | ' + time_str) if time_str else ''}</div>
                            </div>
                            <div style="background:{sent_color}20; color:{sent_color}; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap; margin-left:12px;">
                                {sent['sentiment']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                company_name = info.get('shortName', ticker).replace(' ', '+')
                st.markdown(f"""
                <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                    <a href="https://finance.yahoo.com/quote/{ticker}/news" target="_blank" style="color:{BLUE}; font-size:14px; text-decoration:none; font-weight:500;">
                        {ticker} News on Yahoo Finance
                    </a>
                </div>
                <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:12px; margin-bottom:10px;">
                    <a href="https://www.google.com/search?q={ticker}+{company_name}+stock+news&tbm=nws" target="_blank" style="color:{BLUE}; font-size:14px; text-decoration:none; font-weight:500;">
                        {ticker} on Google News
                    </a>
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
                if analyst_data.get("dcf") is not None:
                    valuation = analyze_valuation(analyst_data)
                    val_color = GREEN if valuation["signal"] == "Undervalued" else RED if valuation["signal"] == "Overvalued" else YELLOW
                    st.markdown(f"""
                    <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:8px; padding:16px;">
                        <div style="font-size:1.4rem; font-weight:bold; color:{val_color};">{valuation['signal']}</div>
                        <div style="color:{TEXT2}; margin-top:4px;">DCF: ${valuation['dcf']:,.2f} | Margin: {valuation['margin_of_safety']:+.1f}%</div>
                        <div style="color:{TEXT2}; font-size:12px; margin-top:8px;">{valuation['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif analyst_data.get("error") == "no_key":
                    st.info("Add FMP_API_KEY in .streamlit/secrets.toml to enable DCF valuation.")
                else:
                    st.warning(f"DCF data unavailable. {analyst_data.get('error', 'FMP API may be rate-limited.')}")

# ==================== ANALYTICS ====================
elif page == "Analytics":
    st.header("Portfolio Analytics")

    holdings = get_holdings()

    if not holdings:
        st.info("👋 Add positions to see analytics. Go to **Trade** to get started.")
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

        # Benchmark Overlay
        st.subheader("Portfolio vs S&P 500")
        ph = st.session_state.data.get("portfolio_history", [])
        if len(ph) >= 2 and not spy.empty:
            eq_df = pd.DataFrame(ph)
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            port_start = eq_df["value"].iloc[0]

            spy_daily = spy["Close"].reset_index()
            spy_daily.columns = ["date", "close"]
            spy_start = spy_daily["close"].iloc[0]

            fig_bm = go.Figure()
            fig_bm.add_trace(go.Scatter(
                x=eq_df["date"], y=(eq_df["value"] / port_start - 1) * 100,
                name="Portfolio", line=dict(color=BLUE, width=2)
            ))
            fig_bm.add_trace(go.Scatter(
                x=spy_daily["date"], y=(spy_daily["close"] / spy_start - 1) * 100,
                name="S&P 500", line=dict(color=YELLOW, width=2, dash="dash")
            ))
            fig_bm.update_layout(
                height=300, margin=dict(t=20,b=30,l=40,r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Return %", showgrid=True, gridcolor=BORDER),
                xaxis=dict(showgrid=False),
                font=dict(color=TEXT2), hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_bm, width="stretch")
        else:
            st.info("Need at least 2 days of portfolio history for benchmark comparison.")

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

        # Trade Journal Analytics
        st.subheader("Trade Journal Analytics")
        journal = st.session_state.data.get("journal", [])
        stats = get_trade_stats(journal)

        if stats["total"] > 0:
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            tc1.metric("Total Trades", stats["total"])
            tc2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            tc3.metric("Avg Win", f"{stats['avg_win']:+.1f}%")
            tc4.metric("Avg Loss", f"{stats['avg_loss']:+.1f}%")
            tc5.metric("Profit Factor", f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞")

            tc6, tc7 = st.columns(2)
            tc6.metric("Best Trade", f"{stats['best']:+.1f}%")
            tc7.metric("Worst Trade", f"{stats['worst']:+.1f}%")
        else:
            st.info("Complete some buy/sell round trips to see trade analytics.")

        st.divider()

        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation (1-Year Projection)")
        if not portfolio_returns.empty:
            mc = compute_monte_carlo(portfolio_returns, total, days=252, sims=500)
            if mc:
                days_axis = list(range(1, 253))
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mc["p95"], name="95th %ile",
                                           line=dict(color=GREEN, width=1), fill=None))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mc["p75"], name="75th %ile",
                                           line=dict(color=GREEN, width=1), fill="tonexty",
                                           fillcolor="rgba(37,99,235,0.08)"))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mc["p50"], name="Median",
                                           line=dict(color=BLUE, width=2)))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mc["p25"], name="25th %ile",
                                           line=dict(color=RED, width=1), fill="tonexty",
                                           fillcolor="rgba(37,99,235,0.08)"))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mc["p5"], name="5th %ile",
                                           line=dict(color=RED, width=1)))
                fig_mc.add_hline(y=total, line_dash="dash", line_color=TEXT2, annotation_text="Current")
                fig_mc.update_layout(
                    height=350, margin=dict(t=20,b=40,l=40,r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridcolor=BORDER, tickprefix="$"),
                    xaxis=dict(title="Trading Days", showgrid=False),
                    font=dict(color=TEXT2), hovermode="x unified",
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_mc, width="stretch")

        st.divider()

        # Correlation Heatmap
        st.subheader("Correlation Matrix")
        if len(holdings) >= 2:
            corr = compute_correlation_matrix(holdings, get_history)
            if not corr.empty:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="Blues", zmin=-1, zmax=1,
                    text=corr.round(2).values, texttemplate="%{text}",
                    textfont=dict(size=12),
                ))
                fig_corr.update_layout(
                    height=350, margin=dict(t=20,b=20,l=20,r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT2),
                )
                st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("Need 2+ holdings for correlation analysis.")

        st.divider()

        # AI Coaching Tips
        st.subheader("AI Trading Coach")
        tips = generate_coaching_tips(st.session_state.data, holdings)
        for tip in tips:
            st.markdown(f"""
            <div style="background:{BG2}; border-left:3px solid {BLUE}; padding:12px 16px; margin-bottom:8px; border-radius:0 8px 8px 0;">
                <span style="color:{TEXT};">💡 {tip}</span>
            </div>
            """, unsafe_allow_html=True)

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
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("Trade Journal")

        if st.session_state.data["journal"]:
            st.dataframe(pd.DataFrame(st.session_state.data["journal"]), width="stretch", hide_index=True)
        else:
            st.info("No trades logged yet")

# ==================== REBALANCE ====================
elif page == "Rebalance":
    st.header("Portfolio Rebalancer")

    holdings = get_holdings()

    if not holdings:
        st.info("Add positions first. Go to **Trade** to place your first order.")
    else:
        # Current allocation
        current = compute_current_weights(holdings, get_price)
        total_invested = current.pop("_total", 0)
        total_account = total_invested + st.session_state.data["cash"]

        st.subheader("Current Allocation")
        cur_col1, cur_col2 = st.columns([2, 1])

        with cur_col1:
            cur_data = []
            for tk, info_w in current.items():
                cur_data.append({
                    "Ticker": tk,
                    "Shares": f"{info_w['shares']:.2f}",
                    "Price": f"${info_w['price']:.2f}",
                    "Value": f"${info_w['value']:,.0f}",
                    "Weight": f"{info_w['weight_pct']:.1f}%",
                })
            st.dataframe(pd.DataFrame(cur_data), width="stretch", hide_index=True)

        with cur_col2:
            st.metric("Invested", f"${total_invested:,.0f}")
            st.metric("Cash", f"${st.session_state.data['cash']:,.0f}")
            st.metric("Total", f"${total_account:,.0f}")

        st.divider()

        # Target allocation
        st.subheader("Target Allocation")

        strategy = st.radio("Strategy", ["Equal Weight", "Custom Weights"], horizontal=True, key="rebal_strategy")

        tickers = list(holdings.keys())

        if strategy == "Equal Weight":
            targets = generate_equal_weight_targets(tickers)
            st.caption(f"Each position targets {100/len(tickers):.1f}% of invested value")
        else:
            st.caption("Set target weight for each holding (will be normalized to 100%)")
            custom_weights = {}
            cols = st.columns(min(len(tickers), 4))
            for i, tk in enumerate(tickers):
                with cols[i % min(len(tickers), 4)]:
                    default_w = current.get(tk, {}).get("weight_pct", 0)
                    custom_weights[tk] = st.number_input(
                        f"{tk} %", min_value=0.0, max_value=100.0,
                        value=round(default_w, 1), step=1.0, key=f"cw_{tk}"
                    )
            targets = generate_custom_targets(custom_weights)

        deploy_cash = st.checkbox("Include cash in rebalance (deploy idle cash)", value=True, key="deploy_cash")

        # Current vs Target chart
        chart_tickers = sorted(set(list(current.keys()) + list(targets.keys())))
        cur_weights = [current.get(t, {}).get("weight_pct", 0) for t in chart_tickers]
        tgt_weights = [targets.get(t, 0) for t in chart_tickers]

        fig_rebal = go.Figure()
        fig_rebal.add_trace(go.Bar(name="Current", x=chart_tickers, y=cur_weights, marker_color=BLUE))
        fig_rebal.add_trace(go.Bar(name="Target", x=chart_tickers, y=tgt_weights, marker_color=GREEN))
        fig_rebal.update_layout(
            barmode="group", height=300, margin=dict(t=20, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Weight %", showgrid=True, gridcolor=BORDER),
            xaxis=dict(showgrid=False),
            font=dict(color=TEXT2), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_rebal, width="stretch")

        st.divider()

        # Suggested trades
        st.subheader("Suggested Trades")

        commission = st.session_state.data["commission_stock"] if st.session_state.data["commission_enabled"] else 0
        trades = compute_rebalance_trades(holdings, targets, st.session_state.data["cash"], get_price, commission, deploy_cash)

        if trades:
            trade_data = []
            for t in trades:
                action_color = GREEN if t["action"] == "buy" else RED
                trade_data.append({
                    "Action": t["action"].upper(),
                    "Ticker": t["ticker"],
                    "Shares": f"{t['shares']:.2f}",
                    "Price": f"${t['price']:.2f}",
                    "Value": f"${abs(t['value_change']):,.0f}",
                    "Current %": f"{t['current_weight']:.1f}%",
                    "Target %": f"{t['target_weight']:.1f}%",
                })
            st.dataframe(pd.DataFrame(trade_data), width="stretch", hide_index=True)

            # Cost summary
            cost = estimate_rebalance_cost(trades, commission)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Total Buys", f"${cost['total_buys']:,.0f}")
            cc2.metric("Total Sells", f"${cost['total_sells']:,.0f}")
            cc3.metric("Net Cash Needed", f"${cost['net_cash_needed']:+,.0f}")
            cc4.metric("Trades", cost['num_trades'])

            if cost["total_commissions"] > 0:
                st.caption(f"Estimated commissions: ${cost['total_commissions']:.2f}")

            # Check feasibility
            can_rebalance = cost["net_cash_needed"] <= st.session_state.data["cash"]

            if not can_rebalance:
                st.error(f"Insufficient cash. Need ${cost['net_cash_needed']:,.0f}, have ${st.session_state.data['cash']:,.0f}. Try unchecking 'Include cash in rebalance'.")

            st.divider()

            if can_rebalance:
                if st.button("Execute Rebalance", type="primary", width="stretch"):
                    errors = []
                    for t in trades:
                        try:
                            execute_trade({
                                "ticker": t["ticker"],
                                "action": t["action"],
                                "order_type": "market",
                                "shares": t["shares"],
                                "price": t["price"],
                                "notes": "Rebalance",
                            })
                        except Exception as e:
                            errors.append(f"{t['ticker']}: {e}")
                    if errors:
                        st.warning(f"Some trades failed: {'; '.join(errors)}")
                    else:
                        st.success(f"Rebalance complete! Executed {len(trades)} trades.")
                    st.rerun()
        else:
            st.success("Portfolio is already balanced to target weights.")

# ==================== AI ASSISTANT ====================
elif page == "AI Assistant":
    st.header("🤖 AI Trading Assistant")
    st.caption("Powered by Llama 3.3 70B via Groq — portfolio-aware, real-time data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested prompts (only show when no history)
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="background:{BG2}; border:1px solid {BORDER}; border-radius:12px; padding:24px; margin:16px 0; text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">🤖</div>
            <div style="color:{TEXT}; font-size:16px; font-weight:bold;">How can I help you today?</div>
            <div style="color:{TEXT2}; font-size:13px; margin-top:6px;">I know your portfolio, real-time prices, and can help you navigate the app.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='color:{TEXT2}; font-size:13px; margin-bottom:8px;'>Try one of these:</div>", unsafe_allow_html=True)
        prompts = [
            ("📊 Analyze my portfolio", "Analyze my current portfolio. What are my biggest positions, how am I performing, and what risks should I be aware of?"),
            ("📈 Market outlook", "Based on my holdings, what's the current market regime and how are my stocks positioned? Any concerns?"),
            ("🎯 What should I look at next?", "Based on my portfolio composition and cash balance, what opportunities should I explore? Consider diversification and risk."),
            ("🧠 Explain risk metrics", "Explain what Sharpe ratio, VaR, and max drawdown mean in the context of my portfolio. How do I find them in the app?"),
            ("📉 Review my trades", "Review my recent trade history. What patterns do you see? Am I overtrading or making good entries?"),
            ("🔍 How do I use this app?", "Give me a tour of this trading simulator. What are all the features available and how do I use them effectively?"),
        ]
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(prompts):
            with cols[i % 2]:
                if st.button(label, key=f"prompt_{i}", width="stretch"):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    holdings_summary, prices_summary = build_portfolio_context(
                        st.session_state.data, get_price, get_holdings
                    )
                    with st.spinner("Thinking..."):
                        response = get_ai_response(
                            prompt, st.session_state.data, holdings_summary, prices_summary
                        )
                    st.session_state.chat_history.append({"role": "model", "content": response})
                    st.rerun()

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about trading, your portfolio, or the app...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        holdings_summary, prices_summary = build_portfolio_context(
            st.session_state.data, get_price, get_holdings
        )
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = get_ai_response(
                    user_input, st.session_state.data, holdings_summary, prices_summary
                )
            st.markdown(response)
        st.session_state.chat_history.append({"role": "model", "content": response})
        st.rerun()

    # Controls
    if st.session_state.chat_history:
        st.divider()
        cc1, cc2, cc3 = st.columns([1, 1, 4])
        with cc1:
            if st.button("Clear Chat", width="stretch", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        with cc2:
            st.metric("Messages", len(st.session_state.chat_history))

# ==================== SETTINGS ====================
elif page == "Settings":
    st.header("Settings")

    st.subheader("Account")

    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Balance", f"${st.session_state.data['starting_balance']:,.0f}")
    c2.metric("Current Cash", f"${st.session_state.data['cash']:,.0f}")
    c3.metric("Transactions", len(st.session_state.data['portfolio']))

    st.divider()

    st.subheader("Accessibility")
    cb_enabled = st.checkbox("Colorblind Mode (Blue/Orange instead of Green/Red)",
                            value=st.session_state.data.get("colorblind", False))
    if cb_enabled != st.session_state.data.get("colorblind", False):
        st.session_state.data["colorblind"] = cb_enabled
        save_data(st.session_state.data)
        st.rerun()

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
