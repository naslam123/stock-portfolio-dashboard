"""
AI Trading Assistant — Groq (free) primary, Gemini fallback.
Provides portfolio-aware responses using real-time data.
"""

import os
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def _get_key(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def _build_system_prompt(data, holdings_summary, prices_summary):
    """Build system prompt with portfolio context and app navigation."""
    return f"""You are a trading assistant embedded in a Stock Trading Simulator app built with Streamlit. You help users navigate the app, understand their portfolio, and answer trading/investing questions.

CURRENT PORTFOLIO CONTEXT:
- Cash Balance: ${data.get('cash', 0):,.2f}
- Starting Balance: ${data.get('starting_balance', 0):,.2f}
- Account P/L: ${data.get('cash', 0) + sum(h.get('value', 0) for h in holdings_summary.values()) - data.get('starting_balance', 0):,.2f}
- Holdings: {', '.join(f"{tk} ({h['shares']:.1f} shares, ${h['value']:,.0f})" for tk, h in holdings_summary.items()) if holdings_summary else 'None'}
- Watchlist: {', '.join(data.get('watchlist', [])) if data.get('watchlist') else 'Empty'}
- Total Trades: {len(data.get('journal', []))}
- Options Positions: {len(data.get('options', []))}

REAL-TIME PRICES:
{prices_summary if prices_summary else 'No holdings to show prices for.'}

APP NAVIGATION GUIDE:
- Portfolio: View holdings, equity curve, sector allocation, P/L, badges
- Trade: Buy, sell, or short sell S&P 500 stocks with market/limit/stop-loss orders
- Options: Trade calls and puts, view options chains, see P/L payoff diagrams
- Watchlist: Track stocks, set price alerts, quick buy
- Research: Stock charts with technical indicators (RSI, MACD, Bollinger Bands), AI signals (market regime, DCF valuation), multi-timeframe views
- Analytics: Risk metrics (Sharpe, VaR, max drawdown), Monte Carlo simulation, correlation matrix, benchmark vs S&P 500, trade journal stats, AI coaching
- Settings: Dark/light mode, colorblind mode, commissions, account reset

GUIDELINES:
- Be concise (2-4 sentences unless asked to elaborate)
- Reference the user's actual portfolio data when relevant
- For stock-specific questions, use the real-time prices provided
- Explain trading concepts clearly for both beginners and professionals
- If asked about a stock not in their portfolio, still help with general analysis
- Never give definitive buy/sell recommendations — frame as educational
- When helping navigate, reference specific page names"""


def build_portfolio_context(data, get_price_fn, get_holdings_fn):
    """Build holdings and prices context for the system prompt."""
    holdings = get_holdings_fn()
    holdings_summary = {}
    prices_summary_lines = []

    for tk, h in holdings.items():
        try:
            price, chg, pct = get_price_fn(tk)
            val = h["shares"] * price
            holdings_summary[tk] = {"shares": h["shares"], "cost": h["cost"], "value": val}
            pl = val - h["cost"]
            prices_summary_lines.append(
                f"  {tk}: ${price:.2f} ({pct:+.2f}% today) | Holding: {h['shares']:.1f} shares | P/L: ${pl:+,.0f}"
            )
        except Exception:
            holdings_summary[tk] = {"shares": h["shares"], "cost": h["cost"], "value": 0}

    # Also pull watchlist prices
    for tk in data.get("watchlist", []):
        if tk not in holdings:
            try:
                price, chg, pct = get_price_fn(tk)
                prices_summary_lines.append(f"  {tk} (watchlist): ${price:.2f} ({pct:+.2f}% today)")
            except Exception:
                pass

    prices_summary = "\n".join(prices_summary_lines)
    return holdings_summary, prices_summary


def _call_groq(system_prompt, user_msg, history):
    """Call Groq API (free tier: Llama 3.3 70B)."""
    key = _get_key("GROQ_API_KEY")
    if not key:
        return None
    client = Groq(api_key=key)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        role = "assistant" if msg["role"] == "model" else msg["role"]
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt, user_msg, history):
    """Call Gemini API (free tier)."""
    key = _get_key("GEMINI_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    contents = [
        {"role": "user", "parts": [system_prompt + "\n\n(System context — do not repeat. Respond to user below.)"]},
        {"role": "model", "parts": ["Understood. I'm ready to help."]},
    ]
    for msg in history[-10:]:
        contents.append({"role": msg["role"], "parts": [msg["content"]]})
    contents.append({"role": "user", "parts": [user_msg]})
    response = model.generate_content(contents)
    return response.text


def get_ai_response(user_msg, data, holdings_summary, prices_summary):
    """Try Groq first, then Gemini. Returns response text."""
    system_prompt = _build_system_prompt(data, holdings_summary, prices_summary)
    history = st.session_state.get("chat_history", [])

    # Try Groq first (free, fast)
    if GROQ_AVAILABLE:
        try:
            resp = _call_groq(system_prompt, user_msg, history)
            if resp:
                return resp
        except Exception:
            pass

    # Fallback to Gemini
    if GEMINI_AVAILABLE:
        try:
            resp = _call_gemini(system_prompt, user_msg, history)
            if resp:
                return resp
        except Exception:
            pass

    return ("Add a free API key to `.streamlit/secrets.toml`:\n\n"
            "**Groq (recommended):** `GROQ_API_KEY = \"your-key\"`\n"
            "Get one free at [console.groq.com](https://console.groq.com)\n\n"
            "**Or Gemini:** `GEMINI_API_KEY = \"your-key\"`\n"
            "Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
