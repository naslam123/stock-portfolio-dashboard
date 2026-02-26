"""
Configuration constants, theme colors, and CSS for the Trading Simulator.
"""

from sp500_tickers import SP500

TICKER_OPTIONS = [""] + [f"{t} - {n}" for t, n in SP500.items()]
DATA_FILE = "trading_data.json"
DB_FILE = "trading_data.db"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def get_theme_colors(dark: bool, colorblind: bool = False) -> dict:
    """Return color palette. Colorblind mode swaps green/red for blue/orange."""
    if dark:
        colors = {
            "BG": "#0d1117", "BG2": "#161b22",
            "TEXT": "#ffffff", "TEXT2": "#8b949e",
            "BORDER": "#30363d",
            "GREEN": "#3fb950", "RED": "#f85149",
            "BLUE": "#58a6ff", "YELLOW": "#d29922",
        }
    else:
        colors = {
            "BG": "#ffffff", "BG2": "#f6f8fa",
            "TEXT": "#24292f", "TEXT2": "#57606a",
            "BORDER": "#d0d7de",
            "GREEN": "#1a7f37", "RED": "#cf222e",
            "BLUE": "#0969da", "YELLOW": "#9a6700",
        }
    if colorblind:
        colors["GREEN"] = "#2563eb"  # Blue for positive
        colors["RED"] = "#d97706"    # Orange for negative
    return colors


def render_css(colors: dict) -> str:
    """Return the full CSS block for Streamlit theming."""
    c = colors
    return f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: {c['BG']} !important;
}}
section[data-testid="stSidebar"] > div {{
    background: {c['BG2']} !important;
}}
h1, h2, h3 {{
    color: {c['TEXT']} !important;
}}
p, span, label {{
    color: {c['TEXT']} !important;
}}
[data-testid="stMetricValue"] {{
    color: {c['TEXT']} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {c['TEXT2']} !important;
}}
.stTextInput input, .stNumberInput input {{
    background: {c['BG2']} !important;
    border: 1px solid {c['BORDER']} !important;
    color: {c['TEXT']} !important;
}}
.stSelectbox > div > div {{
    background: {c['BG2']} !important;
    border: 1px solid {c['BORDER']} !important;
    color: {c['TEXT']} !important;
}}
.stSelectbox input {{
    color: {c['TEXT']} !important;
}}
[data-testid="stTextInput"] input:disabled {{
    color: {c['TEXT']} !important;
    -webkit-text-fill-color: {c['TEXT']} !important;
}}
.stButton > button {{
    background: {c['BG2']} !important;
    color: {c['TEXT']} !important;
    border: 1px solid {c['BORDER']} !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}}
.stButton > button[kind="primary"] {{
    background: {c['BLUE']} !important;
    color: white !important;
    border: none !important;
}}
hr {{
    border-color: {c['BORDER']} !important;
}}
.stDataFrame {{
    border: 1px solid {c['BORDER']} !important;
}}
.stCheckbox > label {{
    color: {c['TEXT']} !important;
}}
.stRadio > div > label {{
    color: {c['TEXT']} !important;
}}
</style>
"""
