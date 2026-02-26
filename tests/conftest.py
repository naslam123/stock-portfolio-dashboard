"""Shared fixtures and Streamlit mock for test suite."""

import sys
import types
import os
import pytest  # noqa: F401 — used by fixtures


# --------------- Streamlit mock ---------------
# Must be installed BEFORE any app module imports st

def _install_streamlit_mock():
    """Install a fake 'streamlit' module so app code can import it."""
    if "streamlit" in sys.modules:
        return  # real streamlit available, nothing to do

    st = types.ModuleType("streamlit")
    st.session_state = types.SimpleNamespace(data=None)
    st.secrets = {}
    st.cache_data = lambda f=None, **kw: f if f else (lambda fn: fn)
    st.cache_resource = lambda f=None, **kw: f if f else (lambda fn: fn)
    st.toast = lambda *a, **kw: None
    st.sidebar = types.SimpleNamespace(
        selectbox=lambda *a, **kw: None,
        number_input=lambda *a, **kw: 0,
    )
    sys.modules["streamlit"] = st


_install_streamlit_mock()


# --------------- Fixtures ---------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary DB path and clean up after."""
    db_path = str(tmp_path / "test_trading.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def tmp_json(tmp_path):
    """Provide a temporary JSON path."""
    return str(tmp_path / "test_trading.json")


@pytest.fixture
def sample_data():
    """Return a minimal valid account data dict."""
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
        "portfolio_history": [],
        "badges": [],
        "price_alerts": {},
        "colorblind": False,
        "rebalance_targets": {},
    }
