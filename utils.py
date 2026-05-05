# ─────────────────────────────────────────────────────────────────────────────
# utils.py  —  Data Helpers · Symbol Resolution · Shared Chart Layout
#
# Imported by app.py:
#   from utils import to_period, get_symbol_from_name, fetch_stock_data, CHART_LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

import requests
import streamlit as st
import yfinance as yf


# ──────────────────────────────────────────────
# Timeline helper
# ──────────────────────────────────────────────

def to_period(unit: str, val: int) -> str:
    """
    Convert a human-readable timeline (e.g. 3 Months) to a
    yfinance period string (e.g. '3mo').
    """
    days = {"Days": val, "Weeks": val * 7, "Months": val * 30, "Years": val * 365}[unit]
    if days <= 30:    return "1mo"
    if days <= 90:    return "3mo"
    if days <= 180:   return "6mo"
    if days <= 365:   return "1y"
    if days <= 730:   return "2y"
    if days <= 1095:  return "3y"
    return "5y"


# ──────────────────────────────────────────────
# Symbol resolver
# ──────────────────────────────────────────────

def get_symbol_from_name(query: str, market_type: str) -> str:
    """
    Resolve a company name or partial ticker to a full Yahoo Finance
    ticker symbol, respecting the selected market/exchange.

    Falls back to a suffix-based guess if the Yahoo search API is
    unavailable or returns no matching quote.
    """
    query = query.strip()
    url   = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            if quotes:
                # Prefer exchange-matched quote
                if market_type == "Indian Market (NSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "NSI" or str(q.get("symbol")).endswith(".NS")),
                        None,
                    )
                    if match: return match["symbol"]

                elif market_type == "Indian Market (BSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "BSE" or str(q.get("symbol")).endswith(".BO")),
                        None,
                    )
                    if match: return match["symbol"]

                elif market_type == "US Market":
                    match = next(
                        (q for q in quotes if q.get("exchange") in ["NMS", "NYQ"]),
                        None,
                    )
                    if match: return match["symbol"]

                elif market_type == "UK Market (LSE)":
                    match = next(
                        (q for q in quotes if str(q.get("symbol")).endswith(".L")),
                        None,
                    )
                    if match: return match["symbol"]

                elif market_type == "Cryptocurrency":
                    match = next(
                        (q for q in quotes if q.get("quoteType") == "CRYPTOCURRENCY"),
                        None,
                    )
                    if match: return match["symbol"]

                # Auto-detect: return top result
                return quotes[0]["symbol"]

    except Exception:
        pass

    # Fallback: append exchange suffix
    ticker = query.upper()
    if market_type == "Indian Market (NSE)"  and not ticker.endswith(".NS"):  return f"{ticker}.NS"
    if market_type == "Indian Market (BSE)"  and not ticker.endswith(".BO"):  return f"{ticker}.BO"
    if market_type == "UK Market (LSE)"      and not ticker.endswith(".L"):   return f"{ticker}.L"
    if market_type == "Cryptocurrency"       and not ticker.endswith("-USD"): return f"{ticker}-USD"
    return ticker


# ──────────────────────────────────────────────
# Data fetcher  (cached 1 h)
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str):
    """
    Download OHLCV data from Yahoo Finance and return a DataFrame.
    Results are cached for 1 hour to avoid redundant API calls.
    """
    return yf.download(ticker, period=period, auto_adjust=True, progress=False)


# ──────────────────────────────────────────────
# Shared Plotly layout  (dark, professional)
# ──────────────────────────────────────────────

CHART_LAYOUT = dict(
    height=460,
    margin=dict(l=8, r=8, t=28, b=8),

    # Transparent paper so the CSS card background shows through
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,35,0.95)",

    font=dict(family="DM Sans, sans-serif", color="#cbd5e1", size=12),

    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(255,255,255,0.15)",
        mirror=True,
        tickfont=dict(color="#cbd5e1", size=12),
        rangeslider=dict(visible=False),
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(255,255,255,0.15)",
        mirror=True,
        tickfont=dict(color="#e2e8f0", size=12),
    ),

    legend=dict(
        bgcolor="rgba(15,23,42,0.85)",
        bordercolor="rgba(255,255,255,0.12)",
        borderwidth=1,
        font=dict(color="#e2e8f0", size=12),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),

    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="rgba(37,99,235,0.5)",
        font=dict(family="DM Sans, sans-serif", color="#f1f5f9", size=12),
    ),
)
