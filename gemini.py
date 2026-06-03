import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
import requests
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    /* ── Global dark background ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #060d1f !important;
        color: #f8fafc !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        background-color: #060d1f !important;
    }
    section[data-testid="stMain"] > div {
        background-color: #060d1f !important;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #0a1628 0%, #0f2044 50%, #0a1628 100%);
        border: 1px solid rgba(251,191,36,0.30);
        border-radius: 20px;
        padding: 2.4rem 3rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(251,191,36,0.06), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -60px; left: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(251,191,36,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(251,191,36,0.3);
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0;
    }

    /* ── Inputs ── */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div > div {
        background: #0d1b35 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: rgba(251,191,36,0.50) !important;
        box-shadow: 0 0 0 3px rgba(251,191,36,0.10) !important;
    }
    label, .stSelectbox label, [data-testid="stWidgetLabel"], [data-testid="stRadio"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    /* ── Predict button ── */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 4px 20px rgba(251,191,36,0.35) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(251,191,36,0.50) !important;
    }

    /* ── Metric cards — bright white professional ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #0d1b35, #0f2248) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.06) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 32px rgba(251,191,36,0.15), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        border-color: rgba(251,191,36,0.30) !important;
    }

    /* Metric LABEL */
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"],
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] *,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
    }

    /* Metric VALUE — MAXIMUM brightness */
    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    div[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
    div[data-testid="metric-container"] [data-testid="stMetricValue"] div,
    div[data-testid="metric-container"] [data-testid="stMetricValue"] span,
    div[data-testid="metric-container"] [data-testid="stMetricValue"] p,
    div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
        color: #ffffff !important;
        font-size: 1.80rem !important;
        font-weight: 800 !important;
        font-family: 'Space Mono', monospace !important;
        letter-spacing: -0.01em !important;
        text-shadow: 0 0 8px rgba(255,255,255,1.00), 0 0 20px rgba(255,255,255,0.80), 0 0 40px rgba(200,220,255,0.60), 0 2px 4px rgba(0,0,0,0.80) !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        filter: brightness(1.4) !important;
    }

    /* Metric DELTA */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"],
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] > div,
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] * {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        color: #fbbf24 !important;
        -webkit-text-fill-color: #fbbf24 !important;
        opacity: 1 !important;
    }

    /* ── Ticker heading ── */
    .ticker-head {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .ticker-badge {
        background: rgba(251,191,36,0.15);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.40);
        border-radius: 8px;
        padding: 3px 12px;
        font-size: 0.9rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        letter-spacing: 0.05em;
    }

    /* ── Section heading ── */
    .section-head {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-left: 3px solid #fbbf24;
        padding-left: 0.75rem;
    }

    /* ── Alert / success / error boxes ── */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: #0d1b35 !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(251,191,36,0.35) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(251,191,36,0.10) !important;
        border-color: rgba(251,191,36,0.60) !important;
    }

    /* ── Spinner text ── */
    [data-testid="stSpinner"] p { color: #94a3b8 !important; }

    /* ── Hide streamlit default header branding ── */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📈 Stock Price Predictor</h1>
    <p>· Live market data &nbsp;·&nbsp; LSTM Neural Network &nbsp;·&nbsp; 30-day outlook</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
with st.container():
    c1, c2 = st.columns(2)

    with c1:
        market = st.selectbox("🌍 Market / Asset Type", [
            "Auto-detect / Global",
            "US Market",
            "Indian Market (NSE)",
            "Indian Market (BSE)",
            "UK Market (LSE)",
            "Cryptocurrency"
        ])
        timeline_unit = st.selectbox("📅 Timeline Unit", ["Days", "Weeks", "Months", "Years"])

    with c2:
        user_input = st.text_input(
            "🔎 Search Company Name or Symbol",
            placeholder="e.g., Amazon, Apple, AAPL, Bitcoin, TCS"
        )
        timeline_map = {
            "Days":   [7, 14, 30, 60, 90],
            "Weeks":  [1, 2, 4, 8, 12, 26, 52],
            "Months": [1, 3, 6, 12, 24],
            "Years":  [1, 2, 3, 5],
        }
        timeline_val = st.selectbox(f"Count ({timeline_unit})", timeline_map[timeline_unit])

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Fetch Data & Predict", use_container_width=True)

# ─────────────────────────────────────────────
# SHARED PLOTLY LAYOUT
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    height=460,
    margin=dict(l=8, r=8, t=28, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,13,31,0.98)",
    font=dict(family="DM Sans, sans-serif", color="#f8fafc", size=13),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.07)",
        gridwidth=1, zeroline=False,
        showline=True, linewidth=1, linecolor="rgba(255,255,255,0.18)", mirror=True,
        tickfont=dict(color="#cbd5e1", size=11),
        rangeslider=dict(visible=False),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.07)",
        gridwidth=1, zeroline=False,
        showline=True, linewidth=1, linecolor="rgba(255,255,255,0.18)", mirror=True,
        tickfont=dict(color="#cbd5e1", size=11),
    ),
    legend=dict(
        bgcolor="rgba(6,13,31,0.92)",
        bordercolor="rgba(251,191,36,0.30)",
        borderwidth=1,
        font=dict(color="#f8fafc", size=12),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0d1b35",
        bordercolor="#fbbf24",
        font=dict(family="DM Sans, sans-serif", color="#ffffff", size=14),
    ),
)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def to_period(unit, val):
    days = {"Days": val, "Weeks": val * 7, "Months": val * 30, "Years": val * 365}[unit]
    if days <= 30:   return "1mo"
    if days <= 90:   return "3mo"
    if days <= 180:  return "6mo"
    if days <= 365:  return "1y"
    if days <= 730:  return "2y"
    if days <= 1095: return "3y"
    return "5y"

@st.cache_data(ttl=86400, show_spinner=False)
def get_symbol_from_name(query, market_type):
    query = query.strip()
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            if quotes:
                if market_type == "Indian Market (NSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "NSI" or str(q.get("symbol", "")).endswith(".NS")),
                        None,
                    )
                    if match: return match["symbol"]
                elif market_type == "Indian Market (BSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "BSE" or str(q.get("symbol", "")).endswith(".BO")),
                        None,
                    )
                    if match: return match["symbol"]
                elif market_type == "US Market":
                    match = next((q for q in quotes if q.get("exchange") in ["NMS", "NYQ"]), None)
                    if match: return match["symbol"]
                elif market_type == "UK Market (LSE)":
                    match = next((q for q in quotes if str(q.get("symbol", "")).endswith(".L")), None)
                    if match: return match["symbol"]
                elif market_type == "Cryptocurrency":
                    match = next((q for q in quotes if q.get("quoteType") == "CRYPTOCURRENCY"), None)
                    if match: return match["symbol"]
                return quotes[0]["symbol"]
    except Exception:
        pass

    ticker = query.upper()
    if market_type == "Indian Market (NSE)" and not ticker.endswith(".NS"): return f"{ticker}.NS"
    if market_type == "Indian Market (BSE)" and not ticker.endswith(".BO"): return f"{ticker}.BO"
    if market_type == "UK Market (LSE)"     and not ticker.endswith(".L"):  return f"{ticker}.L"
    if market_type == "Cryptocurrency"      and not ticker.endswith("-USD"):return f"{ticker}-USD"
    return ticker

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SeqDataset(Dataset):
    def __init__(self, X, seq_len):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.X[i + self.seq_len]


@st.cache_data(ttl=3600, show_spinner=False)
def train_and_predict(prices_tuple, future_days=30):
    prices = np.array(prices_tuple)
    SEQ_LEN = min(30, max(5, len(prices) // 4))
    EPOCHS  = 30
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(prices.reshape(-1, 1))
    split   = int(len(scaled) * 0.85)
    tr_ds   = SeqDataset(scaled[:split].flatten(), SEQ_LEN)
    tr_dl   = DataLoader(tr_ds, batch_size=32, shuffle=False)

    model   = LSTMModel()
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb.unsqueeze(-1)).squeeze(), yb.squeeze())
            loss.backward()
            opt.step()

    last_seq   = scaled[-SEQ_LEN:].flatten().tolist()
    future_raw = []
    model.eval()

    with torch.no_grad():
        for _ in range(future_days):
            x   = torch.tensor(last_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out = model(x).item()
            future_raw.append(out)
            last_seq.append(out)

    return scaler.inverse_transform(np.array(future_raw).reshape(-1, 1)).flatten()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a stock symbol or company name.")
        st.stop()
        
    st.session_state['active_pred'] = True
    st.session_state['p_user_input'] = user_input
    st.session_state['p_market'] = market
    st.session_state['p_period'] = to_period(timeline_unit, timeline_val)

if st.session_state.get('active_pred'):
    p_user_input = st.session_state['p_user_input']
    p_market = st.session_state['p_market']
    p_period = st.session_state['p_period']

    with st.spinner("Resolving symbol and fetching live market data…"):
        fetch_ticker = get_symbol_from_name(p_user_input, p_market)
        df = fetch_stock_data(fetch_ticker, p_period)

    if df is None or df.empty or "Close" not in df.columns:
        st.error(
            f"Could not retrieve valid historical data for **{p_user_input}** (Resolved Ticker: `{fetch_ticker}`). "
            "Please check the spelling or ensure the market selection is correct."
        )
        st.session_state['active_pred'] = False
        st.stop()

    close = df["Close"].dropna().values.flatten().astype(float)
    dates = df.index[: len(close)]

    if len(close) < 10:
        st.error(f"Not enough historical data to train the model for `{fetch_ticker}`. Try a longer timeline.")
        st.stop()

    if "US" in p_market or p_market == "Cryptocurrency": currency = "USD"
    elif "Indian" in p_market:                           currency = "INR"
    elif "UK" in p_market:                               currency = "GBP"
    else:                                                currency = ""

    try:
        info     = yf.Ticker(fetch_ticker).info
        name     = info.get("longName") or info.get("shortName") or p_user_input.upper()
        currency = info.get("currency", currency)
    except Exception:
        name = p_user_input.upper()

    with st.spinner("Training LSTM model on historical data…"):
        future_prices = train_and_predict(tuple(close))

    future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=30)

    # ── Stats ──────────────────────────────────
    curr        = close[-1]
    chg         = ((close[-1] - close[-2]) / close[-2]) * 100 if len(close) > 1 else 0
    trend       = ((future_prices[-1] - curr) / curr) * 100
    week_close  = close[-5:]  if len(close) >= 5  else close
    month_close = close[-21:] if len(close) >= 21 else close
    week_high   = week_close.max()
    week_low    = week_close.min()
    month_high  = month_close.max()

    # ── Heading ────────────────────────────────
    st.markdown(
        f'<div class="ticker-head">{name}'
        f'<span class="ticker-badge">{fetch_ticker}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Metric cards ───────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price",  f"{currency} {curr:,.2f}",       f"{chg:+.2f}% today")
    m2.metric("Week High",      f"{currency} {week_high:,.2f}",  f"{((week_high - curr) / curr) * 100:+.2f}% vs now")
    m3.metric("Week Low",       f"{currency} {week_low:,.2f}",   f"{((week_low  - curr) / curr) * 100:+.2f}% vs now")
    m4.metric("Month High",     f"{currency} {month_high:,.2f}", f"{((month_high - curr) / curr) * 100:+.2f}% vs now")

    st.write("")

    # ── Forecast alert ─────────────────────────
    if future_prices[-1] > curr:
        st.success(
            f"**▲ Expected to GO UP** · {trend:+.2f}% over 30 days · "
            f"Target: {currency} {future_prices[-1]:,.2f}"
        )
    else:
        st.error(
            f"**▼ Expected to FALL DOWN** · {trend:+.2f}% over 30 days · "
            f"Target: {currency} {future_prices[-1]:,.2f}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical Chart ───────────────────────
    st.markdown("<div class='section-head'>📊 Historical Price Movement</div>", unsafe_allow_html=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=dates, y=close,
        mode="lines",
        name="Close Price",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: " + currency + " %{y:,.2f}<extra></extra>",
    ))
    layout1 = {**CHART_LAYOUT}
    layout1["yaxis"] = {**layout1["yaxis"], "title": dict(text=f"Price ({currency})", font=dict(color="#64748b", size=11))}
    fig1.update_layout(**layout1)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # ── Forecast Chart ─────────────────────────
    st.markdown("<div class='section-head'>🔮 30-Day Forecast Prediction</div>", unsafe_allow_html=True)

    fig2      = go.Figure()
    hist_x    = dates[-60:]
    hist_y    = close[-60:]
    is_up     = future_prices[-1] > curr
    color_pred = "#22c55e" if is_up else "#ef4444"
    fill_rgba  = "rgba(34,197,94,0.08)"  if is_up else "rgba(239,68,68,0.08)"
    glow_rgba  = "rgba(34,197,94,0.22)"  if is_up else "rgba(239,68,68,0.22)"

    fig2.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode="lines", name="Recent Market",
        line=dict(color="#3b82f6", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: " + currency + " %{y:,.2f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines",
        line=dict(color=glow_rgba, width=9),
        showlegend=False, hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines+markers", name="Forecast Trend",
        line=dict(color=color_pred, width=2.5, dash="dash"),
        marker=dict(size=5, symbol="circle", color=color_pred,
                    line=dict(width=1.5, color="rgba(255,255,255,0.25)")),
        fill="tozeroy", fillcolor=fill_rgba,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: " + currency + " %{y:,.2f}<extra></extra>",
    ))

    today_x = str(dates[-1].date())
    fig2.add_shape(
        type="line", x0=today_x, x1=today_x, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(148,163,184,0.45)", width=1.5, dash="dot"),
    )
    fig2.add_annotation(
        x=today_x, y=0.97, yref="paper", text="Today",
        showarrow=False, xanchor="left",
        font=dict(color="#94a3b8", size=11, family="DM Sans"),
    )

    layout2 = {**CHART_LAYOUT}
    layout2["yaxis"] = {**layout2["yaxis"], "title": dict(text=f"Price ({currency})", font=dict(color="#e2e8f0", size=13))}
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # ── Forecast Table ─────────────────────────
    st.markdown("<div class='section-head'>📅 Predicted Prices — Next 30 Days</div>", unsafe_allow_html=True)

    pred_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        f"Predicted ({currency})": np.round(future_prices, 2),
        "Change from today": [
            f"{'▲' if p >= curr else '▼'} {((p - curr) / curr) * 100:+.2f}%"
            for p in future_prices
        ],
    })

    def style_trend(val):
        if "▲" in str(val): return "color: #22c55e; font-weight: bold;"
        if "▼" in str(val): return "color: #ef4444; font-weight: bold;"
        return ""

    styled_df = pred_df.set_index("Date").style.map(style_trend, subset=["Change from today"])
    st.dataframe(styled_df, use_container_width=True, height=350)

    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"{fetch_ticker}_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Lumpsum & SIP Calculator ────────────────
    st.markdown("<div class='section-head'>💰 Long-Term Returns Calculator</div>", unsafe_allow_html=True)
    st.markdown("Calculate potential long-term returns for this asset based on the 30-day forecasted trend.")

    calc_type = st.radio("Investment Mode", ["Lumpsum", "SIP (Monthly)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    calc_c1, calc_c2, calc_c3 = st.columns(3)

    with calc_c1:
        if calc_type == "Lumpsum":
            investment = st.number_input(f"Total Investment ({currency})", min_value=100.0, value=10000.0, step=1000.0)
        else:
            investment = st.number_input(f"Monthly Investment ({currency})", min_value=100.0, value=1000.0, step=500.0)

    with calc_c2:
        expected_return = 15.0 if trend >= 0 else -5.0
        st.text_input(
            "Expected Annual Return (%)", 
            value=f"{expected_return}% (Based on Trend)", 
            disabled=True
        )

    with calc_c3:
        duration_years = st.selectbox("Investment Duration (Years)", [1, 2, 3, 5, 10, 15, 20, 25, 30], index=3)

    # Mathematical Calculations
    if calc_type == "Lumpsum":
        total_invested = investment
        rate = expected_return / 100
        total_value = investment * ((1 + rate) ** duration_years)
        est_returns = total_value - total_invested
    else: # SIP Calculation
        total_invested = investment * 12 * duration_years
        
        if expected_return == 0:
            total_value = total_invested
            est_returns = 0.0
        else:
            monthly_rate = expected_return / 12 / 100
            months = duration_years * 12
            total_value = investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
            est_returns = total_value - total_invested

    # Metric Display
    st.markdown("<br>", unsafe_allow_html=True)
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Total Invested", f"{currency} {total_invested:,.2f}")
    
    sign = "+" if est_returns >= 0 else ""
    res_c2.metric("Estimated Returns", f"{currency} {est_returns:,.2f}", f"{sign}{expected_return}% CAGR")
    res_c3.metric("Total Projected Value", f"{currency} {total_value:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if est_returns > 0:
        st.success(
            f"### PROFIT! 🚀\n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years** "
            f"at an expected rate of **{expected_return}%**, your money is projected to grow to **{currency} {total_value:,.2f}**.\n\n"
            f"**Total Profit Earned:** {currency} {est_returns:,.2f}"
        )
    elif est_returns < 0:
        st.error(
            f"### LOSS! ⚠️\n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years** "
            f"at an expected rate of **{expected_return}%**, your money is projected to shrink to **{currency} {total_value:,.2f}**.\n\n"
            f"**Total Loss Incurred:** {currency} {abs(est_returns):,.2f}"
        )
    else:
        st.info(
            f"### BREAK EVEN ⚖️\n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years**, "
            f"your money will not grow or shrink.\n\n"
            f"**Final Value:** {currency} {total_value:,.2f}"
        )