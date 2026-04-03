"""
app.py — Options Pricing Library Dashboard
-------------------------------------------
Entry point for the Streamlit multi-page dashboard.

Run with:
    cd OptionsPricingLibrary
    streamlit run dashboard/app.py

Pages
-----
1. Home           — Live pricer + model comparison
2. Vol Surface    — SVI calibration, 3D surface, RND
3. Greek Surface  — Heatmaps for all 8 Greeks
4. Vega Matrix    — Pillar-by-pillar vol surface sensitivity
5. Model Risk     — Local vol vs BS for barriers, Heston vs BS for Asians
6. P&L Attribution — Taylor decomposition + backtest
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st

st.set_page_config(
    page_title="Options Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;700;800&display=swap');

:root {
    --bg:        #0a0b0d;
    --surface:   #111318;
    --surface2:  #181c23;
    --border:    #1e2530;
    --accent:    #00d4b8;
    --accent2:   #7c6af7;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --text:      #e8eaf0;
    --muted:     #5a6278;
    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--display);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide default header */
header[data-testid="stHeader"] { display: none; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; font-size: 22px !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 12px !important; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.stSlider > div > div { background: var(--border) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-family: var(--mono) !important;
    letter-spacing: 0.05em !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* Code blocks */
code { font-family: var(--mono) !important; background: var(--surface2) !important;
       color: var(--accent) !important; padding: 2px 6px !important; border-radius: 4px !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* DataFrames */
.dataframe { background: var(--surface2) !important; color: var(--text) !important;
             font-family: var(--mono) !important; font-size: 12px !important; }
.dataframe th { background: var(--surface) !important; color: var(--accent) !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; }

/* Success / warning / error banners */
.stAlert { border-radius: 6px !important; font-family: var(--mono) !important; }

/* Remove padding from main */
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation header ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 28px;">
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                    color:#00d4b8; letter-spacing:-0.02em;">◈ OPTIONS</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                    color:#e8eaf0; letter-spacing:-0.02em; margin-top:-4px;">PRICER</div>
        <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a6278;
                    letter-spacing:0.12em; margin-top:8px;">QUANT RESEARCH SUITE</div>
    </div>
    <hr style="border-color:#1e2530; margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown("**Navigate**", help="Select a module from the pages in the sidebar")
    st.page_link("app.py",              label="⌂  Home — Live Pricer",       )
    st.page_link("pages/1_Vol_Surface.py",     label="◎  Vol Surface",       )
    st.page_link("pages/2_Greek_Surface.py",   label="△  Greek Surface",      )
    st.page_link("pages/3_Vega_Matrix.py",     label="⊞  Vega Matrix",       )
    st.page_link("pages/4_Model_Risk.py",      label="⚡  Model Risk",       )
    st.page_link("pages/5_PnL_Attribution.py", label="◑  P&L Attribution",  )

    st.markdown("<hr style='border-color:#1e2530;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:10px; color:#5a6278; line-height:1.8;">
    Models available:<br>
    · Black-Scholes (analytical)<br>
    · Heston (FFT + QE MC)<br>
    · Longstaff-Schwartz MC<br>
    · Crank-Nicolson FD<br>
    · Dupire Local Vol MC<br>
    · Heston Asian MC
    </div>
    """, unsafe_allow_html=True)


# ── Home page ─────────────────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from options_lib import BlackScholes, EuropeanOption, AmericanOption, MarketData, OptionType, Heston, HestonParams
from options_lib.numerics import CrankNicolson, LongstaffSchwartz
from options_lib.models.implied_vol import implied_vol_bs


PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=40, b=40),
)

# Page title
st.markdown("""
<h1 style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800;
           letter-spacing:-0.03em; margin-bottom:4px;">
    Live Option Pricer
</h1>
<p style="font-family:'DM Mono',monospace; font-size:12px; color:#5a6278; margin-bottom:32px;">
    Multi-model pricing · BS / Heston / LSMC / Crank-Nicolson
</p>
""", unsafe_allow_html=True)

# ── Pricer inputs ─────────────────────────────────────────────────────────────
col_params, col_model, col_results = st.columns([1.2, 1, 1.8])

with col_params:
    st.markdown("**Market inputs**")
    S     = st.number_input("Spot (S₀)", value=100.0, step=1.0, format="%.1f")
    K     = st.number_input("Strike (K)", value=100.0, step=1.0, format="%.1f")
    T     = st.number_input("Expiry (years)", value=1.0, step=0.25, min_value=0.01)
    r     = st.number_input("Rate (r)", value=0.05, step=0.005, format="%.3f")
    q     = st.number_input("Div yield (q)", value=0.0, step=0.005, format="%.3f")
    sigma = st.number_input("Volatility (σ)", value=0.20, step=0.01, min_value=0.01, max_value=3.0, format="%.2f")
    opt_type = st.selectbox("Option type", ["Call", "Put"])

with col_model:
    st.markdown("**Model & instrument**")
    instrument_type = st.selectbox("Instrument", ["European", "American"])
    models_selected = st.multiselect(
        "Price with",
        ["Black-Scholes", "Heston", "LSMC", "Crank-Nicolson"],
        default=["Black-Scholes", "Heston"]
    )

    st.markdown("**Heston params**")
    v0    = st.number_input("v₀ (init var)", value=sigma**2, step=0.001, format="%.4f", min_value=0.0001)
    kappa = st.number_input("κ (mean rev)", value=2.0, step=0.1, format="%.1f", min_value=0.01)
    v_bar = st.number_input("v̄ (long-run)", value=sigma**2, step=0.001, format="%.4f", min_value=0.0001)
    xi    = st.number_input("ξ (vol-of-vol)", value=0.30, step=0.05, format="%.2f", min_value=0.01)
    rho   = st.number_input("ρ (corr)", value=-0.70, step=0.05, format="%.2f", min_value=-0.99, max_value=0.99)

# ── Compute prices ────────────────────────────────────────────────────────────
otype = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt   = MarketData(spot=S, rate=r, div_yield=q)

prices = {}
if "Black-Scholes" in models_selected:
    if instrument_type == "European":
        inst = EuropeanOption(K, T, otype)
        prices["Black-Scholes"] = BlackScholes(sigma=sigma).price(inst, mkt)
    else:
        fd = CrankNicolson(sigma=sigma, M=200, N=200)
        inst = AmericanOption(K, T, otype)
        prices["Black-Scholes (FD)"] = fd.price(inst, spot=S, r=r, q=q)

if "Heston" in models_selected and instrument_type == "European":
    try:
        params = HestonParams(v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho)
        inst_eu = EuropeanOption(K, T, otype)
        prices["Heston"] = Heston(params).price(inst_eu, mkt)
    except Exception as e:
        st.warning(f"Heston error: {e}")

if "LSMC" in models_selected:
    try:
        lsmc = LongstaffSchwartz(sigma=sigma, n_paths=30_000, n_steps=50, seed=42)
        inst_am = AmericanOption(K, T, otype)
        result = lsmc.price(inst_am, spot=S, rate=r, div_yield=q)
        prices["LSMC"] = result.price
    except Exception as e:
        st.warning(f"LSMC error: {e}")

if "Crank-Nicolson" in models_selected:
    try:
        fd = CrankNicolson(sigma=sigma, M=200, N=200)
        inst_am = AmericanOption(K, T, otype)
        prices["Crank-Nicolson"] = fd.price(inst_am, spot=S, r=r, q=q)
    except Exception as e:
        st.warning(f"FD error: {e}")

# Also always compute BS for Greeks reference
bs_model = BlackScholes(sigma=sigma)
inst_ref  = EuropeanOption(K, T, otype)
bs_ref    = bs_model.price(inst_ref, mkt)

with col_results:
    st.markdown("**Prices**")
    if prices:
        pcols = st.columns(min(len(prices), 2))
        for idx, (model_name, price) in enumerate(prices.items()):
            with pcols[idx % 2]:
                st.metric(model_name, f"{price:.4f}")
    else:
        st.info("Select at least one model")

    st.markdown("**BS Greeks**")
    g_cols = st.columns(4)
    with g_cols[0]: st.metric("Delta", f"{bs_model.delta(inst_ref, mkt):.4f}")
    with g_cols[1]: st.metric("Gamma", f"{bs_model.gamma(inst_ref, mkt):.4f}")
    with g_cols[2]: st.metric("Vega",  f"{bs_model.vega(inst_ref, mkt):.2f}")
    with g_cols[3]: st.metric("Theta", f"{bs_model.theta(inst_ref, mkt):.4f}/d")
    g2_cols = st.columns(4)
    with g2_cols[0]: st.metric("Vanna",  f"{bs_model.vanna(inst_ref, mkt):.4f}")
    with g2_cols[1]: st.metric("Volga",  f"{bs_model.volga(inst_ref, mkt):.4f}")
    with g2_cols[2]: st.metric("Charm",  f"{bs_model.charm(inst_ref, mkt):.6f}/d")
    with g2_cols[3]: st.metric("Rho",    f"{bs_model.rho(inst_ref, mkt):.4f}")

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
chart1, chart2, chart3 = st.columns(3)

# 1. Price vs Spot
with chart1:
    spots = np.linspace(max(S * 0.5, 1), S * 1.5, 120)
    bs_prices  = [BlackScholes(sigma=sigma).price(EuropeanOption(K, T, otype), MarketData(s, r, q)) for s in spots]
    intrinsics = [max(s - K, 0) if opt_type == "Call" else max(K - s, 0) for s in spots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=bs_prices, name="BS price",
                             line=dict(color='#00d4b8', width=2.5)))
    fig.add_trace(go.Scatter(x=spots, y=intrinsics, name="Intrinsic",
                             line=dict(color='#5a6278', width=1.5, dash='dot')))
    fig.add_vline(x=S, line_color='#7c6af7', line_dash='dash', line_width=1.5,
                  annotation_text=f"S={S}", annotation_font_color='#7c6af7')
    fig.add_vline(x=K, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                  annotation_text=f"K={K}", annotation_font_color='#f59e0b')
    fig.update_layout(**PLOTLY_LAYOUT, title="Price vs Spot", height=280,
                      legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

# 2. Delta vs Spot
with chart2:
    deltas = [BlackScholes(sigma=sigma).delta(EuropeanOption(K, T, otype), MarketData(s, r, q)) for s in spots]
    gammas_scaled = [BlackScholes(sigma=sigma).gamma(EuropeanOption(K, T, otype), MarketData(s, r, q)) * 10 for s in spots]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spots, y=deltas, name="Delta",
                              line=dict(color='#00d4b8', width=2.5)))
    fig2.add_trace(go.Scatter(x=spots, y=gammas_scaled, name="Gamma ×10",
                              line=dict(color='#7c6af7', width=2), yaxis='y2'))
    fig2.add_vline(x=S, line_color='#5a6278', line_dash='dash', line_width=1)
    fig2.update_layout(**PLOTLY_LAYOUT, title="Delta & Gamma vs Spot", height=280,
                       yaxis2=dict(overlaying='y', side='right', gridcolor='#1e2530',
                                   showgrid=False, color='#7c6af7'),
                       legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig2, use_container_width=True)

# 3. Vega vs Vol smile
with chart3:
    sigmas = np.linspace(0.05, 0.80, 100)
    vegas  = [BlackScholes(sv).vega(inst_ref, mkt) for sv in sigmas]
    prices_vs_vol = [BlackScholes(sv).price(inst_ref, mkt) for sv in sigmas]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sigmas * 100, y=prices_vs_vol, name="Price",
                              line=dict(color='#00d4b8', width=2.5)))
    fig3.add_trace(go.Scatter(x=sigmas * 100, y=vegas, name="Vega",
                              line=dict(color='#f59e0b', width=2), yaxis='y2'))
    fig3.add_vline(x=sigma * 100, line_color='#7c6af7', line_dash='dash', line_width=1.5)
    fig3.update_layout(**PLOTLY_LAYOUT, title="Price & Vega vs σ (%)", height=280,
                       xaxis_title="Implied Vol (%)",
                       yaxis2=dict(overlaying='y', side='right', gridcolor='#1e2530',
                                   showgrid=False, color='#f59e0b'),
                       legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig3, use_container_width=True)

# ── Model comparison table ────────────────────────────────────────────────────
if len(prices) > 1:
    st.markdown("**Model comparison**")
    import pandas as pd
    ref_price = list(prices.values())[0]
    rows = []
    for name, p in prices.items():
        rows.append({
            "Model": name,
            "Price": f"{p:.4f}",
            "vs BS": f"{p - bs_ref:+.4f}",
            "% diff": f"{(p / bs_ref - 1) * 100:+.2f}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
