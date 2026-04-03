"""
pages/1_Vol_Surface.py
-----------------------
Volatility surface page:
  · Fetch live option chain from yfinance
  · Calibrate SVI per expiry slice
  · 3D interactive vol surface
  · IV smile per expiry
  · Risk-neutral density (Breeden-Litzenberger)
  · Calendar & butterfly arbitrage checks
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from options_lib.market_data.option_chain import fetch_option_chain
from options_lib.market_data.vol_surface import calibrate_vol_surface, SVIParams, VolSurface

# ── Layout constants ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLORS = ['#00d4b8', '#7c6af7', '#f59e0b', '#ef4444', '#10b981', '#3b82f6']

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">
    Volatility Surface
</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    SVI calibration · Arbitrage checks · Risk-neutral density
</p>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Surface controls")
    mode = st.radio("Data source", ["Live (yfinance)", "Synthetic (demo)"])

    if mode == "Live (yfinance)":
        ticker = st.text_input("Ticker", value="SPY")
        rate   = st.number_input("Risk-free rate", value=0.05, step=0.005, format="%.3f")
        div_q  = st.number_input("Dividend yield", value=0.013, step=0.001, format="%.3f")
        n_exp  = st.slider("Max expiries", 2, 8, 4)
        fetch_btn = st.button("⟳  Fetch & Calibrate")
    else:
        spot_s = st.number_input("Spot", value=100.0)
        rate   = st.number_input("Risk-free rate", value=0.05, step=0.005, format="%.3f")
        fetch_btn = True   # always show synthetic

    st.markdown("---")
    show_3d     = st.checkbox("3D Surface", value=True)
    show_smile  = st.checkbox("IV Smiles",  value=True)
    show_rnd    = st.checkbox("Risk-Neutral Density", value=True)
    show_arb    = st.checkbox("Arbitrage Checks", value=True)


# ── Helper: build synthetic surface ─────────────────────────────────────────
def make_synthetic_surface(spot=100.0, rate=0.05):
    slices = {
        '1M': SVIParams(a=0.012, b=0.10, rho=-0.45, m=0.0, sigma=0.08, expiry=1/12),
        '3M': SVIParams(a=0.016, b=0.12, rho=-0.50, m=0.0, sigma=0.10, expiry=0.25),
        '6M': SVIParams(a=0.019, b=0.14, rho=-0.55, m=0.0, sigma=0.12, expiry=0.50),
        '1Y': SVIParams(a=0.022, b=0.16, rho=-0.60, m=0.0, sigma=0.15, expiry=1.00),
        '2Y': SVIParams(a=0.026, b=0.18, rho=-0.62, m=0.0, sigma=0.18, expiry=2.00),
    }
    forwards = {d: spot * np.exp(rate * s.expiry) for d, s in slices.items()}
    return VolSurface(svi_slices=slices, forwards=forwards, spot=spot, rate=rate, ticker='DEMO')


@st.cache_data(ttl=300)
def fetch_and_calibrate(ticker, rate, div_q, n_exp):
    chain   = fetch_option_chain(ticker, rate=rate, div_yield=div_q,
                                  n_expiries=n_exp, min_volume=5,
                                  max_spread_pct=0.40)
    surface = calibrate_vol_surface(chain, verbose=False)
    return chain, surface


# ── Main flow ─────────────────────────────────────────────────────────────────
surface = None
chain   = None

if mode == "Synthetic (demo)":
    surface = make_synthetic_surface(spot_s, rate)

elif mode == "Live (yfinance)" and fetch_btn:
    with st.spinner(f"Fetching {ticker} options and calibrating SVI..."):
        try:
            chain, surface = fetch_and_calibrate(ticker, rate, div_q, n_exp)
            st.success(f"Calibrated {len(surface.svi_slices)} expiry slices "
                       f"from {len(chain.quotes)} quotes.")
        except Exception as e:
            st.error(f"Error: {e}")

if surface is None:
    st.info("Configure the sidebar and click **Fetch & Calibrate** to load data.")
    st.stop()

spot    = surface.spot
dates   = surface.expiry_dates
expiries = surface.expiries

# ── Summary metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
atm_iv_short = float(surface.svi_slices[dates[0]].implied_vol(np.array([0.0]))[0])
atm_iv_long  = float(surface.svi_slices[dates[-1]].implied_vol(np.array([0.0]))[0])
skew_short   = float(surface.svi_slices[dates[0]].rho)
cal_check    = surface.check_calendar_arbitrage()
but_check    = surface.check_butterfly_arbitrage()

m1.metric("Spot",          f"{spot:.2f}")
m2.metric("ATM Vol (near)", f"{atm_iv_short:.1%}")
m3.metric("ATM Vol (far)",  f"{atm_iv_long:.1%}")
m4.metric("Near-term skew ρ", f"{skew_short:.3f}")
m5.metric("Slices calibrated", f"{len(dates)}")

st.markdown("---")

# ── 3D Vol Surface ─────────────────────────────────────────────────────────────
if show_3d:
    st.markdown("### 3D Implied Volatility Surface")

    # Build the grid
    k_grid  = np.linspace(-0.4, 0.4, 60)     # log-moneyness
    T_grid  = np.linspace(expiries.min(), expiries.max(), 50)
    KK, TT  = np.meshgrid(k_grid, T_grid)
    IV_grid = np.zeros_like(KK)

    for i, T_val in enumerate(T_grid):
        F = spot * np.exp(rate * T_val)
        K_strikes = F * np.exp(k_grid)
        for j, Kj in enumerate(K_strikes):
            try:
                IV_grid[i, j] = surface.implied_vol(Kj, T_val) * 100
            except Exception:
                IV_grid[i, j] = np.nan

    # Convert log-moneyness to % OTM for display
    K_display = np.exp(k_grid) * 100 - 100  # % from ATM

    fig_3d = go.Figure(data=[go.Surface(
        x=K_display,
        y=T_grid,
        z=IV_grid,
        colorscale=[
            [0.0,  '#0f172a'],
            [0.2,  '#1e3a5f'],
            [0.4,  '#0f6e56'],
            [0.6,  '#00d4b8'],
            [0.8,  '#7c6af7'],
            [1.0,  '#f59e0b'],
        ],
        colorbar=dict(
            title=dict(text='IV %', font=dict(color='#e8eaf0', size=11)),
            tickfont=dict(color='#e8eaf0', size=10),
            thickness=12,
            len=0.6,
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor='#fff', project_z=False)
        ),
        opacity=0.92,
        hovertemplate='Moneyness: %{x:.1f}%<br>Expiry: %{y:.2f}yr<br>IV: %{z:.1f}%<extra></extra>',
    )])

    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(10,11,13,1)',
            xaxis=dict(title='Moneyness (%)', gridcolor='#1e2530', color='#5a6278', titlefont=dict(size=11)),
            yaxis=dict(title='Expiry (yr)', gridcolor='#1e2530', color='#5a6278', titlefont=dict(size=11)),
            zaxis=dict(title='IV (%)', gridcolor='#1e2530', color='#5a6278', titlefont=dict(size=11)),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        font=dict(family='DM Mono', color='#e8eaf0', size=11),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── IV Smiles per expiry ───────────────────────────────────────────────────────
if show_smile:
    st.markdown("### Implied Volatility Smile")

    k_fine = np.linspace(-0.45, 0.45, 200)
    fig_sm = go.Figure()

    for idx, (date, svi) in enumerate(surface.svi_slices.items()):
        F  = surface.forwards.get(date, spot * np.exp(rate * svi.expiry))
        iv = svi.implied_vol(k_fine) * 100
        K_pct = np.exp(k_fine) * 100 - 100

        fig_sm.add_trace(go.Scatter(
            x=K_pct, y=iv,
            name=f"{date} (T={svi.expiry:.2f})",
            line=dict(color=COLORS[idx % len(COLORS)], width=2.5),
            hovertemplate='Moneyness: %{x:.1f}%<br>IV: %{y:.2f}%<extra></extra>',
        ))

        # Add market quotes if available
        if chain:
            slice_quotes = chain.get_slice(date)
            if slice_quotes:
                mq_k  = [np.log(q.strike / F) for q in slice_quotes]
                mq_iv = [q.iv * 100 for q in slice_quotes]
                mq_k_pct = [(np.exp(k) - 1) * 100 for k in mq_k]
                fig_sm.add_trace(go.Scatter(
                    x=mq_k_pct, y=mq_iv,
                    mode='markers',
                    marker=dict(color=COLORS[idx % len(COLORS)], size=5, symbol='circle'),
                    name=f"{date} market",
                    hovertemplate='K: %{x:.1f}%<br>IV: %{y:.2f}%<extra></extra>',
                    showlegend=False,
                ))

    fig_sm.add_vline(x=0, line_color='#5a6278', line_dash='dash', line_width=1,
                     annotation_text='ATM', annotation_font_color='#5a6278')
    fig_sm.update_layout(
        **PLOTLY_LAYOUT,
        title="IV Smile by Expiry (SVI fit vs market quotes)",
        xaxis_title="Moneyness (% from ATM)",
        yaxis_title="Implied Vol (%)",
        height=380,
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', x=0.01, y=0.99),
    )
    st.plotly_chart(fig_sm, use_container_width=True)

    # Skew & ATM vol term structure
    ts_col1, ts_col2 = st.columns(2)

    with ts_col1:
        atm_vols  = [float(svi.implied_vol(np.array([0.0]))[0]) * 100 for svi in surface.svi_slices.values()]
        fig_atm   = go.Figure()
        fig_atm.add_trace(go.Scatter(
            x=list(expiries), y=atm_vols,
            mode='lines+markers',
            line=dict(color='#00d4b8', width=2.5),
            marker=dict(size=7, color='#00d4b8'),
            fill='tozeroy',
            fillcolor='rgba(0,212,184,0.08)',
        ))
        fig_atm.update_layout(**PLOTLY_LAYOUT, title="ATM Vol Term Structure",
                              xaxis_title="Expiry (yr)", yaxis_title="ATM IV (%)", height=260)
        st.plotly_chart(fig_atm, use_container_width=True)

    with ts_col2:
        rhos  = [svi.rho for svi in surface.svi_slices.values()]
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Bar(
            x=dates, y=rhos,
            marker_color=['#ef4444' if r < 0 else '#10b981' for r in rhos],
            text=[f"{r:.3f}" for r in rhos],
            textposition='outside',
            textfont=dict(size=10, color='#e8eaf0'),
        ))
        fig_rho.update_layout(**PLOTLY_LAYOUT, title="Skew ρ (SVI) by Expiry",
                              yaxis_title="ρ", height=260)
        st.plotly_chart(fig_rho, use_container_width=True)


# ── Risk-Neutral Density ───────────────────────────────────────────────────────
if show_rnd:
    st.markdown("### Risk-Neutral Density  *(Breeden-Litzenberger)*")

    rnd_date = st.selectbox("Expiry for RND", dates, index=min(2, len(dates)-1))
    K_grid_rnd, density = surface.risk_neutral_density(rnd_date)

    svi_T   = surface.svi_slices[rnd_date]
    fwd     = surface.forwards.get(rnd_date, spot * np.exp(rate * svi_T.expiry))
    atm_iv  = float(svi_T.implied_vol(np.array([0.0]))[0])

    # Lognormal benchmark
    from scipy.stats import lognorm
    sigma_ln = atm_iv * np.sqrt(svi_T.expiry)
    mu_ln    = np.log(fwd) - 0.5 * sigma_ln**2
    ln_density = lognorm.pdf(K_grid_rnd, s=sigma_ln, scale=np.exp(mu_ln))
    norm_factor = np.trapezoid(ln_density, K_grid_rnd)
    if norm_factor > 0:
        ln_density /= norm_factor

    fig_rnd = go.Figure()
    fig_rnd.add_trace(go.Scatter(
        x=K_grid_rnd, y=density,
        name="Risk-neutral (market)",
        line=dict(color='#00d4b8', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,184,0.12)',
    ))
    fig_rnd.add_trace(go.Scatter(
        x=K_grid_rnd, y=ln_density,
        name="Lognormal (BS benchmark)",
        line=dict(color='#5a6278', width=1.5, dash='dash'),
    ))
    fig_rnd.add_vline(x=spot, line_color='#7c6af7', line_dash='dash', line_width=1.5,
                      annotation_text='Spot', annotation_font_color='#7c6af7')
    fig_rnd.add_vline(x=fwd, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                      annotation_text='Forward', annotation_font_color='#f59e0b')
    fig_rnd.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Risk-Neutral Density — {rnd_date}",
        xaxis_title="Strike (K)", yaxis_title="Density",
        height=340,
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_rnd, use_container_width=True)

    # Density statistics
    d1, d2, d3, d4 = st.columns(4)
    mean_rnd = np.trapezoid(K_grid_rnd * density, K_grid_rnd)
    var_rnd  = np.trapezoid((K_grid_rnd - mean_rnd)**2 * density, K_grid_rnd)
    skew_rnd = np.trapezoid(((K_grid_rnd - mean_rnd) / np.sqrt(max(var_rnd, 1e-8)))**3 * density, K_grid_rnd)
    kurt_rnd = np.trapezoid(((K_grid_rnd - mean_rnd) / np.sqrt(max(var_rnd, 1e-8)))**4 * density, K_grid_rnd) - 3
    d1.metric("E[S_T]",   f"{mean_rnd:.2f}")
    d2.metric("Std dev",  f"{np.sqrt(var_rnd):.2f}")
    d3.metric("Skewness", f"{skew_rnd:.3f}", delta="vs 0 (normal)")
    d4.metric("Exc. kurtosis", f"{kurt_rnd:.3f}", delta="vs 0 (normal)")


# ── Arbitrage Checks ───────────────────────────────────────────────────────────
if show_arb:
    st.markdown("### Arbitrage Checks")
    arb_col1, arb_col2 = st.columns(2)

    with arb_col1:
        cal_ok = cal_check['is_arbitrage_free']
        status = "✓  Calendar spread arb-free" if cal_ok else f"✗  {cal_check['n_violations']} calendar violations"
        color  = "#10b981" if cal_ok else "#ef4444"
        st.markdown(f"""
        <div style="background:#111318;border:1px solid {'#10b981' if cal_ok else '#ef4444'};
                    border-radius:8px;padding:16px;">
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:{color};font-weight:500;">
                {status}
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;margin-top:6px;">
                Total variance non-decreasing in T for fixed log-moneyness k
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arb_col2:
        but_ok = but_check['is_arbitrage_free']
        n_viol = sum(1 for v in but_check['slice_results'].values() if not v)
        status2 = "✓  Butterfly arb-free (all slices)" if but_ok else f"✗  {n_viol} slices violate butterfly"
        color2  = "#10b981" if but_ok else "#ef4444"
        st.markdown(f"""
        <div style="background:#111318;border:1px solid {'#10b981' if but_ok else '#ef4444'};
                    border-radius:8px;padding:16px;">
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:{color2};font-weight:500;">
                {status2}
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;margin-top:6px;">
                Risk-neutral density g(k) ≥ 0 across all slices
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Per-slice butterfly status
    slice_names  = list(but_check['slice_results'].keys())
    slice_status = [1 if v else 0 for v in but_check['slice_results'].values()]
    fig_arb = go.Figure(go.Bar(
        x=slice_names, y=slice_status,
        marker_color=['#10b981' if s else '#ef4444' for s in slice_status],
        text=['OK' if s else 'VIOLATION' for s in slice_status],
        textposition='inside',
        textfont=dict(size=11, color='#000'),
    ))
    fig_arb.update_layout(
        **PLOTLY_LAYOUT,
        title="Butterfly Arbitrage Status per Slice",
        yaxis=dict(range=[0, 1.3], tickvals=[0, 1], ticktext=['Violation', 'OK']),
        height=220,
    )
    st.plotly_chart(fig_arb, use_container_width=True)
