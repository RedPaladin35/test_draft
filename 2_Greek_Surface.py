"""
pages/2_Greek_Surface.py
-------------------------
Greek surface dashboard:
  · Delta, Gamma, Vega, Theta, Vanna, Volga, Charm, Rho
  · Heatmaps across (strike × expiry) grid
  · 3D Greek surface for selected Greek
  · BS PDE verification
  · Model: BS (analytical) or Heston (bump-and-reprice)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType, Heston, HestonParams
from options_lib.risk import GreekEngine, GreekSurface

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    margin=dict(l=40, r=20, t=50, b=40),
)

GREEK_COLORMAPS = {
    'delta'  : [[0, '#1e3a5f'], [0.5, '#00d4b8'], [1, '#f59e0b']],
    'gamma'  : [[0, '#0f172a'], [0.5, '#7c6af7'], [1, '#f59e0b']],
    'vega'   : [[0, '#0f172a'], [0.5, '#10b981'], [1, '#00d4b8']],
    'theta'  : [[0, '#f59e0b'], [0.5, '#ef4444'], [1, '#0f172a']],
    'vanna'  : [[0, '#ef4444'], [0.5, '#111318'], [1, '#00d4b8']],
    'volga'  : [[0, '#0f172a'], [0.5, '#7c6af7'], [1, '#f59e0b']],
    'charm'  : [[0, '#f59e0b'], [0.5, '#111318'], [1, '#7c6af7']],
    'rho'    : [[0, '#0f172a'], [0.5, '#3b82f6'], [1, '#00d4b8']],
}

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">Greek Surface</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    Δ  Γ  V  Θ  Vanna  Volga  Charm  ρ  across all strikes &amp; expiries
</p>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Grid parameters")
    S         = st.number_input("Spot", value=100.0, step=1.0)
    r         = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
    q         = st.number_input("Div yield", value=0.0, step=0.005, format="%.3f")
    sigma     = st.number_input("Volatility σ", value=0.20, step=0.01, min_value=0.01)
    opt_type  = st.selectbox("Option type", ["Call", "Put"])

    st.markdown("---")
    k_lo      = st.number_input("Min strike (% spot)", value=70.0) / 100
    k_hi      = st.number_input("Max strike (% spot)", value=130.0) / 100
    n_strikes = st.slider("N strikes", 5, 30, 15)
    t_lo      = st.number_input("Min expiry (yr)", value=0.1, step=0.05, min_value=0.05)
    t_hi      = st.number_input("Max expiry (yr)", value=2.0, step=0.25)
    n_exp     = st.slider("N expiries", 3, 15, 8)

    st.markdown("---")
    model_choice = st.selectbox("Model", ["Black-Scholes (analytical)", "Heston"])

    if model_choice == "Heston":
        st.markdown("**Heston params**")
        v0    = st.number_input("v₀", value=sigma**2, format="%.4f", min_value=0.0001)
        kappa = st.number_input("κ", value=2.0, step=0.1)
        v_bar = st.number_input("v̄", value=sigma**2, format="%.4f", min_value=0.0001)
        xi    = st.number_input("ξ", value=0.30, step=0.05, min_value=0.01)
        rho   = st.number_input("ρ", value=-0.70, step=0.05, min_value=-0.99, max_value=0.99)

    st.markdown("---")
    selected_greek_3d = st.selectbox("Greek for 3D view",
        ["delta", "gamma", "vega", "theta", "vanna", "volga"])


# ── Build model ────────────────────────────────────────────────────────────────
otype = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt   = MarketData(spot=S, rate=r, div_yield=q)

if model_choice == "Heston":
    params = HestonParams(v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho)
    model  = Heston(params)
else:
    model  = BlackScholes(sigma=sigma)


# ── Compute Greek surface ──────────────────────────────────────────────────────
strikes  = np.linspace(S * k_lo, S * k_hi, n_strikes)
expiries = np.linspace(t_lo, t_hi, n_exp)

with st.spinner("Computing Greek surfaces..."):
    surface = GreekSurface(model, mkt, strikes, expiries, otype)
    surface.compute()


# ── ATM cross-section metrics ──────────────────────────────────────────────────
atm_idx = np.argmin(np.abs(strikes - S))
mid_exp = expiries[len(expiries) // 2]
inst_atm = EuropeanOption(S, mid_exp, otype)
bs_atm   = BlackScholes(sigma=sigma)
g_atm    = bs_atm.vanna(inst_atm, mkt)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("ATM Delta",  f"{surface.delta_surface[n_exp//2, atm_idx]:.4f}")
m2.metric("ATM Gamma",  f"{surface.gamma_surface[n_exp//2, atm_idx]:.4f}")
m3.metric("ATM Vega",   f"{surface.vega_surface[n_exp//2, atm_idx]:.2f}")
m4.metric("ATM Theta",  f"{surface.theta_surface[n_exp//2, atm_idx]:.4f}")
m5.metric("ATM Vanna",  f"{surface.vanna_surface[n_exp//2, atm_idx]:.4f}")
m6.metric("ATM Volga",  f"{surface.volga_surface[n_exp//2, atm_idx]:.4f}")

st.markdown("---")


# ── Helper: heatmap trace ──────────────────────────────────────────────────────
def make_heatmap(z_data, title, cmap_key):
    cmap = GREEK_COLORMAPS.get(cmap_key, 'RdBu_r')
    return go.Heatmap(
        x=np.round(strikes, 1),
        y=[f"{t:.2f}yr" for t in expiries],
        z=z_data,
        colorscale=cmap,
        colorbar=dict(thickness=8, len=0.8,
                      tickfont=dict(size=9, color='#e8eaf0'),
                      title=dict(text=title, font=dict(size=10, color='#5a6278'))),
        hovertemplate='K=%{x}<br>T=%{y}<br>Value=%{z:.4f}<extra></extra>',
        xgap=0.5, ygap=0.5,
    )


# ── 2×4 Heatmap grid ──────────────────────────────────────────────────────────
st.markdown("### Greek Heatmaps  —  Strike (x) × Expiry (y)")

fig_grid = make_subplots(
    rows=2, cols=4,
    subplot_titles=["Δ Delta", "Γ Gamma", "V Vega", "Θ Theta",
                    "Vanna", "Volga", "Charm", "ρ Rho"],
    horizontal_spacing=0.06,
    vertical_spacing=0.12,
)

greek_data = [
    (surface.delta_surface, 'delta',  1, 1),
    (surface.gamma_surface, 'gamma',  1, 2),
    (surface.vega_surface,  'vega',   1, 3),
    (surface.theta_surface, 'theta',  1, 4),
    (surface.vanna_surface, 'vanna',  2, 1),
    (surface.volga_surface, 'volga',  2, 2),
]

# Compute charm and rho on-the-fly for BS
charm_surf = np.zeros((n_exp, n_strikes))
rho_surf   = np.zeros((n_exp, n_strikes))
bs_for_cr  = BlackScholes(sigma=sigma)
for i, T in enumerate(expiries):
    for j, K in enumerate(strikes):
        inst = EuropeanOption(float(K), float(T), otype)
        charm_surf[i, j] = bs_for_cr.charm(inst, mkt)
        rho_surf[i, j]   = bs_for_cr.rho(inst, mkt)

greek_data += [
    (charm_surf, 'charm', 2, 3),
    (rho_surf,   'rho',   2, 4),
]

for z_data, cmap_key, row, col in greek_data:
    trace = make_heatmap(z_data, cmap_key, cmap_key)
    fig_grid.add_trace(trace, row=row, col=col)
    fig_grid.update_xaxes(tickfont=dict(size=8), row=row, col=col)
    fig_grid.update_yaxes(tickfont=dict(size=8), row=row, col=col)

fig_grid.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=10),
    height=560,
    margin=dict(l=30, r=20, t=50, b=20),
)
for ann in fig_grid.layout.annotations:
    ann.font.size = 12
    ann.font.color = '#00d4b8'

st.plotly_chart(fig_grid, use_container_width=True)

st.markdown("---")

# ── 3D Greek surface ───────────────────────────────────────────────────────────
st.markdown(f"### 3D — {selected_greek_3d.title()} Surface")

z_map = {
    'delta' : surface.delta_surface,
    'gamma' : surface.gamma_surface,
    'vega'  : surface.vega_surface,
    'theta' : surface.theta_surface,
    'vanna' : surface.vanna_surface,
    'volga' : surface.volga_surface,
}
z_3d = z_map[selected_greek_3d]
cmap_3d = GREEK_COLORMAPS.get(selected_greek_3d, 'RdBu_r')

fig_3d = go.Figure(data=[go.Surface(
    x=np.round(strikes, 1),
    y=expiries,
    z=z_3d,
    colorscale=cmap_3d,
    colorbar=dict(thickness=12, tickfont=dict(size=10, color='#e8eaf0'),
                  title=dict(text=selected_greek_3d.title(), font=dict(size=11))),
    contours=dict(z=dict(show=True, usecolormap=True, project_z=False)),
    hovertemplate='K=%{x}<br>T=%{y:.2f}yr<br>Value=%{z:.4f}<extra></extra>',
)])

fig_3d.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        bgcolor='rgba(10,11,13,1)',
        xaxis=dict(title='Strike (K)', gridcolor='#1e2530', color='#5a6278'),
        yaxis=dict(title='Expiry (yr)', gridcolor='#1e2530', color='#5a6278'),
        zaxis=dict(title=selected_greek_3d.title(), gridcolor='#1e2530', color='#5a6278'),
        camera=dict(eye=dict(x=1.8, y=-1.4, z=1.0)),
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=460,
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# ── Cross-sections ─────────────────────────────────────────────────────────────
st.markdown("### Cross-sections")
cs1, cs2 = st.columns(2)

with cs1:
    st.markdown("**Greeks vs Strike (fixed expiry)**")
    exp_idx = st.slider("Expiry index", 0, n_exp - 1, n_exp // 2,
                         format=f"T=%.2f", help="Select expiry slice")
    fig_cs = go.Figure()
    for greek, data, color in [
        ("Δ Delta",  surface.delta_surface[exp_idx], '#00d4b8'),
        ("Γ Gamma×10", surface.gamma_surface[exp_idx] * 10, '#7c6af7'),
        ("V Vega/100", surface.vega_surface[exp_idx] / 100, '#f59e0b'),
    ]:
        fig_cs.add_trace(go.Scatter(x=strikes, y=data, name=greek,
                                     line=dict(color=color, width=2)))
    fig_cs.add_vline(x=S, line_color='#5a6278', line_dash='dash', line_width=1)
    fig_cs.update_layout(**PLOTLY_LAYOUT,
                          title=f"Greeks vs Strike (T={expiries[exp_idx]:.2f}yr)",
                          xaxis_title="Strike", height=280,
                          legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_cs, use_container_width=True)

with cs2:
    st.markdown("**Greeks vs Expiry (fixed strike)**")
    K_idx = st.slider("Strike index", 0, n_strikes - 1, n_strikes // 2)
    fig_te = go.Figure()
    for greek, data, color in [
        ("Δ Delta",    surface.delta_surface[:, K_idx], '#00d4b8'),
        ("V Vega/100", surface.vega_surface[:, K_idx] / 100, '#f59e0b'),
        ("Θ Theta×(-100)", -surface.theta_surface[:, K_idx] * 100, '#ef4444'),
    ]:
        fig_te.add_trace(go.Scatter(x=expiries, y=data, name=greek,
                                     line=dict(color=color, width=2)))
    fig_te.update_layout(**PLOTLY_LAYOUT,
                          title=f"Greeks vs Expiry (K={strikes[K_idx]:.1f})",
                          xaxis_title="Expiry (yr)", height=280,
                          legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_te, use_container_width=True)

# ── BS PDE verification ────────────────────────────────────────────────────────
with st.expander("BS PDE verification  Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0"):
    bs_check = BlackScholes(sigma=sigma)
    residuals = []
    for T_val in expiries[::2]:
        for K_val in strikes[::3]:
            inst = EuropeanOption(float(K_val), float(T_val), otype)
            residuals.append(bs_check.verify_pde(inst, mkt))

    fig_pde = go.Figure(go.Histogram(
        x=residuals, nbinsx=30,
        marker_color='#00d4b8', marker_line_color='#0a0b0d', marker_line_width=0.5,
    ))
    fig_pde.add_vline(x=0, line_color='#f59e0b', line_dash='dash', line_width=2)
    fig_pde.update_layout(
        **PLOTLY_LAYOUT,
        title="PDE Residual Distribution (should be ~0)",
        xaxis_title="Residual", yaxis_title="Count", height=240,
    )
    st.plotly_chart(fig_pde, use_container_width=True)
    max_r = max(abs(r) for r in residuals)
    if max_r < 1e-6:
        st.success(f"✓  Max |residual| = {max_r:.2e}  — PDE satisfied")
    else:
        st.warning(f"Max |residual| = {max_r:.2e}")
