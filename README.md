# Options Pricing Dashboard

A multi-page Streamlit dashboard for the Options Pricing Library.

## Setup

```bash
cd OptionsPricingLibrary
pip install -r dashboard/requirements.txt
pip install -e .
```

## Run

```bash
streamlit run dashboard/app.py
```

## Pages

| Page | Description |
|---|---|
| **Home** | Live multi-model pricer — BS, Heston, LSMC, Crank-Nicolson. Price vs spot, Delta/Gamma chart, Vega vs σ. |
| **Vol Surface** | Fetch live option chain (yfinance), calibrate SVI, 3D interactive surface, IV smiles, risk-neutral density, arb checks. |
| **Greek Surface** | Full 8-Greek heatmap grid (Δ Γ V Θ Vanna Volga Charm ρ) across all strikes and expiries. 3D surface for any Greek. BS PDE verification. |
| **Vega Matrix** | Pillar-by-pillar vol surface sensitivity. Expiry bucketing. Strike exposure. Scenario P&L waterfall. |
| **Model Risk** | Barrier: local vol MC vs flat BS as barrier varies. Asian: Heston MC vs flat BS across strikes. Vol premium surface. |
| **P&L Attribution** | Taylor decomposition (Δ Γ V Θ Vanna Volga). Interactive single-step. GBM backtest with stacked cumulative P&L. |

## Architecture

```
dashboard/
├── app.py                    ← Home + navigation + global CSS
├── requirements.txt
└── pages/
    ├── 1_Vol_Surface.py
    ├── 2_Greek_Surface.py
    ├── 3_Vega_Matrix.py
    ├── 4_Model_Risk.py
    └── 5_PnL_Attribution.py
```

All pages import from `options_lib` — the library must be installed
(`pip install -e .` from the `OptionsPricingLibrary` root).
