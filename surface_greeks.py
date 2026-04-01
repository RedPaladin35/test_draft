"""
risk/surface_greeks.py
-----------------------
Volatility surface Greeks: sensitivities of option prices to movements
in the entire implied vol surface, not just a single sigma.

The Problem with Scalar Vega
------------------------------
Black-Scholes Vega = dV/dσ measures sensitivity to a parallel shift of
the entire vol surface by one unit. In reality, the vol surface can move
in many ways:
  1. Parallel shift     — all vols up/down by Δσ (ATM level changes)
  2. Skew move          — slope of the smile changes (OTM puts vs calls)
  3. Curvature move     — convexity of the smile changes
  4. Term structure move — short-dated vs long-dated vols move differently

A proper vol risk system bucketes sensitivities by BOTH strike dimension
(ATM level, skew, curvature) and time dimension (per expiry bucket).
This is what sell-side risk systems actually compute.

The SVI Parametrisation Makes This Natural
-------------------------------------------
SVI parametrises each expiry slice as:
    w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

This gives us natural risk parameters directly tied to market observables:
  - a (level): shifts ATM total variance → "Parallel Vega"
  - b (slope/curvature): scales the wing width
  - ρ (skew): tilts the smile left/right → "Skew Sensitivity"
  - σ_svi (smile width): controls curvature → "Curvature Sensitivity"

We define surface Greeks by bumping each SVI parameter and
repricing via the vol surface → option price chain.

Vega Buckets
------------
The most useful decomposition for a trader:
  - Per-expiry vega: how much do I make/lose if T=0.5yr vol moves by 1%?
  - This shows WHERE along the term structure you are long or short vol
  - Bucketing by expiry is the standard DV01-equivalent for vol risk

Dollar Vega
-----------
All Greeks are reported as dollar sensitivities:
  dollar_vega[T] = dV/dσ(T) × 1%   (per 1% parallel shift of that expiry)

This is directly comparable across instruments regardless of notional.

Usage
-----
>>> from options_lib.risk.surface_greeks import SurfaceGreekEngine
>>> engine = SurfaceGreekEngine(vol_surface, market)
>>> result = engine.compute(instrument)
>>> result.vega_parallel        # parallel shift vega
>>> result.vega_by_expiry       # dict {expiry_date: vega}
>>> result.skew_sensitivity     # dV/d(skew parameter ρ)
>>> result.curvature_sensitivity  # dV/d(σ_svi parameter)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import copy

from options_lib.market_data.vol_surface import VolSurface, SVIParams, VolSurface
from options_lib.instruments.base import Instrument, MarketData
from options_lib.instruments.european import EuropeanOption
from options_lib.models.black_scholes import BlackScholes


# ------------------------------------------------------------------
# Bump sizes (calibrated for numerical stability)
# ------------------------------------------------------------------
BUMP_PARALLEL   = 0.001   # 0.1% parallel vol shift
BUMP_SKEW       = 0.01    # bump in ρ (SVI skew parameter)
BUMP_CURVATURE  = 0.01    # bump in σ_svi (SVI curvature parameter)
BUMP_LEVEL      = 0.001   # bump in SVI a parameter


@dataclass
class SurfaceGreeks:
    """
    Full set of vol surface Greeks for one instrument.

    All values are dollar sensitivities (option price change per unit bump).

    Attributes
    ----------
    price               : float   Option price at current surface
    vega_parallel       : float   dV per 1% parallel shift of entire surface
    vega_by_expiry      : dict    {expiry_date: dV per 1% shift of that slice}
    skew_sensitivity    : float   dV per unit bump of ρ across all slices
    curvature_sensitivity : float dV per unit bump of σ_svi across all slices
    skew_by_expiry      : dict    {expiry_date: skew sensitivity}
    curvature_by_expiry : dict    {expiry_date: curvature sensitivity}
    term_structure_dv01 : dict    {expiry_date: dV per 1bp shift of that slice}
    """
    price                 : float
    vega_parallel         : float
    vega_by_expiry        : dict
    skew_sensitivity      : float
    curvature_sensitivity : float
    skew_by_expiry        : dict        = field(default_factory=dict)
    curvature_by_expiry   : dict        = field(default_factory=dict)
    term_structure_dv01   : dict        = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'='*52}",
            f"Vol Surface Greeks",
            f"{'='*52}",
            f"Price:                {self.price:.4f}",
            f"Parallel vega (+1%):  {self.vega_parallel:+.4f}",
            f"Skew sensitivity:     {self.skew_sensitivity:+.4f}",
            f"Curvature sensitivity:{self.curvature_sensitivity:+.4f}",
            f"{'─'*52}",
            f"Vega by expiry (per 1% shift):",
        ]
        for exp, vega in sorted(self.vega_by_expiry.items()):
            lines.append(f"  {exp}: {vega:+.4f}")
        lines.append(f"{'='*52}")
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {
            'price'                  : self.price,
            'vega_parallel'          : self.vega_parallel,
            'vega_by_expiry'         : self.vega_by_expiry,
            'skew_sensitivity'       : self.skew_sensitivity,
            'curvature_sensitivity'  : self.curvature_sensitivity,
            'skew_by_expiry'         : self.skew_by_expiry,
            'curvature_by_expiry'    : self.curvature_by_expiry,
            'term_structure_dv01'    : self.term_structure_dv01,
        }


class SurfaceGreekEngine:
    """
    Computes vol surface Greeks by bumping SVI parameters and repricing.

    The engine prices options by:
      1. Extracting the implied vol from the vol surface at (K, T)
      2. Pricing via BS with that implied vol
      3. Bumping surface parameters and re-extracting the implied vol
      4. Central differencing to get sensitivities

    This gives the correct sensitivity to the surface, not to a
    hypothetical constant-vol world.

    Parameters
    ----------
    vol_surface : VolSurface
        Calibrated SVI vol surface.
    market : MarketData
        Current spot, rate, dividend yield.
    """

    def __init__(self, vol_surface: VolSurface, market: MarketData):
        self.vol_surface = vol_surface
        self.market      = market

    def _price_on_surface(
        self,
        instrument   : EuropeanOption,
        surface      : VolSurface,
        market       : MarketData,
    ) -> float:
        """
        Price a European option by extracting IV from the surface and using BS.

        This is the core pricing step: surface → IV → BS price.
        It means our surface Greeks capture how option prices respond
        to real vol surface movements, not hypothetical sigma bumps.
        """
        iv = surface.implied_vol(K=instrument.strike, T=instrument.expiry)
        iv = max(iv, 0.001)  # floor for numerical safety
        bs = BlackScholes(sigma=iv)
        return bs.price(instrument, market)

    def _bump_surface_parallel(
        self,
        surface  : VolSurface,
        bump     : float,
    ) -> VolSurface:
        """
        Return a new surface with ALL SVI slices shifted by `bump`
        in the `a` parameter (total variance level).

        Bumping `a` shifts the total variance level uniformly across
        all strikes for a given expiry, which is a parallel shift of
        the implied vol smile (since IV = √(w/T) and w = a + ...).
        """
        new_slices = {}
        for date, svi in surface.svi_slices.items():
            new_slices[date] = SVIParams(
                a      = svi.a + bump,   # parallel shift
                b      = svi.b,
                rho    = svi.rho,
                m      = svi.m,
                sigma  = svi.sigma,
                expiry = svi.expiry,
            )
        return VolSurface(
            svi_slices = new_slices,
            forwards   = surface.forwards,
            spot       = surface.spot,
            rate       = surface.rate,
            ticker     = surface.ticker,
        )

    def _bump_surface_slice(
        self,
        surface      : VolSurface,
        expiry_date  : str,
        param        : str,
        bump         : float,
    ) -> VolSurface:
        """
        Return a new surface with ONE slice's parameter bumped.

        Used for per-expiry vega buckets and skew/curvature sensitivities.

        Parameters
        ----------
        param : str   One of 'a', 'b', 'rho', 'm', 'sigma'
        """
        new_slices = {}
        for date, svi in surface.svi_slices.items():
            if date == expiry_date:
                kwargs = dict(
                    a=svi.a, b=svi.b, rho=svi.rho,
                    m=svi.m, sigma=svi.sigma, expiry=svi.expiry
                )
                kwargs[param] = kwargs[param] + bump
                # Clamp to valid ranges
                kwargs['b']     = max(kwargs['b'],     1e-4)
                kwargs['sigma'] = max(kwargs['sigma'], 1e-4)
                kwargs['rho']   = np.clip(kwargs['rho'], -0.99, 0.99)
                new_slices[date] = SVIParams(**kwargs)
            else:
                new_slices[date] = svi

        return VolSurface(
            svi_slices = new_slices,
            forwards   = surface.forwards,
            spot       = surface.spot,
            rate       = surface.rate,
            ticker     = surface.ticker,
        )

    def compute(
        self,
        instrument : EuropeanOption,
    ) -> SurfaceGreeks:
        """
        Compute all vol surface Greeks for a European option.

        Parameters
        ----------
        instrument : EuropeanOption

        Returns
        -------
        SurfaceGreeks
        """
        surface = self.vol_surface
        market  = self.market
        K       = instrument.strike
        T       = instrument.expiry

        # Base price
        price = self._price_on_surface(instrument, surface, market)

        # ------------------------------------------------------------------
        # Parallel Vega: bump all slices' `a` parameter up and down
        # dV/d(parallel shift of 1%)
        # ------------------------------------------------------------------
        # Convert 1% vol bump to equivalent variance bump
        # If IV = √(w/T), then d(IV) = 0.01 means d(w) ≈ 2*IV*T*0.01
        iv_base = surface.implied_vol(K, T)
        dw      = 2 * iv_base * T * BUMP_PARALLEL   # d(total variance)

        surf_up   = self._bump_surface_parallel(surface, +dw)
        surf_down = self._bump_surface_parallel(surface, -dw)

        p_up   = self._price_on_surface(instrument, surf_up,   market)
        p_down = self._price_on_surface(instrument, surf_down, market)

        vega_parallel = (p_up - p_down) / (2 * BUMP_PARALLEL)

        # ------------------------------------------------------------------
        # Vega by expiry: bump each slice independently
        # This gives the term structure of vega exposure
        # ------------------------------------------------------------------
        vega_by_expiry     = {}
        term_structure_dv01 = {}

        for exp_date, svi in surface.svi_slices.items():
            # Convert 1% vol bump to `a` bump for this slice
            iv_slice = surface.implied_vol(K, svi.expiry)
            dw_slice = 2 * iv_slice * svi.expiry * BUMP_PARALLEL

            s_up   = self._bump_surface_slice(surface, exp_date, 'a', +dw_slice)
            s_down = self._bump_surface_slice(surface, exp_date, 'a', -dw_slice)

            p_up_s   = self._price_on_surface(instrument, s_up,   market)
            p_down_s = self._price_on_surface(instrument, s_down, market)

            vega_by_expiry[exp_date]     = (p_up_s - p_down_s) / (2 * BUMP_PARALLEL)
            term_structure_dv01[exp_date] = (p_up_s - p_down_s) / (2 * BUMP_PARALLEL / 100)

        # ------------------------------------------------------------------
        # Skew Sensitivity: bump ρ (the SVI skew parameter) on each slice
        # dV/dρ — how much does the option gain/lose when the smile tilts?
        # ------------------------------------------------------------------
        skew_by_expiry = {}
        skew_total     = 0.0

        for exp_date in surface.svi_slices:
            s_up   = self._bump_surface_slice(surface, exp_date, 'rho', +BUMP_SKEW)
            s_down = self._bump_surface_slice(surface, exp_date, 'rho', -BUMP_SKEW)

            p_up_sk   = self._price_on_surface(instrument, s_up,   market)
            p_down_sk = self._price_on_surface(instrument, s_down, market)

            sk = (p_up_sk - p_down_sk) / (2 * BUMP_SKEW)
            skew_by_expiry[exp_date] = sk
            skew_total += sk

        # ------------------------------------------------------------------
        # Curvature Sensitivity: bump σ_svi (smile width parameter)
        # dV/d(σ_svi) — sensitivity to smile convexity
        # ------------------------------------------------------------------
        curvature_by_expiry = {}
        curvature_total     = 0.0

        for exp_date in surface.svi_slices:
            s_up   = self._bump_surface_slice(surface, exp_date, 'sigma', +BUMP_CURVATURE)
            s_down = self._bump_surface_slice(surface, exp_date, 'sigma', -BUMP_CURVATURE)

            p_up_cv   = self._price_on_surface(instrument, s_up,   market)
            p_down_cv = self._price_on_surface(instrument, s_down, market)

            cv = (p_up_cv - p_down_cv) / (2 * BUMP_CURVATURE)
            curvature_by_expiry[exp_date] = cv
            curvature_total += cv

        return SurfaceGreeks(
            price                  = price,
            vega_parallel          = vega_parallel,
            vega_by_expiry         = vega_by_expiry,
            skew_sensitivity       = skew_total,
            curvature_sensitivity  = curvature_total,
            skew_by_expiry         = skew_by_expiry,
            curvature_by_expiry    = curvature_by_expiry,
            term_structure_dv01    = term_structure_dv01,
        )

    def compute_portfolio(
        self,
        instruments  : list,
        positions    : Optional[list] = None,
    ) -> SurfaceGreeks:
        """
        Compute aggregated vol surface Greeks for a portfolio.

        The portfolio Greeks are just the weighted sum of individual Greeks —
        linearity of differentiation means we can aggregate directly.

        Parameters
        ----------
        instruments : list of EuropeanOption
        positions   : list of float, optional
            Position sizes (+ = long, - = short). Defaults to 1.0 each.

        Returns
        -------
        SurfaceGreeks   Aggregated portfolio surface Greeks.
        """
        if positions is None:
            positions = [1.0] * len(instruments)

        if len(positions) != len(instruments):
            raise ValueError("positions must have same length as instruments")

        # Compute individual Greeks
        individual = [self.compute(inst) for inst in instruments]

        # Aggregate
        total_price          = sum(pos * g.price for pos, g in zip(positions, individual))
        total_vega_parallel  = sum(pos * g.vega_parallel for pos, g in zip(positions, individual))
        total_skew           = sum(pos * g.skew_sensitivity for pos, g in zip(positions, individual))
        total_curvature      = sum(pos * g.curvature_sensitivity for pos, g in zip(positions, individual))

        # Aggregate by-expiry dictionaries
        all_dates = set()
        for g in individual:
            all_dates.update(g.vega_by_expiry.keys())

        agg_vega_by_expiry   = {}
        agg_skew_by_expiry   = {}
        agg_curv_by_expiry   = {}
        agg_ts_dv01          = {}

        for date in sorted(all_dates):
            agg_vega_by_expiry[date] = sum(
                pos * g.vega_by_expiry.get(date, 0.0)
                for pos, g in zip(positions, individual)
            )
            agg_skew_by_expiry[date] = sum(
                pos * g.skew_by_expiry.get(date, 0.0)
                for pos, g in zip(positions, individual)
            )
            agg_curv_by_expiry[date] = sum(
                pos * g.curvature_by_expiry.get(date, 0.0)
                for pos, g in zip(positions, individual)
            )
            agg_ts_dv01[date] = sum(
                pos * g.term_structure_dv01.get(date, 0.0)
                for pos, g in zip(positions, individual)
            )

        return SurfaceGreeks(
            price                  = total_price,
            vega_parallel          = total_vega_parallel,
            vega_by_expiry         = agg_vega_by_expiry,
            skew_sensitivity       = total_skew,
            curvature_sensitivity  = total_curvature,
            skew_by_expiry         = agg_skew_by_expiry,
            curvature_by_expiry    = agg_curv_by_expiry,
            term_structure_dv01    = agg_ts_dv01,
        )


@dataclass
class VolSurfaceScenario:
    """
    P&L under a specific vol surface scenario.

    Used for scenario analysis: what happens to a portfolio if
    the vol surface moves in a particular way?

    Parameters
    ----------
    parallel_shift  : float   Parallel vol shift in decimal (e.g. 0.01 = +1%)
    skew_shift      : float   Change in ρ parameter per slice
    curvature_shift : float   Change in σ_svi parameter per slice
    """
    parallel_shift  : float = 0.0
    skew_shift      : float = 0.0
    curvature_shift : float = 0.0

    @classmethod
    def vol_spike(cls, magnitude: float = 0.05) -> "VolSurfaceScenario":
        """Classic vol spike: parallel up + skew steepens."""
        return cls(parallel_shift=magnitude, skew_shift=-0.1)

    @classmethod
    def vol_crush(cls, magnitude: float = 0.05) -> "VolSurfaceScenario":
        """Vol crush: parallel down + skew flattens."""
        return cls(parallel_shift=-magnitude, skew_shift=0.05)

    @classmethod
    def skew_steepening(cls) -> "VolSurfaceScenario":
        """Skew steepens without ATM change."""
        return cls(skew_shift=-0.15)


def scenario_pnl(
    engine        : SurfaceGreekEngine,
    instrument    : EuropeanOption,
    scenario      : VolSurfaceScenario,
) -> dict:
    """
    Estimate option P&L under a vol surface scenario using surface Greeks.

    P&L ≈ vega_parallel × parallel_shift
          + skew_sensitivity × skew_shift
          + curvature_sensitivity × curvature_shift

    This is a first-order (linear) approximation. For large moves,
    second-order terms (vol gamma, vega convexity) matter.

    Parameters
    ----------
    engine     : SurfaceGreekEngine
    instrument : EuropeanOption
    scenario   : VolSurfaceScenario

    Returns
    -------
    dict with 'pnl_parallel', 'pnl_skew', 'pnl_curvature', 'total_pnl'
    """
    greeks = engine.compute(instrument)

    pnl_parallel   = greeks.vega_parallel * scenario.parallel_shift
    pnl_skew       = greeks.skew_sensitivity * scenario.skew_shift
    pnl_curvature  = greeks.curvature_sensitivity * scenario.curvature_shift
    total_pnl      = pnl_parallel + pnl_skew + pnl_curvature

    return {
        'pnl_parallel'   : pnl_parallel,
        'pnl_skew'       : pnl_skew,
        'pnl_curvature'  : pnl_curvature,
        'total_pnl'      : total_pnl,
        'base_price'     : greeks.price,
    }
