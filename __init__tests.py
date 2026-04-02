"""
options_lib.tests
-----------------
Test suite for the options pricing library.

185 tests across 9 test files. Run with:
    pytest options_lib/tests/ -v

Test files
----------
test_black_scholes.py         BS pricing, Greeks, IV round-trip (32 tests)
test_heston.py                Heston pricing, smile, calibration (14 tests)
test_finite_difference.py     Crank-Nicolson European + American (17 tests)
test_monte_carlo.py           MC variance reduction, barriers, Asians (13 tests)
test_greeks.py                Scalar Greeks + P&L attribution (27 tests)
test_vol_surface.py           SVI calibration, arb checks, Dupire (25 tests)
test_lsmc_and_localvol_mc.py  LSMC + local vol barrier (18 tests)
test_heston_asian.py          Heston simulator + Asian pricer (17 tests)
test_surface_greeks.py        Vol surface Greeks + scenarios (22 tests)
"""
