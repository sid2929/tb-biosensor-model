"""
model.py -- Kinetic core for the TB whole-cell biosensor.

The biosensor is a two-stage ODE system:

  1. Binding stage. Surface-displayed nanobodies bind free antigen at
     concentration Ag (assumed constant -- the sample volume is large
     relative to a single cell, so antigen depletion by one cell's
     nanobodies is negligible). This is a standard pseudo-first-order
     bimolecular binding model:

         dC/dt = kon * Ag * (N_total - C) - koff * C
         theta = C / N_total

     C is the number of bound nanobody-antigen complexes per cell,
     N_total is the total number of nanobodies expressed per cell, and
     theta is fractional receptor occupancy.

  2. Cascade/reporter stage. Bound receptor drives a downstream signaling
     cascade that ultimately produces a visible reporter (e.g. a
     chromogenic or fluorescent output). The cascade turns on
     sigmoidally with the ABSOLUTE number of bound complexes (Hill
     kinetics on C, not on the fraction bound) and is throttled by the
     metabolic cost of having expressed N_total nanobodies in the first
     place:

         dR/dt = k_eff * Hill(C, K_abs, n) - delta * R
         Hill(C, K_abs, n) = C^n / (K_abs^n + C^n)
         k_eff = k_cat_max / (1 + N_total / N_half)

     k_eff is the burden-derated maximum cascade rate: as N_total grows
     past N_half, the cell's cascade capacity is throttled towards zero
     (ribosomes/chaperones/ATP diverted to nanobody production). This is
     the capture-vs-burden trade-off the sweep is designed to resolve.

     Note on Hill's argument (C vs. theta): an earlier version of this
     model used theta = C/N_total as the Hill input. Because Ag is held
     constant (non-depleting), theta(t) turns out to be exactly
     independent of N_total (see README.md "Appendix: fractional-
     occupancy version"), which collapsed the intended capture/burden
     trade-off. Each bound complex is a molecular event capable of
     triggering the cascade independently of how many *other* nanobodies
     a cell happens to also be expressing, so the cascade should read
     absolute count C, not the population fraction theta -- that is the
     model used here.

All simulations run in MINUTES. kon and koff are supplied to this module
in their native SPR units (M^-1 s^-1, s^-1) and converted internally.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Fixed parameters. See README.md for the "grounded / assumed / swept" table
# and the justification for each value. These are not touched by the sweep.
# ---------------------------------------------------------------------------

KON_PER_S = 5.44e5          # M^-1 s^-1, SPR-measured nanobody-antigen on-rate
KOFF_PER_S = 1.499e-4       # s^-1, SPR-measured nanobody-antigen off-rate
KON = KON_PER_S * 60.0      # M^-1 min^-1
KOFF = KOFF_PER_S * 60.0    # min^-1

AG = 0.1e-9                 # M, free antigen concentration (assumed constant)

# Hill threshold in ABSOLUTE bound-complex units (complexes/cell), not a
# fraction. Chosen to sit in the middle of the C range the sweep actually
# produces: at Ag = 0.1 nM and Kd = koff/kon ~ 0.276 nM, equilibrium
# occupancy theta_ss ~ 0.27, so C = theta_ss * N_total spans roughly
# 27 (N_total=100) to 270,000 (N_total=1e6) across the N_total sweep --
# K_ABS = 1000 sits inside that range rather than at either edge, so the
# cascade neither saturates nor stays silent across the whole sweep.
K_ABS = 1000                 # bound complexes/cell giving half-maximal cascade drive
N_HILL = 2                  # Hill coefficient
DELTA = 0.025                # min^-1, reporter decay/dilution rate

T_SPAN = (0.0, 240.0)        # minutes
Y0 = (0.0, 0.0)               # C(0) = 0, R(0) = 0

# Integrator tolerances: tight enough that the threshold-crossing time
# extracted downstream isn't limited by integration error.
RTOL = 1e-7
ATOL = 1e-9


def hill(x, K=K_ABS, n=N_HILL):
    """Hill activation function, in [0, 1).

    `x` is the absolute number of bound complexes C (not a fraction), and
    `K` is the half-maximal threshold in the same absolute-count units.
    """
    x_n = x ** n
    return x_n / (K ** n + x_n)


def k_eff(k_cat_max, N_total, N_half):
    """Burden-derated maximum cascade rate (min^-1).

    Approaches k_cat_max when N_total << N_half (cheap expression, no
    burden) and decays towards zero as N_total >> N_half (expression
    load saturates/overwhelms the cell's synthetic capacity).
    """
    return k_cat_max / (1.0 + N_total / N_half)


def _rhs(t, y, N_total, k_cat_max, N_half, kon=KON, koff=KOFF, Ag=AG,
         K_abs=K_ABS, n=N_HILL, delta=DELTA):
    """Right-hand side of the coupled binding + cascade ODE system."""
    C, R = y

    dC_dt = kon * Ag * (N_total - C) - koff * C
    # Cascade is driven by the absolute bound-complex count C, not by
    # occupancy theta -- see the module docstring for why.
    dR_dt = k_eff(k_cat_max, N_total, N_half) * hill(C, K_abs, n) - delta * R
    return [dC_dt, dR_dt]


def simulate(N_total, k_cat_max, N_half, t_span=T_SPAN, t_eval=None,
             y0=Y0, rtol=RTOL, atol=ATOL):
    """Integrate the binding + cascade ODEs for one parameter combination.

    Returns the scipy OdeResult (sol.t, sol.y[0] = C(t), sol.y[1] = R(t)).
    Use `theta_of(sol, N_total)` to recover occupancy from the result.
    """
    return solve_ivp(
        _rhs, t_span, y0, method="LSODA", t_eval=t_eval,
        args=(N_total, k_cat_max, N_half), rtol=rtol, atol=atol,
    )


def theta_of(sol, N_total):
    """Fractional receptor occupancy theta(t) = C(t) / N_total from a solve_ivp result."""
    return np.clip(sol.y[0] / N_total, 0.0, 1.0)
