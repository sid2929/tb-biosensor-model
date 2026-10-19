"""
sweep.py -- Grid search over (N_total, k_cat_max, N_half).

For every combination on the grid we simulate the binding + cascade ODEs
(model.py) over 0-240 min and record how long the reporter R(t) takes to
cross 50% of the single largest R value seen anywhere on the whole grid.
That shared threshold is what makes "time to visible color change"
comparable across scenarios -- each scenario is judged against the same
target signal level, not its own private ceiling.

Run this first:

    python sweep.py

It writes outputs/sweep_results.npz, which plot.py then reads. Re-run
this script any time model.py or the grids below change.
"""

import time

import numpy as np

from model import T_SPAN, simulate

# ---------------------------------------------------------------------------
# Sweep grids. These are the three parameters we don't have measurements
# for (see README.md, "Swept due to being unknown").
# ---------------------------------------------------------------------------

N_TOTAL = np.logspace(2, 6, 20)            # nanobodies per cell
K_CAT_MAX = np.logspace(-2.301, -0.301, 10)  # min^-1  (~0.005 to 0.5)
N_HALF = np.logspace(3, 5, 10)             # nanobodies per cell

# Time resolution used for the threshold-crossing search. 0.1 min is well
# below the ~10-100 min timescales set by koff and delta, so linear
# interpolation between samples introduces negligible timing error.
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 2401)

OUTPUT_PATH = "outputs/sweep_results.npz"


def time_to_threshold(t, R, threshold):
    """First time R(t) reaches `threshold`, via linear interpolation.

    Returns np.inf if R never reaches the threshold within the simulated
    window -- i.e. this scenario would not produce a visible signal in
    time, which is itself an important (and plot-able) outcome.
    """
    reached = R >= threshold
    if not reached.any():
        return np.inf

    idx = int(np.argmax(reached))  # index of first sample at/above threshold
    if idx == 0:
        return float(t[0])

    t0, t1 = t[idx - 1], t[idx]
    R0, R1 = R[idx - 1], R[idx]
    if R1 == R0:
        return float(t1)
    frac = (threshold - R0) / (R1 - R0)
    return float(t0 + frac * (t1 - t0))


def run_sweep(verbose=True):
    """Simulate every (N_total, k_cat_max, N_half) combination on the grid.

    Returns (time_to_threshold_grid, global_max_R, threshold), where
    time_to_threshold_grid has shape (len(N_TOTAL), len(K_CAT_MAX), len(N_HALF)).
    """
    n_N, n_k, n_h = len(N_TOTAL), len(K_CAT_MAX), len(N_HALF)

    # R(t) for every combination, held in memory just long enough to find
    # the grid-wide max and then extract crossing times. Not persisted --
    # only the summary (time-to-threshold) grid is saved to disk.
    R_all = np.empty((n_N, n_k, n_h, T_EVAL.size))

    t_start = time.time()
    for i, N_total in enumerate(N_TOTAL):
        for j, k_cat_max in enumerate(K_CAT_MAX):
            for l, N_half in enumerate(N_HALF):
                sol = simulate(N_total, k_cat_max, N_half, t_eval=T_EVAL)
                R_all[i, j, l, :] = sol.y[1]
        if verbose:
            elapsed = time.time() - t_start
            print(f"  N_total {i + 1:2d}/{n_N} done ({elapsed:5.1f}s elapsed)")

    global_max_R = float(R_all.max())
    threshold = 0.5 * global_max_R

    ttt = np.empty((n_N, n_k, n_h))
    for i in range(n_N):
        for j in range(n_k):
            for l in range(n_h):
                ttt[i, j, l] = time_to_threshold(T_EVAL, R_all[i, j, l, :], threshold)

    return ttt, global_max_R, threshold


def main():
    import os

    os.makedirs("outputs", exist_ok=True)

    print(f"Sweeping {len(N_TOTAL)} x {len(K_CAT_MAX)} x {len(N_HALF)} = "
          f"{len(N_TOTAL) * len(K_CAT_MAX) * len(N_HALF)} parameter combinations...")
    ttt, global_max_R, threshold = run_sweep()

    n_reached = np.isfinite(ttt).sum()
    n_total = ttt.size
    print(f"Global max R across sweep: {global_max_R:.4g}")
    print(f"Threshold (50% of max):    {threshold:.4g}")
    print(f"Combinations reaching threshold within 240 min: {n_reached}/{n_total}")

    np.savez(
        OUTPUT_PATH,
        N_total=N_TOTAL,
        k_cat_max=K_CAT_MAX,
        N_half=N_HALF,
        time_to_threshold=ttt,
        global_max_R=global_max_R,
        threshold=threshold,
    )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
