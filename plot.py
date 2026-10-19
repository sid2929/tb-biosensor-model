"""
plot.py -- Figures for the nanobody expression-level trade-off.

Reads outputs/sweep_results.npz (written by sweep.py) and produces three
PNGs in outputs/:

  heatmap.png      optimal N_total vs. (k_cat_max, N_half)
  timecourses.png   theta(t) and R(t) for four representative N_total
  sensitivity.png   how much the optimum moves under each swept parameter

Run after sweep.py:

    python plot.py
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from model import T_SPAN, simulate, theta_of
from sweep import K_CAT_MAX, N_HALF, N_TOTAL

OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "sweep_results.npz")

# ---------------------------------------------------------------------------
# Palette + chrome (validated colorblind-safe categorical/sequential set).
# Sequential ramp for magnitude (heatmap); fixed-order categorical hues for
# identity (lines/bars). Colors never carry meaning alone in the text -- axis
# labels and legends spell out what each color is.
# ---------------------------------------------------------------------------

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NO_CROSSING = "#c3c2b7"  # fill for grid cells that never reach threshold

CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"
CAT_YELLOW = "#eda100"

SEQ_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STEPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_PRIMARY,
    "axes.edgecolor": AXIS_LINE,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "font.size": 11,
})


def nearest_index(grid, value):
    """Index of the grid point closest to `value`."""
    return int(np.argmin(np.abs(grid - value)))


def median_index(grid):
    """Index of the grid point closest to the array's median.

    Used instead of np.median directly so that "the median scenario" is an
    actual simulated grid point, not an interpolated value between two.
    """
    return nearest_index(grid, np.median(grid))


def optimal_N_total(ttt_slice, N_total_grid):
    """N_total minimizing time-to-threshold along a 1D slice.

    Returns NaN if the threshold is never reached anywhere on the slice --
    there is no well-defined optimum for a scenario that never lights up.
    """
    if not np.isfinite(ttt_slice).any():
        return np.nan
    return N_total_grid[np.argmin(ttt_slice)]


# ---------------------------------------------------------------------------
# Figure 1: heatmap of optimal N_total over (k_cat_max, N_half)
# ---------------------------------------------------------------------------

def plot_heatmap(ttt):
    n_k, n_h = len(K_CAT_MAX), len(N_HALF)
    opt_N = np.full((n_h, n_k), np.nan)  # [row=N_half, col=k_cat_max] for imshow-style layout
    for j in range(n_k):
        for l in range(n_h):
            opt_N[l, j] = optimal_N_total(ttt[:, j, l], N_TOTAL)

    fig, ax = plt.subplots(figsize=(8.5, 6.6))

    masked = np.ma.masked_invalid(opt_N)
    cmap = SEQ_BLUE.copy()
    cmap.set_bad(NO_CROSSING)

    mesh = ax.pcolormesh(
        K_CAT_MAX, N_HALF, masked, shading="nearest", cmap=cmap,
        norm=LogNorm(vmin=N_TOTAL.min(), vmax=N_TOTAL.max()),
        edgecolors=SURFACE, linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_{cat,max}$ (min$^{-1}$)  --  cascade speed")
    ax.set_ylabel(r"$N_{half}$ (nanobodies/cell)  --  burden onset")
    ax.set_title("Optimal nanobody expression level, by cascade scenario", pad=12)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"Optimal $N_{total}$ (nanobodies per cell)")
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)

    # Gray cells never reach the shared detection threshold within 240 min
    # at any N_total -- there is no "optimal" expression level to report.
    caption_lines = [
        "Gray = threshold never reached within 240 min at any $N_{total}$ tested."
    ]

    # Report honestly whether the optimum is an interior trade-off or still
    # pinned to an edge of the swept range -- don't assume either outcome.
    finite_opt = opt_N[np.isfinite(opt_N)]
    if finite_opt.size:
        at_low_edge = np.isclose(finite_opt, N_TOTAL.min())
        at_high_edge = np.isclose(finite_opt, N_TOTAL.max())
        if at_low_edge.all():
            caption_lines.append(
                "Non-gray region is uniformly the smallest $N_{total}$ tested -- "
                "still a corner solution."
            )
        elif at_high_edge.all():
            caption_lines.append(
                "Non-gray region is uniformly the largest $N_{total}$ tested -- "
                "still a corner solution."
            )
        else:
            n_interior = np.sum(~at_low_edge & ~at_high_edge)
            caption_lines.append(
                f"{n_interior}/{finite_opt.size} scenarios reaching threshold land on an "
                "interior optimum (neither swept edge) -- a real capture/burden trade-off."
            )
    caption = "\n".join(
        textwrap.fill(line, width=78) for line in caption_lines
    )

    if masked.mask.any() or finite_opt.size:
        ax.text(
            0.0, -0.20, caption,
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="top",
        )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: theta(t) and R(t) time courses at one mid-range scenario
# ---------------------------------------------------------------------------

def plot_timecourses():
    k_cat_med = K_CAT_MAX[median_index(K_CAT_MAX)]
    n_half_med = N_HALF[median_index(N_HALF)]

    rep_N_totals = [10 ** 2.5, 10 ** 3.5, 10 ** 4.5, 10 ** 5.5]
    colors = [CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW]

    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], 1201)

    fig, (ax_theta, ax_R) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)

    for N_total, color in zip(rep_N_totals, colors):
        sol = simulate(N_total, k_cat_med, n_half_med, t_eval=t_eval)
        theta = theta_of(sol, N_total)
        R = sol.y[1]
        label = fr"$N_{{total}}$ = {N_total:,.0f}"
        ax_theta.plot(sol.t, theta, color=color, lw=2, label=label)
        ax_R.plot(sol.t, R, color=color, lw=2, label=label)

    ax_theta.set_ylabel(r"Occupancy $\theta = C / N_{total}$")
    ax_theta.set_title(
        fr"Time courses at $k_{{cat,max}}$ = {k_cat_med:.3g} min$^{{-1}}$, "
        fr"$N_{{half}}$ = {n_half_med:,.0f} nanobodies/cell",
        pad=12,
    )
    ax_theta.grid(True, alpha=0.6, lw=0.5)
    ax_theta.legend(frameon=False, fontsize=9)
    # The four curves are exact overlaps, not a plotting error: dividing the
    # binding ODE by N_total shows theta's dynamics never involve N_total
    # (Ag is held constant/non-depleting). That's still true here -- it's
    # *only* because the cascade below now reads absolute C instead of this
    # theta that the R panel shows a real, N_total-dependent trade-off. See
    # README "Appendix: fractional-occupancy version" for the contrast.
    ax_theta.text(
        0.98, 0.05,
        "All 4 curves overlap exactly -- theta(t) does not depend on $N_{total}$.",
        transform=ax_theta.transAxes, fontsize=9, color=INK_MUTED,
        ha="right", va="bottom",
    )

    ax_R.set_ylabel("Reporter signal $R$ (a.u.)")
    ax_R.set_xlabel("Time (min)")
    ax_R.grid(True, alpha=0.6, lw=0.5)
    ax_R.legend(frameon=False, fontsize=9)

    for ax in (ax_theta, ax_R):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS_LINE)
        ax.spines["bottom"].set_color(AXIS_LINE)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "timecourses.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: sensitivity of the optimum to each swept parameter
# ---------------------------------------------------------------------------

def plot_sensitivity(ttt):
    k_med_idx = median_index(K_CAT_MAX)
    h_med_idx = median_index(N_HALF)

    # Sweep k_cat_max alone, N_half held at its median.
    opt_vs_kcat = np.array([
        optimal_N_total(ttt[:, j, h_med_idx], N_TOTAL) for j in range(len(K_CAT_MAX))
    ])
    # Sweep N_half alone, k_cat_max held at its median.
    opt_vs_nhalf = np.array([
        optimal_N_total(ttt[:, k_med_idx, l], N_TOTAL) for l in range(len(N_HALF))
    ])

    def log10_span(values):
        """max(log10) - min(log10) over finite entries, and how many there were.

        With 0 finite entries there's nothing to measure. With exactly 1,
        the span is trivially 0 -- not evidence the optimum doesn't move,
        just that only one point on the slice produced a signal at all.
        Both cases are reported honestly in the caption below rather than
        folded into a bare "0.00".
        """
        finite = values[np.isfinite(values)]
        n = finite.size
        span = float(np.log10(finite).max() - np.log10(finite).min()) if n else 0.0
        return span, n

    span_kcat, n_kcat = log10_span(opt_vs_kcat)
    span_nhalf, n_nhalf = log10_span(opt_vs_nhalf)

    fig, ax = plt.subplots(figsize=(7, 6.4))
    labels = [r"$k_{cat,max}$ swept" + "\n(N_half fixed at median)",
              r"$N_{half}$ swept" + "\n(k_cat_max fixed at median)"]
    values = [span_kcat, span_nhalf]
    colors = [CAT_BLUE, CAT_ORANGE]

    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                 ha="center", va="bottom", color=INK_PRIMARY, fontsize=10)

    ax.set_ylabel(r"Shift in optimal $N_{total}$ (log$_{10}$ nanobodies/cell)")
    ax.set_title("Which uncertainty moves the optimum more?", pad=12)
    ax.grid(True, axis="y", alpha=0.6, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=max(0.5, max(values) * 1.3))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS_LINE)
    ax.spines["bottom"].set_color(AXIS_LINE)
    ax.tick_params(axis="x", length=0)

    # Report data coverage honestly for each bar -- a 0.00 span means very
    # different things depending on whether it came from 0, 1, or many
    # finite (threshold-reached) points on that slice.
    def coverage_note(label, n, n_grid):
        if n == 0:
            return f"{label}: 0/{n_grid} reach threshold at the other parameter's median -- no span measurable."
        if n == 1:
            return f"{label}: only 1/{n_grid} reaches threshold -- span is trivially 0, not evidence of a flat optimum."
        return f"{label}: {n}/{n_grid} reach threshold."

    caption = "\n".join(textwrap.fill(line, width=80) for line in [
        coverage_note(r"$k_{cat,max}$ sweep", n_kcat, len(K_CAT_MAX)),
        coverage_note(r"$N_{half}$ sweep", n_nhalf, len(N_HALF)),
        "The median scenario sits outside the region of (k_cat_max, N_half) that "
        "reaches threshold within 240 min at all -- see heatmap.png.",
    ])
    ax.text(
        0.0, -0.24, caption,
        transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="top",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sensitivity.png"), dpi=200)
    plt.close(fig)


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(
            f"{RESULTS_PATH} not found -- run `python sweep.py` first."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = np.load(RESULTS_PATH)
    ttt = data["time_to_threshold"]

    plot_heatmap(ttt)
    plot_timecourses()
    plot_sensitivity(ttt)

    print(f"Wrote heatmap.png, timecourses.png, sensitivity.png to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
