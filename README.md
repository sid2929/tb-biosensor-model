# TB whole-cell biosensor -- nanobody expression-level model

A kinetic model of an *E. coli* whole-cell biosensor for TB detection: a
surface-displayed nanobody captures antigen, receptor occupancy drives a
signaling cascade, and the cascade produces a visible reporter. The question
this model was built to answer: **how much nanobody should the cell express
per cell to minimize time to a visible color change**, given that more
nanobody means more antigen capture capacity but also more metabolic burden?

This is the **second version** of the model. The first version drove the
cascade with fractional receptor occupancy (`theta = C/N_total`) and turned
out, on inspection, to have no real capture/burden trade-off at all -- see
[Appendix: fractional-occupancy version](#appendix-fractional-occupancy-version)
for why, and `outputs/appendix_theta_based/` for its preserved results. The
model below fixes that by driving the cascade with the *absolute* number of
bound complexes instead.

## Equations

**Binding** (nanobody-antigen complex formation at the cell surface,
unchanged from v1):

```
dC/dt = kon * Ag * (N_total - C) - koff * C
theta = C / N_total
```

**Cascade + burden** (downstream signal production, throttled by expression
load):

```
dR/dt = k_eff * Hill(C, K_abs, n) - delta*R
Hill(C, K_abs, n) = C^n / (K_abs^n + C^n)
k_eff = k_cat_max / (1 + N_total / N_half)
```

| Symbol | Meaning |
|---|---|
| `C` | bound nanobody-antigen complexes per cell |
| `N_total` | total nanobodies expressed per cell |
| `theta` | fractional receptor occupancy, `C / N_total` (still tracked, no longer drives the cascade) |
| `Ag` | free antigen concentration (assumed constant/non-depleting) |
| `K_abs` | Hill threshold, in *absolute* bound-complex units |
| `R` | reporter signal (arbitrary units) |
| `k_eff` | burden-derated maximum cascade rate |

The only change from v1 is the argument to `Hill`: **`C` instead of
`theta`**. Binding, burden, antigen assumptions, solver settings, sweep
ranges, and the success metric are all unchanged. Simulated with
`scipy.integrate.solve_ivp`, method `LSODA`, over `t = 0` to `240` min,
`C(0) = 0`, `R(0) = 0`. `kon`/`koff` are supplied in their native SPR units
(per second) and converted to per-minute (`x60`) before integration.

## Parameters

| Parameter | Value | Grounded in data | Assumed, with justification | Swept due to being unknown |
|---|---|:---:|:---:|:---:|
| `kon` | 5.44e5 M⁻¹s⁻¹ | ✅ SPR | | |
| `koff` | 1.499e-4 s⁻¹ | ✅ SPR | | |
| `Ag` | 0.1 nM | | ✅ | |
| `K_abs` | 1000 complexes/cell | | ✅ | |
| `n` | 2 | | ✅ | |
| `delta` | 0.025 min⁻¹ | | ✅ | |
| `N_total` | 10² -- 10⁶ /cell | | | ✅ |
| `k_cat_max` | 0.005 -- 0.5 min⁻¹ | | | ✅ |
| `N_half` | 10³ -- 10⁵ /cell | | | ✅ |

**Why each assumed value is what it is:**

- **`Ag` = 0.1 nM.** Sub-`Kd` (`Kd = koff/kon ≈ 0.276 nM`), chosen to
  represent a low-concentration clinical sample rather than a saturating
  lab-bench concentration. **This still matters**: it fixes equilibrium
  occupancy at `theta_ss = kon*Ag/(kon*Ag+koff) ≈ 0.266`, meaning a cell
  only ever binds about a quarter of its nanobodies regardless of how many
  it expresses -- antigen availability caps how much signal is even
  achievable, independent of the expression-level question this model
  answers.
- **`K_abs` = 1000 bound complexes/cell.** This replaces v1's fractional
  `K = 0.4`. It's chosen to sit in the middle of the range of `C` the sweep
  actually produces: at `theta_ss ≈ 0.266`, `C = theta_ss * N_total` spans
  roughly **27** (at `N_total = 100`) to **266,000** (at `N_total = 10⁶`)
  across the swept range, so `K_abs = 1000` sits inside that range rather
  than at either edge -- the cascade neither saturates nor stays silent
  across the whole sweep, so the sweep can actually show variation on both
  sides of the threshold. See
  [Appendix](#appendix-fractional-occupancy-version) for why this had to
  become an absolute count rather than a fraction.
- **`n = 2`.** No cascade has been built or characterized yet; a Hill
  coefficient of 2 assumes mild cooperativity/ultrasensitivity typical of a
  short phosphorelay or dimerization-gated step. Unchanged from v1.
- **`delta = 0.025 min⁻¹`** (~28 min half-life) is a generic reporter
  turnover/dilution rate in the range typical of bacterial fluorescent or
  chromogenic reporters, in the absence of a measured value. Unchanged from
  v1.

**Why each swept range is what it is** (unchanged from v1):

- **`k_cat_max` (0.005 -- 0.5 min⁻¹)** spans a cascade characteristic
  timescale (`1/k_cat_max`) of 200 min (slow) down to 2 min (fast). The
  cascade mechanism hasn't been built, so this range reflects genuine
  uncertainty about how fast it could plausibly turn over, not a guess at a
  single "true" value.
- **`N_half` (10³ -- 10⁵ /cell)** spans early-onset burden (throttling
  becomes significant at only a few thousand nanobodies/cell) to late-onset
  burden (the cell tolerates tens of thousands before its cascade capacity
  is meaningfully reduced).
- **`N_total` (10² -- 10⁶ /cell)** is the design variable itself -- the
  full plausible range of per-cell expression level a promoter/RBS choice
  could realistically produce.

## Does the optimum have a genuine interior trade-off now?

**Yes, but only in a narrow corner of the swept scenarios -- reported
here exactly as the sweep found it, not adjusted to fit expectations.**

Across the full 20 x 10 x 10 grid, only **34 of 2,000** combinations reach
the detection threshold (50% of the global max `R`) within 240 min at all
(down from 165/2,000 in the v1/theta model -- the absolute-count cascade is
harder to trigger at low `N_total`, since `C` has to clear `K_abs` on its
own rather than a self-normalizing fraction). Those 34 successes come from
just **7 of the 100** `(k_cat_max, N_half)` scenarios, all clustered in the
fast-cascade / high-burden-tolerance corner of the grid
(`k_cat_max >= ~0.3 min⁻¹`, `N_half >= ~1.3x10⁴`) -- see `heatmap.png`.

**In every one of those 7 scenarios, the optimal `N_total` lands strictly
inside the swept range** -- never at the smallest (100) or largest (10⁶)
`N_total` tested -- specifically between about **12,700 and 33,600
nanobodies/cell**. That is a real, interior capture/burden trade-off: below
that range `C` hasn't cleared `K_abs` yet, so the cascade barely fires no
matter how little burden there is; above it, burden throttles `k_eff`
faster than the (already near-saturated) Hill term can compensate.
`timecourses.png` shows this directly -- `N_total = 31,623` clearly
outperforms both a 100x smaller and a 100x larger value.

The honest caveat: this trade-off only shows up where the cascade is fast
and burden-tolerant enough to produce a signal at all within 240 min. The
*median* `k_cat_max`/`N_half` scenario -- the "typical" guess used for
`timecourses.png` and as the fixed point in `sensitivity.png` -- does
**not** reach threshold at any `N_total` (0/10 combinations when `N_half`
is swept at that median `k_cat_max`; only 1/10 when `k_cat_max` is swept at
that median `N_half`). `sensitivity.png`'s bars are both 0.00, but for two
different reasons that its caption reports explicitly rather than
collapsing into "no sensitivity" -- see the figure. **The practical
reading: under the middle of our current parameter uncertainty, this
design may simply be too slow to produce a visible signal in 240 minutes,
regardless of expression level -- getting cascade speed and burden
tolerance into that upper corner (via cascade/RBS engineering, or a longer
assay window) may matter more than fine-tuning `N_total` at all.**

## Deliverables (`outputs/`)

- **`heatmap.png`** -- optimal `N_total` (minimizing time-to-threshold) as a
  function of `(k_cat_max, N_half)`, log-log axes. Gray cells never reach
  threshold within 240 min at any `N_total`; the caption reports, from the
  actual data, whether the colored cells sit on an interior optimum or a
  swept-range edge.
- **`timecourses.png`** -- `theta(t)` and `R(t)` for four representative
  `N_total` values (10^2.5, 10^3.5, 10^4.5, 10^5.5) at the sweep's median
  `k_cat_max` and `N_half`. `theta(t)` still overlaps exactly across all
  four (binding is unchanged); `R(t)` now clearly separates them.
- **`sensitivity.png`** -- how far the optimal `N_total` moves (max minus
  min, in log10 units) when `k_cat_max` is swept alone vs. when `N_half` is
  swept alone, each with the other fixed at its sweep median. The caption
  reports how many of the 10 points on each slice reached threshold at all,
  since a 0.00 span means something different with 0, 1, or many data
  points behind it.
- **`outputs/appendix_theta_based/`** -- the three PNGs (plus the raw sweep
  `.npz`) from the original fractional-occupancy version, preserved for
  comparison. See the appendix below.

The success metric behind all three (unchanged from v1): for each grid
point, find the first time `R(t)` crosses 50% of the single largest `R`
value observed anywhere on the entire sweep grid. That shared ceiling makes
every scenario judged against the same bar. Not reached within 240 min ->
recorded as `inf`, shown as gray/excluded in the plots.

## How to run

```bash
python sweep.py   # simulates the full grid, writes outputs/sweep_results.npz
python plot.py     # reads that file, writes the three PNGs to outputs/
```

`sweep.py` runs 20 x 10 x 10 = 2,000 ODE solves and takes a few seconds.
`plot.py` reruns four extra simulations directly (for `timecourses.png`)
and otherwise just reads the cached sweep results.

## How to update this model

- **New SPR data for `kon`/`koff`**: edit the two constants at the top of
  [`model.py`](model.py) (`KON_PER_S`, `KOFF_PER_S`). Everything downstream
  (including the unit conversion to per-minute) updates automatically.
- **Cascade gets characterized** (real `K_abs`, `n`, or a non-Hill response
  curve): edit `K_ABS`, `N_HILL`, and/or the `hill()` function in
  [`model.py`](model.py). If the mechanism turns out not to be a Hill
  function at all, replace the body of `hill()` -- `_rhs()` only calls it
  through that one function, so nothing else needs to change. If wet-lab
  data pins down the cascade in terms of fractional occupancy rather than
  absolute count after all, see the note at the end of the
  [Appendix](#appendix-fractional-occupancy-version) for exactly what to
  revert.
- **Burden gets characterized** (real throttling curve instead of the
  assumed `1/(1 + N_total/N_half)` form): edit `k_eff()` in
  [`model.py`](model.py). Once a real `N_half` (or equivalent burden
  parameter) is measured, move it from `sweep.py`'s swept grid into
  `model.py`'s fixed parameters, the same way `kon`/`koff` are handled now.
- **`k_cat_max` gets measured**: same move -- pull it out of `K_CAT_MAX` in
  [`sweep.py`](sweep.py) and set it as a fixed constant in `model.py`,
  collapsing the sweep from 3D to 2D (or to a single run, if all three
  swept parameters end up characterized).
- **Antigen is no longer assumed non-depleting**: this needs a new state
  variable (free `Ag(t)`) and a matching depletion term in `_rhs()` in
  [`model.py`](model.py) -- `theta(t)` would then stop being
  `N_total`-independent even in the original fractional-occupancy
  formulation.
- **Different sweep ranges or resolution**: edit `N_TOTAL`, `K_CAT_MAX`,
  `N_HALF` at the top of [`sweep.py`](sweep.py).
- **Different success threshold** (e.g. not 50% of global max): edit the
  `threshold = 0.5 * global_max_R` line in `sweep.py`'s `main()`.

After any change, re-run `python sweep.py` before `python plot.py` --
`plot.py` reads the cached `.npz` and will silently show stale results
otherwise.

## Appendix: fractional-occupancy version

The first version of this model used fractional occupancy as the Hill
input:

```
dR/dt = k_eff * Hill(theta, K, n) - delta*R
theta = C / N_total,   Hill(theta, K, n) = theta^n / (K^n + theta^n),   K = 0.4
```

with burden (`k_eff`) exactly as in the current model. This looked
reasonable -- occupancy is the natural "how activated is this receptor"
variable -- but produced a heatmap and sensitivity chart that were both
essentially flat (preserved in `outputs/appendix_theta_based/`).

**Why:** dividing the binding equation through by `N_total` gives

```
d(theta)/dt = kon*Ag*(1 - theta) - koff*theta
```

`N_total` cancels out completely. Because `Ag` is held constant
(non-depleting), **`theta(t)` is exactly independent of `N_total`** -- every
cell converges to the same occupancy trajectory
(`theta_ss = kon*Ag/(kon*Ag+koff) ≈ 0.266`) regardless of how many
nanobodies it expresses. Since `Hill(theta, ...)` only ever sees that one
shared trajectory, `N_total` could only affect `dR/dt` through `k_eff`,
which is monotonically *decreasing* in `N_total`. `dR/dt` is linear in `R`
for a given forcing term, so time-to-threshold turned out to be a strictly
increasing function of `N_total` whenever threshold was reached at
all -- making the "optimum" always the smallest `N_total` on the grid, a
corner solution rather than a trade-off. `outputs/appendix_theta_based/heatmap.png`
is uniform wherever it isn't gray, and its `sensitivity.png` shows both
spans at exactly 0.00 for that reason.

That is a mathematically correct consequence of the equations as originally
written, not a bug -- it's documented here as a legitimate step in the
model's development, not an error being smoothed over. It's also the reason
this version exists: **each bound complex is a molecular event capable of
triggering the cascade on its own**, independent of how many *other*
nanobodies a given cell happens to also be expressing, so the cascade
should plausibly read the absolute count `C`, not the population-wide
fraction `theta`. Swapping the Hill input from `theta` to `C` (with a
correspondingly re-scaled threshold, `K_abs` instead of `K`) is what turns
this into a model with a genuine capture/burden trade-off -- see
[above](#does-the-optimum-have-a-genuine-interior-trade-off-now).

To reproduce the appendix version from the current code: in `_rhs()` in
[`model.py`](model.py), change `hill(C, K_abs, n)` back to
`hill(C / N_total, K, n)` and reinstate a fractional threshold
(`K = 0.4`) in place of `K_ABS`.

## Files

```
model.py     Fixed parameters + the ODE right-hand side + simulate()
sweep.py     Sweep grids, the threshold-crossing metric, the grid search
plot.py      The three figures
outputs/     sweep_results.npz + heatmap.png, timecourses.png, sensitivity.png
             appendix_theta_based/   preserved v1 (fractional-occupancy) results
```

## References

1. Dhekale YP, Jothivel K, Ghorui SK, Gupta GD, Singh S, Baghel NS,
   Kulkarni S, Gupta PK. Isolation of a nanobody specific to the PstS-1
   protein and evaluation of its immunoreactivity with structural components
   of *Mycobacterium tuberculosis* granuloma. *Frontiers in Immunology*.
   2025;16:1684904. doi:10.3389/fimmu.2025.1684904 -- source of the measured
   binding constants (`kon`, `koff`, K_D) for the C8Nb nanobody vs. PstS-1.
2. Inouye S, Inouye M. Up-promoter mutations in the lpp gene of
   *Escherichia coli*. *Nucleic Acids Research*. 1985;13(9):3101-3110. --
   informs the upper bound of the nanobody expression sweep (~47% of total
   cellular protein achievable under an optimized lpp promoter variant).

All parameters not attributed to a reference above (antigen concentration,
Hill parameters, reporter decay, and the swept cascade/burden ranges) are
assumptions or unknowns, as documented in the sections above.

## Attribution and context

Developed by Siddharth as an external dry-lab (computational modeling)
contributor to an iGEM 2026 team's tuberculosis biosensor project.

## License

Released under the [MIT License](LICENSE).
