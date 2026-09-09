# Numerical data dictionary

All files are UTF-8 text in open formats. Dimensionless variables use the
proper acceleration $a_{\mathrm c}$ as the scale.

## `circular_detector_response.csv`

The 99 rows used directly in Figure 1:

- `E_over_a_c`: positive detector gap $E/a_{\mathrm c}$;
- `rate_over_a_c_v_*`: excitation rate
  $\dot{\mathcal F}(E)/a_{\mathrm c}$ at the indicated speed;
- `a_c_beta_eff_v_*`: effective inverse temperature
  $a_{\mathrm c}\beta_{\mathrm{eff}}(E)$ at the indicated speed.

## `full_grid_convergence.csv`

One row for each of 99 gaps and three speeds (297 rows total):

- production and tighter-reference rates;
- their absolute difference;
- their relative difference, normalized by the reference rate.

## `cutoff_convergence.csv`

Three cutoff triplets for each speed. For every first cutoff $U$, the file
records $R_U,R_{2U},R_{4U}$, consecutive absolute differences, and

$$
p=\log_2\frac{|R_U-R_{2U}|}{|R_{2U}-R_{4U}|}.
$$

The test is evaluated at $E/a_{\mathrm c}=0$, where the remainder is
nonoscillatory and exhibits the conservative $U^{-3}$ order.

## `independent_quadpack_check.csv`

Five representative comparisons between the production calculation and
QUADPACK QAWFE applied directly on the infinite interval. The final two
columns give QUADPACK's absolute error estimate and number of Fourier cycles.
That estimate is adaptive and is not a guaranteed bound; convergence is
established separately from QUADPACK's per-cycle status codes, and the
mode-sum table below provides an independent comparison over the whole grid.

## `mode_sum_cross_check.csv`

One row for each of 99 gaps and three speeds (297 rows total), comparing the
production rate with the manuscript's mode sum, Eq. (94), evaluated by
`mode_sum_rate.py`. That evaluation shares no step with the Eq. (110)
production path: no Hadamard subtraction, no Taylor branch, no cutoff, no
analytic tail. Columns are the production rate, the mode-sum rate, and their
absolute and relative differences.

## `numerical_error_budget.json`

Machine-readable summary of maxima, their locations, the analytic tail bound,
numerical parameters, and the software/platform versions used for the run.

## `numerical_convergence.txt`

Human-readable summary of the same error budget. This file ends with an
explicit pass statement only when all publication acceptance thresholds are
satisfied.
