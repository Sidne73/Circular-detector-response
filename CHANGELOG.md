# Changelog

All notable changes to this reproduction package are documented here.

## 1.0.0 - 2026-09-03

First public release, accompanying *Field quantization in rotating frames:
coordinate covariance and the circular-detector response*,
[arXiv:2609.10390](https://arxiv.org/abs/2609.10390) [gr-qc].

Contents:

- the production calculation and the source data for Figure 1;
- a numerical error budget with four complementary checks: refinement of the
  quadrature over all 297 plotted values, the conservative cutoff order and
  analytic tail bound, an adaptive QUADPACK Fourier calculation at five
  representative points, and a full-grid cross-check against the manuscript's
  mode sum, Eq. (94), which shares no step with the Eq. (110) production path;
- `verification/verify_manuscript.ipynb`, a notebook presenting symbolic
  derivations and numerical consistency checks of the manuscript's equations
  in manuscript order, with the scope of each check stated explicitly;
- unit and regression tests, artifact validation, an end-to-end reproduction
  command, exact environment pins, citation and archive metadata, and a
  BSD 3-Clause license.

Numerical accuracy on the published grid (massless field in $3+1$ dimensions,
$v\in\{0.30,0.60,0.85\}$, $E/a_{\mathrm c}\in[0.08,4.00]$), measured on
CPython 3.11.9 / NumPy 2.4.6 / SciPy 1.15.1:

| comparison | maximum absolute | maximum relative |
|---|---:|---:|
| production vs refined quadrature, 297 points | 1.2e-12 | 1.6e-7 |
| production vs QUADPACK QAWFE, 5 points | 1.5e-12 | 1.0e-7 |
| production vs mode sum Eq. (94), 297 points | 1.5e-12 | 1.0e-7 |

The mode-sum truncation itself (300 vs 600 modes) is 1.3e-16. These are
measured differences between independent approximations, not certified error
bounds; the conservative analytic bound on the omitted tail is 6.9e-11.

Implementation notes that a reader of the manuscript's Appendix E may want:

- The Taylor branch for $x^2-\sin^2x$ retains terms through $x^{12}$, as
  printed in Eq. (E4). The first omitted term is $-4x^{14}/42567525$, so at
  the $|x|<0.15$ switch the truncation is about `1.63e-15` relative; the
  direct branch just above the switch is limited instead by its own
  double-precision cancellation, at roughly `1e-14`.
- The subtracted correlator is evaluated as

  $$
  \frac{a_{\mathrm c}^2P(x)}{16\pi^2\left[1+u^2P(x)/4\right]},
  \qquad P(x)=\frac{x^2-\sin^2x}{x^4},
  $$

  which is algebraically Eq. (E3) with the fourth power of $u$ cancelled
  analytically. This avoids underflow at very small nonzero proper times and
  removes the need for a special case at $u=0$, where $P(0)=1/3$ gives the
  coincidence limit of Eq. (109) directly.
- The boundary value $s\to s-\mathrm i\epsilon$ in `circular_wightman_g11`
  acts on the temporal term only, as in Eq. (104); that routine is diagnostic and is not
  used by the production path, which needs no regulator.
