# Circular-detector response: reproducibility package

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22678695.svg)](https://doi.org/10.5281/zenodo.22678695)
[![arXiv](https://img.shields.io/badge/arXiv-2609.10390-b31b1b.svg)](https://arxiv.org/abs/2609.10390)

Version 1.0.0 of the code, numerical data, error budget, figure, and
symbolic verification notebook associated
with the manuscript *Field quantization in rotating frames: coordinate
covariance and the circular-detector response* by Sidney Natzuka Junior,
Carlos Augusto Domingues Zarro, and Matheus dos Santos Soares,
[arXiv:2609.10390](https://arxiv.org/abs/2609.10390) [gr-qc].

The archive reproduces Figure 1 and every numerical-accuracy statement in the
manuscript's final appendix. No external input data are required; the software
dependencies are listed below.

## Scientific scope

The calculation concerns a massless scalar field in $(3+1)$-dimensional
Minkowski spacetime and an Unruh--DeWitt detector on a circular worldline. The
proper acceleration is set to $a_{\mathrm c}=1$; consequently, all reported
gaps, rates, and inverse temperatures are dimensionless:

- gap: $E/a_{\mathrm c}$,
- rate: $\dot{\mathcal F}/a_{\mathrm c}$,
- inverse temperature: $a_{\mathrm c}\beta_{\mathrm{eff}}$.

The figure contains 99 gaps from $0.08$ to $4.00$ for each of the speeds
$v=0.30,0.60,0.85$, for a total of 297 response values.

## Repository contents

```text
.
|-- CITATION.cff                    citation metadata for GitHub and archives
|-- CHANGELOG.md                    release notes and implementation notes
|-- LICENSE                         BSD 3-Clause license
|-- README.md                       this reproducibility guide
|-- .github/workflows/              Linux/Windows continuous reproduction
|-- requirements.txt                compatible dependency ranges
|-- requirements-lock.txt           exact environment used for the release
|-- requirements-verification.txt   extra pins for running the notebook
|-- code/
|   |-- hadamard_g11.py             correlation and response implementation
|   |-- mode_sum_rate.py            independent evaluation from Eq. (94)
|   |-- reference_rates.py          optional: regenerates the reference constants
|   |-- publication_config.py       single source of numerical parameters
|   |-- plot_detector_response.py   plotted data and Figure 1
|   |-- numerical_error_budget.py   all-grid, cutoff, and independent checks
|   |-- reproduce_all.py            canonical end-to-end entry point
|   |-- test_hadamard_g11.py        unit and numerical regression tests
|   `-- validate_outputs.py         validation of generated artifacts
|-- data/
|   |-- README.md                   schemas and interpretation
|   |-- circular_detector_response.csv
|   |-- full_grid_convergence.csv
|   |-- cutoff_convergence.csv
|   |-- independent_quadpack_check.csv
|   |-- mode_sum_cross_check.csv
|   |-- numerical_error_budget.json
|   `-- numerical_convergence.txt
|-- figures/
|   |-- circular_detector_response.pdf
|   `-- circular_detector_response.png
`-- verification/
    |-- README.md                   verdict legend and structure
    |-- verify_manuscript.ipynb     SymPy verification, equation by equation
    `-- run_verification.py         headless execution and pass/fail check
```

## Symbolic verification

`verification/verify_manuscript.ipynb` re-derives every equation of the
manuscript with SymPy, in the order in which they appear, and reports one of
seven verdicts for each, keeping symbolic identities, arbitrary-precision
numerical checks, double-precision comparisons of the shipped code, strict
inequalities over a finite sample, consequences, definitions, and statements
that are deliberately not verified in separate categories. Its last section
checks the code in `code/` against the equations it has just verified, and
compares the manuscript's Eq. (94) with Eq. (110) at all 297 plotted points.

```bash
python -m pip install -r requirements-lock.txt -r requirements-verification.txt
python verification/run_verification.py
```

See `verification/README.md` for the verdict legend and for the list of
statements that are deliberately not verified.

## Numerical method

For $u=a_{\mathrm c}s$, $x=u/(2\gamma v)$, and

$$
N=4\gamma^4v^4\left(x^2-\sin^2x\right),
$$

the Hadamard-subtracted correlation function is evaluated in the stable form

$$
\frac{\Delta\mathcal W(u)}{a_{\mathrm c}^2}
=\frac{N}{4\pi^2u^2(u^2+N)}.
$$

For $|x|<0.15$, the code uses the series

$$
x^2-\sin^2x=\frac{x^4}{3}-\frac{2x^6}{45}+\frac{x^8}{315}
-\frac{2x^{10}}{14175}+\frac{2x^{12}}{467775}+\mathcal O(x^{14}),
$$

and sets the coincidence value analytically to $1/(48\pi^2)$. The production
response uses 24-point composite Gauss--Legendre quadrature on unit panels up
to $U=800$, followed by analytic integration of the leading $u^{-2}$ tail.

The error budget has four components:

1. **Complete-grid refinement.** All 297 plotted rates are recomputed with
   32-point quadrature, panel width $0.75$, and $U=1200$.
2. **Cutoff control.** The analytic remainder bound is evaluated at $U=800$,
   and the conservative third-order cutoff behavior is measured at
   $E/a_{\mathrm c}=0$ using $U=100,200,400,800,1600$.
3. **Independent adaptive calculation.** Five representative rates are
   recomputed by QUADPACK's adaptive infinite-interval Fourier algorithm
   (QAWFE), applied directly to the unsplit stable integrand on $[0,\infty)$.
   This check uses neither the finite-panel rule nor the analytic tail
   correction of the production method. Its convergence is taken from
   QUADPACK's own per-cycle diagnostics, not from the absence of a warning,
   and its adaptive error estimate is treated as an estimate rather than a
   bound.
4. **Independent mode-sum calculation.** All 297 rates are recomputed from the
   manuscript's Eq. (94) by `mode_sum_rate.py`, which shares no step with the
   Eq. (110) production path: no Hadamard subtraction, no Taylor branch, no
   cutoff and no analytic tail. Its own truncation is measured by doubling the
   mode count and the angular order rather than estimated adaptively.

## Exact reproduction

Use Python 3.11 for the archival reproduction. The archived results were
generated with Python 3.11.9, NumPy 2.4.6, SciPy 1.15.1, Matplotlib 3.10.0,
SymPy 1.13.1, and mpmath 1.3.0, on Windows x86-64.

Create an isolated environment from the package root:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Generate and validate every artifact with one command:

```bash
python code/reproduce_all.py
```

The script resolves all output paths relative to this package, so it may be
called from any working directory. A successful run ends with
`Reproduction completed successfully.`

Run the unit and regression test suite separately with:

```bash
python -m unittest discover -s code -p "test_*.py" -v
```

Run the symbolic verification of the manuscript with:

```bash
python -m pip install -r requirements-lock.txt -r requirements-verification.txt
python verification/run_verification.py
```

No LaTeX installation is required. The vector PDF uses Matplotlib's STIX math
renderer, and the PNG is exported at 600 dpi.

## Expected numerical results

Small last-bit differences can occur across operating systems or math
libraries. With the locked environment, the checks should reproduce:

| Check | Maximum absolute difference | Maximum relative difference |
|---|---:|---:|
| 297-point refined quadrature | approximately `1.2e-12` | approximately `1.6e-7` |
| Five-point QUADPACK Fourier check | approximately `1.5e-12` | approximately `1.0e-7` |
| 297-point mode-sum cross-check | approximately `1.5e-12` | approximately `1.0e-7` |

These are measured differences between independent approximations. They are
not certified error bounds, and they are distinct from the conservative
analytic bound on the omitted tail. Do not describe the plotted rates as
accurate to 30 digits.

The observed conservative cutoff orders lie between `2.95` and `3.03`. The
analytic remainder bound at the production cutoff is at most `6.9e-11`, or
`1.3e-3` relative to the smallest plotted rate. Exact values and their
locations are recorded in `data/numerical_error_budget.json`.

The mode-sum truncation itself is `1.3e-16`, far below the differences above.

## Mapping between claims and archived files

| Manuscript result | Reproduction source | Archived evidence |
|---|---|---|
| Figure 1 curves | `plot_detector_response.py` | figure PDF/PNG and `circular_detector_response.csv` |
| 297-point refinement | `numerical_error_budget.py` | `full_grid_convergence.csv` |
| Third-order cutoff convergence | `numerical_error_budget.py` | `cutoff_convergence.csv` |
| Analytic tail-error bound | `numerical_error_budget.py` | `numerical_error_budget.json` |
| Independent Fourier check | `numerical_error_budget.py` | `independent_quadpack_check.csv` |
| Mode-sum cross-check, Eq. (94) vs Eq. (110) | `mode_sum_rate.py` | `mode_sum_cross_check.csv` |
| Human-readable summary | `numerical_error_budget.py` | `numerical_convergence.txt` |
| Manuscript equations, one by one | `verification/verify_manuscript.ipynb` | executed notebook with per-equation verdicts |

## Citation and archival release

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub uses
this file to display a **Cite this repository** entry. For publication, create
a tagged GitHub release (`v1.0.0`) and archive that immutable release with
Zenodo. Cite the Zenodo record and its DOI rather than a mutable branch. The
included `.zenodo.json` supplies the release metadata.

This package is what the manuscript's data availability statement refers to.
That statement reads:

> The data that support the findings of this study, and the code that
> generates them, are openly available in the reproducibility package of
> Ref. [N]. That package contains the implementation of Eq. (110), the
> response values plotted in Fig. 1, the numerical error budget reported in
> Appendix E, and a notebook of symbolic derivations and numerical
> consistency checks of the equations presented here. No other data were
> generated or analyzed in this study.

The same statement is entered in the journal's submission form, with `[N]`
resolving to the citation of this archived release.

## License

The source code and accompanying files are distributed under the
[BSD 3-Clause License](LICENSE). Cite the archived release when reusing the
software or numerical data.
