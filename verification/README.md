# Symbolic verification of the manuscript

`verify_manuscript.ipynb` re-derives the equations of *Field quantization in
rotating frames: coordinate covariance and the circular-detector response* with
SymPy, in the order in which they appear in the paper, and reports a verdict for
each one. It closes with a section that checks the code in `../code/` against
the equations it has just verified.

## Running it

From the package root, install both requirement files; the lock file supplies
NumPy, SciPy, SymPy and mpmath, and the verification file supplies the notebook
tooling. Neither is sufficient on its own.

```bash
python -m pip install -r requirements-lock.txt -r requirements-verification.txt
python verification/run_verification.py
```

`run_verification.py` executes the notebook from a temporary directory and
leaves the tracked file untouched. Use `--inplace` to store the outputs back
into `verify_manuscript.ipynb`, or `--output PATH` to write the executed
notebook elsewhere; continuous integration uses `--output` so that the uploaded
artifact carries the outputs of that run rather than the committed ones.

Alternatively, open `verify_manuscript.ipynb` in Jupyter and run all cells.
Execution takes about two minutes.

The notebook raises an `AssertionError` on the first identity that fails, so a
clean execution is part of the pass criterion. It is not the whole of it: a
deleted or skipped cell would also execute cleanly, so `run_verification.py`
additionally checks that no cell errored and that the statement inventory
matches the manifest it carries.

## What the verdicts mean

Each check prints one line carrying one of the verdicts below, and under it
the statement being checked, typeset in the notation of the manuscript.

| verdict | kind of evidence |
|---|---|
| `PASS [symbolic identity]` | SymPy reduced the difference between the manuscript's expression and an independently rebuilt one to exactly zero. |
| `PASS [arbitrary precision]` | A convergent integral, special-function identity, or transcendental limit, evaluated with `mpmath`. The working precision, the achieved agreement and the tolerance are all printed. |
| `PASS [implementation]` | A double-precision comparison between the shipped code and a reference value. This is a floating-point agreement, not a proof, and is bounded by float64 rounding however good the reference is. |
| `PASS [inequality]` | A strict inequality checked over a stated finite sample. The sample and the achieved margin are printed. A finite sample is evidence, not a proof for every index. |
| `PASS [consequence]` | Follows immediately from an equation already verified above. Nothing new is computed; the dependency is printed. |
| `PASS [definition]` | Introduces notation or names an object. Any algebraic content it carries is still checked. |
| `NOT VERIFIED` | Distributional (Dirac deltas, $s-\mathrm i\epsilon$ boundary values, Fourier representations of $\delta$) or operator valued. SymPy has no faithful model for these, so no claim is made. The reason is printed, together with whatever surrogate check is possible. |

`NOT VERIFIED` is a statement about the notebook, not about the manuscript.

**The counts below are recorded statements, not independent proofs.** Several
equations contribute more than one line and many follow from one another; the
breakdown by category is the informative part, not the total.

**The expressions are transcribed from the manuscript by hand.** The notebook
does not parse the LaTeX source, so a clean run certifies agreement with the
revision whose SHA-256 it records at the top, and not with a later one.

## Structure

The notebook mirrors the paper. Its table of contents links to one section per
manuscript section and one per appendix. Where a main-text equation is a quoted
result whose algebra the paper carries out in an appendix, the main-text cell
says so and links to the corresponding appendix section of the notebook, where
the derivation is verified step by step. This applies to Eqs. (11)-(12)
(Appendix A), Eq. (70) (Appendix B), Eq. (74) (Appendix C), and Eqs. (76)-(79)
(Appendix D).

## Result of the shipped run

208 recorded statements:

- 138 symbolic identities reduced to exact zero,
- 5 arbitrary-precision numerical checks,
- 11 double-precision comparisons of the shipped code,
- 2 strict inequalities over a stated finite sample,
- 15 immediate consequences of equations verified above,
- 25 definitional,
- 12 not verified, all of them distributional or operator valued.

`run_verification.py` compares this breakdown against a manifest, so a deleted
or skipped cell fails the run rather than quietly reducing the totals.

The twelve unverified statements are Eqs. (5), (6), (7), (14) (axial part),
(15), (57), (74), (93), (99), (106), (C5) and (C7). Every one of them is a
Dirac delta, a Fock-space operator identity, or a contour integral of a
boundary value. Where a surrogate exists it is checked: the delta-function
normalisations are backed by the exact coefficient arithmetic of Eqs. (16) and
(C6) and by the mode sums of Eqs. (65) and (76); Eq. (106) is backed by the
exact identity $\dot{\mathcal F}(-E)-\dot{\mathcal F}(E)=E/2\pi$, which
is what that term produces.

## Notable checks

- **Eq. (24)** is verified by applying the manuscript's own subtraction,
  Eq. (110), to the uniformly accelerated correlator and recovering the Planck
  form in closed form, **at gaps of both signs**. This exercises the Hadamard
  subtraction, the finite/tail split and the restored inertial term of
  Eq. (106) on a case whose answer is known independently.
- **Eqs. (94) and (110)** are compared directly at all 297 plotted points: the
  mode sum uses no Hadamard subtraction, no Taylor branch, no cutoff and no
  analytic tail, so the agreement relates two manuscript equations through two
  disjoint numerical routes.
- **Eq. (36)** (the TT metric) is rebuilt entry by entry the way the
  manuscript obtains it, by differentiating the coordinate transformation
  (35) and substituting it into the Minkowski line element. **Eq. (B2)**, the
  identity that drives the whole of Appendix B, is proved from $C^2-S^2=1$
  alone.
- **Eq. (72)** is checked by direct substitution of the TT mode into the exact
  TT wave operator, using only Bessel's equation for the radial factor.
- $\rho_{\mathrm{rig}}$ and $\rho_{\mathrm{TT}}$ (Eqs. (67) and (78)) are
  rebuilt from Cartesian components at the image points, so those checks are
  independent of Eq. (19) and of the manuscript's own manipulation.

## Corrections this notebook contributed

Two errors were found here and are fixed in the checked revision of the
manuscript.

1. **Orientation reversal, Sec. III B.** Reversing
   $\Omega_{\mathrm{TT}}\to-\Omega_{\mathrm{TT}}$ leaves $C=\cosh\eta$
   unchanged and flips $S=\sinh\eta$, so the *time* shift reverses,
   $-2\pi rS\mapsto+2\pi rS$, while the *angular* shift does not: it is
   $+2\pi C$ in both orientations. An earlier revision printed
   $\theta-2\pi C$.
2. **Support of the energy-conserving delta, Eqs. (95) and (101).** The delta
   is supported on an *equality*,
   $m\Omega_{\mathrm{rig}}=\omega+E/\gamma$, not on the strict inequality
   an earlier revision printed. The physically relevant consequence is stated
   separately: for $E>0$ every contributing mode has corotating frequency
   exactly $-E/\gamma$, hence negative. The notebook now
   checks the equality and the consequence as two separate statements, so that
   its label and its assertion describe the same thing.
