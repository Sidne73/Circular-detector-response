"""Regenerate the high-precision reference rates used by the test suite.

The tests in :mod:`test_hadamard_g11` compare the production quadrature with
five reference values of Eq. (110).  Those values are stored as ordinary
double-precision literals, but they are *rounded* arbitrary-precision results,
and this module is the procedure that produces them, so that they are not
unexplained constants.

Two independent arbitrary-precision methods are implemented and cross-checked:

``quadosc``
    ``mpmath.quadosc`` applied to
    ``2 int_0^inf cos(E u) DeltaW(u) du`` with ``DeltaW`` evaluated from the
    manuscript's Eq. (E3).  This is the same integral the production code
    evaluates, at 35 working digits and without the finite cutoff or the
    analytic tail split.

``halfperiod``
    the same integral accumulated one half-period of ``cos(E u)`` at a time up
    to a finite ``U``, plus the leading tail of Eq. (E5) integrated in closed
    form by Eq. (E6), and an arbitrary-precision remainder beyond ``U``.

The values quoted in the tests are the ``quadosc`` ones.  ``halfperiod`` is a
corroboration with a different failure mode, not an equal-accuracy competitor:
it truncates the oscillatory remainder beyond a finite range, so its agreement
with ``quadosc`` degrades from about 1e-21 at the smallest gap to about 3e-9 at
the largest.  The independent QUADPACK and mode-sum calculations in
:mod:`numerical_error_budget` corroborate the same numbers along a third and a
fourth route.

This module is optional: it is not imported by the production path and is not
executed by ``reproduce_all.py``, because ``mpmath.quadosc`` takes minutes.
Run it directly to regenerate the constants:

    python code/reference_rates.py
"""

from __future__ import annotations

import sys

import mpmath as mp

from publication_config import INDEPENDENT_POINTS

WORKING_DIGITS = 35


def _subtracted(u, speed):
    """DeltaW(u)/a_c^2 from Eq. (E3), in arbitrary precision."""

    v = mp.mpf(speed)
    gamma = 1 / mp.sqrt(1 - v**2)
    x = u / (2 * gamma * v)
    numerator = 4 * gamma**4 * v**4 * (x**2 - mp.sin(x) ** 2)
    return numerator / (4 * mp.pi**2 * u**2 * (u**2 + numerator))


def reference_rate_quadosc(energy_over_a_c, speed):
    """Eq. (110) for E > 0 via mpmath's oscillatory quadrature."""

    gap = mp.mpf(energy_over_a_c)
    return 2 * mp.quadosc(
        lambda u: _subtracted(u, speed) * mp.cos(gap * u), [0, mp.inf], omega=gap
    )


def reference_rate_halfperiod(energy_over_a_c, speed, cutoff=200):
    """Eq. (110) for E > 0 by half-period accumulation plus an analytic tail."""

    gap = mp.mpf(energy_over_a_c)
    v = mp.mpf(speed)
    upper = mp.mpf(cutoff)
    half = mp.pi / gap

    total = mp.mpf(0)
    left = mp.mpf(0)
    while left < upper:
        right = min(left + half, upper)
        total += mp.quad(
            lambda u: _subtracted(u, speed) * mp.cos(gap * u), [left, right]
        )
        left = right

    leading = (v**2 / (4 * mp.pi**2)) * (
        mp.cos(gap * upper) / upper - gap * (mp.pi / 2 - mp.si(gap * upper))
    )

    remainder = mp.mpf(0)
    left = upper
    while left < upper + 4000 * half:
        right = left + half
        piece = mp.quad(
            lambda u: (_subtracted(u, speed) - v**2 / (4 * mp.pi**2 * u**2))
            * mp.cos(gap * u),
            [left, right],
        )
        remainder += piece
        left = right
        if abs(piece) < mp.mpf(10) ** (-(WORKING_DIGITS - 3)) and left > upper + 10 * half:
            break

    return 2 * (total + leading + remainder)


def generate(points=INDEPENDENT_POINTS, digits=WORKING_DIGITS):
    """Return {(speed, gap): (quadosc, halfperiod, agreement)} for `points`."""

    mp.mp.dps = digits
    table = {}
    for speed, gap in points:
        first = reference_rate_quadosc(gap, speed)
        second = reference_rate_halfperiod(gap, speed)
        table[(speed, gap)] = (first, second, abs(first - second) / abs(first))
    return table


def main() -> int:
    table = generate()
    print(f"Eq. (110) reference rates at {WORKING_DIGITS} working digits")
    print("Paste the rounded values into test_hadamard_g11.py.\n")
    worst = mp.mpf(0)
    for (speed, gap), (first, second, agreement) in sorted(table.items()):
        worst = max(worst, agreement)
        print(f"    ({speed:.2f}, {gap:.2f}): {mp.nstr(first, 17)},"
              f"   # methods agree to {float(agreement):.1e}")
    print()
    print(f"worst cross-method disagreement: {float(worst):.3e}")
    # The half-period method truncates its oscillatory tail, so at the
    # largest gap it is only good to a few times 1e-9 relative.  A
    # disagreement much larger than that would indicate a real problem in
    # one of the two methods rather than a known limitation of one.
    if worst > mp.mpf(10) ** -8:
        print("WARNING: the two methods disagree more than expected",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
