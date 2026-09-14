"""Hadamard-subtracted G11 and circular-detector response.

This module is intentionally self-contained.  It implements the single-detector
Wightman function restricted to a uniformly rotating worldline, subtracts the
universal inertial Hadamard double pole in a cancellation-free form, and
Fourier-transforms the regular remainder.

The conventions are those of the associated manuscript:

    x(s) = (gamma*s, R, phi_0 + gamma*Omega*s, z_0),
    a_c  = gamma**2 * v**2 / R,
    v    = R*Omega,

with metric signature (+---).  The response convention is

    Fdot(E) = integral ds exp(-i E s) G11(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sici


FOUR_PI_SQ = 4.0 * np.pi**2


@dataclass(frozen=True)
class CircularDetector:
    """Kinematics of a circular detector at fixed proper acceleration."""

    speed: float
    proper_acceleration: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.speed) or not 0.0 < self.speed < 1.0:
            raise ValueError("speed must be finite and satisfy 0 < v < 1")
        if (not np.isfinite(self.proper_acceleration)
                or self.proper_acceleration <= 0.0):
            raise ValueError("proper_acceleration must be finite and positive")

    @property
    def gamma(self) -> float:
        return 1.0 / np.sqrt(1.0 - self.speed**2)

    @property
    def radius(self) -> float:
        return self.gamma**2 * self.speed**2 / self.proper_acceleration

    @property
    def angular_velocity(self) -> float:
        return self.proper_acceleration / (self.gamma**2 * self.speed)


def circular_wightman_g11(
    proper_time_difference,
    detector: CircularDetector,
    epsilon: float = 1.0e-8,
):
    """Boundary-value representation of the unrenormalized circular G11.

    This is manuscript Eq. (104),

        W_circ(s) = -1/(4 pi^2) / [gamma^2 (s - i eps)^2
                                   - 4 R^2 sin^2(gamma Omega s / 2)],

    in which the boundary-value shift ``s -> s - i eps`` descends from
    ``Delta T -> Delta T - i eps`` in Eq. (9) and therefore acts on the
    temporal term only; the spatial chord ``4 R^2 sin^2(gamma Omega s / 2)``
    is evaluated at real ``s``.

    ``epsilon`` exposes the usual i-epsilon prescription for diagnostic use.
    Numerical response calculations should use :func:`hadamard_subtracted_g11`
    instead; it removes the double pole analytically and needs no regulator.
    """

    s = np.asarray(proper_time_difference, dtype=float)
    gamma = detector.gamma
    radius = detector.radius
    omega = detector.angular_velocity
    temporal = (gamma * (s - 1j * epsilon)) ** 2
    chord = 4.0 * radius**2 * np.sin(0.5 * gamma * omega * s) ** 2
    return -1.0 / (FOUR_PI_SQ * (temporal - chord))


def inertial_hadamard_g11(proper_time_difference, epsilon: float = 1.0e-8):
    """Universal inertial Hadamard double pole at equal spatial position."""

    s = np.asarray(proper_time_difference, dtype=np.complex128) - 1j * epsilon
    return -1.0 / (FOUR_PI_SQ * s**2)


SERIES_THRESHOLD = 0.15
"""Switch between the Taylor series and direct evaluation of x^2-sin^2(x).

Below the threshold the direct difference cancels; above it the truncated
series would be the less accurate branch.  With the terms retained in
:func:`_x2_minus_sin2` the two branches meet at a relative accuracy of about
1e-14, set by the double-precision cancellation of the direct branch, so the
switch point is no longer a visible feature of the integrand.
"""


def _p_ratio(x: np.ndarray) -> np.ndarray:
    r"""Return P(x) = (x^2 - sin^2 x) / x^4, with P(0) = 1/3.

    This is Eq. (E4) divided by x^4.  Working with ``P`` rather than with
    ``x^2 - sin^2 x`` keeps :func:`hadamard_subtracted_g11` free of any
    fourth power of the proper time, which would otherwise underflow to zero
    for very small nonzero arguments (``u = 1e-90`` gave 0 instead of the
    coincidence value in versions before this one).  The production quadrature
    never samples that regime, so no published number depends on the change.

    Below ``SERIES_THRESHOLD`` the series of :func:`_x2_minus_sin2` is used
    with the leading ``x^4`` divided out; above it the direct quotient is
    accurate because the numerator no longer cancels.
    """

    # P is even in x, and evaluating it at |x| makes that exact rather than
    # only mathematical.  The direct branch squares np.sin(x), and a vectorised
    # sin is not guaranteed to satisfy sin(-x) == -sin(x) to the last bit:
    # NumPy picks a SIMD kernel from the host's CPU features, so P(x) and P(-x)
    # could differ by one ULP on some machines and not on others.
    x = np.abs(np.asarray(x, dtype=float))
    x2 = x * x
    small = x < SERIES_THRESHOLD
    series = (
        1.0 / 3.0
        - x2 * (
            2.0 / 45.0
            - x2 * (
                1.0 / 315.0
                - x2 * (2.0 / 14175.0 - x2 * 2.0 / 467775.0)
            )
        )
    )
    safe = np.where(small, 1.0, x)
    direct = (safe**2 - np.sin(safe) ** 2) / safe**4
    return np.where(small, series, direct)


def _x2_minus_sin2(x: np.ndarray) -> np.ndarray:
    r"""Return x^2-sin^2(x) without cancellation near the origin.

    This is manuscript Eq. (E4).  Writing x^2-sin^2(x) = x^2-(1-cos 2x)/2 and
    expanding the cosine gives the alternating series

        x^4/3 - 2x^6/45 + x^8/315 - 2x^10/14175 + 2x^12/467775 + O(x^14),

    which is evaluated in Horner form for ``|x| < SERIES_THRESHOLD``.  The
    first omitted term is -4 x^14 / 42567525, so at the threshold the
    truncation is 1.6e-15 relative -- an order of magnitude below the ~1e-14
    cancellation error of the direct branch just above the threshold, which is
    therefore what limits the accuracy of this routine.

    Retaining only the x^8 term, as in prerelease versions of this package,
    left a 4.8e-9 relative error at the threshold that propagated to about
    1.7e-12 absolute in the response, and so exceeded the quadrature error
    budget of Appendix E that it was supposed to sit below.
    """

    # Even in x; see the note in _p_ratio on why |x| is taken here.
    x = np.abs(np.asarray(x, dtype=float))
    x2 = x * x
    small = x < SERIES_THRESHOLD
    series = x2 * x2 * (
        1.0 / 3.0
        - x2 * (
            2.0 / 45.0
            - x2 * (
                1.0 / 315.0
                - x2 * (2.0 / 14175.0 - x2 * 2.0 / 467775.0)
            )
        )
    )
    # The placeholder keeps the discarded branch away from the cancellation
    # region; np.where evaluates both arguments.
    direct = x2 - np.sin(np.where(small, 1.0, x)) ** 2
    return np.where(small, series, direct)


def hadamard_subtracted_g11(proper_time_difference, detector: CircularDetector):
    r"""Return Delta G11 = G11_circ - G11_in in stable closed form.

    With ``u=a_c*s``, ``x=u/(2 gamma v)``, and

        N = 4 gamma^4 v^4 (x^2 - sin^2 x),

    the direct difference

        a_c^2/(4 pi^2) [1/u^2 - 1/(u^2+N)]

    is evaluated as ``a_c^2 N/[4 pi^2 u^2(u^2+N)]``.  Substituting
    ``N = u^4 P(x)/4`` with ``P(x) = (x^2 - sin^2 x)/x^4`` cancels the
    fourth power of ``u`` analytically and leaves

        a_c^2 P(x) / [16 pi^2 (1 + u^2 P(x)/4)],

    which is what is evaluated here.  This form has no underflow at small
    nonzero ``u`` and needs no special case at ``u = 0``, where ``P(0) = 1/3``
    returns the coincidence limit a_c^2/(48 pi^2) of Eq. (109) directly.
    """

    s = np.asarray(proper_time_difference, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("proper_time_difference must be finite")
    acceleration = detector.proper_acceleration
    u = acceleration * s
    x = u / (2.0 * detector.gamma * detector.speed)
    ratio = _p_ratio(x)
    return acceleration**2 * ratio / (
        16.0 * np.pi**2 * (1.0 + u * u * ratio / 4.0)
    )


def dimensionless_subtracted_g11(u, speed: float):
    """Return Delta G11/a_c^2 as a function of u=a_c*s."""

    detector = CircularDetector(speed=speed, proper_acceleration=1.0)
    return hadamard_subtracted_g11(u, detector)


@lru_cache(maxsize=8)
def composite_gauss_legendre_rule(
    cutoff: float = 800.0,
    panel_width: float = 1.0,
    order: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for a composite rule on [0, cutoff]."""

    if cutoff <= 0.0 or panel_width <= 0.0 or order < 2:
        raise ValueError("cutoff, panel_width, and order must be positive")
    panel_count_float = cutoff / panel_width
    panel_count = int(round(panel_count_float))
    if not np.isclose(panel_count_float, panel_count, rtol=0.0, atol=1.0e-12):
        raise ValueError("cutoff must be an integer multiple of panel_width")

    local_x, local_w = leggauss(order)
    left = panel_width * np.arange(panel_count, dtype=float)
    nodes = left[:, None] + 0.5 * panel_width * (local_x[None, :] + 1.0)
    weights = np.broadcast_to(0.5 * panel_width * local_w, nodes.shape)
    return nodes.ravel(), weights.ravel().copy()


def _leading_tail_integral(energy_over_a, cutoff: float) -> np.ndarray:
    r"""Integral_U^infinity cos(e*u)/u^2 du for non-negative e."""

    energy = np.atleast_1d(np.asarray(energy_over_a, dtype=float))
    frequency = np.abs(energy)
    argument = frequency * cutoff
    sine_integral = sici(argument)[0]
    tail = np.cos(argument) / cutoff - frequency * (0.5 * np.pi - sine_integral)
    return np.where(frequency == 0.0, 1.0 / cutoff, tail)


def circular_response_rate(
    energy_over_a,
    speed: float,
    *,
    cutoff: float = 800.0,
    panel_width: float = 1.0,
    order: int = 24,
) -> np.ndarray:
    r"""Return Fdot(E)/a_c using the Hadamard-subtracted representation.

    The leading ``v^2/(4 pi^2 u^2)`` tail is integrated analytically.  The
    inertial spontaneous-emission term is restored for negative gaps.

    Supported domain
    ----------------
    The default rule (``cutoff=800``, ``panel_width=1``, ``order=24``) has
    been characterised only on the published grid: a massless field in 3+1
    dimensions, ``speed`` in {0.30, 0.60, 0.85} and ``|E|/a_c`` in
    [0.08, 4.00].  There it agrees with a refined rule to 1.2e-12 absolutely,
    and the omitted tail is bounded in Appendix E of the manuscript.

    The function accepts any finite gap and any ``0 < speed < 1``, but those
    guarantees do not extend automatically.  Larger gaps oscillate faster and
    need a smaller ``panel_width`` or a larger ``order``; speeds approaching
    one lengthen the oscillation period of the integrand in ``u`` and need a
    larger ``cutoff``.  When moving off the published grid, re-run the
    refinement comparison of ``numerical_error_budget.full_grid_refinement``
    with the new parameters instead of assuming the shipped tolerance.
    """

    scalar_input = np.ndim(energy_over_a) == 0
    energy = np.atleast_1d(np.asarray(energy_over_a, dtype=float))
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy_over_a must contain only finite values")
    nodes, weights = composite_gauss_legendre_rule(cutoff, panel_width, order)
    regular = dimensionless_subtracted_g11(nodes, speed)

    # Chunking keeps the routine lightweight if a dense energy grid is used.
    integral = np.empty(energy.size, dtype=float)
    for start in range(0, energy.size, 128):
        stop = min(start + 128, energy.size)
        cosine = np.cos(np.abs(energy[start:stop, None]) * nodes[None, :])
        integral[start:stop] = cosine @ (weights * regular)

    leading_amplitude = speed**2 / FOUR_PI_SQ
    integral += leading_amplitude * _leading_tail_integral(energy, cutoff)
    inertial = np.where(energy < 0.0, -energy / (2.0 * np.pi), 0.0)
    result = inertial + 2.0 * integral
    return result[0] if scalar_input else result


def effective_inverse_temperature(positive_energy_over_a, excitation_rate):
    r"""Return a_c*beta_eff(E) from circular detailed balance."""

    energy = np.asarray(positive_energy_over_a, dtype=float)
    excitation = np.asarray(excitation_rate, dtype=float)
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(excitation)):
        raise ValueError("energies and rates must be finite")
    if np.any(energy <= 0.0) or np.any(excitation <= 0.0):
        raise ValueError("positive energies and excitation rates are required")
    deexcitation = excitation + energy / (2.0 * np.pi)
    return np.log(deexcitation / excitation) / energy
