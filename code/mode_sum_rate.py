"""Independent evaluation of the circular response from the mode sum, Eq. (94).

The production routine in :mod:`hadamard_g11` evaluates the manuscript's
Eq. (110): a Fourier transform of the Hadamard-subtracted worldline
correlator, with the double pole removed analytically, a composite
Gauss--Legendre rule on a finite interval, and an analytic leading tail.

This module evaluates the *same* rate from a different manuscript equation and
with no step in common.  Starting from the mode-sum form Eq. (94),

    Fdot(E) = sum_m int_0^inf dq int_-inf^inf dk
              [q J_m(qR)^2 / (4 pi omega)] delta[E + gamma (omega - m Omega)],

set ``omega = sqrt(q^2 + k^2)`` for the massless field and change variables to
``q = omega sin(theta)``, ``k = omega cos(theta)`` with ``theta`` in (0, pi),
whose Jacobian is ``omega``.  The delta function then fixes

    omega = A_m,      A_m = m Omega - E / gamma,

and contributes ``1/gamma``.  Only modes with ``A_m > 0`` survive, so for
``E > 0`` every contributing mode has negative corotating frequency
``omega - m Omega = -E/gamma``, which is the statement made below Eq. (95).
Using the symmetry of the integrand about ``theta = pi/2``,

    Fdot(E)/a_c = (1 / (2 pi gamma)) sum_{m : A_m > 0} A_m
                  int_0^{pi/2} sin(theta) J_m(A_m R sin theta)^2 d(theta),

with ``a_c = 1``.  This is what :func:`mode_sum_response_rate` computes.

Because ``A_m R <= m v < m`` for ``v < 1``, the Bessel functions are evaluated
below their turning point and the sum converges geometrically in ``m``; the
mode count needed is modest and is checked by
:func:`mode_sum_convergence_report`.

The routine shares no code path with :mod:`hadamard_g11`: no Hadamard
subtraction, no Taylor branch, no cutoff, no analytic tail.  Agreement between
the two is therefore evidence about the manuscript's own consistency
(Eq. (94) against Eq. (110)) as well as about the implementation.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv

from hadamard_g11 import CircularDetector


def _angular_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on theta in [0, pi/2]."""

    if order < 8:
        raise ValueError("angular order must be at least 8")
    nodes, weights = leggauss(order)
    half = 0.25 * np.pi
    return half * (nodes + 1.0), half * weights


def mode_sum_response_rate(
    energy_over_a,
    speed: float,
    *,
    modes: int = 300,
    angular_order: int = 96,
) -> np.ndarray:
    r"""Return Fdot(E)/a_c for E > 0 from Eq. (94), with a_c = 1.

    Parameters
    ----------
    energy_over_a
        Positive dimensionless gaps ``E / a_c``.
    speed
        Tangential speed ``v = R Omega``, in (0, 1).
    modes
        Highest azimuthal number retained.  The summand decays geometrically
        once ``m`` exceeds ``A_m R``; see :func:`mode_sum_convergence_report`.
    angular_order
        Gauss--Legendre order for the ``theta`` integral.

    Notes
    -----
    Only ``E > 0`` is implemented.  For ``E < 0`` the manuscript's rate carries
    the separate inertial term of Eq. (106), which this representation does not
    contain; use :func:`hadamard_g11.circular_response_rate` there.
    """

    scalar_input = np.ndim(energy_over_a) == 0
    energy = np.atleast_1d(np.asarray(energy_over_a, dtype=float))
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy_over_a must contain only finite values")
    if np.any(energy <= 0.0):
        raise ValueError("the mode-sum form is implemented for E > 0 only")
    if modes < 1:
        raise ValueError("modes must be positive")

    detector = CircularDetector(speed=speed, proper_acceleration=1.0)
    gamma = detector.gamma
    radius = detector.radius
    omega_rot = detector.angular_velocity

    theta, weight = _angular_rule(angular_order)
    sin_theta = np.sin(theta)

    orders = np.arange(1, modes + 1, dtype=float)
    result = np.empty(energy.size, dtype=float)
    for index, gap in enumerate(energy):
        amplitude = orders * omega_rot - gap / gamma          # A_m
        keep = amplitude > 0.0
        if not np.any(keep):
            result[index] = 0.0
            continue
        active_m = orders[keep]
        active_a = amplitude[keep]
        argument = active_a[:, None] * radius * sin_theta[None, :]
        bessel = jv(active_m[:, None], argument)
        angular = (bessel**2 * (sin_theta * weight)[None, :]).sum(axis=1)
        result[index] = (active_a * angular).sum() / (2.0 * np.pi * gamma)
    return result[0] if scalar_input else result


def mode_sum_convergence_report(
    energy_over_a,
    speed: float,
    *,
    coarse: tuple[int, int] = (300, 96),
    fine: tuple[int, int] = (600, 160),
) -> dict:
    """Compare two truncations of the mode sum at the same gaps.

    Returns the maximum absolute and relative difference between a coarse and
    a fine truncation, which bounds the truncation error of the coarse one.
    """

    low = mode_sum_response_rate(
        energy_over_a, speed, modes=coarse[0], angular_order=coarse[1]
    )
    high = mode_sum_response_rate(
        energy_over_a, speed, modes=fine[0], angular_order=fine[1]
    )
    absolute = np.abs(np.atleast_1d(low) - np.atleast_1d(high))
    relative = absolute / np.abs(np.atleast_1d(high))
    return {
        "coarse": {"modes": coarse[0], "angular_order": coarse[1]},
        "fine": {"modes": fine[0], "angular_order": fine[1]},
        "maximum_absolute_difference": float(absolute.max()),
        "maximum_relative_difference": float(relative.max()),
    }


def main() -> int:
    """Print the mode-sum rates on the published grid and their truncation."""

    from publication_config import ENERGIES, SPEEDS

    for speed in SPEEDS:
        rate = mode_sum_response_rate(ENERGIES, speed)
        report = mode_sum_convergence_report(ENERGIES[::14], speed)
        print(
            f"v = {speed:.2f}:  rate spans "
            f"[{rate.min():.6e}, {rate.max():.6e}];  truncation "
            f"<= {report['maximum_absolute_difference']:.2e} absolutely"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
