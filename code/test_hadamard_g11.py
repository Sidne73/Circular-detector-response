"""Unit and regression tests for the circular-response implementation."""

from __future__ import annotations

import unittest

import numpy as np

from hadamard_g11 import (
    CircularDetector,
    SERIES_THRESHOLD,
    _p_ratio,
    _x2_minus_sin2,
    circular_response_rate,
    circular_wightman_g11,
    dimensionless_subtracted_g11,
    effective_inverse_temperature,
    hadamard_subtracted_g11,
)
from mode_sum_rate import mode_sum_convergence_report, mode_sum_response_rate
from publication_config import PRODUCTION, SPEEDS


class CircularDetectorTests(unittest.TestCase):
    def test_invalid_kinematics_are_rejected(self) -> None:
        for speed in (-0.1, 0.0, 1.0, 1.1, np.nan, np.inf):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                CircularDetector(speed)
        for acceleration in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(a=acceleration), self.assertRaises(ValueError):
                CircularDetector(0.5, proper_acceleration=acceleration)

    def test_radius_and_angular_velocity_recover_speed(self) -> None:
        for speed in SPEEDS:
            detector = CircularDetector(speed, proper_acceleration=1.7)
            self.assertAlmostEqual(
                detector.radius * detector.angular_velocity, speed, places=15
            )


class SubtractedCorrelationTests(unittest.TestCase):
    def test_coincidence_limit(self) -> None:
        expected = 1.0 / (48.0 * np.pi**2)
        for speed in SPEEDS:
            got = float(hadamard_subtracted_g11(0.0, CircularDetector(speed)))
            self.assertAlmostEqual(got, expected, places=15)

    def test_small_nonzero_times_do_not_underflow(self) -> None:
        """Eq. (E3) evaluated through P(x) must not lose the coincidence value.

        The earlier ``N / [4 pi^2 u^2 (u^2 + N)]`` form built a fourth power of
        u explicitly, which underflows to zero well before u itself does.
        """

        expected = 1.0 / (48.0 * np.pi**2)
        for speed in SPEEDS:
            for exponent in (-30, -60, -90, -150):
                got = float(dimensionless_subtracted_g11(10.0**exponent, speed))
                with self.subTest(speed=speed, exponent=exponent):
                    self.assertAlmostEqual(got, expected, places=15)

    def test_nonfinite_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hadamard_subtracted_g11([0.1, np.nan], CircularDetector(0.5))

    def test_epsilon_enters_only_the_temporal_term(self) -> None:
        """Eq. (104) shifts s -> s - i eps in gamma^2 s^2, not in the chord."""

        detector = CircularDetector(0.60)
        gamma = detector.gamma
        radius = detector.radius
        omega = detector.angular_velocity
        s, epsilon = 1.0, 1.0e-8
        chord = 4.0 * radius**2 * np.sin(0.5 * gamma * omega * s) ** 2
        expected = -1.0 / (
            4.0 * np.pi**2 * ((gamma * (s - 1j * epsilon)) ** 2 - chord)
        )
        got = complex(circular_wightman_g11(s, detector, epsilon=epsilon))
        self.assertAlmostEqual(got.real, expected.real, delta=1.0e-18)
        self.assertAlmostEqual(got.imag, expected.imag, delta=1.0e-18)

    def test_evenness_and_positivity(self) -> None:
        points = np.array([0.0, 1.0e-8, 0.1, 1.0, 25.0])
        for speed in SPEEDS:
            positive = dimensionless_subtracted_g11(points, speed)
            negative = dimensionless_subtracted_g11(-points, speed)
            self.assertTrue(np.all(positive > 0.0))
            np.testing.assert_allclose(positive, negative, rtol=0.0, atol=0.0)


class ResponseTests(unittest.TestCase):
    def test_detailed_balance_identity(self) -> None:
        energies = np.array([0.08, 0.50, 1.52, 4.00])
        for speed in SPEEDS:
            excitation = circular_response_rate(energies, speed, **PRODUCTION)
            deexcitation = circular_response_rate(-energies, speed, **PRODUCTION)
            np.testing.assert_allclose(
                deexcitation - excitation,
                energies / (2.0 * np.pi),
                rtol=2.0e-13,
                atol=2.0e-13,
            )
            beta = effective_inverse_temperature(energies, excitation)
            self.assertTrue(np.all(np.isfinite(beta)))
            self.assertTrue(np.all(beta > 0.0))

    def test_nonfinite_energy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            circular_response_rate([0.1, np.nan], 0.5)
        with self.assertRaises(ValueError):
            effective_inverse_temperature([0.1, np.nan], [1.0, 1.0])

    def test_regression_values(self) -> None:
        expected = {
            (0.30, 0.08): 7.3258384316054080e-03,
            (0.60, 1.52): 3.0169864486386504e-04,
            (0.85, 4.00): 5.1277283809400297e-08,
        }
        for (speed, energy), wanted in expected.items():
            got = float(circular_response_rate(energy, speed, **PRODUCTION))
            self.assertAlmostEqual(got, wanted, delta=2.0e-15)

    def test_agrees_with_high_precision_reference(self) -> None:
        """Compare against 35-digit mpmath evaluations of Eq. (110).

        The reference values are produced by ``reference_rates.py``, which
        implements two arbitrary-precision methods and prints their mutual
        agreement.  They are also corroborated by the QUADPACK and mode-sum
        checks in ``numerical_error_budget.py``.  Because ``mpmath.quadosc``
        takes minutes, the values are stored here as rounded literals rather
        than recomputed on every test run.

        The tolerance is the documented accuracy of the production rule, whose
        error is dominated by the U = 800 cutoff and is largest at v = 0.85 and
        small gap.  It is not a tolerance on the reference values.
        """

        reference = {
            (0.30, 0.08): 7.3258384316279267e-03,
            (0.30, 4.00): 1.0927184824474443e-05,
            (0.60, 1.52): 3.0169864493984160e-04,
            (0.85, 0.08): 1.6393839481959767e-02,
            (0.85, 4.00): 5.1277288938765972e-08,
        }
        for (speed, energy), wanted in reference.items():
            got = float(circular_response_rate(energy, speed, **PRODUCTION))
            with self.subTest(speed=speed, energy=energy):
                self.assertAlmostEqual(got, wanted, delta=3.0e-12)


class ModeSumTests(unittest.TestCase):
    """Eq. (94) against Eq. (110), sharing no step of the production path."""

    def test_matches_the_production_rate(self) -> None:
        energies = np.array([0.08, 0.52, 1.52, 2.68, 4.00])
        for speed in SPEEDS:
            spectral = mode_sum_response_rate(energies, speed)
            production = circular_response_rate(energies, speed, **PRODUCTION)
            absolute = np.abs(spectral - production)
            relative = absolute / np.abs(spectral)
            with self.subTest(speed=speed):
                self.assertLess(absolute.max(), 2.5e-12)
                self.assertLess(relative.max(), 2.0e-7)

    def test_truncation_is_controlled(self) -> None:
        report = mode_sum_convergence_report([0.08, 1.52, 4.00], 0.85)
        self.assertLess(report["maximum_absolute_difference"], 1.0e-14)

    def test_negative_gaps_are_rejected(self) -> None:
        for bad in ([-0.5], [0.0], [0.5, np.nan]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mode_sum_response_rate(bad, 0.5)


class SeriesTests(unittest.TestCase):
    def test_series_and_direct_branches_agree_at_the_threshold(self) -> None:
        """Eq. (E4) must not leave a step at |x| = SERIES_THRESHOLD.

        Both branch formulas are evaluated at the same point, so the
        comparison isolates the branch disagreement from the (nonzero) slope
        of x^2-sin^2(x) there.
        """

        x = SERIES_THRESHOLD
        x2 = x * x
        series = x2 * x2 * (
            1.0 / 3.0
            - x2 * (
                2.0 / 45.0
                - x2 * (1.0 / 315.0 - x2 * (2.0 / 14175.0 - x2 * 2.0 / 467775.0))
            )
        )
        direct = x2 - np.sin(x) ** 2
        self.assertLess(abs(series - direct) / direct, 1.0e-13)

    def test_series_matches_exact_values(self) -> None:
        """Spot values of x^2-sin^2(x) from an independent 30-digit source."""

        exact = {
            0.01: 3.3332888892063478e-09,
            0.10: 3.3288920620815562e-05,
            0.1499: 1.6779702990727090e-04,
            0.1501: 1.6869298896902992e-04,
            1.00: 2.9192658172642881e-01,
        }
        for x, wanted in exact.items():
            got = float(_x2_minus_sin2(np.array([x]))[0])
            with self.subTest(x=x):
                self.assertLess(abs(got - wanted) / wanted, 1.0e-13)

    def test_p_ratio_is_the_series_divided_by_x_fourth(self) -> None:
        """P(x) = (x^2 - sin^2 x)/x^4 on both sides of the switch, and P(0)=1/3."""

        self.assertAlmostEqual(float(_p_ratio(np.array([0.0]))[0]), 1.0 / 3.0,
                               places=15)
        for x in (1.0e-6, 0.05, 0.1499, 0.1501, 1.0, 25.0):
            wanted = float(_x2_minus_sin2(np.array([x]))[0]) / x**4
            got = float(_p_ratio(np.array([x]))[0])
            with self.subTest(x=x):
                self.assertLess(abs(got - wanted) / abs(wanted), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
