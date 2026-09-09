"""Reproduce every numerical-error statement in the manuscript appendix.

The production calculation is checked in four complementary ways:

1. all 297 plotted values are recomputed with a tighter composite
   Gauss--Legendre rule;
2. the conservative U^{-3} cutoff order and an analytic tail bound are
   evaluated;
3. five representative values are recomputed with QUADPACK's adaptive
   infinite-interval Fourier algorithm, independently of the finite-panel
   rule and analytic leading-tail correction used in production; and
4. all 297 plotted values are recomputed from the manuscript's mode sum,
   Eq. (94), which shares no step with the Eq. (110) production path.

Checks 1 and 2 quantify the production method against itself.  Checks 3 and 4
are independent formulations; of the two, check 4 is the stronger, because
QUADPACK's adaptive error estimate is not a guaranteed bound (see
:func:`independent_fourier_rate`) whereas the mode sum's truncation error is
measured directly by comparing two truncations.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad

from hadamard_g11 import circular_response_rate, dimensionless_subtracted_g11
from mode_sum_rate import mode_sum_convergence_report, mode_sum_response_rate
from publication_config import (
    CUTOFF_REFINEMENT,
    CUTOFFS,
    DATA_DIR,
    ENERGIES,
    INDEPENDENT_EPSABS,
    INDEPENDENT_LIMIT,
    INDEPENDENT_LIMLST,
    INDEPENDENT_MAX_ERROR_ESTIMATE,
    INDEPENDENT_POINTS,
    MAX_CONSERVATIVE_TAIL_BOUND,
    MAX_CUTOFF_ORDER,
    MAX_FULL_GRID_ABSOLUTE_DIFFERENCE,
    MAX_FULL_GRID_RELATIVE_DIFFERENCE,
    MAX_INDEPENDENT_ABSOLUTE_DIFFERENCE,
    MAX_INDEPENDENT_RELATIVE_DIFFERENCE,
    MAX_MODE_SUM_ABSOLUTE_DIFFERENCE,
    MAX_MODE_SUM_RELATIVE_DIFFERENCE,
    MAX_MODE_SUM_TRUNCATION,
    MAX_RELATIVE_TAIL_BOUND,
    MIN_CUTOFF_ORDER,
    MODE_SUM,
    MODE_SUM_FINE,
    PACKAGE_VERSION,
    PRODUCTION,
    REFERENCE,
    SPEEDS,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _location(record: dict) -> dict[str, float]:
    return {
        "speed": float(record["speed"]),
        "energy_over_a_c": float(record["energy_over_a_c"]),
    }


def full_grid_refinement() -> tuple[list[dict], dict]:
    """Compare production and reference rules at every plotted point."""

    rows: list[dict] = []
    for speed in SPEEDS:
        production = circular_response_rate(ENERGIES, speed, **PRODUCTION)
        reference = circular_response_rate(ENERGIES, speed, **REFERENCE)
        absolute = np.abs(production - reference)
        relative = absolute / np.abs(reference)
        for energy, got, wanted, abs_difference, rel_difference in zip(
            ENERGIES, production, reference, absolute, relative
        ):
            rows.append(
                {
                    "speed": f"{speed:.2f}",
                    "energy_over_a_c": f"{energy:.2f}",
                    "production_rate_over_a_c": f"{got:.17e}",
                    "reference_rate_over_a_c": f"{wanted:.17e}",
                    "absolute_difference": f"{abs_difference:.17e}",
                    "relative_difference": f"{rel_difference:.17e}",
                }
            )

    maximum_absolute = max(rows, key=lambda row: float(row["absolute_difference"]))
    maximum_relative = max(rows, key=lambda row: float(row["relative_difference"]))
    smallest_rate = min(abs(float(row["reference_rate_over_a_c"])) for row in rows)
    summary = {
        "point_count": len(rows),
        "maximum_absolute_difference": float(maximum_absolute["absolute_difference"]),
        "maximum_absolute_difference_location": _location(maximum_absolute),
        "maximum_relative_difference": float(maximum_relative["relative_difference"]),
        "maximum_relative_difference_location": _location(maximum_relative),
        "smallest_reference_rate_over_a_c": smallest_rate,
    }
    if summary["maximum_absolute_difference"] > MAX_FULL_GRID_ABSOLUTE_DIFFERENCE:
        raise RuntimeError("full-grid absolute-error threshold exceeded")
    if summary["maximum_relative_difference"] > MAX_FULL_GRID_RELATIVE_DIFFERENCE:
        raise RuntimeError("full-grid relative-error threshold exceeded")
    return rows, summary


def conservative_tail_bound(speed: float, cutoff: float) -> float:
    r"""Bound the response contribution from the omitted O(u^{-4}) tail.

    The exact difference between the stable integrand and its leading
    v^2/(4 pi^2 u^2) term is bounded using sin^2(x) <= 1.  The returned
    value includes the factor of two in the response Fourier transform.
    """

    gamma_squared = 1.0 / (1.0 - speed**2)
    denominator_factor = 1.0 - 4.0 * gamma_squared * speed**4 / cutoff**2
    if denominator_factor <= 0.0:
        raise ValueError("cutoff is too small for the tail bound")
    return (
        2.0
        * speed**4
        / (3.0 * np.pi**2 * denominator_factor * cutoff**3)
    )


def cutoff_convergence(smallest_rate: float) -> tuple[list[dict], dict]:
    """Measure the conservative cutoff order at zero dimensionless gap."""

    rows: list[dict] = []
    bounds = []
    for speed in SPEEDS:
        responses = {
            cutoff: float(
                circular_response_rate(
                    0.0,
                    speed,
                    cutoff=cutoff,
                    **CUTOFF_REFINEMENT,
                )
            )
            for cutoff in CUTOFFS
        }
        for first, second, third in zip(CUTOFFS[:-2], CUTOFFS[1:-1], CUTOFFS[2:]):
            first_difference = abs(responses[first] - responses[second])
            second_difference = abs(responses[second] - responses[third])
            order = float(np.log2(first_difference / second_difference))
            rows.append(
                {
                    "speed": f"{speed:.2f}",
                    "U": f"{first:.0f}",
                    "R_U": f"{responses[first]:.17e}",
                    "R_2U": f"{responses[second]:.17e}",
                    "R_4U": f"{responses[third]:.17e}",
                    "abs_R_U_minus_R_2U": f"{first_difference:.17e}",
                    "abs_R_2U_minus_R_4U": f"{second_difference:.17e}",
                    "observed_order": f"{order:.17e}",
                }
            )
        bounds.append(
            {
                "speed": speed,
                "absolute_bound": conservative_tail_bound(speed, PRODUCTION["cutoff"]),
            }
        )

    orders = [float(row["observed_order"]) for row in rows]
    maximum_bound = max(item["absolute_bound"] for item in bounds)
    maximum_relative_bound = maximum_bound / smallest_rate
    summary = {
        "energy_over_a_c": 0.0,
        "cutoffs": list(CUTOFFS),
        "minimum_observed_order": min(orders),
        "maximum_observed_order": max(orders),
        "tail_bounds_at_production_cutoff": bounds,
        "maximum_absolute_tail_bound": maximum_bound,
        "maximum_tail_bound_relative_to_smallest_plotted_rate": maximum_relative_bound,
    }
    if summary["minimum_observed_order"] < MIN_CUTOFF_ORDER:
        raise RuntimeError("observed cutoff order is below the acceptance range")
    if summary["maximum_observed_order"] > MAX_CUTOFF_ORDER:
        raise RuntimeError("observed cutoff order is above the acceptance range")
    if maximum_bound > MAX_CONSERVATIVE_TAIL_BOUND:
        raise RuntimeError("conservative tail bound exceeds the acceptance threshold")
    if maximum_relative_bound > MAX_RELATIVE_TAIL_BOUND:
        raise RuntimeError("relative tail bound exceeds the acceptance threshold")
    return rows, summary


def independent_fourier_rate(speed: float, energy_over_a_c: float) -> tuple[float, float, int]:
    """Evaluate the unsplit infinite-interval cosine transform with QAWFE.

    Convergence is established from QUADPACK's own diagnostics, not from the
    absence of a warning.  With ``full_output=1`` SciPy suppresses the
    ``IntegrationWarning`` and instead *appends* a message (and an explanation
    mapping) to the returned tuple, so a wrapper that only catches warnings
    and reads ``result[:3]`` accepts nonconverged results silently.  This one

    * rejects any appended convergence message (``len(result) > 3``),
    * rejects any nonzero per-cycle status in ``ierlst[:lst]``,
    * rejects nonfinite values, and
    * rejects a result whose own absolute error estimate exceeds
      ``INDEPENDENT_MAX_ERROR_ESTIMATE``.

    The last test is a sanity limit, not a guarantee: an adaptive estimate can
    be optimistic.  That is why the mode-sum check of
    :func:`mode_sum_refinement` is carried alongside this one.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        result = quad(
            lambda u: float(dimensionless_subtracted_g11(u, speed)),
            0.0,
            np.inf,
            weight="cos",
            wvar=energy_over_a_c,
            epsabs=INDEPENDENT_EPSABS,
            limit=INDEPENDENT_LIMIT,
            limlst=INDEPENDENT_LIMLST,
            full_output=1,
        )
    where = f"v={speed:.2f}, E/a_c={energy_over_a_c:.2f}"
    if caught:
        messages = "; ".join(str(item.message) for item in caught)
        raise RuntimeError(f"QUADPACK warning at {where}: {messages}")
    if len(result) > 3:
        raise RuntimeError(f"QUADPACK did not converge at {where}: {result[3]}")

    integral, error_estimate, information = result[:3]
    cycles = int(information.get("lst", 0))
    statuses = np.asarray(information.get("ierlst", []))[:cycles]
    if statuses.size and np.any(statuses != 0):
        failing = np.flatnonzero(statuses != 0).tolist()
        raise RuntimeError(
            f"QUADPACK reported nonzero ierlst at {where}: "
            f"cycles {failing} -> {statuses[statuses != 0].tolist()}"
        )
    if not (np.isfinite(integral) and np.isfinite(error_estimate)):
        raise RuntimeError(f"QUADPACK returned a nonfinite value at {where}")
    if 2.0 * float(error_estimate) > INDEPENDENT_MAX_ERROR_ESTIMATE:
        raise RuntimeError(
            f"QUADPACK error estimate too large at {where}: "
            f"{2.0 * float(error_estimate):.3e} > "
            f"{INDEPENDENT_MAX_ERROR_ESTIMATE:.1e}"
        )
    return 2.0 * float(integral), 2.0 * float(error_estimate), cycles


def independent_check() -> tuple[list[dict], dict]:
    """Compare production with an algorithmically independent transform."""

    rows: list[dict] = []
    for speed, energy in INDEPENDENT_POINTS:
        production = float(circular_response_rate(energy, speed, **PRODUCTION))
        independent, estimator, cycles = independent_fourier_rate(speed, energy)
        absolute = abs(production - independent)
        relative = absolute / abs(independent)
        rows.append(
            {
                "speed": f"{speed:.2f}",
                "energy_over_a_c": f"{energy:.2f}",
                "production_rate_over_a_c": f"{production:.17e}",
                "quadpack_rate_over_a_c": f"{independent:.17e}",
                "absolute_difference": f"{absolute:.17e}",
                "relative_difference": f"{relative:.17e}",
                "quadpack_absolute_error_estimate": f"{estimator:.17e}",
                "fourier_cycles": str(cycles),
            }
        )

    maximum_absolute = max(rows, key=lambda row: float(row["absolute_difference"]))
    maximum_relative = max(rows, key=lambda row: float(row["relative_difference"]))
    summary = {
        "point_count": len(rows),
        "maximum_absolute_difference": float(maximum_absolute["absolute_difference"]),
        "maximum_absolute_difference_location": _location(maximum_absolute),
        "maximum_relative_difference": float(maximum_relative["relative_difference"]),
        "maximum_relative_difference_location": _location(maximum_relative),
        "maximum_quadpack_absolute_error_estimate": max(
            float(row["quadpack_absolute_error_estimate"]) for row in rows
        ),
    }
    if summary["maximum_absolute_difference"] > MAX_INDEPENDENT_ABSOLUTE_DIFFERENCE:
        raise RuntimeError("independent-check absolute-error threshold exceeded")
    if summary["maximum_relative_difference"] > MAX_INDEPENDENT_RELATIVE_DIFFERENCE:
        raise RuntimeError("independent-check relative-error threshold exceeded")
    return rows, summary


def mode_sum_refinement() -> tuple[list[dict], dict]:
    """Recompute every plotted rate from Eq. (94) and compare with production.

    This is the strongest independent check in the package: the mode sum uses
    no Hadamard subtraction, no Taylor branch, no finite cutoff and no
    analytic tail, so it shares no step with the production evaluation of
    Eq. (110).  Its own truncation error is measured by comparing two mode
    counts rather than estimated adaptively.
    """

    rows: list[dict] = []
    truncations: list[dict] = []
    for speed in SPEEDS:
        production = circular_response_rate(ENERGIES, speed, **PRODUCTION)
        spectral = mode_sum_response_rate(ENERGIES, speed, **MODE_SUM)
        if not np.all(np.isfinite(spectral)) or np.any(spectral <= 0.0):
            raise RuntimeError(f"invalid mode-sum rate for v={speed:.2f}")
        absolute = np.abs(production - spectral)
        relative = absolute / np.abs(spectral)
        for energy, got, wanted, abs_d, rel_d in zip(
            ENERGIES, production, spectral, absolute, relative
        ):
            rows.append(
                {
                    "speed": f"{speed:.2f}",
                    "energy_over_a_c": f"{energy:.2f}",
                    "production_rate_over_a_c": f"{got:.17e}",
                    "mode_sum_rate_over_a_c": f"{wanted:.17e}",
                    "absolute_difference": f"{abs_d:.17e}",
                    "relative_difference": f"{rel_d:.17e}",
                }
            )
        report = mode_sum_convergence_report(
            ENERGIES[::14],
            speed,
            coarse=(MODE_SUM["modes"], MODE_SUM["angular_order"]),
            fine=(MODE_SUM_FINE["modes"], MODE_SUM_FINE["angular_order"]),
        )
        truncations.append({"speed": speed, **report})

    maximum_absolute = max(rows, key=lambda row: float(row["absolute_difference"]))
    maximum_relative = max(rows, key=lambda row: float(row["relative_difference"]))
    worst_truncation = max(
        item["maximum_absolute_difference"] for item in truncations
    )
    summary = {
        "point_count": len(rows),
        "mode_sum_parameters": MODE_SUM,
        "maximum_absolute_difference": float(maximum_absolute["absolute_difference"]),
        "maximum_absolute_difference_location": _location(maximum_absolute),
        "maximum_relative_difference": float(maximum_relative["relative_difference"]),
        "maximum_relative_difference_location": _location(maximum_relative),
        "mode_sum_truncation": truncations,
        "maximum_mode_sum_truncation": worst_truncation,
    }
    if summary["maximum_absolute_difference"] > MAX_MODE_SUM_ABSOLUTE_DIFFERENCE:
        raise RuntimeError("mode-sum absolute-error threshold exceeded")
    if summary["maximum_relative_difference"] > MAX_MODE_SUM_RELATIVE_DIFFERENCE:
        raise RuntimeError("mode-sum relative-error threshold exceeded")
    if worst_truncation > MAX_MODE_SUM_TRUNCATION:
        raise RuntimeError("mode-sum truncation threshold exceeded")
    return rows, summary


def software_environment() -> dict[str, str]:
    return {
        "package_version": PACKAGE_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def _write_text_report(path: Path, summary: dict) -> None:
    grid = summary["full_grid_refinement"]
    cutoff = summary["cutoff_convergence_and_tail_bound"]
    independent = summary["independent_quadpack_check"]
    spectral = summary["mode_sum_cross_check"]
    environment = summary["software_environment"]
    lines = [
        "Circular detector response: publication numerical error budget",
        "",
        f"package version: {PACKAGE_VERSION}",
        "",
        "1. Complete plotted-grid refinement",
        f"production: {PRODUCTION}",
        f"reference:  {REFERENCE}",
        f"points checked: {grid['point_count']}",
        f"maximum absolute difference: {grid['maximum_absolute_difference']:.6e}",
        "maximum absolute-difference location: "
        f"v={grid['maximum_absolute_difference_location']['speed']:.2f}, "
        f"E/a_c={grid['maximum_absolute_difference_location']['energy_over_a_c']:.2f}",
        f"maximum relative difference: {grid['maximum_relative_difference']:.6e}",
        "maximum relative-difference location: "
        f"v={grid['maximum_relative_difference_location']['speed']:.2f}, "
        f"E/a_c={grid['maximum_relative_difference_location']['energy_over_a_c']:.2f}",
        f"smallest reference rate: {grid['smallest_reference_rate_over_a_c']:.6e}",
        "",
        "2. Conservative cutoff convergence and tail bound",
        f"cutoffs: {', '.join(f'{value:.0f}' for value in CUTOFFS)}",
        "observed order range: "
        f"{cutoff['minimum_observed_order']:.6f} to "
        f"{cutoff['maximum_observed_order']:.6f}",
        f"maximum absolute tail bound: {cutoff['maximum_absolute_tail_bound']:.6e}",
        "tail bound relative to smallest plotted rate: "
        f"{cutoff['maximum_tail_bound_relative_to_smallest_plotted_rate']:.6e}",
        "",
        "3. Independent adaptive QUADPACK Fourier check",
        "integrand: unsplit stable subtracted integrand on [0, infinity)",
        f"points checked: {independent['point_count']}",
        f"maximum absolute difference: {independent['maximum_absolute_difference']:.6e}",
        f"maximum relative difference: {independent['maximum_relative_difference']:.6e}",
        "maximum QUADPACK absolute error estimate: "
        f"{independent['maximum_quadpack_absolute_error_estimate']:.6e}",
        "",
        "4. Independent mode-sum cross-check, Eq. (94) against Eq. (110)",
        f"mode-sum parameters: {spectral['mode_sum_parameters']}",
        f"points checked: {spectral['point_count']}",
        f"maximum absolute difference: {spectral['maximum_absolute_difference']:.6e}",
        "maximum absolute-difference location: "
        f"v={spectral['maximum_absolute_difference_location']['speed']:.2f}, "
        f"E/a_c={spectral['maximum_absolute_difference_location']['energy_over_a_c']:.2f}",
        f"maximum relative difference: {spectral['maximum_relative_difference']:.6e}",
        "mode-sum truncation (coarse vs fine): "
        f"{spectral['maximum_mode_sum_truncation']:.6e}",
        "",
        "5. Software environment",
    ]
    lines.extend(f"{key}: {value}" for key, value in environment.items())
    lines.extend(["", "All publication acceptance checks passed.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_error_budget() -> list[Path]:
    """Generate the CSV, JSON, and human-readable validation artifacts."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows, grid_summary = full_grid_refinement()
    cutoff_rows, cutoff_summary = cutoff_convergence(
        grid_summary["smallest_reference_rate_over_a_c"]
    )
    independent_rows, independent_summary = independent_check()
    spectral_rows, spectral_summary = mode_sum_refinement()

    grid_path = DATA_DIR / "full_grid_convergence.csv"
    cutoff_path = DATA_DIR / "cutoff_convergence.csv"
    independent_path = DATA_DIR / "independent_quadpack_check.csv"
    spectral_path = DATA_DIR / "mode_sum_cross_check.csv"
    json_path = DATA_DIR / "numerical_error_budget.json"
    report_path = DATA_DIR / "numerical_convergence.txt"

    _write_csv(grid_path, list(grid_rows[0]), grid_rows)
    _write_csv(cutoff_path, list(cutoff_rows[0]), cutoff_rows)
    _write_csv(independent_path, list(independent_rows[0]), independent_rows)
    _write_csv(spectral_path, list(spectral_rows[0]), spectral_rows)

    summary = {
        "full_grid_refinement": grid_summary,
        "cutoff_convergence_and_tail_bound": cutoff_summary,
        "independent_quadpack_check": independent_summary,
        "mode_sum_cross_check": spectral_summary,
        "numerical_parameters": {
            "production": PRODUCTION,
            "reference": REFERENCE,
            "independent_epsabs": INDEPENDENT_EPSABS,
            "independent_limit": INDEPENDENT_LIMIT,
            "independent_limlst": INDEPENDENT_LIMLST,
            "independent_max_error_estimate": INDEPENDENT_MAX_ERROR_ESTIMATE,
            "mode_sum": MODE_SUM,
            "mode_sum_fine": MODE_SUM_FINE,
        },
        "software_environment": software_environment(),
    }
    # allow_nan=False makes a NaN anywhere in the budget a hard failure rather
    # than a silently written "NaN" token that later compares false against
    # every threshold.
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_text_report(report_path, summary)
    return [grid_path, cutoff_path, independent_path, spectral_path,
            json_path, report_path]


def main() -> int:
    for path in generate_error_budget():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
