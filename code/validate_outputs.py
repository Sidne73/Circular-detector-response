"""Numerical and file-level validation for all archived research artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from hadamard_g11 import (
    CircularDetector,
    circular_response_rate,
    effective_inverse_temperature,
    hadamard_subtracted_g11,
)
from publication_config import (
    DATA_DIR,
    ENERGIES,
    FIGURE_DIR,
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
    PACKAGE_ROOT,
    PACKAGE_VERSION,
    PRODUCTION,
    SPEEDS,
)

# Every stored difference is recomputed from the stored rates and required to
# agree with the recorded value to this relative accuracy.  It guards against
# a CSV whose columns no longer describe each other.
STORED_DIFFERENCE_TOLERANCE = 1.0e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    """Return a CSV column as finite floats, or fail loudly."""

    if name not in rows[0]:
        raise AssertionError(f"missing CSV column: {name}")
    try:
        values = np.array([float(row[name]) for row in rows], dtype=float)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"non-numeric entry in column {name}") from error
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"non-finite entry in column {name}")
    return values


def _require_close(got, wanted, tolerance: float, what: str) -> None:
    got = np.asarray(got, dtype=float)
    wanted = np.asarray(wanted, dtype=float)
    if not np.all(np.isfinite(got)) or not np.all(np.isfinite(wanted)):
        raise AssertionError(f"non-finite value while checking {what}")
    scale = np.maximum(np.abs(wanted), np.finfo(float).tiny)
    if not np.all(np.abs(got - wanted) / scale <= tolerance):
        raise AssertionError(f"stored value disagrees with recomputation: {what}")


def _require_finite_number(value, what: str) -> float:
    """Reject NaN before it is compared with a threshold.

    ``nan > threshold`` is false, so a NaN summary statistic would otherwise
    pass every acceptance test in this module.
    """

    number = float(value)
    if not np.isfinite(number):
        raise AssertionError(f"non-finite summary value: {what}")
    return number


def _require_nonempty(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty output: {path}")


def _top_level_block(text: str, key: str) -> str:
    """Return the indented body of a top-level YAML key.

    CITATION.cff carries author lists in more than one place: the software's
    own `authors`, and the `authors` of each work under `references`. Counting
    a field across the whole file would conflate them, so callers slice out the
    block they mean first.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = index
            break
    if start is None:
        raise AssertionError(f"CITATION.cff has no top-level '{key}' key")

    body: list[str] = []
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break
        body.append(line)
    return "\n".join(body)


def validate_release_metadata() -> None:
    """Check that the archive contains internally consistent release metadata."""

    citation_path = PACKAGE_ROOT / "CITATION.cff"
    zenodo_path = PACKAGE_ROOT / ".zenodo.json"
    license_path = PACKAGE_ROOT / "LICENSE"
    readme_path = PACKAGE_ROOT / "README.md"
    lock_path = PACKAGE_ROOT / "requirements-lock.txt"
    workflow_path = PACKAGE_ROOT / ".github" / "workflows" / "reproducibility.yml"
    _require_nonempty(
        [
            citation_path,
            zenodo_path,
            license_path,
            readme_path,
            lock_path,
            workflow_path,
        ]
    )

    citation = citation_path.read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in citation:
        raise AssertionError("CITATION.cff has an unexpected schema version")
    if f"version: {PACKAGE_VERSION}" not in citation:
        raise AssertionError("CITATION.cff package version is inconsistent")
    if "symbolic verification notebook" not in citation:
        raise AssertionError("CITATION.cff title omits the verification notebook")
    if _top_level_block(citation, "authors").count("family-names:") != 3:
        raise AssertionError("CITATION.cff must list exactly three authors")
    if "license: BSD-3-Clause" not in citation:
        raise AssertionError("CITATION.cff license is inconsistent")

    zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
    if zenodo["version"] != PACKAGE_VERSION or len(zenodo["creators"]) != 3:
        raise AssertionError("Zenodo metadata are inconsistent")
    if "notebook" not in zenodo["description"]:
        raise AssertionError("Zenodo description omits the verification notebook")
    if zenodo["license"] != "bsd-3-clause":
        raise AssertionError("Zenodo license metadata are inconsistent")

    license_text = license_path.read_text(encoding="utf-8")
    if not license_text.startswith("BSD 3-Clause License"):
        raise AssertionError("release license is missing or inconsistent")

    lock = lock_path.read_text(encoding="utf-8")
    required_versions = (
        "numpy==2.4.6",
        "scipy==1.15.1",
        "matplotlib==3.10.0",
        "sympy==1.13.1",
        "mpmath==1.3.0",
    )
    if any(requirement not in lock for requirement in required_versions):
        raise AssertionError("locked numerical environment is incomplete")


def validate_numerics() -> None:
    """Check analytic limits, positivity, and detailed balance."""

    expected_limit = 1.0 / (48.0 * np.pi**2)
    for speed in SPEEDS:
        detector = CircularDetector(speed)
        at_zero = float(hadamard_subtracted_g11(0.0, detector))
        if not np.isclose(at_zero, expected_limit, rtol=0.0, atol=1.0e-15):
            raise AssertionError(f"incorrect coincidence limit for v={speed}")

        excitation = circular_response_rate(ENERGIES, speed)
        deexcitation = circular_response_rate(-ENERGIES, speed)
        if np.any(excitation <= 0.0) or np.any(deexcitation <= excitation):
            raise AssertionError(f"response positivity failed for v={speed}")
        beta = effective_inverse_temperature(ENERGIES, excitation)
        reconstructed = np.log(deexcitation / excitation) / ENERGIES
        if not np.allclose(beta, reconstructed, rtol=2.0e-12, atol=2.0e-12):
            raise AssertionError(f"detailed balance failed for v={speed}")


def validate_artifacts() -> tuple[int, int]:
    """Validate artifact structure, dimensions, and reported error thresholds."""

    png_path = FIGURE_DIR / "circular_detector_response.png"
    pdf_path = FIGURE_DIR / "circular_detector_response.pdf"
    plotted_path = DATA_DIR / "circular_detector_response.csv"
    grid_path = DATA_DIR / "full_grid_convergence.csv"
    cutoff_path = DATA_DIR / "cutoff_convergence.csv"
    independent_path = DATA_DIR / "independent_quadpack_check.csv"
    spectral_path = DATA_DIR / "mode_sum_cross_check.csv"
    summary_path = DATA_DIR / "numerical_error_budget.json"
    report_path = DATA_DIR / "numerical_convergence.txt"
    _require_nonempty(
        [
            png_path,
            pdf_path,
            plotted_path,
            grid_path,
            cutoff_path,
            independent_path,
            spectral_path,
            summary_path,
            report_path,
        ]
    )

    image = mpimg.imread(png_path)
    height, width = image.shape[:2]
    if width < 4000 or height < 1700:
        raise AssertionError(f"PNG resolution too low: {width}x{height}")
    if not np.all(np.isfinite(image)):
        raise AssertionError("PNG contains non-finite pixels")

    # ---- plotted data: grid, finiteness, positivity, internal consistency
    plotted_rows = _read_csv(plotted_path)
    expected_plot_columns = 1 + 2 * len(SPEEDS)
    if len(plotted_rows) != len(ENERGIES):
        raise AssertionError("plotted-data CSV has an unexpected row count")
    if len(plotted_rows[0]) != expected_plot_columns:
        raise AssertionError("plotted-data CSV has an unexpected column count")
    stored_energies = _column(plotted_rows, "E_over_a_c")
    _require_close(stored_energies, ENERGIES, 1.0e-12, "plotted energy grid")
    for speed in SPEEDS:
        rates = _column(plotted_rows, f"rate_over_a_c_v_{speed:.2f}")
        betas = _column(plotted_rows, f"a_c_beta_eff_v_{speed:.2f}")
        if np.any(rates <= 0.0):
            raise AssertionError(f"non-positive plotted rate for v={speed:.2f}")
        if np.any(betas <= 0.0):
            raise AssertionError(f"non-positive beta_eff for v={speed:.2f}")
        # beta_eff must be reconstructible from the stored rates alone.
        deexcitation = rates + stored_energies / (2.0 * np.pi)
        _require_close(betas, np.log(deexcitation / rates) / stored_energies,
                       1.0e-12, f"beta_eff column for v={speed:.2f}")
        # and the stored rates must be the ones the shipped code produces.
        _require_close(rates,
                       circular_response_rate(stored_energies, speed, **PRODUCTION),
                       1.0e-12, f"plotted rate column for v={speed:.2f}")

    # ---- refinement grid: recompute the stored differences and their maxima
    grid_rows = _read_csv(grid_path)
    if len(grid_rows) != len(ENERGIES) * len(SPEEDS):
        raise AssertionError("full-grid convergence CSV must contain 297 rows")
    production = _column(grid_rows, "production_rate_over_a_c")
    refined = _column(grid_rows, "reference_rate_over_a_c")
    stored_abs = _column(grid_rows, "absolute_difference")
    stored_rel = _column(grid_rows, "relative_difference")
    if np.any(production <= 0.0) or np.any(refined <= 0.0):
        raise AssertionError("non-positive rate in the refinement CSV")
    _require_close(stored_abs, np.abs(production - refined),
                   STORED_DIFFERENCE_TOLERANCE, "full-grid absolute differences")
    _require_close(stored_rel, np.abs(production - refined) / np.abs(refined),
                   STORED_DIFFERENCE_TOLERANCE, "full-grid relative differences")

    # ---- independent QUADPACK check
    independent_rows = _read_csv(independent_path)
    if len(independent_rows) != len(INDEPENDENT_POINTS):
        raise AssertionError("independent-check CSV must contain five rows")
    ind_prod = _column(independent_rows, "production_rate_over_a_c")
    ind_quad = _column(independent_rows, "quadpack_rate_over_a_c")
    _require_close(_column(independent_rows, "absolute_difference"),
                   np.abs(ind_prod - ind_quad), STORED_DIFFERENCE_TOLERANCE,
                   "independent absolute differences")

    # ---- mode-sum cross-check
    spectral_rows = _read_csv(spectral_path)
    if len(spectral_rows) != len(ENERGIES) * len(SPEEDS):
        raise AssertionError("mode-sum CSV must contain 297 rows")
    spec_prod = _column(spectral_rows, "production_rate_over_a_c")
    spec_mode = _column(spectral_rows, "mode_sum_rate_over_a_c")
    if np.any(spec_mode <= 0.0):
        raise AssertionError("non-positive mode-sum rate")
    _require_close(_column(spectral_rows, "absolute_difference"),
                   np.abs(spec_prod - spec_mode), STORED_DIFFERENCE_TOLERANCE,
                   "mode-sum absolute differences")
    # the two full-grid tables must describe the same production calculation
    _require_close(spec_prod, production, 1.0e-15,
                   "production rates shared by the refinement and mode-sum tables")

    cutoff_rows = _read_csv(cutoff_path)
    if len(cutoff_rows) != 3 * len(SPEEDS):
        raise AssertionError("cutoff-convergence CSV must contain nine rows")
    orders = _column(cutoff_rows, "observed_order")
    if np.any(orders < MIN_CUTOFF_ORDER) or np.any(orders > MAX_CUTOFF_ORDER):
        raise AssertionError("a stored cutoff order lies outside its interval")

    # ---- JSON summary: reject NaN, and check it against the CSVs
    summary_text = summary_path.read_text(encoding="utf-8")
    if "NaN" in summary_text or "Infinity" in summary_text:
        raise AssertionError("the JSON error budget contains a non-finite token")
    summary = json.loads(summary_text)
    if summary["software_environment"]["package_version"] != PACKAGE_VERSION:
        raise AssertionError("error-budget package version is inconsistent")
    grid = summary["full_grid_refinement"]
    cutoff = summary["cutoff_convergence_and_tail_bound"]
    independent = summary["independent_quadpack_check"]
    spectral = summary["mode_sum_cross_check"]
    if grid["point_count"] != 297 or spectral["point_count"] != 297:
        raise AssertionError("a JSON summary does not cover the full plotted grid")

    checks = (
        ("full-grid absolute", grid["maximum_absolute_difference"],
         float(stored_abs.max()), MAX_FULL_GRID_ABSOLUTE_DIFFERENCE),
        ("full-grid relative", grid["maximum_relative_difference"],
         float(stored_rel.max()), MAX_FULL_GRID_RELATIVE_DIFFERENCE),
        ("independent absolute", independent["maximum_absolute_difference"],
         float(np.abs(ind_prod - ind_quad).max()),
         MAX_INDEPENDENT_ABSOLUTE_DIFFERENCE),
        ("independent relative", independent["maximum_relative_difference"],
         float((np.abs(ind_prod - ind_quad) / np.abs(ind_quad)).max()),
         MAX_INDEPENDENT_RELATIVE_DIFFERENCE),
        ("mode-sum absolute", spectral["maximum_absolute_difference"],
         float(np.abs(spec_prod - spec_mode).max()),
         MAX_MODE_SUM_ABSOLUTE_DIFFERENCE),
        ("mode-sum relative", spectral["maximum_relative_difference"],
         float((np.abs(spec_prod - spec_mode) / np.abs(spec_mode)).max()),
         MAX_MODE_SUM_RELATIVE_DIFFERENCE),
    )
    for what, stored, recomputed, threshold in checks:
        value = _require_finite_number(stored, what)
        _require_close(value, recomputed, 1.0e-9,
                       f"{what} maximum recomputed from the CSV")
        if value > threshold:
            raise AssertionError(f"{what} difference exceeds its threshold")

    if not MIN_CUTOFF_ORDER <= _require_finite_number(
            cutoff["minimum_observed_order"], "minimum cutoff order"):
        raise AssertionError("minimum cutoff order is outside the accepted interval")
    if _require_finite_number(cutoff["maximum_observed_order"],
                              "maximum cutoff order") > MAX_CUTOFF_ORDER:
        raise AssertionError("maximum cutoff order is outside the accepted interval")
    if _require_finite_number(cutoff["maximum_absolute_tail_bound"],
                              "tail bound") > MAX_CONSERVATIVE_TAIL_BOUND:
        raise AssertionError("analytic tail bound exceeds its threshold")
    if _require_finite_number(
            cutoff["maximum_tail_bound_relative_to_smallest_plotted_rate"],
            "relative tail bound") > MAX_RELATIVE_TAIL_BOUND:
        raise AssertionError("relative tail bound exceeds its threshold")
    if _require_finite_number(spectral["maximum_mode_sum_truncation"],
                              "mode-sum truncation") > MAX_MODE_SUM_TRUNCATION:
        raise AssertionError("mode-sum truncation exceeds its threshold")
    if independent["point_count"] != len(INDEPENDENT_POINTS):
        raise AssertionError("JSON independent-check point count is inconsistent")

    report = report_path.read_text(encoding="utf-8")
    if "All publication acceptance checks passed." not in report:
        raise AssertionError("human-readable convergence report is incomplete")

    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
        raise AssertionError("PDF header or end-of-file marker is invalid")
    return width, height


def main() -> int:
    validate_release_metadata()
    validate_numerics()
    width, height = validate_artifacts()
    print("All numerical and artifact checks passed.")
    print(f"PNG dimensions: {width} x {height} pixels")
    print("PDF structure: header and end-of-file marker present")
    print("Full-grid, cutoff-order, tail-bound, and independent checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
