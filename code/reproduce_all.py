"""Generate and validate every numerical artifact in the archive."""

from __future__ import annotations

import sys
from time import perf_counter

from numerical_error_budget import generate_error_budget
from plot_detector_response import generate_figure_artifacts
from validate_outputs import (
    validate_artifacts,
    validate_numerics,
    validate_release_metadata,
)


def main() -> int:
    started = perf_counter()
    print("Generating Figure 1 and plotted data...")
    for path in generate_figure_artifacts():
        print(f"Wrote {path}")

    print("Generating the complete numerical error budget...")
    for path in generate_error_budget():
        print(f"Wrote {path}")

    print("Validating numerical identities and generated artifacts...")
    validate_release_metadata()
    validate_numerics()
    width, height = validate_artifacts()
    elapsed = perf_counter() - started
    print("Reproduction completed successfully.")
    print(f"PNG dimensions: {width} x {height} pixels")
    print(f"Elapsed time: {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
