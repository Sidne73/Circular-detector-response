"""Numerical configuration used for the published figure and error budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PACKAGE_VERSION = "1.0.0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
FIGURE_DIR = PACKAGE_ROOT / "figures"

# Figure 1 uses 99 positive gaps for each of three tangential speeds.
SPEEDS = (0.30, 0.60, 0.85)
ENERGIES = np.arange(2, 101, dtype=float) / 25.0

PRODUCTION = {"cutoff": 800.0, "panel_width": 1.0, "order": 24}
REFERENCE = {"cutoff": 1200.0, "panel_width": 0.75, "order": 32}

# These five points include both ends of the plotted gap interval and all
# plotted speeds.  The middle point lies on the figure's regular energy grid.
INDEPENDENT_POINTS = (
    (0.30, 0.08),
    (0.30, 4.00),
    (0.60, 1.52),
    (0.85, 0.08),
    (0.85, 4.00),
)
INDEPENDENT_EPSABS = 2.0e-14
INDEPENDENT_LIMIT = 500
INDEPENDENT_LIMLST = 500
# QUADPACK reports an absolute error estimate per call.  A nominally
# successful QAWFE result whose own estimate exceeds this is rejected rather
# than reported, because an adaptive estimate is not a guaranteed bound.
INDEPENDENT_MAX_ERROR_ESTIMATE = 1.0e-12

# Mode-sum cross-check: the manuscript's Eq. (94) evaluated directly, sharing
# no step with the Eq. (110) production path.
MODE_SUM = {"modes": 300, "angular_order": 96}
MODE_SUM_FINE = {"modes": 600, "angular_order": 160}

# At e=0 the omitted remainder is nonoscillatory and exposes the conservative
# U^{-3} cutoff behavior without phase-dependent cancellations.
CUTOFFS = (100.0, 200.0, 400.0, 800.0, 1600.0)
CUTOFF_REFINEMENT = {"panel_width": 0.5, "order": 32}

# Acceptance thresholds.
#
# Each is set from the value actually observed on the reference platform
# (CPython 3.11.9 / NumPy 2.4.6 / SciPy 1.15.1, Windows x86-64), with a factor
# of roughly two for platform variation in the last bits.  They are NOT
# cross-platform guarantees; if a new platform fails one of these, measure the
# variation there before relaxing the limit.
#
#   quantity                                observed      threshold
#   full-grid absolute                      1.21e-12      2.0e-12
#   full-grid relative                      1.58e-07      3.0e-07
#   five-point QUADPACK absolute            1.51e-12      2.5e-12
#   five-point QUADPACK relative            1.00e-07      2.0e-07
#   full-grid mode-sum absolute             1.51e-12      2.5e-12
#   full-grid mode-sum relative             1.00e-07      2.0e-07
#   mode-sum truncation (300 vs 600 modes)  1.28e-16      1.0e-14
#
# These comparison thresholds are distinct in meaning from the conservative
# analytic tail bound below, which controls the omitted part of the integral
# rather than the difference between two approximations.
MAX_FULL_GRID_ABSOLUTE_DIFFERENCE = 2.0e-12
MAX_FULL_GRID_RELATIVE_DIFFERENCE = 3.0e-7
MIN_CUTOFF_ORDER = 2.90
MAX_CUTOFF_ORDER = 3.10
MAX_INDEPENDENT_ABSOLUTE_DIFFERENCE = 2.5e-12
MAX_INDEPENDENT_RELATIVE_DIFFERENCE = 2.0e-7
MAX_MODE_SUM_ABSOLUTE_DIFFERENCE = 2.5e-12
MAX_MODE_SUM_RELATIVE_DIFFERENCE = 2.0e-7
MAX_MODE_SUM_TRUNCATION = 1.0e-14
MAX_CONSERVATIVE_TAIL_BOUND = 7.0e-11
MAX_RELATIVE_TAIL_BOUND = 1.5e-3
