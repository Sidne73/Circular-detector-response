"""Reproduce publication-ready Figure 1 and its underlying numerical data.

Run from any directory with

    python path/to/plot_detector_response.py

Outputs are written relative to this package, not the current working directory.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from hadamard_g11 import circular_response_rate, effective_inverse_temperature
from publication_config import (
    DATA_DIR,
    ENERGIES,
    FIGURE_DIR,
    PACKAGE_VERSION,
    PRODUCTION,
    SPEEDS,
)


def configure_publication_style() -> None:
    """Use journal-scale typography and LaTeX-like STIX math rendering."""

    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "0.86",
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def calculate_curves():
    rates: dict[float, np.ndarray] = {}
    inverse_temperatures: dict[float, np.ndarray] = {}
    for speed in SPEEDS:
        rate = circular_response_rate(ENERGIES, speed, **PRODUCTION)
        if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
            raise RuntimeError(f"invalid excitation rate for v={speed:.2f}")
        rates[speed] = rate
        inverse_temperatures[speed] = effective_inverse_temperature(ENERGIES, rate)
    return rates, inverse_temperatures


def write_data(rates, inverse_temperatures) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "circular_detector_response.csv"
    header = ["E_over_a_c"]
    header += [f"rate_over_a_c_v_{v:.2f}" for v in SPEEDS]
    header += [f"a_c_beta_eff_v_{v:.2f}" for v in SPEEDS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index, energy in enumerate(ENERGIES):
            writer.writerow(
                [f"{energy:.2f}"]
                + [f"{rates[v][index]:.17e}" for v in SPEEDS]
                + [f"{inverse_temperatures[v][index]:.17e}" for v in SPEEDS]
            )
    return path


def make_figure(rates, inverse_temperatures) -> tuple[Path, Path]:
    configure_publication_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    colors = ("#0072B2", "#D55E00", "#009E73")  # color-vision-safe Okabe-Ito
    dashes = (None, (5.0, 2.3), (1.2, 1.5))
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.18))
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.92, wspace=0.27)

    for speed, color, dash in zip(SPEEDS, colors, dashes):
        line_a, = axes[0].plot(
            ENERGIES,
            rates[speed],
            color=color,
            label=rf"$v={speed:.2f}$",
            solid_capstyle="round",
        )
        line_b, = axes[1].plot(
            ENERGIES,
            inverse_temperatures[speed],
            color=color,
            solid_capstyle="round",
        )
        if dash is not None:
            line_a.set_dashes(dash)
            line_b.set_dashes(dash)

    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0, 4.0)
    axes[0].set_ylim(1.0e-8, 1.0e-1)
    axes[0].set_xticks(np.arange(0.0, 4.1, 1.0))
    axes[0].set_xlabel(r"$E/a_{\mathrm{c}}$")
    axes[0].set_ylabel(r"$\dot{\mathcal{F}}(E)/a_{\mathrm{c}}$")
    # The upper-right corner is free of data on the logarithmic scale, so the
    # legend remains fully readable without masking or crossing any curve.
    axes[0].legend(loc="upper right", handlelength=3.0, borderaxespad=0.65)

    axes[1].set_xlim(0.0, 4.0)
    axes[1].set_ylim(0.0, 14.0)
    axes[1].set_xticks(np.arange(0.0, 4.1, 1.0))
    axes[1].set_yticks(np.arange(0.0, 14.1, 2.0))
    axes[1].set_xlabel(r"$E/a_{\mathrm{c}}$")
    axes[1].set_ylabel(r"$a_{\mathrm{c}}\,\beta_{\mathrm{eff}}(E)$")
    axes[1].axhline(2.0 * np.pi, color="0.35", linewidth=1.15, dashes=(4.0, 2.5))
    axes[1].text(
        3.90,
        2.0 * np.pi + 0.27,
        r"$2\pi$ (linear acceleration)",
        ha="right",
        va="bottom",
        color="0.30",
        fontsize=8.0,
    )

    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(
            0.02,
            1.045,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            clip_on=False,
        )

    pdf_path = FIGURE_DIR / "circular_detector_response.pdf"
    png_path = FIGURE_DIR / "circular_detector_response.png"
    metadata = {
        "Title": "Circular-detector response in the Minkowski vacuum",
        "Author": (
            "Sidney Natzuka Junior; Carlos Augusto Domingues Zarro; "
            "Matheus dos Santos Soares"
        ),
        "Subject": "Reproduction package for Figure 1",
        "Keywords": "rotating detector, Wightman function, quantum field theory",
        "Creator": f"rotating-vacuum reproduction package {PACKAGE_VERSION}",
        # Suppress run-time timestamps so repeated PDF builds are deterministic
        # when the software environment and fonts are unchanged.
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf_path, format="pdf", metadata=metadata)
    figure.savefig(
        png_path,
        format="png",
        dpi=600,
        metadata={
            "Title": metadata["Title"],
            "Author": metadata["Author"],
            "Software": metadata["Creator"],
        },
    )
    plt.close(figure)
    return pdf_path, png_path


def generate_figure_artifacts() -> list[Path]:
    rates, inverse_temperatures = calculate_curves()
    csv_path = write_data(rates, inverse_temperatures)
    pdf_path, png_path = make_figure(rates, inverse_temperatures)
    return [pdf_path, png_path, csv_path]


def main() -> int:
    for path in generate_figure_artifacts():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
