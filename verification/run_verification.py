"""Execute the symbolic verification notebook headlessly and check the result.

Run from anywhere with

    python path/to/verification/run_verification.py

The notebook raises on the first failed identity, so a clean execution is part
of the pass criterion.  It is not the whole of it: a deleted or silently
skipped cell would also execute cleanly.  This script therefore also compares
the notebook's machine-readable ``INVENTORY`` lines against the manifest below,
so that the number of statements in each verdict category must match.

Options
-------
``--inplace``
    Store the executed notebook, with its outputs, back over
    ``verify_manuscript.ipynb``.
``--output PATH``
    Write the executed notebook to PATH.  Use this in CI, so that the uploaded
    artifact carries the outputs of *that* run rather than the committed ones.

With neither option the notebook is executed from a managed temporary
directory that is removed afterwards, and the tracked notebook is untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "verify_manuscript.ipynb"
TIMEOUT_SECONDS = 1800

# The expected shape of a complete run.  Update this deliberately when checks
# are added or removed; an accidental change makes the run fail loudly instead
# of quietly reporting fewer statements.
EXPECTED_INVENTORY = {
    "symbolic_identity": 138,
    "arbitrary_precision": 5,
    "implementation": 11,
    "inequality": 2,
    "consequence": 15,
    "definition": 25,
    "not_verified": 12,
    "total": 208,
}

# Equations the notebook must explicitly record as unverified.  If one of these
# quietly acquires a PASS, the disclosure this package makes about its own
# scope would no longer be accurate.
EXPECTED_NOT_VERIFIED = {
    "5", "6", "7", "14", "15", "57", "74", "93", "99", "106", "C5", "C7",
}


def execute(notebook_path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    return notebook


def collect_text(notebook: nbformat.NotebookNode) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    errors: list[str] = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                text_parts.append(output.text)
            elif output.output_type == "error":
                errors.append(f"cell {index}: {output.ename}: {output.evalue}")
    return "".join(text_parts), errors


def parse_inventory(text: str) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("INVENTORY "):
            _, key, value = line.split()
            inventory[key] = int(value)
    return inventory


def parse_not_verified(text: str) -> set[str]:
    equations: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("NOT VERIFIED"):
            marker = "Eq. ("
            if marker in stripped:
                start = stripped.index(marker) + len(marker)
                equations.add(stripped[start:stripped.index(")", start)])
    return equations


def check(text: str) -> list[str]:
    problems: list[str] = []

    inventory = parse_inventory(text)
    if not inventory:
        problems.append("the notebook printed no INVENTORY lines")
    for key, expected in EXPECTED_INVENTORY.items():
        got = inventory.get(key)
        if got is None:
            problems.append(f"inventory is missing the category {key!r}")
        elif got != expected:
            problems.append(
                f"inventory mismatch for {key!r}: got {got}, expected {expected}"
            )

    recorded = parse_not_verified(text)
    missing = EXPECTED_NOT_VERIFIED - recorded
    extra = recorded - EXPECTED_NOT_VERIFIED
    if missing:
        problems.append(
            "these equations are no longer disclosed as unverified: "
            + ", ".join(sorted(missing))
        )
    if extra:
        problems.append(
            "these equations became unverified unexpectedly: "
            + ", ".join(sorted(extra))
        )

    if "recorded statements" not in text:
        problems.append("the notebook did not reach its summary cell")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="write the executed notebook back over verify_manuscript.ipynb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the executed notebook to this path (for CI artifacts)",
    )
    arguments = parser.parse_args()

    if not NOTEBOOK.is_file():
        print(f"missing notebook: {NOTEBOOK}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        if arguments.inplace:
            target = NOTEBOOK
        else:
            target = Path(scratch) / NOTEBOOK.name
            shutil.copy2(NOTEBOOK, target)

        notebook = execute(target)

        if arguments.inplace:
            nbformat.write(notebook, NOTEBOOK)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, arguments.output)

    text, errors = collect_text(notebook)
    problems = errors + check(text)
    if problems:
        for message in problems:
            print(message, file=sys.stderr)
        return 1

    summary = text[text.index("=" * 78):]
    print(summary.rstrip())
    print()
    print("Verification notebook executed with no failed identity, and its")
    print("statement inventory matches the expected manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
