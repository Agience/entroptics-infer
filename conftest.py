"""Make `pytest` find the package without the package being installed.

Why this is needed
--------------------
`pyproject.toml` puts the package under `src/` and points pytest at `tests/`, so running
`pytest` from the repository root collects the tests and then cannot import what they test: every
test module fails at collection, before a single assertion runs. Without this file the suite passes
only if the caller already knows to set `PYTHONPATH`, or has done an editable install first.

A failure of that kind is loud while its cause is not, and a green run on one machine says nothing
about another. A suite that does not run from the obvious command has not been run.

What this does, and what it does not
--------------------------------------
Puts `src/` on the path for the test session only. It does not install anything, shadow an
installed copy, or change what is imported when the package is properly installed — if
`entroptics_infer` is already importable, this changes nothing, because the working tree's `src/`
is appended rather than prepended.

The research checks under `research/checks/` do not go through pytest; each puts `src/` and
`research/` on the path itself, because they are run as programs rather than collected.
"""
from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    # appended, not inserted: an installed copy still wins, so this cannot silently make the
    # tests run against the working tree when the caller meant to test what they installed
    sys.path.append(str(SRC))
