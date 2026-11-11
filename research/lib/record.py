"""Write what a test measured, so a claim about it can be checked against it.

Why this exists
-----------------
`CLAIMS.md` enforces one rule: every figure in prose has a row naming the script that produced it,
and `prose_check.py` checks the figure against that script's stored results. The experiments under
`experiments/` store theirs in `results/`. The reader's tests print theirs and store nothing, so
**nothing measured about the reader can go in the ledger** — a row for it would cite a script whose
numbers the gate cannot see, which is the same as citing prose.

So each test records what it measured. Nothing else changes: the tests still print, still decide
their own verdicts, and the recording is a side effect of running them.

How the gate finds it
-----------------------
`prose_check.py` reads every `results/*.json` and keys it by the file's stem, and a prose line
naming one script is checked against that script's file alone. So the record has to be named after
the test that wrote it, which is what :func:`figures` does -- the caller does not pass a name and
cannot get it wrong.

What belongs in a record
--------------------------
The numbers a claim would quote, at the precision it would quote them. Not the whole computation:
a record is what the test concluded, so that a sentence in `CLAIMS.md` can be held against it.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import sys
import time

import numpy

#: research/ -- this module lives in research/lib/, and the records go to research/results/
HERE = pathlib.Path(__file__).resolve().parents[1]

#: Where a recording script may live: the reader's checks and the reports beside the experiments.
SCRIPTS = (HERE / "checks", HERE / "experiments")


#: What each script has recorded in *this* run, so a second call adds to the first.
_THIS_RUN: dict = {}


def _the_machine() -> dict:
    """What produced these figures, so a gate can tell a changed machine from changed code.

    **A record that does not say what ran it cannot be checked anywhere but where it was written.**
    `verify_records.py` re-runs each script and compares; run on another machine it reports every
    platform difference as drift, which is true and useless. Measured: one store's check reports **1** statement changing no answer under Python 3.12 and **0** under 3.14,
    on byte-identical code and the same numpy -- because three of that store's questions are decided
    by a gap smaller than the rounding bound of the dot product that produced them, and a question
    decided inside the arithmetic resolves whichever way the last bit falls.

    So the interpreter and the array library go in the record, excluded from the figures, and a
    drift report says which of the two changed.
    """
    return {"python": ".".join(str(n) for n in sys.version_info[:3]),
            "numpy": getattr(numpy, "__version__", "absent")}


def figure(name: str, value):
    """Record one value **and return it**, so printing it and recording it are the same act.

    The figures a test prints and does not record are mostly one shape: the test records the counts
    and prints the rate, and a claim then quotes the rate. `prose_check.py` refuses such a claim,
    which is the gate working, but it costs a round trip every time.

        print(f"{record.figure('kept_joined', kept / seen):.0%}")

    There is no gate demanding this. A rule that every printed number be recorded would mostly
    catch numbers nobody will ever quote, and the existing loop -- write the claim, have it
    refused, record the figure -- is self-correcting. This only makes the shorter path available.
    """
    figures(_depth=2, **{name: value})
    return value


def figures(_depth: int = 1, **values) -> pathlib.Path:
    """Store ``values`` under the calling script's own name and return where they went.

    Called from ``test_foo.py`` it writes ``results/test_foo.json``. The name is taken from the
    caller rather than passed in, because a record filed under the wrong name is worse than no
    record: the gate would check a claim against a script that did not make it.

    **Calling it twice from one script adds to the record rather than replacing it.** A test that
    records its main table and then a second table keeps both; a record that overwrote would drop
    the first, and its figures would stay in the prose backed by nothing.

    Accumulation is per run, not per file: a fresh run of a script starts its record again, so a
    figure a script has stopped producing stops being recorded.
    """
    # _depth lets `figure` delegate here without filing everything under record.py:
    # inspect.stack()[1] from inside this module IS this module.
    caller = pathlib.Path(inspect.stack()[_depth].filename).stem
    if not any((d / f"{caller}.py").is_file() for d in SCRIPTS):
        # the caller is not a script here -- an interactive session, or `python -c`, whose
        # filename is literally "<string>". Filing under that name would write a record no
        # gate could ever check against a script, so say what happened instead.
        raise RuntimeError(
            f"record.figures was called from {caller!r}, which is not a script in "
            f"{HERE}. A record is filed under the script that made it, so the gates can "
            f"check a claim against the thing that measured it.")
    out = HERE / "results" / f"{caller}.json"
    out.parent.mkdir(exist_ok=True)
    _THIS_RUN.setdefault(caller, {}).update(values)
    out.write_text(json.dumps({"script": f"{caller}.py",
                               "recorded": time.strftime("%Y-%m-%d"),
                               "measured_on": _the_machine(),
                               **_THIS_RUN[caller]}, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _where(path):
    """Where the instrument was imported from, with no machine's home directory in it:
    ``<checkout>/src/entroptics`` from a source checkout, ``site-packages/entroptics`` installed."""
    return "/".join(path.parts[-3:] if path.parent.name == "src" else path.parts[-2:])


def instrument():
    """Path and digests of the instrument that produced these figures, never the version string.

    Two byte-identical checkouts of `entroptics` reported 0.2.3 and 0.2.2, because ``__version__``
    reads installed distribution metadata rather than the tree being imported.
    """
    import entroptics
    path = pathlib.Path(entroptics.__file__).parent
    digests = {}
    for name in ("projection.py", "reads.py", "aperture.py", "screen.py"):
        f = path / name
        if f.exists():
            digests[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return {"version": getattr(entroptics, "__version__", "unknown"),
            "path": _where(path), "digests": digests}
