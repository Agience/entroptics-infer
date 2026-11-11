"""Re-run every test that records, and check the record it writes is the one already there.

The other half of the gate
----------------------------
`prose_check.py` checks a figure in `CLAIMS.md` against the record its script wrote. This checks the
record against the script. Without it, a record written before its test changed keeps backing a
claim that the code does not produce, and the gate reports a clean run: the claim matches the
record, the record matches nothing.

The same holds for any change to how a record is written. A `record.figures` that overwrote rather
than accumulated would make a test recording twice silently discard its first table, and figures
that were measured would stop being backed while staying in the prose. Re-running the script and
comparing is what catches that.

This is `verify.py`'s job for Phase A, done for the reader's tests.

What it does
--------------
For each `results/test_*.json`, run the script that names itself in it, and compare what comes back
with what was there. Floats are compared at the precision a claim would quote them to, because that
is the precision anything downstream depends on -- a figure that moves in the sixth decimal has not
drifted for any purpose here.

Three outcomes, and they mean different things
------------------------------------------------
``same``      the record reproduces. What the ledger cites is what the code does.
``drifted``   the script now produces a different number under the same key. Whatever cites that
              key is stale, and the ledger has to be corrected rather than the record quietly
              rewritten.
``missing``   a key that was recorded is no longer produced. A claim citing it is citing something
              the code stopped measuring.

A record that has never been written is not a failure here -- it means the test does not record, and
`prose_check.py` already reports which claims fall back to the pooled set for that reason.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: Keys that are expected to differ between runs and say nothing about a figure.
#:
#: ``timings`` is where a test puts a wall clock. A wall clock is a figure about the box -- it moves
#: with what else is running on it and with the machine it is running on -- and a gate that re-runs a
#: test and compares seconds would report drift on every run forever and mean nothing by it. What
#: must reproduce is what the code computes about the data, which is everything else in the record.
#: `prose_check.py` still reads these, so a claim quoting a speed still has to name a measured one.
NOT_A_FIGURE = frozenset({"recorded", "script", "timings", "measured_on"})

#: A wall clock recorded beside the figures rather than under ``timings``, at any depth: a key named
#: ``seconds``, ``seconds_<what>``, ``<what>_seconds`` or ``ms_<what>``. Ten tests record their clocks
#: this way, and re-run on any box they report every one of them as drift -- the same thing
#: ``timings`` exists to keep out of the comparison, found by name because it was not filed there.
#: ``second_only`` and ``*_times`` do not match: those are counts and ratios of counts, and they are
#: compared.
WALL_CLOCK = re.compile(r"^(?:seconds|ms)(?:_|$)|_seconds$")


def flatten(obj, prefix=""):
    """``{path: value}`` for every number in a record, so drift is reported per figure."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if prefix == "" and k in NOT_A_FIGURE:
                continue
            if WALL_CLOCK.search(str(k)):
                continue
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool) or obj is None:
        out[prefix] = obj
    elif isinstance(obj, (int, float)):
        v = float(obj)
        if v != v:
            # a NaN is a real value here -- "this arm could not be asked" -- and it has to compare
            # equal to itself, which NaN does not. Comparing raw floats reported every NaN as
            # drifted from itself: eight of them, on the first run of this file.
            out[prefix] = "not a number"
        else:
            # the precision a claim would quote; below that, movement is not drift for any purpose
            out[prefix] = round(v, 4)
    else:
        out[prefix] = obj
    return out


def main(argv=None) -> int:
    only = set(argv or [])
    records = sorted(RESULTS.glob("test_*.json"))
    if only:
        records = [p for p in records if p.stem in only]
    if not records:
        print("no records to verify; run the tests first")
        return 0

    print(f"{len(records)} records, each checked against a fresh run of the script that wrote it\n")
    print(f"  {'record':<44}{'figures':>9}{'drifted':>9}{'missing':>9}")
    print("  " + "-" * 71)

    bad, elsewhere = {}, {}
    for path in records:
        found = path.read_bytes()
        before = json.loads(found.decode("utf-8"))
        script = HERE / before.get("script", f"{path.stem}.py")
        if not script.is_file():
            script = HERE / "checks" / script.name
        if not script.is_file():
            print(f"  {path.stem:<44}{'':>9}{'':>9}{'no script':>9}")
            bad[path.stem] = [("(the script named in the record is gone)", None, None)]
            continue

        was = flatten(before)
        run = subprocess.run([sys.executable, str(script.relative_to(HERE))], cwd=HERE,
                             capture_output=True, text=True)
        if run.returncode != 0:
            print(f"  {path.stem:<44}{len(was):>9}{'':>9}{'failed':>9}")
            bad[path.stem] = [("(the script no longer runs)", None, None)]
            path.write_bytes(found)
            continue

        after = json.loads(path.read_text(encoding="utf-8"))
        now = flatten(after)
        # **Which changed, the code or the machine.** A record says what ran it, and a figure that
        # moves when the interpreter moves is not a claim drifting from its script -- it is a claim
        # that was never reproducible anywhere else. Measured: one test reports 1 statement changing
        # no answer under Python 3.12 and 0 under 3.14, on byte-identical code.
        was_on = before.get("measured_on") or {}
        now_on = after.get("measured_on") or {}
        moved = [k for k in set(was_on) | set(now_on) if was_on.get(k) != now_on.get(k)]
        if moved:
            elsewhere[path.stem] = [(k, was_on.get(k), now_on.get(k)) for k in sorted(moved)]
        # **Put the record back byte for byte as it was found.** Running the script rewrote it, so
        # without this the check repairs the very thing it reports: the drift is printed once and
        # then gone, a second run passes, and the old value survives only in that one printout.
        # This file's own verdict says a drifted record is not fixed by re-recording it -- so it
        # must not re-record it. Restored, the gate keeps failing until the claim is dealt with,
        # which is what a gate is for.
        path.write_bytes(found)
        drifted = [(k, was[k], now[k]) for k in was if k in now and was[k] != now[k]]
        missing = [(k, was[k], None) for k in was if k not in now]
        if drifted or missing:
            bad[path.stem] = drifted + missing
        print(f"  {path.stem:<44}{len(was):>9}{len(drifted):>9}{len(missing):>9}")
    print("  " + "-" * 71)
    print()

    if not bad:
        print("  Every record reproduces. What `CLAIMS.md` cites is what the code does now, and not")
        print("  what it did when the figure was written down.")
        return 0

    print("  **Records that no longer reproduce:**")
    for name, rows in bad.items():
        print(f"    {name}")
        for key, was_v, now_v in rows[:8]:
            if now_v is None and was_v is None:
                print(f"      {key}")
            elif now_v is None:
                print(f"      {key}: was {was_v}, no longer produced")
            else:
                print(f"      {key}: was {was_v}, now {now_v}")
        if len(rows) > 8:
            print(f"      ... and {len(rows) - 8} more")
    print()
    if elsewhere:
        print("  **Written on a different machine from the one checking them:**")
        for name, moved in elsewhere.items():
            parts = ", ".join(f"{k} {was} then {now}" for k, was, now in moved)
            print(f"    {name}: {parts}")
        print()
        print("  For those, a drifted figure may be the machine rather than the code, and this gate")
        print("  cannot tell which. Re-run them where they were written, or make the figure not")
        print("  depend on the last bit -- a decision inside the arithmetic's own resolution "
              "resolves")
        print("  whichever way the platform falls.")
        print()
    print("  A drifted record is not fixed by re-recording it. Whatever cites the key is stale, and")
    print("  the claim is what has to change.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
