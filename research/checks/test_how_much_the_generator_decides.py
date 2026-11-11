"""Every inference figure here rests on four generated stores. What is the spread across others?

The exposure
--------------
`test_inference_on_a_partly_stated_structure.py` draws its stores from seeds 7000 to 7004, and so
does everything built on it: the corruption sweep, the end-to-end reader, the coordinate comparison,
the relation-choosing. They are not independent measurements of the method -- they are the same four
or five structures, read five different ways.

That is not wrong, but it leaves the headline figures with no interval. "80% at a fifth
observed" is a mean over four draws reported to the nearest per cent, and nothing says whether a
fifth draw would have read 75 or 85. A number quoted that precisely should say how much of it is
the store it came from, which is what this line does for the ontology (§47.25) and has never done
for the generated ones.

What is measured
------------------
The same read, the same grader, the same protocol -- over twenty stores drawn from seeds no
other file uses, reported as the spread across them rather than as a mean. Beside it, the figure
the other files quote, computed from their own seeds, so the two can be compared directly.

An interval from draws is the only kind available here: the seeds are the whole source of variation,
each draw is a different structure and a different observation of it, and there is no formula that
would say what a different structure gives.

What it cannot fix
--------------------
If the spread is wide, the figures elsewhere are still what those stores gave -- they do not become
wrong, they become *less precise than they are written*. The remedy is to quote them with the
interval, which is a thing to do to the rows rather than to the code.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

import ask
import operators
import place

from test_inference_on_a_partly_stated_structure import _ops, observed, structure

SHARES = (0.2, 0.4, 0.6)
#: seeds no other file in this line uses, so the draws are fresh rather than the same four again
FRESH = range(4100, 4120)
#: what `test_inference_on_a_partly_stated_structure.py` and everything built on it draw from
THEIRS = range(7000, 7005)


def read_once(seed, share):
    """One store: the share of unstated pairs the read puts in the right group."""
    rng = np.random.default_rng(seed)
    concepts, relations, group, sends = structure(rng)
    statements = observed(concepts, relations, group, sends, share, rng)
    if not statements:
        return None
    stated: dict = {}
    for a, r, b in statements:
        stated.setdefault((a, r), set()).add(b)

    vectors, _ = place.places(statements=statements)
    ops = _ops(vectors, statements)
    present = [c for c in concepts if c in vectors]
    if not present:
        return None
    at = {c: i for i, c in enumerate(present)}
    allp = np.stack([vectors[c] for c in present])

    got, floors = [], []
    for a in concepts:
        if a not in vectors:
            continue
        for r in relations:
            if r not in ops:
                continue
            unstated = [c for c in present if c not in stated.get((a, r), ()) and c != a]
            if not unstated:
                continue
            want = {c for c in unstated if group[c] == sends[r][group[a]]}
            if not want or len(want) == len(unstated):
                continue
            carried = operators.carry(vectors, r, a, ops)
            if carried is None:
                continue
            idx = [at[c] for c in unstated]
            scores = ask._along(allp, carried)[idx]
            got.append(unstated[int(np.argmax(scores))] in want)
            floors.append(len(want) / len(unstated))
    if not got:
        return None
    return float(np.mean(got)), float(np.mean(floors)), len(got)


def main() -> int:
    print(f"the same read on {len(FRESH)} stores from seeds no other file uses, against the "
          f"{len(THEIRS)}")
    print("that every inference figure in this line is drawn from\n")
    print(f"  {'observed':>9}{'stores':>8}{'least':>9}{'median':>9}{'most':>9}"
          f"{'the spread':>13}{'the floor':>11}{'what the others quote':>24}")
    print("  " + "-" * 92)

    rows = {}
    for share in SHARES:
        fresh = [read_once(s, share) for s in FRESH]
        fresh = [x for x in fresh if x is not None]
        theirs = [read_once(s, share) for s in THEIRS]
        theirs = [x for x in theirs if x is not None]
        if not fresh or not theirs:
            continue
        reads = [x[0] for x in fresh]
        rows[f"{share:.1f}"] = {
            "observed": float(share), "stores": len(fresh),
            "least": min(reads), "median": float(np.median(reads)), "most": max(reads),
            "spread": max(reads) - min(reads),
            "floor": float(np.mean([x[1] for x in fresh])),
            "what_the_others_quote": float(np.mean([x[0] for x in theirs])),
            "questions": int(np.mean([x[2] for x in fresh]))}
        v = rows[f"{share:.1f}"]
        print(f"  {share:>9.0%}{v['stores']:>8}{v['least']:>9.0%}{v['median']:>9.0%}"
              f"{v['most']:>9.0%}{v['spread']:>12.0%} {v['floor']:>11.0%}"
              f"{v['what_the_others_quote']:>24.0%}")
    print()

    record.figures(across_stores=rows)

    inside = [k for k, v in rows.items()
              if v["least"] <= v["what_the_others_quote"] <= v["most"]]
    widest = max(rows, key=lambda k: rows[k]["spread"])
    w = rows[widest]

    if len(inside) == len(rows):
        print("  **The figures quoted elsewhere sit inside the spread of fresh stores at every")
        print("  share**, so they are not a lucky draw — but they are means over four stores")
        print(f"  reported to the per cent, and the widest spread here is {w['spread']:.0%}, at")
        print(f"  {w['observed']:.0%} observed, from {w['least']:.0%} to {w['most']:.0%}.")
        print()
        print("  **So the inference figures in this line carry a range of about that width**, and")
        print("  should be read as the store they came from rather than as the method's number.")
        print("  That is the same caution §47.25 states for the ontology, which the generated")
        print("  stores have never had.")
    else:
        outside = [k for k in rows if k not in inside]
        print(f"  **At {', '.join(outside)} observed the figure quoted elsewhere falls outside the")
        print("  spread of twenty fresh stores.** The stores those files draw are not typical of")
        print("  the generator, so the figures resting on them say what those structures gave and")
        print("  not what the method gives, and every row that quotes one needs the interval")
        print("  beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
