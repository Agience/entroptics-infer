"""Every figure here comes from one store of 84 statements. How much would another one differ?

What every other number reported is missing
---------------------------------------------
Stage 44 of `CLAIMS.md` carries about thirty figures and not one of them has an uncertainty.
They are all measured on `ontology.py` as it stands, and reported as though the store were the
population rather than one draw from it. A figure with no spread beside it cannot be argued with:
0.26 might be what this method does, or what these 84 statements happen to give.

There is no second store to check against, and inventing one would measure the generator. What
there is, is this store's own subsets.

The check
-----------
Drop a share of the statements at random, re-run the headline measurement on what is left, and do
it many times. The spread across subsets is what a differently-written store of the same kind would
be expected to give. It is not a confidence interval for the population of all ontologies -- there
is only one store here and subsets of it are not independent of it -- but it is the answer to "how
much of 0.26 is the method and how much is these particular statements", which is the question
nobody has asked of any figure in the ledger.

Two shares are dropped rather than one, because the spread should grow as the store shrinks. If it
does not, the resampling is not doing what it is supposed to.

What is measured
------------------
The core read: hold each statement out of the operator
and of the places, apply its relation to its source, and ask whether the true destination comes
first. Reported as the median across subsets with the range, against the whole store's value.

What each outcome means
-------------------------
A whole-store value sitting inside a narrow spread is a figure that means something about the
method. A wide spread says the number is a property of these statements, and every figure quoted
from this store should carry the range rather than the point.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections

import numpy as np

import record

import ask
import ontology
import operators
import place


def read_score(statements, concepts):
    """Top-1 over every held-out statement of this store."""
    hits = []
    for i, parts in enumerate(statements):
        if len(parts) != 3:
            continue
        src, relation, dst = parts
        held = [s for j, s in enumerate(statements) if j != i]
        vectors, _ = place.places(drop=[i], statements=statements)
        if any(c not in vectors for c in (src, dst, relation)):
            continue
        ops = operators.all_operators(vectors, statements=held)
        if relation not in ops:
            continue
        present = [c for c in concepts if c in vectors]
        at = {c: i2 for i2, c in enumerate(present)}
        if dst not in at:
            continue
        allp = np.stack([vectors[c] for c in present])
        landed = operators.carry(vectors, relation, src, ops)
        if landed is None:
            continue
        scores = ask._along(allp, landed)
        if src in at:
            scores[at[src]] = -np.inf
        hits.append(present[int(np.argmax(scores))] == dst)
    return (float(np.mean(hits)) if hits else float("nan")), len(hits)


def main() -> int:
    whole = list(ontology.STATEMENTS)
    concepts = list(ontology.concepts())
    rng = np.random.default_rng(len(whole))

    full, asked = read_score(whole, concepts)
    print(f"the whole store: {len(whole)} statements, {asked} askable, top-1 {full:.2f}\n")

    print(f"  {'dropped':>9}{'subsets':>9}{'statements':>12}{'askable':>9}"
          f"{'median':>9}{'range':>16}")
    print("  " + "-" * 65)

    rows = {}
    for share_dropped, subsets in ((4, 12), (8, 12)):
        # a share written as one-in-k, so the number in the code is a count and not a rate
        got, sizes, askables = [], [], []
        for _ in range(subsets):
            keep = sorted(rng.permutation(len(whole))[: len(whole) - len(whole) // share_dropped])
            kept = [whole[i] for i in keep]
            s, n = read_score(kept, concepts)
            if not np.isnan(s):
                got.append(s)
                sizes.append(len(kept))
                askables.append(n)
        label = f"1 in {share_dropped}"
        rows[label] = {"subsets": len(got), "dropped_one_in": share_dropped,
                       "statements": float(np.mean(sizes)),
                       "askable": float(np.mean(askables)),
                       "median": float(np.median(got)),
                       "low": float(min(got)), "high": float(max(got))}
        print(f"  {label:>9}{len(got):>9}{np.mean(sizes):>12.0f}{np.mean(askables):>9.0f}"
              f"{np.median(got):>9.2f}{min(got):>10.2f} to {max(got):.2f}")
    print("  " + "-" * 65)
    print()

    record.figures(whole_store=full, askable=asked, statements=len(whole), subsets=rows)

    narrow, wide = rows["1 in 8"], rows["1 in 4"]
    grew = (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])
    print(f"  dropping one statement in 8 gives {narrow['low']:.2f} to {narrow['high']:.2f}; "
          f"one in 4 gives {wide['low']:.2f} to {wide['high']:.2f}")
    print(f"  the spread {'grows' if grew else 'does NOT grow'} as the store shrinks, which is "
          f"{'what resampling should do' if grew else 'not what resampling should do'}")
    print()

    inside = narrow["low"] <= full <= narrow["high"]
    span = max(wide["high"] - wide["low"], narrow["high"] - narrow["low"])
    if not grew:
        print("  **The resampling is not behaving**, so nothing should be read off the spread until")
        print("  that is understood: a smaller store must be more variable, and if it is not then")
        print("  the subsets are not varying what they are supposed to.")
    elif span > full:
        print(f"  **The figure is a property of these statements as much as of the method.** The")
        print(f"  spread across subsets, {span:.2f}, is as large as the value itself, {full:.2f}.")
        print("  Every number quoted from this store should carry a range, and the ledger's Stage")
        print("  44 currently carries none.")
    else:
        print(f"  The whole store reads {full:.2f} and subsets of it run {wide['low']:.2f} to "
              f"{wide['high']:.2f}.")
        print(f"  {'That contains the whole-store value' if inside else 'The whole-store value sits outside the narrower spread'}, "
              f"and the spread is smaller than the value.")
        print("  So the figure says something about the method and not only about these 84")
        print("  statements -- but it should still be quoted with the range beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
