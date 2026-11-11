"""If a query names a relation, what should naming it do?

Where this comes from
-----------------------
On 70 held-out statements, a step told which relation to apply lands on the right destination
more often than a step choosing the relation for itself. Choosing the relation is most of the
difficulty.

The reader always has to choose, because the only thing a query contributes is concept names and
the relation word in "where does a salmon live" is "live", which names nothing. But a relation here
is a concept, so a query can name it: ``salmon lives_in`` names two.

That needs the query split on whole names. Splitting on ``[a-z]+`` breaks ``lives_in`` into "lives"
and "in", and the six concepts whose names carry an underscore are exactly the six used as
relations, so under that split the only concepts that could say which relation was wanted would be
the only ones a query could never name.

Two ways to use a named relation, and they are not the same
-------------------------------------------------------------
``summed``    add the relation's place to the query's, as `ask.land` does for every named concept.
              This says the relation is somewhere to move toward.

``selecting`` apply only that relation's operator. This says the relation is what acts, and the
              query has given a context and an operator, leaving the content to be found.

The triple is the argument for the second: content, context and operator, and given any two the
third is determined. A relation is not a location, and a relation-concept's place is the hub every
statement using it passes through -- so adding it to the query direction should make the direction
*less* specific, not more.

That is a prediction, and it is what is measured here.

The arms
----------
``source only``          what the reader does now: apply every operator, keep the best landing.
``source + summed``      the relation named and added to the query direction, operator still chosen.
``source, selecting``    the relation named and used to pick the operator.

All three are asked the same held-out statements with the places rebuilt each time, and scored on
whether the true destination comes first.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

from ontology import Engine                                    # the shipped engine

import record

import ask
import ontology
import operators
import place


def main() -> int:
    concepts = list(ontology.concepts())
    statements = list(ontology.STATEMENTS)
    arms = {"source only": {}, "source + summed": {}, "source, selecting": {}}

    for si, parts in enumerate(statements):
        if len(parts) != 3:
            continue
        src, relation, dst = parts
        held = [s for i, s in enumerate(statements) if i != si]
        vectors, _ = place.places(drop=[si])
        if any(c not in vectors for c in (src, dst, relation)):
            continue
        ops = operators.all_operators(vectors, statements=held)
        if relation not in ops:
            continue
        present = [c for c in concepts if c in vectors]
        allp = np.stack([vectors[c] for c in present])
        at = {c: i for i, c in enumerate(present)}

        def best_over(labels, start_from):
            """The best landing over the given relations, applied at ``start_from``."""
            best = None
            for label in labels:
                landed = operators.carry(vectors, label, start_from, ops)
                if landed is None:
                    continue
                scores = ask._along(allp, landed)
                if src in at:
                    scores[at[src]] = -np.inf
                i = int(np.argmax(scores))
                if best is None or scores[i] > best[0]:
                    best = (float(scores[i]), present[i])
            return best

        got = best_over(ops, src)
        if got:
            arms["source only"][si] = got[1] == dst

        # summed: the query direction is the source and the relation together, and the step is
        # taken from that direction rather than from the source alone
        mixed = vectors[src] + vectors[relation]
        best = None
        for label in ops:
            landed = Engine._land(mixed, *ops[label]["sides"])
            scores = ask._along(allp, landed)
            if src in at:
                scores[at[src]] = -np.inf
            i = int(np.argmax(scores))
            if best is None or scores[i] > best[0]:
                best = (float(scores[i]), present[i])
        if best:
            arms["source + summed"][si] = best[1] == dst

        got = best_over([relation], src)
        if got:
            arms["source, selecting"][si] = got[1] == dst

    print(f"{len(statements)} statements, each held out of the operator and of the places\n")
    print(f"  {'what the query gives':<24}{'asked':>7}{'reached first':>15}")
    print("  " + "-" * 46)
    for name in ("source only", "source + summed", "source, selecting"):
        v = list(arms[name].values())
        print(f"  {name:<24}{len(v):>7}{np.mean(v):>15.0%}")
    print("  " + "-" * 46)
    print()

    record.figures(
        top1={k: float(np.mean(list(v.values()))) for k, v in arms.items()},
        asked={k: len(v) for k, v in arms.items()},
    )

    import math
    print(f"    {'pair':<40}{'wins':>6}{'losses':>8}{'p':>9}")
    print("    " + "-" * 64)
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sorted(set(arms[a]) & set(arms[b]))
            w = sum(1 for si in both if arms[a][si] and not arms[b][si])
            l = sum(1 for si in both if arms[b][si] and not arms[a][si])
            m = w + l
            pv = (min(1.0, 2 * sum(math.comb(m, j) for j in range(0, min(w, l) + 1)) / 2 ** m)
                  if m else 1.0)
            print(f"    {a + ' vs ' + b:<40}{w:>6}{l:>8}{pv:>9.3f}")
    print()

    sel = float(np.mean(list(arms["source, selecting"].values())))
    sm = float(np.mean(list(arms["source + summed"].values())))
    only = float(np.mean(list(arms["source only"].values())))
    if sel > sm and sel > only:
        print("  **A named relation should select the operator, not join the landing.** Summing it")
        print("  into the query direction is the worse of the two ways to use it, which is what the")
        print("  triple predicts: a relation is what acts, not somewhere to go, and its place is the")
        print("  hub every statement using it passes through.")
    elif sm > sel:
        print("  Summing the relation into the query beats selecting on it, so a relation does")
        print("  behave as a place to move toward and the triple's reading is wrong here.")
    else:
        print("  Naming the relation does not help either way, so the difficulty in choosing it is")
        print("  not resolved by being told.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
