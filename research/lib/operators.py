"""What a relation does to a location, derived from the statements that use it.

The triple
------------
**Content** is what a vertex holds — its gloss. Context is where it sits, which is how it is
found: `place.py` computes that from the edges. Operator is what acts: given a location and a
relation, where does that relation take you?

A relation is a concept here, not a label
-------------------------------------------
``is_a`` is a vertex. Its operator is derived from every statement that contains it — no list of
relation names exists anywhere, so anything used as a relation has an operator by the same rule, and
a relation that has never been used has none.

How it is derived, and why nothing is fitted
----------------------------------------------
Take the statements a relation appears in. Stack the places of what sits before it as the rows of
``A`` and what sits after it as the rows of ``B``. The statement list is the shared ordered axis
— row ``i`` of both is the same statement — which is the shape
:func:`entroptics_infer.bind.association_matrix` asks for, and the operator is ``Ã^T B̃``.

Observed, not fitted: no least squares, no target, no convergence. It is a statistic of the
statements, computed in one pass, and it changes when a statement arrives because the statement
enters the sum.

Before and after, rather than subject and object
--------------------------------------------------
The roles come from position, and the code never says which position means what. A relation is
found wherever it sits in a statement, and what precedes and follows it are read off from there. So
a statement written in another order works without anything being added, and a statement that uses a
relation in two places contributes both.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

import ontology
import place

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.bind import association_matrix          # noqa: E402
from entroptics_infer.wrapper import Engine                   # noqa: E402
from measure import along as _along  # noqa: E402,F401


def uses(relation, *, statements=None):
    """``[(before, after), ...]`` — every place this concept stands between two others."""
    out = []
    for parts in (statements or ontology.STATEMENTS):
        for i, p in enumerate(parts):
            if p == relation and 0 < i < len(parts) - 1:
                out.append((parts[i - 1], parts[i + 1]))
    return out


def relations(*, statements=None):
    """Every concept used as a relation, counted off the statements. Nothing declares them."""
    seen: dict = {}
    for parts in (statements or ontology.STATEMENTS):
        for i, p in enumerate(parts):
            if 0 < i < len(parts) - 1:
                seen[p] = seen.get(p, 0) + 1
    return dict(sorted(seen.items()))


def operator_from_pairs(A, B) -> dict:
    """``{"matrix", "uses", "sides"}`` -- an operator over stacked sides, kept with the sides.

    The matrix is the shipped :func:`entroptics_infer.bind.association_matrix`; the sides are kept
    because a landing is taken from them by the shipped :meth:`Engine._land`, never from the matrix.
    A control that re-pairs or permutes the sides builds its operator here too, so it is read by the
    same arithmetic as the operator it is a control for.
    """
    A, B = np.asarray(A), np.asarray(B)
    return {"matrix": np.asarray(association_matrix(A, B)), "uses": len(A), "sides": (A, B)}


def operator_of(relation, vectors, *, statements=None):
    """``(axes, axes)`` — what this relation does to a location, and how many uses it came from.

    The shipped engine's operator. ``None`` where the relation is used fewer than twice: a
    cross-covariance over one row is that row, and predicts that row and nothing else.
    """
    pairs = uses(relation, statements=statements)
    if len(pairs) < 2:
        return None, len(pairs)
    engine = Engine(statements or ontology.STATEMENTS)
    return engine.operator(relation, vectors), len(pairs)


def all_operators(vectors, *, statements=None):
    """Every relation's operator, derived now. Nothing is stored between calls."""
    engine = Engine(statements or ontology.STATEMENTS)
    out = {}
    for relation in relations(statements=statements):
        before, after = engine._sides(relation, vectors, ())
        if len(before) >= 2:
            out[relation] = operator_from_pairs(np.stack(before), np.stack(after))
    return out


def operators_over(vectors, statements, rng=None):
    """Every relation's operator from ``statements``, over the pairs both of whose ends are placed.

    The pairs are the statements' own (`uses`), so a store the places were built without is read
    the same way as the one they were built from. ``rng`` re-pairs each relation's sides first --
    the null that destroys which source goes with which destination and nothing else.
    """
    out = {}
    for relation in relations(statements=statements):
        pairs = [p for p in uses(relation, statements=statements)
                 if p[0] in vectors and p[1] in vectors]
        if len(pairs) < 2:
            continue
        A = np.stack([vectors[a] for a, _ in pairs])
        order = rng.permutation(len(pairs)) if rng is not None else np.arange(len(pairs))
        B = np.stack([vectors[pairs[int(j)][1]] for j in order])
        out[relation] = operator_from_pairs(A, B)
    return out


def carry(vectors, relation, concept, ops=None):
    """Where ``relation`` takes ``concept``: the shipped landing, from the operator's own sides."""
    ops = ops or all_operators(vectors)
    if relation not in ops:
        return None
    A, B = ops[relation]["sides"]
    return Engine._land(vectors[concept], A, B)


def predict(vectors, relation, concept, *, count, ops=None, among=None, exclude_self=True):
    """The vertices nearest where the relation takes this one. The inference, ranked."""
    landed = carry(vectors, relation, concept, ops)
    if landed is None:
        return []
    names = [c for c in (among or vectors) if not (exclude_self and c == concept)]
    scores = _along(np.stack([vectors[c] for c in names]), landed)
    return sorted(zip(scores, names), reverse=True)[:count]


def truth(relation, concept, *, statements=None):
    """What the ontology actually states, for checking a prediction against."""
    return sorted({b for a, b in uses(relation, statements=statements) if a == concept})


if __name__ == "__main__":
    vectors, axes = place.places()
    concepts = ontology.concepts()
    used = relations()
    ops = all_operators(vectors)
    print(f"{len(used)} concepts are used as relations; {len(ops)} have an operator "
          f"(a relation used once has none)\n")
    print(f"{'relation':<12}{'uses':>6}   where it takes a concept the ontology already places")
    print("-" * 92)
    for relation, n in used.items():
        pairs = uses(relation)
        src = pairs[0][0]
        got = predict(vectors, relation, src, ops=ops, count=3, among=concepts)
        says = truth(relation, src)
        mark = "hit " if got and got[0][1] in says else "    "
        shown = ", ".join(f"{c} {s:.2f}" for s, c in got) or "(no operator)"
        print(f"{relation:<12}{n:>6}   {src} -> {shown}")
        print(f"{'':<18}   {mark}the ontology says: {', '.join(says) or '(nothing)'}")
