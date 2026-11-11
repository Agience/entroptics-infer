"""The two structures, built from the same facts and scored the same way.

Why this exists
-----------------
The vertex model is reported as costing top-1 against typed edges -- 0.54 to 0.40 -- and those two
figures are not taken side by side. The typed one comes from a typed build of the ontology, so two
things differ that are not the structure:

* **The candidate set.** Typed edges are scored over 60 concepts. The vertex model has 66, because
  the six relations are concepts, and a destination in this ontology is never one of them. So six
  impossible distractors sit on one side only.
* **Chance.** 1 in 59 against 1 in 65, for the same reason.

The statements *are* the typed edges -- ``("dog", "is_a", "mammal")`` reads as a triple either way --
so both structures can be built from the same source here and scored against the same candidates.
Anything left after that is the structure.

The two structures
--------------------
``typed``   an edge is ``(src, label, dst)``; a place has one axis per ``(label, direction,
            concept)``; a relation is a name drawn from a closed set.
``vertex``  a relation is a concept, a statement is a vertex, an edge carries position and no type;
            a place has one axis per ``(position, vertex)``.

Both are placed by the shipped walk, and only the basis differs. Each structure is written
as an adjacency in the shipped form -- every vertex's ``(neighbour, axis)`` entries -- and each place
is `Engine._resolved` for the vertices the walk keeps and `Engine._row` for the place they make,
the two steps `Engine.places` takes. Nothing here walks on its own, so a difference between the
structures is the basis and not a second walk. The relation's own axes can still be zeroed out of
both ends before the landing (``blank``), which is a read-side change and takes no walk.

A walk that carries an edge weight or stops at a floor is `walk.carried_places`. The shipped walk
does neither, and `walk.carried_places` is never the arm a basis contrast is read on.

What is being scored
----------------------
Every statement held out in turn: derive the relation's operator from the others, apply it to what
precedes the relation, and see where the true destination ranks. Candidates are the concepts that
ever appear as a destination, which is the same set for both and excludes what neither could
answer.

Held out means held out of the places too
-------------------------------------------
The places are rebuilt for each question from the store *without* that statement. Holding it out of
only the operator is not enough, and the difference is not small: with the places built once from
everything, all three structures score about 0.6, because a concept's place still carries the
statement being asked about. That is recall, not inference. Rebuilding costs 84 builds per structure
instead of one and is the only version of this comparison that means anything.

Some questions become unaskable that way -- removing a statement can take a concept out of the store
entirely, leaving nothing to ask from. Those are reported as skipped rather than scored.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import sys

import numpy as np

from ontology import Engine                                    # the shipped engine

import ontology
from walk import carried_places, triples, typed_basis, vertex_basis  # noqa: F401  (read as fair.*)

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.bind import association_matrix          # noqa: E402
from measure import along as _along  # noqa: E402,F401


def answerable(statements):
    """The concepts that ever stand as a destination. The only things either structure could say."""
    return sorted({b for _, _, b in triples(statements)})


# --------------------------------------------------------------------------- the shipped walk

def shipped_walk(starts, adj, axes):
    """``{start: place}`` -- each place the two steps `Engine.places` takes, on any adjacency.

    ``adj`` is ``{vertex: [(neighbour, axis), ...]}``, the shape `Engine._adjacency` returns.
    `Engine._resolved` settles the walk and stops it at its own aperture; `Engine._row` adds each
    kept vertex's value into every axis it points at. The basis is the only thing a caller supplies.
    """
    index = {a: i for i, a in enumerate(axes)}
    return {start: Engine._row(adj, index, Engine._resolved(adj, start)) for start in starts}


def _only_the_shipped_walk(weighted, floored):
    if weighted or floored:
        raise ValueError("the shipped walk carries no edge weight and no floor; "
                         "a weight or floor study reads carried_places")


# --------------------------------------------------------------------------- typed edges

def typed_places(statements, *, weighted=False, floored=False):
    """Typed edges placed by the shipped walk."""
    _only_the_shipped_walk(weighted, floored)
    starts, adj, axes = typed_basis(statements)
    return shipped_walk(starts, adj, axes), axes


# --------------------------------------------------------------------------- vertex model

def vertex_places(statements, *, weighted=False, floored=False):
    """The unbound ``(position, vertex)`` basis placed by the shipped walk."""
    _only_the_shipped_walk(weighted, floored)
    starts, adj, axes = vertex_basis(statements)
    return shipped_walk(starts, adj, axes), axes


# --------------------------------------------------------------------------- scoring

def score(build, *, weighted, floored, blank_relation, candidates):
    """Top-1, mean rank, and how many could be asked -- places rebuilt without the held statement."""
    all_statements = list(ontology.STATEMENTS)
    top1, ranks, skipped = [], [], 0

    by_relation: dict = {}
    for i, (a, r, b) in enumerate(triples(all_statements)):
        by_relation.setdefault(r, []).append((i, a, b))

    for relation, items in by_relation.items():
        if len(items) < 3:
            continue
        for si, src, dst in items:
            held = [p for j, p in enumerate(all_statements) if j != si]
            pairs = [(a, b) for a, r, b in triples(held) if r == relation]
            if len(pairs) < 2:
                skipped += 1
                continue

            # the store without this statement: nothing about it is in the coordinate, the basis
            # or the operator
            vectors, axes = build(held, weighted=weighted, floored=floored)
            if src not in vectors or dst not in vectors:
                skipped += 1
                continue
            present = [c for c in candidates if c in vectors]
            index = {c: i for i, c in enumerate(present)}
            if dst not in index:
                skipped += 1
                continue
            allp = np.stack([vectors[c] for c in present])

            drop = ([i for i, ax in enumerate(axes) if ax[-1] == relation]
                    if blank_relation else [])

            def rows(which):
                M = np.stack([vectors[p[which]] for p in pairs]).copy()
                if drop:
                    M[:, drop] = 0.0
                return M

            v = vectors[src].copy()
            if drop:
                v[drop] = 0.0
            scores = _along(allp, Engine._land(v, rows(0), rows(1)))
            if src in index:
                scores[index[src]] = -np.inf
            rank = int(np.where(np.argsort(-scores) == index[dst])[0][0]) + 1
            top1.append(rank == 1)
            ranks.append(rank)
    return float(np.mean(top1)), float(np.mean(ranks)), len(ranks)


def main() -> int:
    candidates = answerable(ontology.STATEMENTS)
    print(f"{len(ontology.STATEMENTS)} statements, built both ways from the same source")
    print(f"candidates: the {len(candidates)} concepts that ever stand as a destination -- "
          f"the same set for both, chance 1 in {len(candidates)}\n")

    print(f"{'structure':<10}{'blank':>7}{'top-1':>9}{'mean rank':>12}{'asked':>7}")
    print("-" * 45)
    best = {}
    for name, build in (("typed", typed_places), ("vertex", vertex_places)):
        for blank in (False, True):
            t, r, n = score(build, weighted=False, floored=False,
                            blank_relation=blank, candidates=candidates)
            print(f"{name:<10}{str(blank):>7}{t:>9.2f}{r:>12.1f}{n:>7}")
            if name not in best or t > best[name][0]:
                best[name] = (t, r, blank)
    print("-" * 45)
    print()
    for name, (t, r, b) in best.items():
        print(f"  {name:<7} best {t:.2f} top-1, mean rank {r:.1f} (blank={b})")
    gap = best["typed"][0] - best["vertex"][0]
    print()
    if abs(gap) < 0.02:
        print("  **The structures are level once scored the same way.** The reported gap was the")
        print("  candidate set and the chance level differing, not the structure.")
    elif gap > 0:
        print(f"  Typed edges are still ahead by {gap:.2f} on a like-for-like comparison, so the")
        print("  cost of the vertex model is real and not an artefact of how it was scored.")
    else:
        print(f"  The vertex model is ahead by {-gap:.2f} once both are scored the same way -- the")
        print("  reported gap was an artefact of the comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
