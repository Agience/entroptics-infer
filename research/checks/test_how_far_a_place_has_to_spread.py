"""The spread visits every vertex from every vertex. Does the read need the far half of it?

Why this is the question for read-time
----------------------------------------
§47.51 and §47.61 put numbers on what a question costs: a read of this store is under a tenth of a
second, about four fifths of it is placing the store, and inside that 99% is the spread -- a
walk from every vertex out over every other, filling a dense array whose rows and columns both grow
with the statements. The whole thing goes as about the square of how much the store says, which is
why a read that is quick here would not be quick on an ontology of any size.

Nothing about that is fixable by tuning, and the method's own rule forbids picking a cut-off. But
there is a prior question that is pure measurement: does the read depend on the far part of the
spread at all? The weight a vertex carries divides by its fanout at every hop, so what arrives
from far away is small. If it is also *inert* -- if the same question gets the same answer whether
the spread runs two hops or all of them -- then a bounded spread is not an approximation of this
method, it is this method, computed without the part that costs.

What is measured
------------------
The coordinate built with the spread stopped after each of one, two, three ... hops, against the
coordinate with the spread run to exhaustion, on both stores. For each:

``agrees``      the share of reads whose top-1 is the same vertex as the full spread's, over
                every question the store can put -- nothing is held out, so the answer to most
                of them is a pair the store states and both depths retrieve it. That is the right
                population for "does the far spread change the coordinate", and the wrong one for
                "does it change an inference": on held-out questions the two agree far less often
                and the shallow one reads better (CLAIMS §47.126).
``identical``   the largest difference between a vertex's place under the two, after each is scaled
                to unit length, which is what the read actually compares.
``edge visits`` the work the spread does, counted rather than clocked: one per edge followed.
                A timing cannot reproduce to the four places the record gate compares at, and an
                edge traversal is what the cost *is*.

Every depth is reported and none is chosen. If the curve flattens, the depth at which it flattens is
a property of the store that a reader could read off rather than be told -- and finding that rule is
a separate question from knowing the curve exists.

The coordinate, and the hop-limited spreads beside it
-------------------------------------------------------
``depth=None`` is the shipped coordinate: `Engine.places`, the settled walk stopped at its own
aperture, with its work counted off the shipped walk (the vertices each place settled and the
neighbour entries they offered). A finite depth is the contrast this file measures against it, and
it is built from the shipped engine's own primitives and nothing else: the settled values come
from `Engine._settle`, the place from `Engine._row`, and the only research-side step is which
settled vertices are kept -- those fewer than ``depth`` edges from the start, counted on the shipped
`Engine._adjacency`. So a hop-limited place is the shipped place with its walk cut at a hop count
instead of at its aperture, every kept vertex at the value the shipped walk settles it at, and a
difference between the two is the cut and nothing else.

"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import time

import numpy as np

import record

import ask
import ontology
import operators
import place

from test_whether_the_same_coordinate_can_be_built_faster import built
from measure import unit as _unit  # noqa: E402

DEPTHS = (1, 2, 3, 4, 6, 8)


def within(adj, start, depth):
    """``{vertex: hops}`` -- every vertex fewer than ``depth`` edges from ``start``.

    Counted on the shipped adjacency with `place.neighbours`; it settles nothing and adds nothing,
    so it decides which of the shipped walk's settled vertices a hop-limited place keeps and has no
    say in their values.
    """
    hops, edge = {start: 0}, [start]
    for step in range(1, depth):
        edge = sorted({o for v in edge for o in place.neighbours(adj, v)} - set(hops))
        if not edge:
            break
        hops.update((v, step) for v in edge)
    return hops


def places_to(statements, depth, drop=None):
    """``(vectors, rounds, traversals)`` -- the shipped coordinate, or the shipped walk cut at ``depth``.

    ``depth=None`` is `Engine.places` itself, with ``traversals`` the neighbour entries its settled
    vertices offered and ``rounds`` the most vertices any one place settled. A finite ``depth`` reads
    the shipped `Engine._settle` from each start, keeps the settled vertices fewer than ``depth``
    edges away at the value the walk settled them at, and adds them with the shipped `Engine._row`.
    ``traversals`` is then the neighbour entries those kept vertices offer -- the same count the
    shipped arm reports -- and ``rounds`` the most hops any place spans, plus one.
    """
    engine = place._engine(statements)
    without = place._without(drop, statements)
    if depth is None:
        work = {}
        vectors = built(engine, without=without, counted=work)
        return vectors, work.get("most_settled", 0), work.get("neighbours", 0)
    axes = engine.basis(without=without)
    index = {a: i for i, a in enumerate(axes)}
    adj = engine._adjacency(without)

    vectors, rounds, traversals = {}, 0, 0
    for start in engine.vertices(without=without):
        ball = within(adj, start, depth)
        kept = []
        for here, carried in engine._settle(adj, start):
            if here in ball:
                kept.append((here, carried))
                if len(kept) == len(ball):
                    break
        traversals += sum(len(adj[h]) for h, _ in kept)
        rounds = max(rounds, max(ball.values()) + 1)
        vectors[start] = engine._row(adj, index, kept)
    return vectors, rounds, traversals


def _reads(vectors, statements, concepts):
    """``{(src, relation): the vertex reached}`` for every question the store can put."""
    ops = operators.all_operators(vectors, statements=statements)
    present = [c for c in concepts if c in vectors]
    if not present:
        return {}
    allp = np.stack([vectors[c] for c in present])
    out = {}
    for src, relation, _ in statements:
        if relation not in ops or src not in vectors or (src, relation) in out:
            continue
        landed = operators.carry(vectors, relation, src, ops)
        if landed is None:
            continue
        scores = ask._along(allp, landed)
        for i, c in enumerate(present):
            if c == src:
                scores[i] = -np.inf
        out[(src, relation)] = present[int(np.argmax(scores))]
    return out


def survey(name, statements, concepts):
    """One store, at every depth, against its own unlimited spread."""
    a = time.perf_counter()
    full, rounds, full_work = places_to(statements, None)
    full_time = time.perf_counter() - a
    theirs, _ = place.places(statements=statements)
    copy_matches = max(float(np.max(np.abs(_unit(full[k]) - _unit(theirs[k]))))
                       for k in full if k in theirs)
    full_reads = _reads(full, statements, concepts)

    print(f"  {name}: {len(statements)} statements, a shipped place settles at most {rounds} "
          f"vertices")
    print(f"  the shipped coordinate against `place.places`: largest difference "
          f"{copy_matches:.2e} per vertex\n")
    print(f"    {'hops':>6}{'agrees with the shipped':>30}{'largest difference':>21}"
          f"{'edge visits':>13}{'of the full':>13}")
    print("    " + "-" * 78)

    rows = {}
    for depth in DEPTHS:
        a = time.perf_counter()
        cut, _, work = places_to(statements, depth)
        took = time.perf_counter() - a
        gap = max(float(np.max(np.abs(_unit(cut[k]) - _unit(full[k])))) for k in full)
        reads = _reads(cut, statements, concepts)
        shared = set(reads) & set(full_reads)
        agrees = (sum(reads[k] == full_reads[k] for k in shared) / len(shared)
                  if shared else float("nan"))
        rows[str(depth)] = {"hops": depth, "agrees": agrees, "largest_difference": gap,
                            "traversals": work, "share_of_full": work / full_work}
        print(f"    {depth:>6}{agrees:>30.0%}{gap:>21.2e}{work:>13}"
              f"{work / full_work:>12.1%}")
    print()
    return {"statements": len(statements), "most_settled_by_a_place": rounds,
            "copy_largest_difference": copy_matches, "full_traversals": full_work,
            "by_depth": rows}


def _generated(concepts, relations, k, rng):
    names = [f"c{i}" for i in range(concepts)]
    labels = [f"r{j}" for j in range(relations)]
    out = []
    for src in names:
        others = [c for c in names if c != src]
        for relation in labels:
            for dst in rng.choice(others, size=k, replace=False):
                out.append((src, relation, str(dst)))
    return tuple(out), names


print_ = print


def main() -> int:
    print("the coordinate, built with the spread stopped after each number of hops,")
    print("against the same coordinate with the spread run out\n")

    out = {}
    out["the ontology"] = survey("the ontology", list(ontology.STATEMENTS),
                                 list(ontology.concepts()))
    statements, names = _generated(36, 3, 2, np.random.default_rng(36))
    out["generated"] = survey("a generated store", list(statements), names)

    record.figures(spread_depth=out)

    for name, v in out.items():
        rows = v["by_depth"]
        one = rows[str(min(DEPTHS))]
        apart_at = [d for d in DEPTHS if rows[str(d)]["agrees"] != 1.0]
        print_(f"  **On {name}, one hop agrees with the shipped coordinate on "
               f"{one['agrees']:.0%} of**")
        print_(f"  **the questions, for {one['share_of_full']:.1%} of the shipped walk's work.** "
               f"A shipped place settles at most {v['most_settled_by_a_place']} vertices.")
        if not apart_at:
            print_("  Every depth measured names the same answers as the shipped coordinate.")
        else:
            print_(f"  Cut at {', '.join(str(d) for d in apart_at)} hops the walk names a different "
                   f"answer on some questions,")
            dearer = [d for d in DEPTHS if rows[str(d)]["share_of_full"] > 1]
            if dearer:
                print_(f"  From {min(dearer)} hops a cut place offers more neighbour entries than the "
                       f"shipped walk does,")
                print_("  so there a hop count costs more than the aperture that stops the shipped walk.")
        print_()

    print_("  **Nothing here is changed and no depth is chosen.** A reader that stopped at a fixed")
    print_("  number of hops would carry a constant, which this method does not permit. What the")
    print_("  table shows is whether such a rule could exist to be derived -- and where the read")
    print_("  stops changing, the far spread is doing arithmetic nobody reads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
