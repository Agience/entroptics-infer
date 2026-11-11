"""Does the order the statements arrived in reach the coordinate, and reach the answer?

What this is asking
---------------------
The coordinate is meant to be the store's own structure. A store is a set of statements: the
order somebody happened to write them down in carries nothing, so nothing derived from the store
should move when that order changes. That is not a preference, it is what "derived from the store"
means -- an insertion order is a chosen thing, and a chosen thing in the coordinate is the same
defect as a constant in the coordinate.

Nothing checked it. This does, by building the same statements in several orders and comparing the
coordinates cell by cell, with the statement axes named by their parts rather than by their
positional ids, so a shuffle does not read as a difference merely by relabelling the columns.

The two walks
---------------
``shipped``             :meth:`Engine.places`. Each vertex contributes once, at the largest value
                        that reaches it: the values are settled by priority queue and accumulated
                        afterwards, which is a max-product shortest path, and a shortest path does
                        not depend on the order the edges were offered in. Each walk stops at its
                        own aperture.
``improving estimate``  the contrast, built from the shipped walk's own primitives: the same
                        kept vertices `Engine._resolved` keeps, each one added by `Engine._row`
                        once for every time its estimate improves as its neighbours offer their
                        settled values (`Engine._settle`) in the order the store lists them. The
                        last improvement is the value the shipped walk settles it at, so the two
                        arms differ only in the earlier, smaller improvements -- and how many of
                        those there are is which neighbour the store happened to list first.

What the read does under it
-----------------------------
The same held-out protocol as everything else: the asked-about pair removed from the operator and
from the coordinate, every statement asked in turn. Reported per order, so the spread across orders
is visible rather than one order's figure standing for all of them.

And how close the decisions are
---------------------------------
A read that names its answer by a margin of one part in a billion has not decided anything -- it has
reported the last rounding error in the sum. So the gap between the first and second candidate is
measured on every question, and the questions sitting inside floating-point noise are counted. Those
are the questions that can flip when nothing about the store has changed.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys

import numpy as np

import ask
import ontology
import record
from checking import shipped

from entroptics_infer import Engine

#: How many other orders to try. Three is enough to show a difference exists or does not; it is not
#: a sample the spread is estimated from, and no figure here is a mean over it.
ORDERS = 3

STATEMENTS = tuple(tuple(s) for s in ontology.STATEMENTS)


def rounding_bound(top, dimension):
    """How far apart two scores must be before the difference is the store rather than the sum.

    A score here is a dot product of two vectors of ``dimension`` entries, divided by two norms.
    Summing ``dimension`` terms in floating point accumulates at worst ``dimension`` roundings, each
    of relative size ``spacing(1)``, so two candidates whose scores differ by less than
    ``dimension * spacing(1) * |score|`` are not distinguished by the arithmetic that produced them
    -- add the same terms in another order and they can change places.

    **This is derived, not chosen.** It is read off the size of the score and the length of the
    vector that produced it, both of which the store fixes; there is no level in it to pick, and it
    moves on its own when the coordinate gets wider or the scores get larger.
    """
    return dimension * float(np.spacing(float(1))) * abs(float(top))


def improving_estimate(engine, *, without=()):
    """The shipped walk's kept vertices, each added once per improving estimate.

    Built from the shipped primitives and nothing else. `Engine._settle` gives every reachable
    vertex its settled value; `Engine._resolved` gives the vertices a place keeps; `Engine._row`
    adds them. The research-side step is the multiplicity: a kept vertex's neighbours, in the order
    the shipped `Engine._adjacency` lists them -- which is the order the statements were given in --
    each offer their settled value divided by their fanout, and the vertex is added at every offer
    that beats the best before it. The last such offer is the value it settles at, so with one
    contribution per vertex this is the shipped place exactly; the earlier improvements are the
    whole of the contrast, and how many there are depends on the store's order. A store is a set,
    and that order carries nothing.
    """
    axes = engine.basis(without=without)
    index = {a: i for i, a in enumerate(axes)}
    adj = engine._adjacency(without)

    vectors = {}
    for start in engine.vertices(without=without):
        value = dict(engine._settle(adj, start))
        added = []
        for here, carried in engine._resolved(adj, start):
            if here == start:
                added.append((here, carried))
                continue
            best = float(0)
            for other, _axis in adj[here]:
                offer = value.get(other, float(0)) / (len(adj[other]) or 1)
                if offer > best:
                    best = offer
                    added.append((here, offer))
        vectors[start] = engine._row(adj, index, added)
    return vectors


def by_name(engine, statements, vectors):
    """The coordinate keyed by what survives a shuffle.

    A statement's id is its position in the store, so every ``("in", id)`` axis is renamed when the
    statements are reordered and every statement vertex with it. Keying both by the statement's own
    parts removes that relabelling, which is the only difference a shuffle is *supposed* to make.
    """
    ids = {engine.statement_id(i): parts for i, parts in enumerate(statements)}
    axes = engine.basis()
    names = [("in", ids[a[1]]) if a[0] == "in" else a for a in axes]
    return {ids.get(v, v): {n: float(x) for n, x in zip(names, row) if x != 0}
            for v, row in vectors.items()}


def apart(one, other):
    """The largest difference between two coordinates, over every cell either of them fills."""
    worst = float(0)
    for key, row in one.items():
        theirs = other.get(key, {})
        for axis in set(row) | set(theirs):
            worst = max(worst, abs(row.get(axis, float(0)) - theirs.get(axis, float(0))))
    return worst


def read(statements, walk):
    """Held-out accuracy, and how close the decisions were.

    The protocol is the published one: every statement joining the asked-about pair is removed from
    the operator and from the coordinate, and the answer is the candidate the carried query lies
    furthest along.
    """
    engine = Engine.on_statements(statements)
    concepts = engine.concepts()
    right = among_right = asked = among_asked = 0
    draw = float(0)
    gaps = []
    for src, relation, dst in statements:
        pair = [s for s in statements if s[0] == src and s[2] == dst]
        held = [s for s in statements if s not in pair]
        if not any(m == relation for _, m, _ in held):
            continue
        vectors = walk(engine, without=pair)
        if src not in vectors or dst not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=pair)
        if landed is None:
            continue
        stated = {b for a, m, b in held if a == src and m == relation}
        pool = [c for c in concepts if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        into = {b for a, m, b in held if m == relation}
        reached = [c for c in pool if c in into]

        asked += 1
        draw += float(1) / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        if pool[int(np.argmax(scores))] == dst:
            right += 1
        ranked = np.sort(scores)[::-1]
        gaps.append((float(ranked[0] - ranked[1]),
                     rounding_bound(ranked[0], len(next(iter(vectors.values()))))))
        if dst in reached and len(reached) > 1:
            among_asked += 1
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            if reached[int(np.argmax(sc))] == dst:
                among_right += 1
    margins = np.array([g for g, _ in gaps])
    bounds = np.array([b for _, b in gaps])
    return {"asked": asked, "names": right / asked, "a_draw_would": draw / asked,
            "among_asked": among_asked, "among": among_right / among_asked,
            "decided_inside_the_rounding": int((margins <= bounds).sum()),
            "decided_by_nothing_at_all": int((margins == float(0)).sum()),
            "closest_margin": float(margins.min()),
            "the_bound_there": float(bounds[int(np.argmin(margins))])}


#: Set in a child run, so the child measures and prints and does not fork again.
CHILD = "ENTROPTICS_ORDER_CHILD"

#: The hash seeds the child runs use. Any distinct values would do -- what is being asked is whether
#: the answer moves when the seed does, and three of them is enough for it to have had the chance.
HASH_SEEDS = ("1", "2", "3")


def digest(vectors, engine):
    """One value standing for a whole coordinate, down to the last bit of every cell.

    Taken over the vertices in the engine's own order, so nothing about how ``vectors`` was keyed
    can reach it, and over the raw bytes, so a difference in the last bit is a difference here.
    """
    run = hashlib.sha256()
    for vertex in engine.vertices():
        run.update(np.asarray(vectors[vertex], dtype=float).tobytes())
    return run.hexdigest()[:16]


def what_it_read(reads):
    """The result, reduced to what would have to agree between two runs of the same store.

    Carries the two coordinates as well as the reads, because that is where the defect being
    guarded against lives: a walk can settle into a container whose iteration order moves and still
    name the same answers on a store where nothing sits close enough to flip.
    """
    engine = Engine.on_statements(STATEMENTS)
    return {"reads": {k: (round(v["names"], 12), round(v["among"], 12),
                          v["decided_inside_the_rounding"], v["decided_by_nothing_at_all"])
                      for k, v in reads.items()},
            "the shipped coordinate": digest(shipped(engine), engine)}


def under_other_hash_seeds(reads):
    """Re-run this file under other hash seeds and return how many agree with what was just read.

    **A file that measures order dependence must not have one.** A walk that holds its settled
    vertices in a `set` and iterates it into a floating-point sum has the last bit of every cell
    decided by `PYTHONHASHSEED`; the shipped walk holds them in a list in the order they settled.
    That is checked here rather than by remembering to check it.

    The child is the same file with ``ENTROPTICS_ORDER_CHILD`` set, which makes it measure, print
    its own table and stop -- no record, no grandchildren.
    """
    if os.environ.get(CHILD):
        return None
    here = pathlib.Path(__file__).resolve()
    want = what_it_read(reads)
    agreeing = {"the read": 0, "the shipped coordinate": 0}
    for seed in HASH_SEEDS:
        env = dict(os.environ, **{CHILD: seed, "PYTHONHASHSEED": seed})
        run = subprocess.run([sys.executable, here.name], cwd=here.parent, env=env,
                             capture_output=True, text=True)
        if run.returncode != 0:
            continue
        got = json.loads(run.stdout.strip().splitlines()[-1])
        if {k: tuple(v) for k, v in got["reads"].items()} == want["reads"]:
            agreeing["the read"] += 1
        agreeing["the shipped coordinate"] += int(got["the shipped coordinate"]
                                                  == want["the shipped coordinate"])
    return agreeing


def main() -> int:
    print("does the order a store was given in reach the coordinate, and reach the answer?\n")
    print(f"  {len(STATEMENTS)} statements from `ontology.py`, and the same statements in "
          f"{ORDERS} other orders\n")

    orders = {"as given": STATEMENTS}
    rng = random.Random(0)
    for trial in range(ORDERS):
        other = list(STATEMENTS)
        rng.shuffle(other)
        orders[f"shuffled {trial}"] = tuple(other)

    walks = {"improving estimate": improving_estimate, "shipped": shipped}
    drift = {}
    for label, walk in walks.items():
        engine = Engine.on_statements(STATEMENTS)
        base = by_name(engine, STATEMENTS, walk(engine))
        cells = sum(len(r) for r in base.values())
        worst = []
        for name, statements in orders.items():
            if name == "as given":
                continue
            other = Engine.on_statements(statements)
            worst.append(apart(base, by_name(other, statements, walk(other))))
        drift[label] = {"cells": cells, "worst_cell_moves": max(worst)}
        print(f"  the {label} walk: {cells} non-zero cells, the worst moves "
              f"**{max(worst):.4g}** across the {ORDERS} orders")
    print()

    reads = {}
    print(f"  {'order':<14}{'walk':<20}{'asked':>7}{'draw':>8}{'names':>8}"
          f"{'reached':>9}{'among':>8}{'inside the rounding':>21}")
    print("  " + "-" * 87)
    for name, statements in orders.items():
        for label, walk in walks.items():
            r = read(statements, walk)
            reads[f"{name}/{label}"] = r
            print(f"  {name:<14}{label:<20}{r['asked']:>7}{r['a_draw_would']:>8.1%}"
                  f"{r['names']:>8.1%}{r['among_asked']:>9}{r['among']:>8.1%}"
                  f"{r['decided_inside_the_rounding']:>21}")
    print()

    if os.environ.get(CHILD):
        # a child run says what it read and stops: no record, no prose, no grandchildren
        print(json.dumps(what_it_read(reads), sort_keys=True))
        return 0

    seeds = under_other_hash_seeds(reads)
    inside_the_rounding = max(r["decided_inside_the_rounding"] for r in reads.values())
    record.figures(orders=ORDERS, statements=len(STATEMENTS), drift=drift, reads=reads,
                   hash_seeds_agreeing=seeds, hash_seeds_tried=len(HASH_SEEDS),
                   # what scoring the undecidable questions either way is worth, which is the
                   # figure the last paragraph quotes and the one a claim would argue over
                   share_decided_inside_the_rounding=(
                       inside_the_rounding / reads["as given/shipped"]["asked"]))

    contrast_moves = drift["improving estimate"]["worst_cell_moves"]
    shipped_moves = drift["shipped"]["worst_cell_moves"]
    contrast_names = sorted({round(r["names"], 6) for k, r in reads.items()
                             if k.endswith("improving estimate")})
    shipped_names = sorted({round(r["names"], 6) for k, r in reads.items() if k.endswith("shipped")})
    inside = max(r["decided_inside_the_rounding"] for r in reads.values())
    asked = reads["as given/shipped"]["asked"]
    closest = reads["as given/shipped"]

    print(f"  **The shipped coordinate moves {shipped_moves:g} across every cell and every order**, "
          f"and the")
    print(f"  shipped read names {', '.join(f'{v:.1%}' for v in shipped_names)} of {asked} on every "
          f"order. A vertex that")
    print(f"  contributes once per improving estimate instead moves the worst cell "
          f"**{contrast_moves:.4g}** and reads")
    print(f"  {', '.join(f'{v:.1%}' for v in contrast_names)}: the multiplicity is which neighbour "
          f"came first, and a store is a set.")
    print()
    print(f"  **{closest['decided_inside_the_rounding']}** of {asked} shipped questions are decided "
          f"by a gap the arithmetic cannot resolve:")
    print(f"  the closest reads **{closest['closest_margin']:.3g}** against a rounding bound of "
          f"**{closest['the_bound_there']:.3g}**, and")
    print(f"  **{closest['decided_by_nothing_at_all']}** are exact ties. Settled, those questions "
          f"land the same way on every order;")
    print(f"  scoring them either way is worth {inside / asked:.1%} of the figure.")
    print()
    tries = len(HASH_SEEDS)
    print(f"  Re-run under {tries} other hash seeds the read agrees **{seeds['the read']}** times "
          f"and the shipped coordinate")
    print(f"  comes back bit for bit **{seeds['the shipped coordinate']}** times out of {tries}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
