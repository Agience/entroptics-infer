"""The read put to a lattice somebody actually wrote, with its own recorded relations.

Why this store and not another
--------------------------------
This program's corpus criterion is **an asymmetric relation that is recorded rather than inferred**.
Four task runs (CLAIMS 7.2, 7.3, 7.11) and a fifth here
(`test_whether_the_read_predicts_the_next_token.py`) put the read on text whose relation is
*manufactured* -- from co-occurrence counts, or from three tokens being adjacent -- and counting beat
it every time. Adjacency records order; it does not record a relation.

A mantle lattice records relations. An edge is there because something wrote it, with a label naming
what it meant, and both ends are artifacts with identity and history. **That is the criterion met by
the substrate rather than by a dataset chosen to meet it**, and it is the store this whole program
is for.

The lattice is named by ``ENTROPTICS_LATTICE`` in the environment, and nothing from it enters this
repository except the counts and rates below -- a store of real content is measured, not published.

**It is read through mantle's own API**, never by opening the file and selecting from it. A lattice
is a sqlite file and it is always tempting to read it as one; doing that reads a schema this package
does not own, past whatever the store does about ordering, partial results and its own counters.
:class:`entroptics_infer.store.Lattice` is the seam, and ``labels()``, ``statements()`` and
``kinds()`` are mantle's ``labels_with_edges``, ``src_ids_by_label`` / ``edges_of`` and ``get_many``.

**Point this at a copy.** A store whose label counters were never backfilled reports no labels at
all -- which is it saying it does not know, not that there are none -- and settling that is a write.
``measure_labels()`` does it and says so in its name.

What is read out of it, and what is left behind
-------------------------------------------------
``force = grant``     left out. ``created`` is provenance -- who wrote a thing -- and every vertex
                      has one, so it is a property of the vertex and not a relation between two
                      pieces of content.
``the rest``          kept, as ``(src, label, dst)``. The label is what the writer recorded.

**Two ways the store can state a held-out fact twice, and both are handled.**

*Reciprocal labels are dropped, and the share is reported.* Where a label holds both ``a -> b`` and
``b -> a``, holding one out leaves the other, and a read that found the inverse would score as
inference. Which labels go is decided by **measuring** the reciprocity, not by knowing what the
words mean, and the share for every label is printed.

*And the hold-out removes every statement joining the two vertices, not just the one being asked
about.* A store may record the same pair under more than one label, and then holding out only the
asked-about one leaves the other saying exactly where to go -- the read would need to notice only
that the two vertices are related at all. That is the same failure as an inverse pair in a shape no
label name gives away, and every run of this file prints how many pairs carry more than one label so
the size of it is visible before the result is.

Then the **2-core**, as everywhere: a vertex named once cannot be held out, so it cannot be asked
about, and leaving it in only inflates the pool.

What is measured
------------------
The same three arms as `test_whether_the_read_survives_its_controls.py`: the held-out read, the same
read with the statement left in (a lookup, and a ceiling), and the read on **permuted** stores, where
the correspondence is gone and everything else is kept. **The permuted arm is the one that can refute
the result**, and the claim is compared to its own draws rather than to another store's.

The unit permuted is the **bundle** -- the set of labels joining one pair -- and not the individual
statement. Shuffling inside each label independently would dissolve the doubling described above,
and a store with fewer pairs under two labels than the real one is a different graph. A null that
differs from the claim by more than the correspondence is not a null for it.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import os
import random

from concurrent import futures

import numpy as np

import record
from heldout import core

import ask

from entroptics_infer.store import open_lattice
from entroptics_infer.wrapper import Engine

#: named in the environment, because a path to real content does not belong in a repository
LATTICE = os.environ.get("ENTROPTICS_LATTICE")
#: Left out by **label**, because that is what the store indexes by. ``created`` is provenance --
#: who wrote a thing -- and every vertex has one, so it is a property of the vertex rather than a
#: relation between two pieces of content.
LEFT_OUT = ("created",)
#: this reader's own name in the store's proper time; mantle refuses a generated one
ORIGIN = "entroptics-infer-reader"
#: how much of the core to read, **in vertices**. A held-out protocol rebuilds the coordinate for
#: every question, so this is what fits rather than what exists, and the whole core is reported
#: beside it so the part read can be seen for the part it is.
MOST = 360
#: How many times the destinations are permuted. **This sets the resolution of the whole test**:
#: with ``d`` draws the smallest obtainable share is ``1 / (d + 1)``, and nothing finer can be said
#: however the numbers fall. At 8 that floor is 0.111; at 20 it is 0.048, which is what the
#: companion controls on WordNet run at, so the two are read at the same resolution.
DRAWS = 20
#: How many arms run at once. The arms are independent measurements, so running them serially is the
#: difference between minutes and an hour and changes no number -- but taking every core is not free
#: either: this box is shared, and a run that saturates it is a run that stops somebody else's.
#: A third of the machine gets most of the speed-up and leaves the box usable. Override with
#: ``ENTROPTICS_WORKERS`` where the machine is yours alone.
WORKERS = int(os.environ.get("ENTROPTICS_WORKERS")
              or max(2, (os.cpu_count() or 4) // 3))


def _arm(what):
    """One arm, in its own process. Top level because a pool cannot pickle a closure."""
    statements, hold_out = what
    return measure(statements, hold_out=hold_out)


def statements_from(path):
    """``(statements, how it was filtered)`` -- through mantle's API, counts out, no content."""
    with open_lattice(path, origin=ORIGIN) as lattice:
        if not lattice.edge_labels_measured():
            lattice.measure_labels()
        labels = [l for l in lattice.labels() if l not in LEFT_OUT]
        rows = [st for l in labels for st in lattice.statements(label=l)]

    by_label = collections.Counter(label for _, label, _ in rows)
    pairs = {(s, l, d) for s, l, d in rows}
    # a label that holds both directions would leave every held-out fact stated backwards
    reciprocity = {}
    for label in by_label:
        mine = [(s, d) for s, l, d in pairs if l == label]
        back = sum(1 for s, d in mine if (d, label, s) in pairs)
        reciprocity[label] = back / len(mine) if mine else 0.0
    kept_labels = [l for l, share in reciprocity.items() if share == 0.0]

    statements = sorted({(s, l, d) for s, l, d in pairs if l in kept_labels and s != d})
    return statements, {"edges_read": len(rows), "labels": dict(by_label),
                        "reciprocity": reciprocity, "labels_kept": sorted(kept_labels),
                        "labels_dropped": sorted(set(by_label) - set(kept_labels))}


def a_connected_part(statements, most):
    """A connected region, walked over **vertices** and keeping every statement inside it.

    Not the whole store: the held-out protocol rebuilds the coordinate for every question, so what is
    read is what fits. Walked rather than sampled, for the reason `lexicon.py` states in full -- a
    random subset of a large store is almost entirely disjoint and a failure on it would say nothing
    about the method.

    **Walked over vertices rather than over statements**, which is not a detail. Taking statements in
    breadth-first order leaves a fringe of vertices named once, and those are exactly what the 2-core
    then strips, until the store left over cannot be asked a single question. Collecting vertices and then
    keeping every statement whose two ends are both inside gives a region that survives its own
    coring.

    ``most`` is a budget in **vertices**. The walk starts from the busiest one, which is a property
    of the store rather than a choice about the answer.
    """
    vertices = {c for a, _, b in statements for c in (a, b)}
    if len(vertices) <= most:
        return tuple(statements)
    touching = collections.defaultdict(list)
    degree = collections.Counter()
    for st in statements:
        a, _, b = st
        touching[a].append(st)
        touching[b].append(st)
        degree[a] += 1
        degree[b] += 1
    start = max(degree, key=lambda v: (degree[v], v))
    seen, queue = {start}, collections.deque([start])
    while queue and len(seen) < most:
        here = queue.popleft()
        for st in touching[here]:
            for other in (st[0], st[2]):
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
                    if len(seen) >= most:
                        break
            if len(seen) >= most:
                break
    return tuple(sorted(st for st in statements if st[0] in seen and st[2] in seen))


def pairs_stated_twice(statements):
    """How much of a store records the same two vertices under more than one label.

    **Printed by every run, because it is the number that says how much a hold-out has to remove.**
    Where a pair carries two labels, holding out one of them leaves the other stating the same
    destination, and the read needs only to notice that the two vertices are related at all. The
    protocol here removes the whole pair, so this is not a fault -- it is the size of the thing the
    protocol is dealing with, and a store where it is large is a store whose figures move a lot when
    it is handled.
    """
    under = collections.defaultdict(set)
    for a, label, b in statements:
        under[(a, b)].add(label)
    twice = {k: v for k, v in under.items() if len(v) > 1}
    return {"pairs": len(under), "pairs_under_two_labels": len(twice),
            "statements_involved": sum(len(v) for v in twice.values()),
            "share_of_statements": (sum(len(v) for v in twice.values()) / len(statements)
                                    if statements else 0.0),
            "combinations": {"+".join(sorted(v)): n for v, n in
                             collections.Counter(tuple(sorted(v))
                                                 for v in twice.values()).items()}}


def permute(statements, rng, *, repairs=200):
    """The same store, every degree and **every pair's label set** kept, the correspondence gone.

    The obvious permutation shuffles destinations inside each label independently. On a store where
    some pairs are recorded under two labels that dissolves the doubling: a real region holds many
    of its pairs under two labels, and an independently-shuffled copy of it almost none. **That is
    not a null for this store** — it
    has a different number of distinct edges, so it is a different graph, and a claim compared
    against it is being compared against something that differs by more than the correspondence.

    So the unit permuted here is the **bundle**: the set of labels joining one pair. Bundles are
    grouped by that set, and destinations are shuffled only among bundles carrying the same set, so
    every label keeps its count, every pair keeps how many labels it carries, and the whole doubling
    structure survives intact. What is gone is which source reaches which destination, which is the
    thing the read is claimed to recover and the only thing that should be.
    """
    bundles = collections.defaultdict(list)
    for a, label, b in statements:
        bundles[(a, b)].append(label)
    by_shape = collections.defaultdict(list)
    for (a, b), labels in bundles.items():
        by_shape[tuple(sorted(labels))].append((a, b))

    out = []
    for shape, pairs in by_shape.items():
        srcs = [a for a, _ in pairs]
        dsts = [b for _, b in pairs]
        rng.shuffle(dsts)
        for _ in range(repairs):
            seen = collections.Counter(zip(srcs, dsts))
            bad = [i for i in range(len(srcs))
                   if srcs[i] == dsts[i] or seen[(srcs[i], dsts[i])] > 1]
            if not bad:
                break
            i, j = rng.choice(bad), rng.randrange(len(dsts))
            dsts[i], dsts[j] = dsts[j], dsts[i]
        for a, b in zip(srcs, dsts):
            for label in shape:
                out.append((a, label, b))
    return tuple(sorted(set(out)))


def measure(statements, *, hold_out=True):
    """The held-out read over a whole store."""
    engine = Engine.on_statements(statements)
    concepts = engine.concepts()
    right = asked = 0
    draw_sum = 0.0
    pools, among = [], []
    for src, label, dst in statements:
        # **Every statement joining these two vertices goes, not just this one.** A store may
        # record the same pair under more than one label, and leaving the other one behind states
        # the answer in the material the read is taken from. The reciprocity check above catches
        # a label that holds both directions; this catches two labels that hold the same
        # direction, which is the same failure wearing a different hat.
        drop = [s for s in statements if s[0] == src and s[2] == dst] if hold_out else []
        held = [s for s in statements if s != (src, label, dst)]
        vectors = engine.places(without=drop)
        if src not in vectors or dst not in vectors:
            continue
        landed = engine.carry(src, label, vectors, without=drop)
        if landed is None:
            continue
        stated = {b for a, r, b in held if a == src and r == label}
        pool = [c for c in concepts if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        into = {b for a, r, b in held if r == label}
        reached = [c for c in pool if c in into]

        asked += 1
        pools.append(len(pool))
        draw_sum += 1.0 / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        right += pool[int(np.argmax(scores))] == dst
        if dst in reached and len(reached) > 1:
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            among.append((reached[int(np.argmax(sc))] == dst, 1.0 / len(reached)))

    if not asked:
        return None
    acc, draw = right / asked, draw_sum / asked
    got = float(np.mean([a for a, _ in among])) if among else float("nan")
    got_draw = float(np.mean([d for _, d in among])) if among else float("nan")
    return {"statements": len(statements), "questions": asked,
            "pool": float(np.mean(pools)), "a_draw_would": draw, "right": acc,
            "times_the_draw": acc / draw if draw else float("nan"),
            "questions_among_the_reached": len(among), "among_the_reached": got,
            "a_draw_among_the_reached": got_draw,
            "times_the_draw_among_the_reached": got / got_draw if got_draw else float("nan")}


def _line(label, v):
    if v is None:
        return f"  {label:>22}{'--':>10}"
    return (f"  {label:>22}{v['statements']:>7}{v['questions']:>11}{v['pool']:>7.0f}"
            f"{v['a_draw_would']:>7.1%}{v['right']:>8.1%}{v['times_the_draw']:>8.1f}"
            f"{v['questions_among_the_reached']:>9}{v['a_draw_among_the_reached']:>12.1%}"
            f"{v['among_the_reached']:>7.1%}{v['times_the_draw_among_the_reached']:>8.1f}")


def main() -> int:
    if not LATTICE or not os.path.exists(LATTICE):
        print("  Skipped: this check reads a private mantle lattice, named by ENTROPTICS_LATTICE, and")
        print("  none is set (or the file is absent). Nothing was measured.")
        return 0

    raw, how = statements_from(LATTICE)
    # core first, then take a connected part of it: coring a slice of the store throws away the
    # very vertices that would have made the slice askable
    whole = core(raw)
    store = core(a_connected_part(whole, MOST))
    if len(store) < 10:
        print(f"  The core of this lattice is {len(store)} statements; there is nothing to ask.")
        return 0

    print("the read on a lattice somebody wrote\n")
    print(f"  {how['edges_read']} edges read, {len(how['labels'])} labels")
    for label, share in sorted(how["reciprocity"].items(), key=lambda kv: -kv[1]):
        mark = "dropped, holds both directions" if share > 0.0 else "kept"
        print(f"    {label:>22}{how['labels'][label]:>7}   reciprocal {share:>6.1%}   {mark}")
    twice = pairs_stated_twice(store)
    print(f"  {len(raw)} statements after that, whole core {len(whole)}, read {len(store)}, "
          f"{len({c for a, _, b in store for c in (a, b)})} vertices")
    print(f"  the part read carries: "
          f"{dict(collections.Counter(l for _, l, _ in store).most_common())}")
    print(f"  **{twice['pairs_under_two_labels']} of {twice['pairs']} pairs are recorded under more "
          f"than one label** -- {twice['share_of_statements']:.0%} of the")
    print(f"  statements read: {twice['combinations'] or 'none'}. The hold-out removes every")
    print("  statement joining the two vertices, so a second label cannot state the answer.")
    print()

    print(f"  {'arm':>22}{'stmts':>7}{'questions':>11}{'pool':>7}{'draw':>7}{'right':>8}"
          f"{'x draw':>8}{'reached':>9}{'draw there':>12}{'named':>7}{'x draw':>8}")
    print("  " + "-" * 104)

    # **The arms are run at once, because they have nothing to do with each other.** Twenty
    # permuted stores and the two real arms are twenty-two independent measurements, and running
    # them one after another on one core is the difference between minutes and an hour: every arm
    # asks one question per statement, and every question rebuilds the coordinate through
    # `Engine.places`, however they are ordered. Nothing about the result changes -- each arm sees
    # the same store and the same seed it would have seen.
    arms = ([("held out (the claim)", store, True), ("stated (a lookup)", store, False)]
            + [(f"shuffled, draw {d}", permute(store, random.Random(d)), True)
               for d in range(DRAWS)])
    with futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        done = list(pool.map(_arm, [(s, h) for _, s, h in arms]))

    held, stated = done[0], done[1]
    shuffled = [v for v in done[2:] if v is not None]
    for (label, _, _), v in zip(arms, done):
        print(_line(label, v))
    print()

    counts = collections.Counter(l for _, l, _ in store)
    biggest = counts.most_common(1)[0] if counts else (None, 0)
    record.figures(left_out=list(LEFT_OUT), most=MOST, draws=DRAWS,
                   how=how, pairs_stated_twice=twice, core=len(store), whole_core=len(whole),
                   vertices=len({c for a, _, b in store for c in (a, b)}),
                   labels_read=dict(counts),
                   biggest_label=biggest[0], biggest_label_share=biggest[1] / len(store),
                   held_out=held, stated=stated, shuffled=shuffled)

    if held is None:
        print("  The held-out arm produced no question. Nothing is established.")
        return 0

    got = held["times_the_draw_among_the_reached"]
    mine = [v["times_the_draw_among_the_reached"] for v in shuffled
            if v["times_the_draw_among_the_reached"] == v["times_the_draw_among_the_reached"]]
    floor = 1.0 / (DRAWS + 1)
    over = sum(1 for m in mine if m >= got) if mine else 0
    share = (over + 1) / (len(mine) + 1) if mine else float("nan")

    print(f"  The read names the held-out destination on **{held['right']:.1%}** of "
          f"{held['questions']} questions")
    print(f"  from a pool of {held['pool']:.0f}, where a draw gives {held['a_draw_would']:.1%} -- "
          f"**{held['times_the_draw']:.1f} times** -- and among the")
    print(f"  {held['questions_among_the_reached']} whose answer the label points into elsewhere it "
          f"names it **{got:.1f} times** a draw among those.")
    if stated:
        print(f"  Left in, the same question is a lookup and scores {stated['right']:.1%}, so the "
              f"protocol is worth {stated['right'] / held['right']:.1f} times.")
    print()
    if mine:
        print(f"  Against its own null: {over} of {len(mine)} permuted stores reached "
              f"{got:.1f}, share {share:.3f},")
        print(f"  at a floor of {floor:.3f}.")
        print()
        if share <= floor:
            print("  **The read clears its own null on a store nobody built for it.** The relations")
            print("  here were recorded by whoever wrote the edges, the vertices are artifacts with")
            print("  identity and history, and the statement asked about is in neither the operator")
            print("  nor the coordinate. **This is the corpus criterion met by the substrate.**")
        else:
            print("  **The read does not clear its own null here.** A store with the correspondence")
            print("  removed reaches the same figure, so what is being read is the shape and not")
            print("  which source goes to which destination -- on this lattice, at this size.")
    print()
    record.figures(left_out=list(LEFT_OUT), most=MOST, draws=DRAWS,
                   how=how, pairs_stated_twice=twice, core=len(store), whole_core=len(whole),
                   vertices=len({c for a, _, b in store for c in (a, b)}),
                   labels_read=dict(counts),
                   biggest_label=biggest[0], biggest_label_share=biggest[1] / len(store),
                   held_out=held, stated=stated, shuffled=shuffled,
                   against_its_own_null={"claim": got, "draws": len(mine),
                                         "null_worst": max(mine) if mine else float("nan"),
                                         "reached_or_beat": over, "share": share, "floor": floor})
    print("  The lattice is read through mantle's API, on a copy; only counts and rates")
    print("  leave it. Reciprocal labels are dropped by measurement rather than by knowing what the")
    print("  words mean, the 2-core is taken because a vertex named once cannot be held out, and")
    print("  the draw is computed over the pools the reader saw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
