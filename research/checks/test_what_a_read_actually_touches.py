"""What a read reads, and what it costs to read only that.

The claim being tested
------------------------
A read carries one source through one relation and scores a set of candidates. The rows of the
coordinate it consults are therefore: the source, the rows on both sides of every statement
that uses the relation — the shared axis the operator is a cross-covariance over — and the
candidates. Nothing else is looked at.

Every figure in this program builds the whole coordinate anyway. On the two-scale prose store that
is 5,882 rows to answer a question that consults a few hundred, and the rest are built and thrown
away. If a read costs what it touches rather than what the store holds, then the cost of answering
is set by the question and not by how much has been written down — which is the property an in-line
engine needs and the one this file measures.

Why this is a restriction and not an approximation
----------------------------------------------------
:meth:`Engine.places` walks each start independently: the loop carries no state between iterations,
and each row is a function of the basis and the adjacency alone. So the row for `dog` is the same
whether it was computed beside 5,881 others or by itself. That is an argument, and arguments have
been wrong here before, so the first table checks it cell for cell on every store — the whole
coordinate against the named rows — and the timing is worth reading only after that passes.

What is measured
------------------
``touched``     how many rows a single held-out read actually consults, against how many the store
                has.
``exact``       the named rows against the same rows of the full build, at every cell.
``cost``        the seconds for each, and — because a clock moves with the box — the work: one
                count per entry of an expanded vertex's neighbour list, which is deterministic.

The question this answers about scale
---------------------------------------
``research/checks/test_what_a_read_costs_as_the_store_grows.py`` measures the read against a growing store
and finds an exponent near two. That measurement builds every row. The candidate set is what a
read scores against, and it does not have to grow with the store: a relation reaches what it
reaches. So the same sweep with ``only`` asks a different question — whether the cost follows the
store or follows the question — and the two exponents side by side is the answer.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import sys
import time

import numpy as np

import ask
import ontology
import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_same_coordinate_can_be_built_faster import built  # noqa: E402

#: the whole of a vertex before any branching divides it
WHOLE = float(1)


def rows_a_read_needs(engine, source, relation, candidates, *, without=()):
    """The vertices a single read consults, and nothing else.

    The source, both sides of every statement using the relation — which is the ordered axis the
    operator is a cross-covariance over, so both ends of each are read — and the candidates.
    """
    needed = {source}
    needed.update(candidates)
    for parts, position in engine.uses(relation, without=without):
        needed.add(parts[position - 1])
        needed.add(parts[position + 1])
    return needed


def stores():
    """The stores this has to hold on: the published one, and the ones that are large."""
    from test_whether_prose_at_two_scales_changes_the_read import prose_stores
    out = {"ontology": tuple(tuple(s) for s in ontology.STATEMENTS)}
    out.update(prose_stores())
    return out


def one_question(statements):
    """A question this store can be asked: the first statement whose relation is used twice."""
    counts = {}
    for _, relation, _ in statements:
        counts[relation] = counts.get(relation, 0) + 1
    for src, relation, dst in statements:
        if counts[relation] > 1:
            return src, relation, dst
    return None


def main() -> int:
    print("what a read reads, and what it costs to read only that\n")
    print(f"  {'store':<20}{'vertices':>10}{'touched':>9}{'share':>8}{'cells':>10}{'differ':>8}"
          f"{'all rows':>10}{'only':>9}{'+array':>10}{'times':>7}")
    print("  " + "-" * 101)

    rows, clocks = {}, {}
    for label, statements in stores().items():
        engine = Engine.on_statements(statements)
        asked = one_question(statements)
        if asked is None:
            continue
        src, relation, dst = asked
        pair = [s for s in statements if s[0] == src and s[2] == dst]
        concepts = engine.concepts(without=pair)
        needed = rows_a_read_needs(engine, src, relation, concepts, without=pair)

        began = time.perf_counter()
        whole = engine.places(without=pair)
        slow = time.perf_counter() - began

        began = time.perf_counter()
        part = engine.places(without=pair, only=needed)
        quick = time.perf_counter() - began

        # **Both savings at once.** The array build does the same walk with its two inner loops done
        # by numpy, and `only` asks it for the rows the question needs. They are independent -- one
        # changes how a row is computed, the other which rows are -- so the pair is the read's actual
        # cost, and it is checked against the whole shipped build like everything else here.
        all_work = {"expansions": 0, "neighbours": 0, "reached": 0, "starts": 0}
        only_work = {"expansions": 0, "neighbours": 0, "reached": 0, "starts": 0}
        built(engine, without=pair, counted=all_work)
        began = time.perf_counter()
        both = built(engine, without=pair, only=needed, counted=only_work)
        quickest = time.perf_counter() - began

        here = [v for v in needed if v in whole]
        worst = float(0)
        differing = cells = 0
        for v in here:
            for got in (part[v], both[v]):
                gap = np.abs(whole[v] - got)
                worst = max(worst, float(gap.max()) if gap.size else float(0))
                differing += int((gap != 0).sum())
                cells += gap.size

        rows[label] = {"statements": len(statements), "vertices": len(whole),
                       "touched": len(here), "share": len(here) / max(len(whole), 1),
                       "cells": cells, "cells_differing": differing, "worst_cell": worst,
                       "candidates": len(concepts),
                       # **The deterministic measure of the saving.** A wall clock moves by nearly
                       # a factor of two between runs of this file on one box, so a claim resting on
                       # it could not be checked against its own record. The step counts are the
                       # same on every machine and in every run, so they are what it rests on.
                       "steps_all_rows": all_work["neighbours"],
                       "steps_only": only_work["neighbours"],
                       "steps_saved": (float(1) - only_work["neighbours"]
                                       / max(all_work["neighbours"], 1)),
                       "times_less_work": (all_work["neighbours"]
                                           / max(only_work["neighbours"], 1))}
        clocks[label] = {"all_rows_seconds": slow, "only_seconds": quick,
                         "only_and_array_seconds": quickest,
                         "times_faster": slow / quick if quick else float(0),
                         "times_faster_both": slow / quickest if quickest else float(0)}
        r = dict(rows[label], **clocks[label])
        print(f"  {label:<20}{len(whole):>10}{len(here):>9}{r['share']:>8.1%}{cells:>10}"
              f"{differing:>8}{slow:>9.2f}s{quick:>8.2f}s{quickest:>9.2f}s"
              f"{r['times_faster_both']:>7.1f}")
    print()

    if not rows:
        print("  No store could be asked. Nothing to report.")
        return 0

    # **The two prose stores are the same corpus at one scale and at two**, so the pair says what
    # happens to a read's footprint when the store grows and nothing else about it changes. Both
    # sides recorded, because the claim is the ratio between them and a ratio computed in prose is a
    # figure with no producer.
    growth = {}
    if "prose spans" in rows and "prose, two scales" in rows:
        small, large = rows["prose spans"], rows["prose, two scales"]
        growth = {"vertices_grew": large["vertices"] / small["vertices"] - float(1),
                  "rows_a_read_needs_grew": large["touched"] / small["touched"] - float(1)}
    record.figures(by_store=rows, timings=clocks, between_the_prose_stores=growth)

    same = all(r["cells_differing"] == 0 for r in rows.values())
    if not same:
        worst = max(rows.values(), key=lambda r: r["cells_differing"])
        print(f"  **The named rows are not the same rows.** {worst['cells_differing']} of "
              f"{worst['cells']} cells differ, worst by")
        print(f"  {worst['worst_cell']:.3g}. The loop is carrying state between starts, which the "
              f"claim above says it does not.")
        return 0

    print(f"  **The named rows are identical to the same rows of the whole build** -- "
          f"{sum(r['cells'] for r in rows.values())} cells,")
    print(f"  none differing. Each start's walk reads the basis and the adjacency and nothing "
          f"another")
    print(f"  start produced, so a row is the same row whether it was built alone or beside every "
          f"other.")
    print()

    biggest = max(rows, key=lambda k: rows[k]["statements"])
    b = dict(rows[biggest], **clocks[biggest])
    print(f"  **A read on {biggest} touches {b['touched']} of {b['vertices']} rows -- "
          f"{b['share']:.1%}.** The other")
    print(f"  {b['vertices'] - b['touched']} are built and discarded. **In walking rather than in "
          f"seconds, asking for")
    print(f"  only those rows does {b['steps_all_rows']} steps against {b['steps_only']} -- "
          f"**{b['times_less_work']:.1f}** times less**, close to the")
    print(f"  {float(1) / b['share']:.1f} times fewer rows, because the work per row is much the "
          f"same whichever rows are asked for.")
    print()
    print(f"  **That is one of the two savings, and it is the one in walking.** The other is in how")
    print(f"  each step is taken: the array build turns a Python step per neighbour into one array")
    print(f"  operation per expanded vertex, which the faster-coordinate test measures at 113.4")
    print(f"  times fewer operations on this store. The two are in different units and do not")
    print(f"  multiply. What they do together is")
    print(f"  what the clock reads, **{b['times_faster_both']:.1f} times on this run**, "
          f"{b['all_rows_seconds']:.2f} s to {b['only_and_array_seconds']:.2f} s. That figure moves")
    print(f"  by nearly a factor of two between runs of this file on one machine, which is why the")
    print(f"  two counts above are what is recorded and it is not. On the smallest store the array")
    print(f"  build costs more than it saves")
    print(f"  -- the saving is in the hubs, and so is this method's use.")
    print()
    if growth:
        print(f"  **The same corpus at one scale and at two** says what happens when the store "
              f"grows and")
        print(f"  nothing else changes: the vertices grow "
              f"**{growth['vertices_grew']:.0%}** and the rows a read needs grow "
              f"**{growth['rows_a_read_needs_grew']:.0%}**.")
        print()
    print(f"  What this buys is the shape of the cost. **The rows a read needs are set by the "
          f"question**")
    print(f"  -- one source, the sides of the statements using one relation, and the candidates -- "
          f"and")
    print(f"  a relation reaches what it reaches however much else is written down. So the store "
          f"may")
    print(f"  grow without the read's row count following it, which is the property an in-line "
          f"engine")
    print(f"  needs. **Whether it does follow is the next measurement**, and it is the growth sweep")
    print(f"  run again with `only` against the same candidate set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
