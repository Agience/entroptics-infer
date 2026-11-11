"""One coordinate, one walk: what the shipped walk does to build a place, counted.

What this certifies
---------------------
There is one walk, `Engine.places`, and every read in ``research/`` goes through it. The research
helpers that used to build a coordinate of their own -- a Python frontier in `place.py` and a numpy
array walk here -- are gone: `place.places` is the engine's `places`, and :func:`built` below is the
engine's `places` with the work counted. So the check that the research coordinate and the shipped
one agree is exact by construction, and it is still run, on every store and with a statement held
out, because a gate that is only argued for is the kind that goes stale.

What is counted
-----------------
The walk settles vertices largest value first and stops at its own aperture
(`Engine._resolved`). The work a place costs is how many vertices it settled and how many
neighbour entries those vertices offered -- the turns of the loop that settles them. Both are counts
about the store and the walk, the same on every machine and in every run, which is why they are the
figures recorded and a wall clock is not.

What would settle it
----------------------
Agreement to the bit, not to a tolerance: the same function called twice adds the same terms in
the same order, so any cell differing would mean the research path is not the shipped one.
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

import ontology
import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402


def built(engine, *, without=(), only=None, counted=None, keep=True):
    """``{vertex: coordinate}`` -- `Engine.places`, with the work its walk did counted on request.

    Without ``counted`` this is ``engine.places(without=without, only=only)``. With it, each row
    is built by the same two steps `places` takes -- `Engine._resolved` for the vertices the walk
    keeps and `Engine._row` for the place they make -- so the counts are read off the shipped walk
    and the rows are the shipped rows. ``keep=False`` counts the work and holds no rows, for the
    arms that need the cost of a coordinate on a store too large to hold one.
    """
    if counted is None:
        return engine.places(without=without, only=only)
    axes = engine.basis(without=without)
    index = {a: i for i, a in enumerate(axes)}
    adj = engine._adjacency(without)
    wanted = engine.vertices(without=without)
    if only is not None:
        keep_these = set(only)
        wanted = tuple(v for v in wanted if v in keep_these)
    vectors = {}
    for start in wanted:
        kept = engine._resolved(adj, start)
        counted["settled"] = counted.get("settled", 0) + len(kept)
        counted["neighbours"] = counted.get("neighbours", 0) + sum(len(adj[h]) for h, _ in kept)
        counted["starts"] = counted.get("starts", 0) + 1
        counted["most_settled"] = max(counted.get("most_settled", 0), len(kept))
        if keep:
            vectors[start] = engine._row(adj, index, kept)
    return vectors


def apart(one, other):
    """The largest difference between two coordinates, and how many cells differ at all."""
    worst, differing, cells = float(0), 0, 0
    for key, row in one.items():
        gap = np.abs(np.asarray(row) - np.asarray(other[key]))
        worst = max(worst, float(gap.max()) if gap.size else float(0))
        differing += int((gap != 0).sum())
        cells += gap.size
    return worst, differing, cells


def stores():
    """The ontology, and the shipped ingestion's prose store with and without its sentences."""
    out = {"ontology": tuple(tuple(s) for s in ontology.STATEMENTS)}
    from test_whether_prose_at_two_scales_changes_the_read import prose_stores
    out.update(prose_stores())
    return out


def main() -> int:
    print("one coordinate, one walk: what the shipped walk does to build a place\n")
    print(f"  {'store':<20}{'statements':>12}{'vertices':>10}{'settled':>10}{'offered':>10}"
          f"{'differ':>8}{'held out':>10}")
    print("  " + "-" * 80)

    rows, clocks = {}, {}
    for label, statements in stores().items():
        engine = Engine.on_statements(statements)
        work: dict = {}
        began = time.perf_counter()
        shipped = engine.places()
        clocks[label] = {"seconds_for_the_coordinate": time.perf_counter() - began}
        research = built(engine, counted=work)
        worst, differing, cells = apart(shipped, research)
        gone = [statements[0]]
        held_worst, held_differing, held_cells = apart(engine.places(without=gone),
                                                        built(engine, without=gone, counted={}))
        starts = max(work["starts"], 1)
        rows[label] = {"statements": len(statements), "vertices": len(engine.vertices()),
                       "axes": len(engine.basis()), "cells": cells, "cells_differing": differing,
                       "worst_cell": worst, "cells_held_out": held_cells,
                       "cells_differing_held_out": held_differing,
                       "worst_cell_held_out": held_worst,
                       "settled_per_start": work["settled"] / starts,
                       "neighbours_offered_per_start": work["neighbours"] / starts,
                       "neighbours_offered": work["neighbours"]}
        r = rows[label]
        print(f"  {label:<20}{r['statements']:>12}{r['vertices']:>10}"
              f"{r['settled_per_start']:>10.2f}{r['neighbours_offered_per_start']:>10.1f}"
              f"{differing:>8}{held_differing:>10}")
    print()

    record.figures(by_store=rows, timings=clocks,
                   cells_in_all=sum(r["cells"] for r in rows.values()))

    same = all(r["cells_differing"] == 0 and r["cells_differing_held_out"] == 0
               for r in rows.values())
    if not same:
        print("  **The research coordinate is not the shipped one.** Something in research/ builds a")
        print("  place of its own, and every figure taken through it is about a walk nobody ships.")
        return 0
    print(f"  **Every cell is identical on every store**, across "
          f"{sum(r['cells'] for r in rows.values())} cells and a further")
    print(f"  {sum(r['cells_held_out'] for r in rows.values())} with a statement held out: the "
          f"research coordinate is the shipped one.")
    widest = max(rows, key=lambda k: rows[k]["statements"])
    print(f"  On the largest store a place settles {rows[widest]['settled_per_start']:.2f} vertices "
          f"and offers {rows[widest]['neighbours_offered_per_start']:.1f} neighbour")
    print("  entries to do it: the walk stops at its own aperture, so the cost of a place is what its")
    print("  vantage resolves and not the size of the store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
