"""Stage 17 — the relation count, with the width it was measured at swept.

What this audits
----------------
`exp03` reports that the number of relation kinds is measured and unbounded — no vocabulary, no
enumeration, no cap — and recovers 32 kinds exactly at 32 used. It runs at an ambient width of 64
and sweeps the kind count and the noise.

"Unbounded" is the word to look at. A relation in this construction owns two orthonormal
directions, a source role and a target role, orthonormalized across the whole set. So `R` relations
need `2R` independent directions and the arithmetic ceiling is `R = D/2`. At `D = 64` that is 32 —
exactly the largest `R` the experiment runs. The stage therefore runs precisely up to its own
ceiling and stops, which is a different statement from being unbounded, and the difference is
invisible at one width.

What is measured
----------------
`exp03`'s part A construction — disjoint vertices, so nothing is confounded by endpoint conflation —
with the ambient width swept, and `R` taken up to each width's own ceiling.

The question is not "is the count unbounded" but the sharper one that can actually be answered:
is the ceiling the basis, or is it something else? If recovery is exact right up to `D/2` at
every width, then the count is limited by how many independent role directions the space holds and
by nothing else — no vocabulary, no cap, and no hidden saturation. That is a claim about a bound
rather than the absence of one, and it is the claim the construction supports.

The ceiling is also checked from the other side: the generator refuses to build `R > D/2` rather
than silently producing fewer distinct relations, and that refusal is exercised here rather than
taken on trust.

What would falsify what
------------------------
Stated before the run. If recovery fails well below `D/2` at some width, the count saturates for a
reason the basis does not explain and "unbounded" is wrong in a way that matters. If recovery holds
*past* `D/2`, the roles were not independent and the corpus was not carrying the kinds it claimed.
If the generator does not refuse past the ceiling, a silently-degenerate corpus could be measured as
though it were sound.

Run
---
    python research/experiments/exp23_audit_relation_count.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import relation_corpus                                  # noqa: E402
from entroptics_infer import relations                              # noqa: E402

#: `exp03`'s design, unchanged.
T = 96
EDGES_PER_RELATION = 12
N_VERTICES = 40
FAR = 0.05
BASE_SEED = 20260915
DRAWS = 6

#: What `exp03` holds, swept here. 64 is its own value.
WIDTHS = (16, 32, 64, 128)
NOISES = (0.05, 0.10, 0.20)

#: A cell's working set, measured rather than guessed: one cell at 128 channels peaks near 7 GB, and
#: the cost grows with the cube of the width, so 64 channels is near 0.85 GB. These size the worker
#: pools and nothing else -- no result depends on them.
GB_PER_CELL_AT_64 = 7.0 / 8.0
MEMORY_BUDGET_GB = 16.0


def ceiling(width):
    """How many relations the basis can hold: two orthonormal role directions each."""
    return width // 2


def kinds_for(width):
    """Kind counts to run at this width, up to and including its ceiling."""
    top = ceiling(width)
    wanted = [r for r in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if r <= top]
    if top not in wanted:
        wanted.append(top)
    return sorted(set(wanted))


def cell(args):
    width, noise, R = args
    counts = []
    for d in range(DRAWS):
        c = relation_corpus(seed=BASE_SEED + d, n_relations=R,
                            edges_per_relation=EDGES_PER_RELATION, n_vertices=N_VERTICES,
                            T=T, D=width, noise=noise, share_vertices=False)
        counts.append(relations.relation_modes(c.records, c.edges, far=FAR))
    return {"width": width, "noise": noise, "R": R,
            "counted": float(np.mean(counts)),
            "exact_rate": float(np.mean([k == R for k in counts])),
            "at_ceiling": R == ceiling(width)}


def run():
    started = time.time()
    # A cell's cost is dominated by the relation coordinates, which are one `D x D` matrix per edge
    # and `12 R` edges with `R` up to `D/2` — so it grows with the cube of the width. Measured: a
    # single cell at 128 channels peaks near 7 GB. Widths are therefore run in separate pools with a
    # worker count scaled to fit, rather than one pool sized for the cheapest cell.
    cores = (os.cpu_count() or 2) - 2
    rows = []
    total = sum(len(kinds_for(w)) * len(NOISES) for w in WIDTHS)
    print(f"{total} cells, {DRAWS} draws each", flush=True)
    for w in WIDTHS:
        grid = [(w, n, r) for n in NOISES for r in kinds_for(w)]
        budget = max(1, int(MEMORY_BUDGET_GB / (GB_PER_CELL_AT_64 * (w / 64) ** 3)))
        workers = max(1, min(len(grid), cores, budget))
        print(f"  {w} channels: {len(grid)} cells, {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows += list(pool.map(cell, grid, chunksize=1))

    for w in WIDTHS:
        ks = kinds_for(w)
        print(f"\n  {w} channels (ceiling {ceiling(w)}): counted kinds against kinds used")
        print("          " + "".join(f"{r:>8}" for r in ks))
        for n in NOISES:
            line = [next(r["counted"] for r in rows
                         if r["width"] == w and r["noise"] == n and r["R"] == k) for k in ks]
            print(f"  n={n:<5.2f} " + "".join(f"{v:>8.2f}" for v in line))

    # ── the ceiling, checked from the other side ────────────────────────────
    refused = {}
    for w in WIDTHS:
        try:
            relation_corpus(seed=BASE_SEED, n_relations=ceiling(w) + 1,
                            edges_per_relation=EDGES_PER_RELATION, n_vertices=N_VERTICES,
                            T=T, D=w, noise=NOISES[0], share_vertices=False)
            refused[w] = False
        except ValueError:
            refused[w] = True

    quiet = [r for r in rows if r["noise"] == min(NOISES)]
    exact = [r for r in quiet if r["exact_rate"] >= 1.0]
    at_ceiling = [r for r in quiet if r["at_ceiling"]]
    exact_at_ceiling = [r for r in at_ceiling if r["exact_rate"] >= 1.0]

    out = {
        "design": {"T": T, "edges_per_relation": EDGES_PER_RELATION, "n_vertices": N_VERTICES,
                   "far": FAR, "draws": DRAWS, "base_seed": BASE_SEED,
                   "widths": WIDTHS, "noises": NOISES,
                   "published_cell": {"width": 64, "source": "exp03 part A"}},
        "rows": rows,
        "cells": len(rows),
        "generator_refuses_past_the_ceiling": refused,
        "all_widths_refuse_past_the_ceiling": all(refused.values()),
        "quiet_cells": len(quiet),
        "quiet_cells_recovered_exactly": len(exact),
        "ceilings_run": len(at_ceiling),
        "ceilings_recovered_exactly": len(exact_at_ceiling),
        "largest_kind_count_recovered": max((r["R"] for r in exact), default=0),
        "seconds": time.time() - started,
    }
    print(f"\n  at the quietest noise, {len(exact)} of {len(quiet)} cells recover the kind count "
          f"exactly in every draw")
    print(f"  at each width's own ceiling: {len(exact_at_ceiling)} of {len(at_ceiling)} exact")
    print(f"  largest kind count recovered exactly: {out['largest_kind_count_recovered']}")
    print(f"  the generator refuses R > D/2 at every width: "
          f"{out['all_widths_refuse_past_the_ceiling']}")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp23.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp23.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
