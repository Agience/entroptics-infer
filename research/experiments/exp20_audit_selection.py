"""Stage 14 — the central selection table, with what it held swept.

What this audits
----------------
`exp01` part B is the paper's selection table (PAPER §5.1). It sweeps the facet count against the gain and
reports that **`z` is 1.000 in all 30 cells** while `cos` collapses past the crossover `g = √(R/m)`.

It holds the noise at 0.10 and the ambient width at 32.

Those are the two parameters `exp15` and `exp18` show to matter. In both, a sweep of one variable
at a fixed value of the other reports a clean dependence that does not survive stepping off the row,
and in `exp15`'s case the held row is the one where the association never loses. The same two are
held here, under the paper's most load-bearing claim, so the same question is asked of it.

What is measured
----------------
`exp01` part B's design exactly — same generator call, same candidate count, same scalars, `m` left
random as it leaves it — with **the noise and the ambient width swept underneath**.

The two claims are audited separately because they are different kinds of claim:

* **`cos` collapses past the crossover** is a *direction* claim, and it is underwritten by arithmetic:
  above `g = √(R/m)` a distractor carries more shared energy than the true key, so any method ranking
  by shared magnitude must fail. This should hold everywhere, and if it does not the arithmetic in
  PAPER §5.1 is wrong.
* **`z` is 1.000** is a *level* claim about one cell of a surface. `exp15` already shows `z` losing
  badly at narrow widths with coupled facets, so the question is where on this surface it holds.

What would falsify what
------------------------
Stated before the run. If `z` is 1.000 across the whole surface, the central claim is stronger than
`exp01` establishes and the held constants are inert. If `z` falls away from the held row, **PAPER §5.1's
table is a cell and not a result**, and it is stated as a surface, as `exp15` and `exp18` are.

If `cos` fails to collapse past the crossover somewhere, that is the more serious finding, because
the crossover is arithmetic rather than measurement.

Run
---
    python research/experiments/exp20_audit_selection.py
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

from corpus import facet_corpus                                     # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from measure import centred as _centred, centred_cos as cos, centred_dot as dot  # noqa: E402,F401

#: `exp01` part B's design, unchanged.
T = 96
N_KEYS = 12
BASE_SEED = 20260915
FAR = 0.05
FACETS = (1, 2, 3, 4, 5, 6)
GAINS = (1.0, 1.5, 2.0, 2.5, 3.0)
TRIALS = 40

#: What `exp01` holds, swept here. 32 and 0.10 are its values and are included so the published row
#: appears in the surface rather than beside it.
WIDTHS = (11, 18, 32, 64)
NOISES = (0.02, 0.05, 0.10, 0.20, 0.35)


def pooled(A, B):
    """The standard-attention analogue: mean over rows, then a dot product."""
    return float(np.dot(_centred(A).mean(axis=0), _centred(B).mean(axis=0)))


def constructible(facets, width):
    """Whether `facet_corpus` can build this cell at all.

    The `R = 1` control gives every distractor its own **disjoint** facet, so it needs one direction
    per key on top of the query's: ``D >= 1 + n_keys``. At `R > 1` the distractors draw from the
    query's own facets and any ``D >= R`` will do. This is a constraint of the generator and not a
    choice made here, so the cells it rules out are skipped and counted rather than quietly dropped.
    """
    return width >= (1 + N_KEYS if facets == 1 else facets + 1)


def cell(args):
    facets, gain, width, noise = args
    hits = {"z": 0, "cos": 0, "dot": 0, "pooled": 0}
    for t in range(TRIALS):
        c = facet_corpus(seed=BASE_SEED + t, facets=facets, n_keys=N_KEYS, T=T, D=width,
                         narrow_gain=gain, noise=noise)
        Q = c.query.frame
        frames = [k.frame for k in c.keys]
        hits["z"] += int(int(np.argmax([engine.associate(Q, f, far=FAR).z
                                        for f in frames])) == c.truth)
        hits["cos"] += int(int(np.argmax([cos(Q, f) for f in frames])) == c.truth)
        hits["dot"] += int(int(np.argmax([dot(Q, f) for f in frames])) == c.truth)
        hits["pooled"] += int(int(np.argmax([pooled(Q, f) for f in frames])) == c.truth)
    row = {"facets": facets, "gain": gain, "width": width, "noise": noise,
           # the crossover for m = 1, which is where a magnitude ranking first must fail
           "crossover_m1": float(np.sqrt(facets)),
           "past_crossover": bool(gain > np.sqrt(facets)),
           **{k: v / TRIALS for k, v in hits.items()}}
    row["best_scalar"] = max(row["cos"], row["dot"], row["pooled"])
    row["edge"] = row["z"] - row["best_scalar"]
    return row


def run():
    started = time.time()
    full = [(f, g, w, n) for w in WIDTHS for n in NOISES for f in FACETS for g in GAINS]
    grid = [c for c in full if constructible(c[0], c[2])]
    skipped = len(full) - len(grid)
    workers = max(1, min(len(grid), (os.cpu_count() or 2) - 2))
    print(f"{len(grid)} cells, {TRIALS} trials each, {workers} workers "
          f"({skipped} cells the generator cannot build)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(cell, grid, chunksize=4))

    # ── claim 1: z is 1.000 ──────────────────────────────────────────────────
    published = [r for r in rows if r["width"] == 32 and r["noise"] == 0.10]
    perfect = [r for r in rows if r["z"] >= 1.0]
    print(f"\n`z` = 1.000 in {len(perfect)} of {len(rows)} cells "
          f"({len(published)} of which are the published row's)")
    print(f"`z` on the published row: min {min(r['z'] for r in published):.3f}")
    print(f"`z` across the surface:   min {min(r['z'] for r in rows):.3f}")

    # the mean `z` of every cell at one noise and one width, over facet counts and gains: PAPER §5.1's
    # width x noise table, recorded so the paper's figures have a producer
    mean_z = {str(n): {str(w): float(np.mean([r["z"] for r in rows
                                              if r["noise"] == n and r["width"] == w]))
                       for w in WIDTHS}
              for n in NOISES}
    print("\nMEAN `z`, rows = noise, columns = ambient width")
    print("        " + "".join(f"{w:>9}" for w in WIDTHS))
    for n in NOISES:
        line = [mean_z[str(n)][str(w)] for w in WIDTHS]
        print(f"n={n:<5.2f} " + "".join(f"{v:>9.3f}" for v in line))

    print("\nEDGE of `z` over the best scalar, rows = noise, columns = ambient width")
    print("        " + "".join(f"{w:>9}" for w in WIDTHS))
    for n in NOISES:
        line = [float(np.mean([r["edge"] for r in rows if r["noise"] == n and r["width"] == w]))
                for w in WIDTHS]
        print(f"n={n:<5.2f} " + "".join(f"{v:>+9.3f}" for v in line))

    # ── claim 2: a magnitude ranking fails past the crossover ────────────────
    # The arithmetic of PAPER §5.1, which should hold everywhere or be wrong. Checked at m = 1, the
    # only subset size whose crossover is known without inspecting the draw.
    past = [r for r in rows if r["past_crossover"] and r["facets"] > 1]
    dot_holds = sum(1 for r in past if r["dot"] <= 1.0 / N_KEYS + 1e-9)
    beaten = [r for r in rows if r["edge"] < 0]

    out = {
        "design": {"T": T, "n_keys": N_KEYS, "trials": TRIALS, "far": FAR, "base_seed": BASE_SEED,
                   "facets": FACETS, "gains": GAINS, "widths": WIDTHS, "noises": NOISES,
                   "chance": 1 / N_KEYS,
                   "published_cell": {"width": 32, "noise": 0.10, "source": "exp01 part B"}},
        "rows": rows,
        "cells": len(rows),
        "cells_the_generator_cannot_build": skipped,
        "cells_where_z_is_perfect": len(perfect),
        "z_min_on_published_row": min(r["z"] for r in published),
        "z_min_over_surface": min(r["z"] for r in rows),
        "z_mean_over_surface": float(np.mean([r["z"] for r in rows])),
        "mean_z_by_noise_and_width": mean_z,
        "cells_where_a_scalar_beats_z": len(beaten),
        "worst_edge": min(r["edge"] for r in rows),
        "cells_past_crossover": len(past),
        "cells_past_crossover_where_dot_is_at_chance": dot_holds,
        "seconds": time.time() - started,
    }
    print(f"\n  past the crossover, `dot` is at or below chance in "
          f"{dot_holds} of {len(past)} cells")
    print(f"  a scalar beats `z` in {len(beaten)} of {len(rows)} cells; "
          f"worst edge {out['worst_edge']:+.3f}")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp20.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp20.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
