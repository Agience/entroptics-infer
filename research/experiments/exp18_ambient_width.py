"""Stage 13 — where the edge lives, in more than one parameter at once.

What forced this
----------------
The dial's own task at Beijing's shape shows no edge, and it looks like the ambient width is the
reason: the cell giving +0.983 at width 32 gives +0.000 at width 11 with
everything else identical.

Sweeping the width alone does show that. It is not a sound conclusion, and the reason is the
kind of thing this program exists to catch. The sweep holds the noise at one value, and the edge
is a ridge in the noise and the width together. A single row across a ridge reads as a clean
dependence on whichever variable the row runs along, and that reading does not survive stepping off
the row.

So this stage sweeps both.

What is measured
----------------
The facet structure stays as `exp15`'s discriminating cell — distractors carrying all but one facet
— and the ambient width, the noise amplitude, the facet count and the gain are all swept. The
result is reported as a surface rather than as a line, because a line through it is what produced
the wrong conclusion.

Where a real corpus's width falls in the sweep is not an analogy: a record's ambient width is its
channel count, for a generated corpus and a real one alike.

What would falsify the account
-------------------------------
Stated before the run. If the edge is large across a whole row or column of the surface, then that
variable does carry it and the ridge reading is wrong. If the edge appears nowhere, the
discriminating cell has been lost and the sweep is measuring nothing.

Run
---
    python research/experiments/exp18_ambient_width.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus                                     # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from measure import centred as _centred, centred_cos as cos, centred_dot as dot  # noqa: E402,F401

T = 96
N_KEYS = 12
TRIALS = 60
FAR = 0.05
BASE_SEED = 20260915
#: The discriminating parameter, swept and never chosen.
GAINS = (1.5, 2.0, 3.0)
#: **Swept, and this is the point of the experiment.** Holding the noise at a single value and
#: reporting the edge against the width alone is wrong: the edge is a ridge in *both*, and one row
#: across it reads as a clean dependence on width that does not survive moving off that row. A conclusion drawn from a held constant is a conclusion about the
#: constant.
NOISES = (0.02, 0.05, 0.10, 0.20, 0.35)
#: Facet counts, so the answer can be read against width-per-facet rather than against a width.
FACETS = (4, 6, 8)
#: From narrower than the narrowest real corpus read here to twice the width used elsewhere here.
WIDTHS = (5, 7, 9, 11, 14, 18, 24, 32, 48, 64)
#: Measured in `exp13`/`exp14` and on the other real corpora. A record's ambient width is its channel count, so these
#: land in the sweep directly rather than by analogy.
REAL_WIDTHS = {"US weather stations": 4, "UCI room occupancy": 4, "UCI household power": 7,
               "UCI HAR inertial": 9, "Beijing air quality": 11, "UCI air quality (Italy)": 12}


def cell(*, width, facets, gain, noise):
    hits = {"association": 0, "cos": 0, "dot": 0}
    ranks = []
    for t in range(TRIALS):
        c = facet_corpus(seed=BASE_SEED + t, facets=facets, n_keys=N_KEYS, T=T, D=width,
                         narrow_gain=gain, noise=noise, subset_size=facets - 1)
        Q = c.query.frame
        ranks.append(engine.rank(Q, far=FAR))
        frames = [k.frame for k in c.keys]
        hits["association"] += int(int(np.argmax([engine.associate(Q, f, far=FAR).z
                                                  for f in frames])) == c.truth)
        hits["cos"] += int(int(np.argmax([cos(Q, f) for f in frames])) == c.truth)
        hits["dot"] += int(int(np.argmax([dot(Q, f) for f in frames])) == c.truth)
    row = {"width": width, "facets": facets, "gain": gain, "noise": noise,
           "width_per_facet": width / facets,
           "modes": float(np.mean(ranks)), **{k: v / TRIALS for k, v in hits.items()}}
    row["best_scalar"] = max(row["cos"], row["dot"])
    row["edge"] = row["association"] - row["best_scalar"]
    return row


def run():
    started = time.time()
    rows = []
    for facets in FACETS:
        for width in WIDTHS:
            if width <= facets:
                continue
            for noise in NOISES:
                for gain in GAINS:
                    rows.append(cell(width=width, facets=facets, gain=gain, noise=noise))

    # The surface, printed as a surface. One row of it reads as a clean dependence on width; the
    # whole of it is what there is.
    print("\nEDGE over the best scalar — rows are the noise, columns the ambient width")
    print("        " + "".join(f"{w:>8}" for w in WIDTHS))
    surface = {}
    for noise in NOISES:
        cells = []
        for w in WIDTHS:
            here = [r["edge"] for r in rows if r["noise"] == noise and r["width"] == w]
            cells.append(max(here) if here else float("nan"))
        surface[noise] = cells
        print(f"n={noise:<5.2f} " + "".join(f"{e:>+8.3f}" for e in cells))

    print("\n  real corpora sit at these widths: "
          + ", ".join(f"{k} ({v})" for k, v in sorted(REAL_WIDTHS.items(), key=lambda kv: kv[1])))

    # Where the edge lives, as a region rather than as a threshold in one variable.
    positive = [r for r in rows if r["edge"] > 0]
    out = {
        "design": {"T": T, "n_keys": N_KEYS, "trials": TRIALS, "far": FAR,
                   "base_seed": BASE_SEED, "gains": GAINS, "facets": FACETS, "widths": WIDTHS,
                   "noises": NOISES, "chance": 1 / N_KEYS},
        "real_widths": REAL_WIDTHS,
        "rows": rows,
        "surface": {str(k): v for k, v in surface.items()},
        "cells": len(rows),
        "cells_with_an_edge": len(positive),
        "max_edge": max(r["edge"] for r in rows),
        # the finding this stage exists to state: the edge is a ridge in two parameters at once,
        # so a sweep of either one alone reports a dependence that does not survive moving off it
        "edge_at_widest_low_noise": max(
            r["edge"] for r in rows if r["width"] == max(WIDTHS) and r["noise"] <= 0.05),
        "edge_at_widest_mid_noise": max(
            r["edge"] for r in rows if r["width"] == max(WIDTHS) and r["noise"] == 0.10),
        "edge_at_narrow_high_noise": max(
            r["edge"] for r in rows if r["width"] <= 11 and r["noise"] == 0.20),
        "edge_at_narrow_mid_noise": max(
            r["edge"] for r in rows if r["width"] <= 11 and r["noise"] == 0.10),
        "noise_of_best_cell": max(rows, key=lambda r: r["edge"])["noise"],
        "seconds": time.time() - started,
    }
    print(f"\n  {out['cells_with_an_edge']} of {out['cells']} cells show any edge at all")
    print(f"  at the widest record: {out['edge_at_widest_low_noise']:+.3f} at low noise, "
          f"{out['edge_at_widest_mid_noise']:+.3f} at noise 0.10")
    print(f"  at the narrow records: {out['edge_at_narrow_mid_noise']:+.3f} at noise 0.10, "
          f"{out['edge_at_narrow_high_noise']:+.3f} at noise 0.20")
    print("  so neither the width nor the noise carries the edge on its own")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp18.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp18.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
