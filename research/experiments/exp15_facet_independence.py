"""Stage 10 — why the advantage does not transfer: are the facets independent?

The question
------------
The association beats every scalar normalization on generated corpora and on none of the three real
ones — including Beijing air quality, which is admitted, multi-facet, and demonstrably contains the
narrow-loud configuration the claim concerns. Three explanations were already ruled out by checking
them first, so what is left is a difference between the corpora themselves.

One difference is visible without looking hard. In the generated corpora the facets are
independent by construction — orthonormal loadings, independent latent time-courses — and in real
data they are not. Eleven Beijing channels resolve 3.5 modes, because pollutants move with pollutants
and temperature moves with dew point. Whether that is the reason is a question, not an answer, and
this makes it one that can be asked.

The prediction, stated before the run
--------------------------------------
`facet_coupling` gives every facet a share of one common time-course, so at 0 they are independent
and as it rises they move together. Nothing else about the corpus changes. Then:

1. The association's edge over the best scalar shrinks as the coupling rises.
2. It shrinks together with the number of distinct coherent structures the record resolves, which is
   the quantity a real corpus can also be measured on.

If the edge is flat in the coupling, facet independence is not what the advantage runs on and the
question is still open. If the edge *grows* with coupling, the account is wrong in a way worth
knowing.

This measures a dependence inside one corpus family, on one task. It does not by itself say what
edge to expect on a real corpus at the same mode count.

Why this matters either way
----------------------------
A positive result gives a second thing to ask of a corpus. The first asks whether the relation
varies along the shared ordered axis; this adds how much distinct structure its records carry for a
matrix read to form subsets of. Both are properties of a corpus, measurable before a task is scored.

Run
---
    python research/experiments/exp15_facet_independence.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus                                    # noqa: E402
from entroptics_infer import engine                                # noqa: E402
from measure import centred as _centred, centred_cos as cos, centred_dot as dot  # noqa: E402,F401

T = 96
#: **Swept.** Holding the width and the noise at one value each reports the edge against the
#: coupling alone, on one row, and the published row is the one where the association never
#: loses; off it the edge goes materially negative, and a sweep that cannot show that is not
#: measuring the dial, it is measuring its own row.
WIDTHS = (11, 18, 32, 64)
NOISES = (0.05, 0.10, 0.20, 0.35)
FACETS = 6
#: The discriminating cell, chosen by measurement rather than by taste: at six facets with
#: distractors carrying five of them, the association scores 1.000 and a cosine 0.017, so there is an
#: edge of +0.983 to watch. At `SUBSET = 2` the cosine also scores 1.000 and the sweep would be
#: measuring nothing — a cell with no edge cannot show one shrinking.
SUBSET = 5
N_KEYS = 12
GAIN = 2.0
TRIALS = 60
FAR = 0.05
BASE_SEED = 20260915
COUPLINGS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.92, 0.97)


def cell(*, coupling, width, noise):
    hits = {"association": 0, "cos": 0, "dot": 0}
    ranks = []
    for t in range(TRIALS):
        corpus = facet_corpus(seed=BASE_SEED + t, facets=FACETS, n_keys=N_KEYS, T=T, D=width,
                              narrow_gain=GAIN, noise=noise, subset_size=SUBSET,
                              facet_coupling=coupling)
        Q = corpus.query.frame
        ranks.append(engine.rank(Q, far=FAR))
        frames = [k.frame for k in corpus.keys]
        hits["association"] += int(int(np.argmax([engine.associate(Q, f, far=FAR).z
                                                  for f in frames])) == corpus.truth)
        hits["cos"] += int(int(np.argmax([cos(Q, f) for f in frames])) == corpus.truth)
        hits["dot"] += int(int(np.argmax([dot(Q, f) for f in frames])) == corpus.truth)
    row = {"coupling": coupling, "width": width, "noise": noise,
           "mean_rank": float(np.mean(ranks)), "rank_ratio": float(np.mean(ranks)) / FACETS,
           **{k: v / TRIALS for k, v in hits.items()}}
    row["best_scalar"] = max(row["cos"], row["dot"])
    row["edge"] = row["association"] - row["best_scalar"]
    return row


def run():
    started = time.time()
    rows = []
    for width in WIDTHS:
        for noise in NOISES:
            for c in COUPLINGS:
                rows.append(cell(coupling=c, width=width, noise=noise))

    for width in WIDTHS:
        print(f"\n=== {width} channels: EDGE, rows = noise, columns = coupling ===")
        print("        " + "".join(f"{c:>9.2f}" for c in COUPLINGS))
        for noise in NOISES:
            line = [next(r["edge"] for r in rows
                         if r["width"] == width and r["noise"] == noise and r["coupling"] == c)
                    for c in COUPLINGS]
            print(f"n={noise:<5.2f} " + "".join(f"{e:>+9.3f}" for e in line))

    edges = [r["edge"] for r in rows]

    def at(width, noise, coupling):
        return next(r["edge"] for r in rows if r["width"] == width
                    and r["noise"] == noise and r["coupling"] == coupling)

    # Does coupling reduce the edge everywhere, or only where it was positive to begin with?
    falls = sum(1 for w in WIDTHS for n in NOISES
                if at(w, n, COUPLINGS[-1]) <= at(w, n, COUPLINGS[0]))
    out = {
        "design": {"T": T, "facets": FACETS, "subset_size": SUBSET, "n_keys": N_KEYS,
                   "gain": GAIN, "trials": TRIALS, "far": FAR, "base_seed": BASE_SEED,
                   "couplings": COUPLINGS, "widths": WIDTHS, "noises": NOISES},
        "rows": rows,
        "cells": len(rows),
        "edge_max": max(edges),
        "edge_min": min(edges),
        "cells_where_the_association_loses": sum(1 for e in edges if e < 0),
        "rows_where_coupling_reduces_the_edge": falls,
        "rows_swept": len(WIDTHS) * len(NOISES),
        # the row a single held width and noise would report, and the two that contradict its reading
        "edge_at_32ch_n010_independent": at(32, 0.10, COUPLINGS[0]),
        "edge_at_32ch_n010_coupled": at(32, 0.10, COUPLINGS[-1]),
        "edge_at_11ch_n010_coupled": at(11, 0.10, COUPLINGS[-1]),
        "edge_at_32ch_n005_coupled": at(32, 0.05, COUPLINGS[-1]),
        "seconds": time.time() - started,
    }
    print(f"\n  {out['cells']} cells; the edge runs {out['edge_min']:+.3f} to "
          f"{out['edge_max']:+.3f}")
    print(f"  the association is BEATEN by a scalar in "
          f"{out['cells_where_the_association_loses']} of {out['cells']} cells")
    print(f"  coupling reduces the edge in {falls} of {out['rows_swept']} width x noise rows")
    print(f"  at 32 channels, noise 0.10: {out['edge_at_32ch_n010_independent']:+.3f} independent, "
          f"{out['edge_at_32ch_n010_coupled']:+.3f} coupled")
    print(f"  at 11 channels, noise 0.10, coupled: {out['edge_at_11ch_n010_coupled']:+.3f}")
    print(f"  at 32 channels, noise 0.05, coupled: {out['edge_at_32ch_n005_coupled']:+.3f}")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp15.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp15.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
