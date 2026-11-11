"""Stage 1b — does the binding refuse where it would be wrong?

The question
------------
An alignment always has a maximum, so a method that returns its argmax always answers. That is the
property this program cannot have: a system whose refusal is a consequence rather than a branch
has to fail closed, and an answer path built on a read that is confidently wrong is one that
confabulates exactly where it is least reliable.

So the decision requires two independent reads to agree, and the second to be unambiguous:

1. **Alignment** proposes — the strongest ``z`` among candidates that resolve.
2. **Breadth** confirms — the proposal is returned only if it is the unique candidate maximizing
   the resolved channels it shares with the query.

Both are measured; the test between them is uniqueness, so nothing is thresholded.

Why breadth is the right second read
-------------------------------------
The failure alignment cannot catch on its own is a narrow, loud, *consistent* partial match. Being
consistent, it is the same wrong answer on every subsample of the record, so a split-half or
jackknife check confirms it rather than exposing it — the error is bias, not variance. Breadth is
evidence of a different kind: a partial match cannot share channels the query has, however loudly it
matches the ones it does.

The design
----------
Two parameters govern the difficulty and both are swept rather than drawn:

* **breadth ratio ``m/R``** — how much of the query a distractor covers. The signal separating the
  true key from a distractor is that the true key carries *every* facet the query carries, so at
  ``m = 1`` of ``R = 6`` that signal is six-to-one, and at ``m = 5`` of ``R = 6`` it is one facet
  wide while the distractor carries each of its five at more than twice the true key's amplitude.
* **noise** — at which point the query stops resolving its own channels, so that the thing the true
  key is supposed to match is no longer fully present in the record.

What would falsify the rule
---------------------------
Stated before the run. The rule fails if precision falls below 1.000 in any cell where it answers
on more than a handful of trials — that is the soundness it exists to provide. It is *not*
falsified by low coverage: refusing correct answers is the price, and the run reports it in full
rather than reporting precision alone.

Run
---
    python research/experiments/exp02_refusal.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus                                    # noqa: E402
from entroptics_infer import bind, engine                          # noqa: E402

T, D = 96, 32
N_KEYS = 12
TRIALS = 80
BASE_SEED = 20260915
FAR = 0.05
GAIN = 2.0
FACETS = (2, 4, 6)
NOISE = (0.1, 0.2, 0.3, 0.5)

#: Coverage below which a precision figure is reported but not read as evidence — it is the count
#: of answered trials that is too small to mean anything, not a tuned cut. Stated so the table can
#: be read without inferring one.
MIN_ANSWERED_TO_JUDGE = 5


def run():
    started = time.time()
    rows = []
    print(f"\n{'R':>3}{'m':>3}{'m/R':>6}{'noise':>7} | {'raw acc':>8}{'coverage':>10}"
          f"{'precision':>11} | refusal reasons")
    for R in FACETS:
        for m in range(1, R):
            for noise in NOISE:
                raw = 0
                answered = 0
                correct = 0
                reasons = Counter()
                missed = 0            # refused, but unguarded alignment would have been right
                for t in range(TRIALS):
                    c = facet_corpus(seed=BASE_SEED + t, facets=R, n_keys=N_KEYS, T=T, D=D,
                                     narrow_gain=GAIN, noise=noise, subset_size=m)
                    Q = c.query.frame
                    keys = [k.frame for k in c.keys]
                    d = bind.decide(Q, keys, far=FAR)

                    live = [b for b in d.bindings if b.association.resolved]
                    unguarded = (max(live, key=lambda b: b.association.z).key
                                 if live else None)
                    hit = int(unguarded == c.truth)
                    raw += hit

                    reasons[d.why] += 1
                    if d.key is None:
                        missed += hit
                    else:
                        answered += 1
                        correct += int(d.key == c.truth)

                row = {
                    "facets": R, "subset_size": m, "breadth_ratio": m / R, "noise": noise,
                    "raw_accuracy": raw / TRIALS,
                    "coverage": answered / TRIALS,
                    "precision": (correct / answered) if answered else None,
                    "answered": answered,
                    "correct_but_refused": missed / TRIALS,
                    "reasons": dict(reasons),
                }
                rows.append(row)
                prec = "    n/a" if row["precision"] is None else f"{row['precision']:>7.3f}"
                flag = "" if (answered < MIN_ANSWERED_TO_JUDGE
                              or row["precision"] == 1.0) else "   <-- UNSOUND"
                top = ", ".join(f"{k}={v}" for k, v in reasons.most_common(2))
                print(f"{R:>3}{m:>3}{m/R:>6.2f}{noise:>7.2f} | {raw/TRIALS:>8.3f}"
                      f"{answered/TRIALS:>10.3f}{prec:>11} | {top}{flag}", flush=True)

    judged = [r for r in rows if r["answered"] >= MIN_ANSWERED_TO_JUDGE]
    unsound = [r for r in judged if r["precision"] < 1.0]

    # The cut is converted rather than trusted. A threshold conceals whatever sits under it, so the
    # verdict is recomputed at every cut the data supports: if the count of unsound cells is the
    # same at all of them, the cut decided nothing and the claim does not rest on it.
    by_cut = []
    for cut in (1, 2, 3, 4, 5, 8, 10, 20):
        j = [r for r in rows if r["answered"] >= cut]
        u = [r for r in j if r["precision"] is not None and r["precision"] < 1.0]
        by_cut.append({"cut": cut, "judged": len(j), "unsound": len(u)})
    under = [r for r in rows if r["answered"] < MIN_ANSWERED_TO_JUDGE]
    under_unsound = [r for r in under
                     if r["precision"] is not None and r["precision"] < 1.0]
    cut_is_inert = all(c["unsound"] == by_cut[0]["unsound"] for c in by_cut)
    print(f"\ncells judged (>= {MIN_ANSWERED_TO_JUDGE} answered): {len(judged)} of {len(rows)}")
    print(f"cells where precision < 1.000: {len(unsound)}")
    print()
    print(f"the cut, converted: {'inert' if cut_is_inert else 'LOAD-BEARING'}")
    print(f"  {'cut':>5}{'judged':>8}{'unsound':>9}")
    for c in by_cut:
        print(f"  {c['cut']:>5}{c['judged']:>8}{c['unsound']:>9}")
    print(f"  it excludes {len(under)} cells, of which {len(under_unsound)} are unsound")
    for r in unsound:
        print(f"  R={r['facets']} m={r['subset_size']} noise={r['noise']} "
              f"precision={r['precision']:.3f} on {r['answered']} answered")

    cost = [r for r in rows if r["correct_but_refused"] > 0]
    print(f"\ncells where a correct answer was refused: {len(cost)} of {len(rows)}; "
          f"worst {max((r['correct_but_refused'] for r in rows), default=0):.3f} of trials")

    out = {
        "design": {"T": T, "D": D, "n_keys": N_KEYS, "trials": TRIALS, "base_seed": BASE_SEED,
                   "far": FAR, "gain": GAIN, "facets": FACETS, "noise": NOISE,
                   "min_answered_to_judge": MIN_ANSWERED_TO_JUDGE},
        "rows": rows,
        "cells_judged": len(judged),
        "soundness_at_every_cut": by_cut,
        "cells_the_cut_excludes": len(under),
        "unsound_cells_the_cut_excludes": len(under_unsound),
        "the_cut_is_inert": bool(cut_is_inert),
        "cells_unsound": len(unsound),
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp02.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp02.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
