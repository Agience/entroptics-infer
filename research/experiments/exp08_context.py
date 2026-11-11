"""Stage 4c — a multi-word context on real text, where ground truth is the text and not a statistic.

Why this task and not the last one
-----------------------------------
`exp07` asked *"which word has the highest co-occurrence statistic in held-out text"*, and PMI —
which estimates exactly that statistic — won. That result stands, and this is not an attempt to
overturn it: it is the test the earlier one could not be, because the answer here is a word that
actually occurs, not a number that has to be estimated.

A position is taken from the second half of a book. The words around it are the context; the word
at it is the answer. Nothing about the answer is a statistic, so no method is the direct estimator of
it by construction.

Why it is the right shape for the claim
----------------------------------------
The claim these reads rest on is about multi-facet relevance: where the evidence is spread over
several independent dimensions, a single scalar cannot carry it. A multi-word context is that case
in natural language — the answer is the word that fits *all* of the context, and the characteristic
failure is a word that fits *one* context word overwhelmingly.

And it puts the forcing where it can be seen. A count-based method must choose how to aggregate its
pairwise statistics across the context — sum, maximum, or mean — and that choice is not given by
the data. All three are run here as separate baselines, because picking one and calling it *the*
baseline would hide the very thing under test. The association-based reads make no such choice:
breadth is counted and magnitude is summed, with nothing weighting one against the other.

The candidates
--------------
The true next word plus frequency-matched distractors, so frequency is not the giveaway, and all
drawn from the words that have a readable record built from the first half.

What would falsify this stage
-----------------------------
Stated before the run. The stage fails if the association reads do not beat random; and it is
reported as no advance over counting if they do not beat the best of the three count
aggregations. Beating one aggregation while losing to another is not a result — the best of them is
the baseline.

Run
---
    python research/experiments/exp08_context.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]
sys.path.insert(0, str(HERE.parent))

import text as TX                                                  # noqa: E402
from entroptics_infer import category, engine                      # noqa: E402
from exp07_real_text import TEXTS, WINDOW, _pmi                    # noqa: E402

N_CANDIDATES = 8
CONTEXT = 6                  # words either side of the position, before the vocabulary filter
QUESTIONS = 250
FAR = 0.05
SEED = 20260915
FREQ_TOLERANCE = 0.5


def one_text(name, url, segments, n_features, min_count):
    rng = np.random.default_rng(SEED)
    text = TX.fetch(url, into=HERE.parents[1] / "corpora")
    tokens = list(text.tokens)
    half = len(tokens) // 2
    first, second = tokens[:half], tokens[half:]

    c_first, c_second = Counter(first), Counter(second)
    vocab = tuple(w for w in c_first if c_first[w] >= min_count and c_second[w] >= min_count)
    first_text = TX.Text(url=url, sha256=text.sha256, characters=0, tokens=tuple(first))
    tr = TX.build_records(first_text, segments=segments, window=WINDOW,
                          n_features=n_features, min_count=min_count)
    readable = [w for w in vocab if w in tr.records and engine.rank(tr.records[w], far=FAR) > 0]
    pool = set(readable)
    pmi_first, _ = _pmi(first, readable, WINDOW)

    print(f"\n=== {name} ===\n  {len(tokens)} tokens, {len(readable)} words with a readable record")

    hits = {k: 0.0 for k in ("random", "pmi_sum", "pmi_max", "pmi_mean",
                             "z_sum", "breadth", "breadth_then_z", "colimit")}
    join_dims = []
    asked = 0
    positions = rng.permutation(len(second))

    for p in positions:
        target = second[p]
        if target not in pool:
            continue
        lo, hi = max(0, p - CONTEXT), min(len(second), p + CONTEXT + 1)
        ctx = [second[q] for q in range(lo, hi) if q != p and second[q] in pool and second[q] != target]
        ctx = list(dict.fromkeys(ctx))
        if len(ctx) < 2:
            continue

        tgt_count = c_first[target]
        distractors = [w for w in readable
                       if w != target and w not in ctx
                       and abs(c_first[w] - tgt_count) <= FREQ_TOLERANCE * tgt_count]
        if len(distractors) < N_CANDIDATES - 1:
            continue
        picked = [distractors[i]
                  for i in rng.choice(len(distractors), size=N_CANDIDATES - 1, replace=False)]
        cands = [target] + picked
        order = rng.permutation(len(cands))
        cands = [cands[i] for i in order]
        gold = cands.index(target)

        # count baselines: three aggregations, none given by the data
        pmi_rows = [[pmi_first.get((c, w), 0.0) for w in ctx] for c in cands]
        hits["pmi_sum"] += int(int(np.argmax([sum(r) for r in pmi_rows])) == gold)
        hits["pmi_max"] += int(int(np.argmax([max(r) for r in pmi_rows])) == gold)
        hits["pmi_mean"] += int(int(np.argmax([sum(r) / len(r) for r in pmi_rows])) == gold)

        # association reads: magnitude, breadth, and breadth with magnitude as the tie-break
        assoc = [[engine.associate(tr.records[c], tr.records[w], far=FAR) for w in ctx]
                 for c in cands]
        z_sum = [sum(a.z for a in row) for row in assoc]
        breadth = [sum(int(a.resolved) for a in row) for row in assoc]
        hits["z_sum"] += int(int(np.argmax(z_sum)) == gold)
        hits["breadth"] += int(int(np.argmax(breadth)) == gold)
        hits["breadth_then_z"] += int(
            max(range(len(cands)), key=lambda i: (breadth[i], z_sum[i])) == gold
        )
        # the colimit read: the context glued into one object, with no aggregation chosen, and
        # each candidate scored by how much of itself lies inside it
        d = category.diagram(tr.records, ctx, far=FAR)
        join_dims.append(d.join_dim)
        if d.join_dim > 0:
            share = []
            for c in cands:
                own = engine.basis(tr.records[c], far=FAR)
                if own.shape[0] == 0:
                    share.append(0.0)
                    continue
                proj = own @ d.join.T
                share.append(float(np.sum(proj * proj) / np.sum(own * own)))
            hits["colimit"] += int(int(np.argmax(share)) == gold)
        else:
            hits["colimit"] += 1.0 / len(cands)

        hits["random"] += 1.0 / len(cands)
        asked += 1
        if asked >= QUESTIONS:
            break

    print(f"  {asked} questions, {N_CANDIDATES} frequency-matched candidates, "
          f"context of up to {CONTEXT} either side\n")
    scores = {k: v / asked for k, v in hits.items()}
    best_count = max(scores["pmi_sum"], scores["pmi_max"], scores["pmi_mean"])
    best_assoc = max(scores["z_sum"], scores["breadth"], scores["breadth_then_z"],
                     scores["colimit"])
    for label, key in (("random", "random"), ("PMI summed", "pmi_sum"), ("PMI max", "pmi_max"),
                       ("PMI mean", "pmi_mean"), ("association, z summed", "z_sum"),
                       ("association, breadth", "breadth"),
                       ("association, breadth then z", "breadth_then_z"),
                       ("colimit of the context", "colimit")):
        print(f"{label:>30}{scores[key]:>9.3f}")
    print(f"\n  best count aggregation {best_count:.3f}   best association read {best_assoc:.3f}"
          f"   edge {best_assoc - best_count:+.3f}")

    return {"name": name, "questions": asked, "scores": scores,
            "join_dim_mean": float(np.mean(join_dims)),
            "join_dim_range": [int(min(join_dims)), int(max(join_dims))],
            "best_count": best_count, "best_association": best_assoc,
            "edge": best_assoc - best_count,
            "design": {"n_candidates": N_CANDIDATES, "context": CONTEXT, "far": FAR,
                       "seed": SEED, "segments": segments, "n_features": n_features,
                       "min_count": min_count, "readable_vocabulary": len(readable)}}


def run():
    started = time.time()
    results = [one_text(*spec) for spec in TEXTS]
    out = {"texts": results, "seconds": time.time() - started}
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp08.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{'text':>12}{'questions':>11}{'random':>9}{'best count':>12}{'best assoc':>12}"
          f"{'edge':>8}")
    for r in results:
        print(f"{r['name']:>12}{r['questions']:>11}{r['scores']['random']:>9.3f}"
              f"{r['best_count']:>12.3f}{r['best_association']:>12.3f}{r['edge']:>+8.3f}")
    print(f"\nwrote {dest / 'exp08.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
