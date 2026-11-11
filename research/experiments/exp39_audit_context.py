"""Stage 32 — the other half of the text negative, with its context span swept.

What is still held
--------------------
`exp38` swept the record window and the text negative survived it: counting beat the association
in 5 of 10 cells, every Moby-Dick window, and the association led in none. But `exp08` holds a second parameter the window sweep did not touch. `CONTEXT`
is how many words either side of a position are taken as the context a candidate must fit, and it is
a different quantity from the span a record's rows are built over.

It decides how much evidence each read is handed. A short context gives both reads little; a long one
gives both more, and gives them more of it from further away, where the connection is weaker. Nothing
says the two reads gain at the same rate, and `exp08`'s conclusion — that the best count beats the
best association on the next-word-from-context task — was measured at one value of it.

What would overturn the claim
-------------------------------
The claim is that counting wins on this task, not that it wins at context 6. It survives only if the
best count beats the best association at every context swept. Both reads are given their best
setting in each cell, because comparing one read's best against another's arbitrary choice is the
forcing this work objects to:

* counting gets `sum`, `max` and `mean`, which is three aggregations none of which the data supplies;
* the association gets magnitude, breadth, and breadth with magnitude as the tie-break.

A single context where the best association leads means the published figure is a property of the
cell, and the claim has to be restated as a range.

`exp08`'s design is otherwise untouched — the same split, the same frequency-matched distractors, the
same candidate count and seed — so context 6 anchors the sweep to the published cell.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import text as TX                                                   # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from exp07_real_text import TEXTS, WINDOW, _pmi                     # noqa: E402
from exp08_context import (FAR, FREQ_TOLERANCE, N_CANDIDATES,       # noqa: E402
                           QUESTIONS, SEED)

SMOKE = os.environ.get("EXP39_SMOKE") == "1"
CONTEXTS = (6,) if SMOKE else (2, 3, 6, 12, 24)
ASKED = 40 if SMOKE else QUESTIONS


def one_cell(name, url, segments, n_features, min_count, context):
    """`exp08`'s task, with the context span supplied rather than held."""
    rng = np.random.default_rng(SEED)
    coin = np.random.default_rng(SEED + 1)
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

    hits = {k: 0 for k in ("random", "pmi_sum", "pmi_max", "pmi_mean",
                           "z_sum", "breadth", "breadth_then_z")}
    # per-question outcomes, because the two reads answer the same questions. Comparing them with
    # an independent-samples error bar would overstate the uncertainty badly; what decides a paired
    # comparison is only the questions where they disagree.
    per_question = {k: [] for k in ("pmi_sum", "pmi_max", "pmi_mean",
                                    "z_sum", "breadth", "breadth_then_z")}
    asked = 0
    for p in rng.permutation(len(second)):
        if asked >= ASKED:
            break
        target = second[p]
        if target not in pool:
            continue
        lo, hi = max(0, p - context), min(len(second), p + context + 1)
        ctx = [second[q] for q in range(lo, hi)
               if q != p and second[q] in pool and second[q] != target]
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
        cands = [cands[i] for i in rng.permutation(len(cands))]
        gold = cands.index(target)
        asked += 1

        pmi_rows = [[pmi_first.get((c, w), 0.0) for w in ctx] for c in cands]
        _ok = int(int(np.argmax([sum(r) for r in pmi_rows])) == gold)
        hits["pmi_sum"] += _ok
        per_question["pmi_sum"].append(_ok)
        _ok = int(int(np.argmax([max(r) for r in pmi_rows])) == gold)
        hits["pmi_max"] += _ok
        per_question["pmi_max"].append(_ok)
        _ok = int(int(np.argmax([sum(r) / len(r) for r in pmi_rows])) == gold)
        hits["pmi_mean"] += _ok
        per_question["pmi_mean"].append(_ok)

        assoc = [[engine.associate(tr.records[c], tr.records[w], far=FAR) for w in ctx]
                 for c in cands]
        z_sum = [sum(a.z for a in row) for row in assoc]
        breadth = [sum(int(a.resolved) for a in row) for row in assoc]
        _ok = int(int(np.argmax(z_sum)) == gold)
        hits["z_sum"] += _ok
        per_question["z_sum"].append(_ok)
        _ok = int(int(np.argmax(breadth)) == gold)
        hits["breadth"] += _ok
        per_question["breadth"].append(_ok)
        _ok = int(max(range(len(cands)), key=lambda i: (breadth[i], z_sum[i])) == gold)
        hits["breadth_then_z"] += _ok
        per_question["breadth_then_z"].append(_ok)
        hits["random"] += int(int(coin.integers(len(cands))) == gold)

    n = max(asked, 1)
    rates = {k: v / n for k, v in hits.items()}
    count_key = max(("pmi_sum", "pmi_max", "pmi_mean"), key=lambda k: rates[k])
    assoc_key = max(("z_sum", "breadth", "breadth_then_z"), key=lambda k: rates[k])
    best_count, best_assoc = rates[count_key], rates[assoc_key]
    # the paired comparison: only the questions the two disagree on carry information
    a = np.asarray(per_question[count_key])
    b = np.asarray(per_question[assoc_key])
    count_only = int(np.sum((a == 1) & (b == 0)))
    assoc_only = int(np.sum((a == 0) & (b == 1)))
    discordant = count_only + assoc_only
    paired_se = (math.sqrt(discordant) / n) if discordant else 0.0
    paired = ((best_assoc - best_count) / paired_se) if paired_se > 0 else 0.0
    return {"text": name, "context": context, "questions": asked, **rates,
            "best_count_read": count_key, "best_association_read": assoc_key,
            "best_count": best_count, "best_association": best_assoc,
            "edge": best_assoc - best_count,
            "count_only": count_only, "association_only": assoc_only,
            "discordant": discordant, "paired_standard_error": paired_se,
            "edge_in_paired_standard_errors": paired,
            # three outcomes, not two. A tie neither confirms the published claim nor overturns it,
            # and folding it into either would be the verdict doing work the measurement did not.
            # a point difference is not a result. Two paired standard errors is the same bar the
            # rest of this program uses, and below it the cell decides nothing either way.
            "outcome": ("counting" if paired <= -2.0
                        else "association" if paired >= 2.0 else "undecided")}


def run():
    started = time.time()
    rows = []
    print(f"{'text':>12}{'context':>9}{'n':>6}{'best count':>12}{'best assoc':>12}"
          f"{'random':>9}{'edge':>9}")
    for name, url, segments, n_features, min_count in TEXTS:
        for context in CONTEXTS:
            r = one_cell(name, url, segments, n_features, min_count, context)
            rows.append(r)
            print(f"{name:>12}{context:>9}{r['questions']:>6}{r['best_count']:>12.3f}"
                  f"{r['best_association']:>12.3f}{r['random']:>9.3f}{r['edge']:>+9.3f}")

    scored = [r for r in rows if r["questions"] > 0]
    wins = sum(1 for r in scored if r["outcome"] == "counting")
    ties = sum(1 for r in scored if r["outcome"] == "undecided")
    leads = sum(1 for r in scored if r["outcome"] == "association")
    best_edge = max((r["edge"] for r in scored), default=float("nan"))
    out = {"design": {"far": FAR, "seed": SEED, "contexts": list(CONTEXTS),
                      "questions": ASKED, "n_candidates": N_CANDIDATES,
                      "record_window": WINDOW, "smoke": SMOKE},
           "cells": rows, "cells_scored": len(scored),
           "cells_where_counting_wins": wins,
           "cells_undecided": ties,
           "cells_where_the_association_leads": leads,
           "best_edge_for_the_association": best_edge,
           # the claim is overturned by the association leading anywhere, not by a tie
           "the_claim_is_overturned_anywhere": bool(leads > 0),
           "seconds": time.time() - started}
    print("\n" + "=" * 74)
    print(f"  at two paired standard errors: counting wins {wins}, undecided {ties}, the "
          f"association leads {leads}, of {len(scored)} cells")
    print(f"  the association's best point edge is {best_edge:+.3f}")
    if leads:
        print("  The context span WAS carrying the result: the association leads somewhere, so the")
        print("  published claim is a property of the cell it was measured in.")
    elif ties:
        print("  The association never leads. Where cells are undecided the task does not separate")
        print("  the two reads at this sample size, which is a limit of the measurement.")
    else:
        print("  The context span was not carrying the result.")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name_out = "exp39.smoke.json" if SMOKE else "exp39.json"
    (dest / name_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / name_out} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
