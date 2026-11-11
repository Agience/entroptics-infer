"""Stage 31 — the text negative, with the window it was measured at swept underneath it.

The audit this closes
-----------------------
`exp07` and `exp08` are where this paper concludes that counting beats the association on real
text — first-half PMI at 0.125 and 0.521 against the association's 0.125 and 0.306. Both were
measured at a single window of 6, held fixed, and a negative result measured at one setting of a
parameter is the shape this program has twice got wrong: held at one value, `exp15` and `exp18`
each report a clean dependence that is one line across a ridge, and both conclusions fall when the
held parameter is swept.

So the same question is put to this one. The window sets two things at once and they pull in
different directions:

* it is the span a record's rows are built over, so it decides what the association has to read;
* it is the span the PMI count is taken over, so it decides what counting has to read.

A window that favoured counting would produce exactly the published result without the association
ever having been given its best setting, and nothing in `exp07` could tell the difference.

What would overturn the claim, and what would confirm it
----------------------------------------------------------
The claim is not that the association is weak at window 6. It is that counting wins on this task.
It survives only if counting wins at every window swept. A single window where the association
leads means the published figure is a property of the cell it was measured in, and the claim has to
be restated as a range rather than a result.

Both texts are carried through, because a claim that held on one and not the other would be a
different finding again. Everything else is `exp07`'s design untouched: the same split, the same
frequency-matched distractors, the same candidate count, the same seed.
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
from exp07_real_text import (FAR, FREQ_TOLERANCE, N_CANDIDATES,     # noqa: E402
                             SEED, TEXTS, _pmi)

SMOKE = os.environ.get("EXP38_SMOKE") == "1"
WINDOWS = (6,) if SMOKE else (2, 3, 6, 12, 24)


def one_cell(name, url, segments, n_features, min_count, window):
    """`exp07`'s task exactly, with the window supplied rather than held."""
    rng = np.random.default_rng(SEED)
    # the random baseline draws from its own generator, so the question stream is bit-for-bit the
    # one exp07 built and window 6 anchors the sweep to the published cell exactly
    coin = np.random.default_rng(SEED + 1)
    text = TX.fetch(url, into=HERE.parents[1] / "corpora")
    tokens = list(text.tokens)
    half = len(tokens) // 2
    first, second = tokens[:half], tokens[half:]

    c_first, c_second = Counter(first), Counter(second)
    vocab = tuple(w for w in c_first
                  if c_first[w] >= min_count and c_second[w] >= min_count)

    first_text = TX.Text(url=text.url, sha256=text.sha256, characters=0, tokens=tuple(first))
    tr = TX.build_records(first_text, segments=segments, window=window,
                          n_features=n_features, min_count=min_count)
    usable = tuple(w for w in vocab if w in tr.records)
    pmi_second, _ = _pmi(second, usable, window)
    pmi_first, _ = _pmi(first, usable, window)
    resolved = [w for w in usable if engine.rank(tr.records[w], far=FAR) > 0]

    hits = {"association": 0, "counting": 0, "random": 0}
    # the two reads answer the same questions, so only the ones they disagree on carry information.
    # An independent-samples error bar on paired data overstates the uncertainty.
    per_question = {"association": [], "counting": []}
    asked = 0
    for q in resolved:
        collocates = [(v, pmi_second.get((q, v), -math.inf)) for v in resolved if v != q]
        collocates = [(v, s) for v, s in collocates if s > -math.inf]
        if not collocates:
            continue
        truth, _ = max(collocates, key=lambda x: x[1])
        target = c_first[truth]
        pool = [v for v, _ in collocates
                if v != truth and abs(c_first[v] - target) <= FREQ_TOLERANCE * target]
        if len(pool) < N_CANDIDATES - 1:
            continue
        picked = [pool[i] for i in rng.choice(len(pool), size=N_CANDIDATES - 1, replace=False)]
        cands = [truth] + picked
        cands = [cands[i] for i in rng.permutation(len(cands))]
        gold = cands.index(truth)
        asked += 1

        zs = [engine.associate(tr.records[q], tr.records[v], far=FAR).z for v in cands]
        _a = int(int(np.argmax(zs)) == gold)
        hits["association"] += _a
        per_question["association"].append(_a)
        counts = [pmi_first.get((q, v), -math.inf) for v in cands]
        _c = int(int(np.argmax(counts)) == gold)
        hits["counting"] += _c
        per_question["counting"].append(_c)
        hits["random"] += int(int(coin.integers(len(cands))) == gold)

    n = max(asked, 1)
    row = {"text": name, "window": window, "questions": asked,
           "association": hits["association"] / n, "counting": hits["counting"] / n,
           "random": hits["random"] / n, "resolved_types": len(resolved)}
    row["edge"] = row["association"] - row["counting"]
    a = np.asarray(per_question["counting"])
    b = np.asarray(per_question["association"])
    count_only = int(np.sum((a == 1) & (b == 0)))
    assoc_only = int(np.sum((a == 0) & (b == 1)))
    discordant = count_only + assoc_only
    se = (math.sqrt(discordant) / n) if discordant else 0.0
    row["count_only"] = count_only
    row["association_only"] = assoc_only
    row["discordant"] = discordant
    row["paired_standard_error"] = se
    row["edge_in_paired_standard_errors"] = (row["edge"] / se) if se > 0 else 0.0
    # a point difference is not a result; two paired standard errors is the program's bar
    row["outcome"] = ("counting" if row["edge_in_paired_standard_errors"] <= -2.0
                      else "association" if row["edge_in_paired_standard_errors"] >= 2.0
                      else "undecided")
    return row


def run():
    started = time.time()
    rows = []
    print(f"{'text':>12}{'window':>8}{'n':>6}{'association':>13}{'counting':>10}"
          f"{'random':>9}{'edge':>9}{'paired SE':>9}{'outcome':>12}")
    for name, url, segments, n_features, min_count in TEXTS:
        for window in WINDOWS:
            r = one_cell(name, url, segments, n_features, min_count, window)
            rows.append(r)
            print(f"{name:>12}{window:>8}{r['questions']:>6}{r['association']:>13.3f}"
                  f"{r['counting']:>10.3f}{r['random']:>9.3f}{r['edge']:>+9.3f}"
                  f"{r['edge_in_paired_standard_errors']:>+9.2f}{r['outcome']:>12}")

    scored = [r for r in rows if r["questions"] > 0]
    wins = sum(1 for r in scored if r["outcome"] == "counting")
    undecided = sum(1 for r in scored if r["outcome"] == "undecided")
    leads = sum(1 for r in scored if r["outcome"] == "association")
    best_edge = max((r["edge"] for r in scored), default=float("nan"))
    out = {
        "design": {"far": FAR, "seed": SEED, "windows": list(WINDOWS),
                   "n_candidates": N_CANDIDATES, "frequency_tolerance": FREQ_TOLERANCE,
                   "smoke": SMOKE},
        "cells": rows,
        "cells_scored": len(scored),
        "cells_where_counting_wins": wins,
        "cells_undecided": undecided,
        "cells_where_the_association_leads": leads,
        "best_edge_for_the_association": best_edge,
        "the_claim_is_overturned_anywhere": bool(leads > 0),
        "seconds": time.time() - started,
    }
    print("\n" + "=" * 74)
    print(f"  at two paired standard errors: counting wins {wins}, undecided {undecided}, the "
          f"association leads {leads}, of {len(scored)} cells")
    print(f"  the association's best point edge is {best_edge:+.3f}")
    if leads:
        print("  The window WAS carrying the result: the association leads somewhere.")
    elif undecided == len(scored):
        print("  No cell separates the two reads at this sample size. The published edge is a")
        print("  point difference inside the paired error, not an established result.")
    elif undecided:
        print("  The association never leads, and some cells do not separate the two reads.")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name_out = "exp38.smoke.json" if SMOKE else "exp38.json"
    (dest / name_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / name_out} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
