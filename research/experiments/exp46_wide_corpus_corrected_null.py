"""Stage 39 — the wide corpus re-scored against a null that delivers its level.

Why this is the measurement to take
-------------------------------------
PAPER §6.1 measures that `associate`'s permutation null does not deliver its stated level on records
whose rows are autocorrelated. The corpus of Stage 13 — nineteen activities over five inertial units,
forty-five channels, the widest real corpus this program has read — sits at a lag-one row
correlation of +0.708, where the realized level is about 0.26 against the 0.05 asked for.

That is also the corpus where the association was measured behind three of five scalar reads, by
−2.32 paired standard errors. So the obvious question is whether it was behind the reads or behind
the null.

**It does not follow automatically that it was.** The task takes an `argmax` over twelve candidate
frames; no cutoff enters it, and an inflation shared by every candidate cancels in a ranking. What
would *not* cancel is the standardization going wrong by different amounts for different candidates
— `z` divides by `tr(C_Q C_f)`, which is a property of the pair, and two candidates at the same
autocorrelation need not be inflated equally. Whether that reorders the argmax is not something to
reason about. It is something to measure, and it has two possible answers, both worth having:

* **the ranking moves** — then Stage 13's comparison was measuring the null and has to be restated;
* **the ranking holds** — then that comparison stands on its own, and the level defect is bounded
  away from it, which is a stronger statement than it had before.

How the correction is applied
-------------------------------
Exactly as `admission` does it, one level down. For each query and candidate the observed `z` is
read against the distribution of `z` between decoupled copies of that same pair: each record's
own covariance and spectrum kept exactly, its phase relationship to the other destroyed. Nothing is
supplied and nothing is fitted; the surrogate is drawn from the pair being scored.

Every other thing is held from `exp19`: the same corpus, the same seed, the same questions, the same
candidates, the same competitors. Only the standardization of the association changes, so a
difference between the two arms is carried by that and nothing else.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import exp19_wide_corpus as W                                       # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from entroptics_infer.admission import _decoupled, _row_autocorrelation  # noqa: E402

FAR = W.FAR
SEED = W.SEED
SMOKE = os.environ.get("EXP46_SMOKE") == "1"
QUERIES = 20 if SMOKE else 120
DRAWS = 8 if SMOKE else 24


def corrected_z(rng, a, b, draws):
    """`z` read against decoupled copies of this very pair, rather than against row re-pairings."""
    obs = engine.associate(a, b, far=FAR).z
    null = np.array([engine.associate(_decoupled(rng, a), _decoupled(rng, b), far=FAR).z
                     for _ in range(draws)])
    sd = float(null.std(ddof=1))
    return (obs - float(null.mean())) / sd if sd > 0 else float(0)


def surrogate_spread(records, keys, rng, pairs, draws):
    """Why the correction subtracts signal: what the surrogate's spread is for each kind of pair.

    `_decoupled` keeps a record's own covariance exactly -- that is the property that makes it
    the right null for the admission margin. But two records of the same activity are similar in
    exactly that, and the similarity survives phase randomization. So the surrogate distribution for
    a genuinely related pair is wider than for an unrelated one, and standardizing by its spread
    divides out the thing the read is looking for.
    """
    pool = list(keys)
    related, unrelated = [], []
    for _ in range(pairs):
        q = pool[int(rng.integers(len(pool)))]
        same = [k for k in pool if k[0] == q[0] and k != q]
        diff = [k for k in pool if k[0] != q[0]]
        if not same or not diff:
            continue
        for bucket, c in ((related, same[int(rng.integers(len(same)))]),
                          (unrelated, diff[int(rng.integers(len(diff)))])):
            Q, f = records[q], records[c]
            null = [engine.associate(_decoupled(rng, Q), _decoupled(rng, f), far=FAR).z
                    for _ in range(draws)]
            bucket.append(float(np.std(null, ddof=1)))
    r, u = float(np.mean(related)), float(np.mean(unrelated))
    return {"pairs": len(related),
            "spread_for_pairs_sharing_an_activity": r,
            "spread_for_pairs_that_do_not": u,
            "ratio": r / u if u else float("nan")}


def evaluate(records, keys, *, same, label):
    """`exp19.evaluate`'s question stream, scored with both standardizations side by side."""
    rng = np.random.default_rng(SEED)
    pool = list(keys)
    hits = {"association": 0, "association, corrected null": 0, "cos": 0, "random": 0}
    per_question = {m: [] for m in hits}
    asked = 0

    order = rng.permutation(len(pool))[:QUERIES]
    surrogate = np.random.default_rng(SEED + 1)
    for qi in order:
        q = pool[qi]
        other = 1 - same
        true_pool = [k for k in pool if k[same] == q[same] and k[other] != q[other]]
        false_pool = [k for k in pool if k[same] != q[same]]
        if not true_pool or len(false_pool) < W.N_CANDIDATES - 1:
            continue
        truth = true_pool[int(rng.integers(len(true_pool)))]
        distractors = [false_pool[i] for i in
                       rng.choice(len(false_pool), W.N_CANDIDATES - 1, replace=False)]
        cands = [truth] + distractors
        perm = rng.permutation(W.N_CANDIDATES)
        cands = [cands[i] for i in perm]
        answer = int(np.where(perm == 0)[0][0])

        Q = records[q]
        frames = [records[c] for c in cands]
        picks = {
            "association": int(np.argmax([engine.associate(Q, f, far=FAR).z for f in frames])),
            "association, corrected null": int(
                np.argmax([corrected_z(surrogate, Q, f, DRAWS) for f in frames])),
            "cos": int(np.argmax([W._flat_cos(Q, f) for f in frames])),
            "random": int(rng.integers(W.N_CANDIDATES)),
        }
        for m, pick in picks.items():
            ok = int(pick == answer)
            hits[m] += ok
            per_question[m].append(ok)
        asked += 1

    out = {"task": label, "asked": asked, "chance": 1.0 / W.N_CANDIDATES,
           "scores": {m: hits[m] / asked for m in hits}}
    # every method answered the same questions, so the comparisons are paired
    base = np.array(per_question["association"])
    for m in ("association, corrected null", "cos"):
        d = np.array(per_question[m]) - base
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        out[f"{m} minus association"] = float(d.mean())
        out[f"{m} minus association, in paired standard errors"] = (
            float(d.mean() / se) if se > 0 else 0.0)
    # and the one the paper's claim turns on
    d = np.array(per_question["association, corrected null"]) - np.array(per_question["cos"])
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    out["corrected minus cos"] = float(d.mean())
    out["corrected minus cos, in paired standard errors"] = float(d.mean() / se) if se > 0 else 0.0
    return out


def run():
    started = time.time()
    path, digest = W.fetch(HERE.parents[1] / "corpora")
    # Both arms are inside the regime here. Removing the corpus mean drops the autocorrelation on
    # weather and Beijing; on this corpus it does not -- `exp19` reads +0.667 raw and +0.708 with
    # the mean removed -- so both are re-scored rather than only the raw one.
    print(f"{QUERIES} queries, {W.N_CANDIDATES} candidates, {DRAWS} surrogate draws per pair")

    tasks = []
    built = {}
    sampler = np.random.default_rng(SEED)
    for arm, strip in (("raw", False), ("corpus mean removed", True)):
        records, keys, width = W.build(path, remove_corpus_mean=strip)
        built[arm] = (records, keys)
        pick = sampler.permutation(len(keys))[:40]
        rho = float(np.mean([_row_autocorrelation(records[keys[int(i)]]) for i in pick]))
        print(f"\n=== {arm} === {len(keys)} segments, {width} channels, "
              f"{records[keys[0]].shape[0]} rows, row autocorrelation {rho:+.3f}\n")
        for same, label in ((0, "same activity"), (1, "same subject")):
            r = evaluate(records, keys, same=same, label=label)
            r["arm"] = arm
            r["row_autocorrelation"] = rho
            tasks.append(r)
            print(f"  {label}  ({r['asked']} questions, chance {r['chance']:.3f})")
            for m, v in r["scores"].items():
                print(f"    {m:<32}{v:>8.3f}")
            k = "association, corrected null minus association"
            print(f"    correcting the null moves the association by {r[k]:+.3f} "
                  f"({r[k + ', in paired standard errors']:+.2f} paired standard errors)")
            print(f"    corrected against cos: {r['corrected minus cos']:+.3f} "
                  f"({r['corrected minus cos, in paired standard errors']:+.2f} paired se)\n")

    # measured on **both** arms and named: reading it off whichever arm the loop happened to leave
    # behind is how a figure ends up attributed to the wrong corpus
    spreads = {}
    print("  the surrogate's spread, by arm")
    for arm, (recs, ks) in built.items():
        sp = surrogate_spread(recs, ks, np.random.default_rng(SEED + 2),
                              10 if SMOKE else 25, 8 if SMOKE else 30)
        spreads[arm] = sp
        print(f"    {arm:<22} sharing an activity {sp['spread_for_pairs_sharing_an_activity']:.2f}"
              f"   not sharing {sp['spread_for_pairs_that_do_not']:.2f}"
              f"   ratio {sp['ratio']:.1f}x")
    print("    wider for exactly the pairs the read has to find, and the correction divides by it")
    print()
    spread = spreads["raw"]

    moved = [t for t in tasks
             if abs(t["association, corrected null minus association, in paired standard errors"])
             >= 1.96]
    out = {"design": {"far": FAR, "seed": SEED, "queries": QUERIES,
                      "candidates": W.N_CANDIDATES, "draws": DRAWS, "smoke": SMOKE},
           "tasks": tasks,
           "tasks_where_correcting_the_null_moved_the_score": len(moved),
           "tasks_scored": len(tasks),
           "surrogate_spread": spread,
           "surrogate_spread_by_arm": spreads,
           "largest_move_in_paired_standard_errors": max(
               abs(t["association, corrected null minus association, in paired standard errors"])
               for t in tasks),
           "seconds": time.time() - started}
    print("=" * 70)
    if moved:
        print(f"  correcting the null moved {len(moved)} of {len(tasks)} tasks by more than the")
        print("  level's own cutoff, so Stage 13's comparison was reading the null in part.")
    else:
        print(f"  correcting the null moved neither task by more than the level's own cutoff")
        print(f"  (largest {out['largest_move_in_paired_standard_errors']:.2f} paired standard "
              f"errors), so Stage 13's")
        print("  comparison stands on its own and the level defect is bounded away from it.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp46.smoke.json" if SMOKE else "exp46.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
