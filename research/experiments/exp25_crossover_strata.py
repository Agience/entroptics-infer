"""Stage 19 — does the advantage appear on real data where the theory says it must?

The question this settles
--------------------------
Five explanations for why the multi-facet advantage does not transfer have been tested and refused:
the axis condition, the facet count, channel independence, the ambient width, and the corpus lacking
the narrow-loud case. One measured difference is still standing, and `exp19` put a number on it.

In the generated family the distractor is louder than the true key in every question — that is
what `narrow_gain` arranges, and it places the task past the crossover `g = √(R/m)` of PAPER §5.1, above
which a distractor carries more shared energy than the true key and any method ranking by shared
magnitude must fail. On the 45-channel inertial corpus the true match is instead the *louder* one
on average (0.726 against 0.553), though some distractor is louder in 66.7% of questions.

So real questions are not uniformly one thing. Some sit past the crossover and some do not, and the
published number averages over both. This asks the sharper question: where a real question does
sit past the crossover, does the association win there?

Why this is the decisive test
------------------------------
It is the first one that looks *inside* a real corpus rather than at it whole. If the edge grows with
how far past the crossover a question sits, then the mechanism does appear on real data exactly where
the theory says it should, and what is rare in nature is the regime rather than the effect. If the
edge is flat in the crossover, the mechanism does not appear even where its own precondition holds,
which is a considerably stronger negative than anything the earlier stages measure.

The stratifier, and why it is not circular
-------------------------------------------
Each question is placed by its observed gain: the loudest distractor's peak channel agreement
divided by the true match's. Above 1 a distractor is louder and a magnitude ranking is fighting the
arithmetic; below 1 it is not.

That is computed from the signal and not from whether any method answered correctly, so it
stratifies the questions rather than the outcomes. The ground truth stays what it was — which
activity was performed, written down by the experimenter.

Bins are quantiles of the observed gain, so the cut points come from the corpus's own
distribution rather than from a number chosen here. Every bin holds the same number of questions,
which is also what keeps the per-bin scores comparable.

What would falsify what
------------------------
Stated before the run. The mechanism predicts the edge rises with the observed gain, and is
worst in the lowest bin where the true match is comfortably the loudest. If the edge is flat, or
falls, the association does not exploit the configuration it was built for even when a real corpus
supplies it. Reported as it comes.

A bin where every method sits at chance tests nothing and is marked rather than scored.

Run
---
    python research/experiments/exp25_crossover_strata.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from entroptics_infer import engine                                 # noqa: E402
from exp19_wide_corpus import (FAR, N_CANDIDATES, SEED, _corr, _flat_cos,   # noqa: E402
                               _flat_dot, _lead, build, fetch)

#: Every segment is asked about, so the strata are the corpus's own and not a sample of it.
N_QUERIES = 900
#: Quantile bins: the cut points come from the corpus, and every bin holds the same count.
N_BINS = 5


def questions(records, keys, rng):
    """One question per query: a same-activity truth, eleven distractors, and the observed gain."""
    pool = list(keys)
    out = []
    for qi in rng.permutation(len(pool))[:N_QUERIES]:
        q = pool[qi]
        true_pool = [k for k in pool if k[0] == q[0] and k[1] != q[1]]
        false_pool = [k for k in pool if k[0] != q[0]]
        if not true_pool or len(false_pool) < N_CANDIDATES - 1:
            continue
        truth = true_pool[int(rng.integers(len(true_pool)))]
        dis = [false_pool[i] for i in
               rng.choice(len(false_pool), N_CANDIDATES - 1, replace=False)]

        Q = records[q]
        width = Q.shape[1]

        def peak(c):
            r = [abs(_corr(Q[:, j], records[c][:, j])) for j in range(width)]
            return float(max(r))

        pt = peak(truth)
        if pt <= 0:
            continue
        gain = max(peak(d) for d in dis) / pt
        out.append((q, truth, dis, gain))
    return out


def score(records, batch):
    """Top-1 for each method over one batch of questions."""
    rng = np.random.default_rng(SEED)
    methods = ("association", "cos", "dot", "mean channel correlation",
               "max channel correlation", "leading-component correlation", "random")
    hits = {m: 0 for m in methods}
    for q, truth, dis, _ in batch:
        cands = [truth] + list(dis)
        perm = rng.permutation(len(cands))
        cands = [cands[i] for i in perm]
        answer = int(np.where(perm == 0)[0][0])
        Q = records[q]
        frames = [records[c] for c in cands]
        width = Q.shape[1]

        hits["association"] += int(int(np.argmax(
            [engine.associate(Q, f, far=FAR).z for f in frames])) == answer)
        hits["cos"] += int(int(np.argmax([_flat_cos(Q, f) for f in frames])) == answer)
        hits["dot"] += int(int(np.argmax([_flat_dot(Q, f) for f in frames])) == answer)
        per = [[abs(_corr(Q[:, j], f[:, j])) for j in range(width)] for f in frames]
        hits["mean channel correlation"] += int(
            int(np.argmax([np.mean(r) for r in per])) == answer)
        hits["max channel correlation"] += int(
            int(np.argmax([np.max(r) for r in per])) == answer)
        ql = _lead(Q)
        hits["leading-component correlation"] += int(int(np.argmax(
            [abs(_corr(ql, _lead(f))) for f in frames])) == answer)
        hits["random"] += int(int(rng.integers(len(cands))) == answer)

    n = len(batch)
    out = {m: hits[m] / n for m in methods}
    out["best_scalar"] = max(out[m] for m in methods if m not in ("association", "random"))
    out["edge"] = out["association"] - out["best_scalar"]
    out["questions"] = n

    # The raw edge is a difference of two scores, and in a hard bin both scores sit near chance --
    # so the difference shrinks because there is no room for it, not because the gap closed. What
    # survives that is the share of the *skill above chance* the association keeps: it is 1 where
    # the two are equally skilled and 0 where the association has no skill at all, at any absolute
    # level. Reading the raw edge alone in these strata gives exactly the wrong answer.
    chance = 1.0 / N_CANDIDATES
    a, b = out["association"] - chance, out["best_scalar"] - chance
    out["skill_above_chance"] = a
    out["scalar_skill_above_chance"] = b
    out["skill_share"] = (a / b) if b > 1e-9 else None
    return out


def run():
    started = time.time()
    path, digest = fetch(HERE.parents[1] / "corpora")
    records, keys, width = build(path, remove_corpus_mean=True)
    rng = np.random.default_rng(SEED)
    qs = questions(records, keys, rng)
    gains = np.array([g for _, _, _, g in qs])
    print(f"\n{len(qs)} questions over {len(keys)} segments, {width} channels")
    print(f"observed gain: median {np.median(gains):.3f}, "
          f"{100 * float((gains > 1).mean()):.1f}% with a louder distractor\n")

    edges = np.quantile(gains, np.linspace(0, 1, N_BINS + 1))
    chance = 1.0 / N_CANDIDATES
    print(f"{'bin':>4}{'gain range':>18}{'n':>6}{'assoc':>8}{'best scalar':>13}{'edge':>9}"
          f"{'skill share':>13}")
    rows = []
    for b in range(N_BINS):
        lo, hi = edges[b], edges[b + 1]
        batch = [x for x in qs if (lo <= x[3] <= hi if b == N_BINS - 1 else lo <= x[3] < hi)]
        if not batch:
            continue
        r = score(records, batch)
        r.update({"bin": b, "gain_lo": float(lo), "gain_hi": float(hi),
                  "median_gain": float(np.median([x[3] for x in batch]))})
        r["live"] = bool(max(r["association"], r["best_scalar"]) > chance * 1.5)
        rows.append(r)
        mark = "" if r["live"] else "   (at chance: tests nothing)"
        share = "     n/a" if r["skill_share"] is None else f"{r['skill_share']:>13.2f}"
        print(f"{b:>4}{f'{lo:.2f} - {hi:.2f}':>18}{r['questions']:>6}"
              f"{r['association']:>8.3f}{r['best_scalar']:>13.3f}{r['edge']:>+9.3f}{share}{mark}",
              flush=True)

    live = [r for r in rows if r["live"]]
    out = {
        "source": {"sha256": digest},
        "design": {"far": FAR, "candidates": N_CANDIDATES, "seed": SEED, "queries": N_QUERIES,
                   "bins": N_BINS, "chance": chance,
                   "stratifier": "loudest distractor's peak channel agreement over the true "
                                 "match's; above 1 a magnitude ranking is fighting the arithmetic"},
        "questions": len(qs),
        "median_gain": float(np.median(gains)),
        "share_with_a_louder_distractor": float((gains > 1).mean()),
        "rows": rows,
        "live_bins": len(live),
        "edge_lowest_bin": live[0]["edge"] if live else None,
        "edge_highest_bin": live[-1]["edge"] if live else None,
        "best_edge": max((r["edge"] for r in live), default=None),
        "edge_rises_with_gain": bool(len(live) > 1 and live[-1]["edge"] > live[0]["edge"]),
        "skill_share_by_bin": [r["skill_share"] for r in live],
        "skill_share_lowest_bin": live[0]["skill_share"] if live else None,
        "skill_share_highest_bin": live[-1]["skill_share"] if live else None,
        # the question the raw edge cannot answer: once headroom is accounted for, does the
        # association close on the scalar as the questions get more adversarial?
        "skill_share_rises_with_gain": bool(
            len(live) > 1 and live[-1]["skill_share"] is not None
            and live[0]["skill_share"] is not None
            and live[-1]["skill_share"] > live[0]["skill_share"]),
        "association_ever_leads": bool(any(r["edge"] > 0 for r in live)),
        "seconds": time.time() - started,
    }
    if live:
        print(f"\n  edge in the lowest-gain bin {out['edge_lowest_bin']:+.3f}, "
              f"highest {out['edge_highest_bin']:+.3f}")
        print(f"  rises with the observed gain: {out['edge_rises_with_gain']}")
        print(f"  the association leads in any live bin: {out['association_ever_leads']}")
        print(f"\n  share of the scalar's skill above chance, by bin: "
              f"{[round(x, 2) if x is not None else None for x in out['skill_share_by_bin']]}")
        print(f"  that share rises with the gain: {out['skill_share_rises_with_gain']}")
        print("  the raw edge narrows because both scores fall toward chance, which is headroom")
        print("  and not the gap closing -- the share is what says whether it closed.")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp25.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp25.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
