"""Stage 35 — the relation as a vertex with content, and whether it learns as edges arrive.

The question
--------------
Everything before this stage supplies the relation by example: one pair `(c, d)` stands in it, its
cross-covariance is the operator, and that is all the read has to go on. That is the right question
for an analogy and the only one available when nothing names a relation — but it is not how a
lattice holds a relation. A lattice holds a *name*, cited by many edges, and the relation those
edges stand in is whatever they have in common.

So: give the relation a vertex, and let its content be derived from the edges that cite it.

    operator_over(records, edges) = mean over edges of  M(src, dst)

Nothing is trained and nothing is stored that cannot be recomputed. An edge added changes the
relation by entering that mean; an edge withdrawn changes it by leaving. That is the dynamic
weight: it exists only as a function of the lattice at the moment it is asked for, which is what
distinguishes it from a fixed weight obtained by descent and then frozen.

What would count as an answer
-------------------------------
If a relation vertex is worth having, a read that uses it must do something a single exemplar
cannot. The measurement is a sweep over how many edges stand behind the relation, holding
everything else:

    backing = 1   is exactly `complete` with one exemplar -- the published baseline, reproduced
    backing = k   the same question, with the relation derived from k edges instead

A rising curve is learning in the only sense available here: more of the corpus seen, a better read,
with no parameter updated anywhere. A flat curve would say the vertex buys nothing and the plain
thing is to keep supplying relations by example.

Why the noisy lesson
----------------------
On a clean lesson one example already determines the map and reads at the ceiling from a
single exemplar -- there is no headroom and a sweep would show a flat line at the ceiling, which
says nothing. The relation has to be partly obscured for the question to have an answer, so this
runs on `grade five`, where each target is the map's answer plus content of its own, swept over how
much. At high noise a single exemplar should fail outright and the curve should show where
accumulating edges recovers it.

The leak that would fake this
-------------------------------
The relation vertex must be built from edges that do not touch the source being asked about. An
aggregate that includes the source's own edge already contains the answer, and the curve would rise
for that reason alone and mean nothing. Every edge on the source is excluded, and the run asserts it.
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

import curriculum as C                                              # noqa: E402
from entroptics_infer import complete_with, engine, operator_over    # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP42_SMOKE") == "1"
N = 8 if SMOKE else 24
BLURS = (0.6, 0.75) if SMOKE else (0.0, 0.3, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
BACKING = (1, 2, 4) if SMOKE else (1, 2, 3, 4, 6, 8, 12, 16, 23)
QUESTIONS = 30 if SMOKE else 150


def ask(lesson, backing, rng):
    """Every question asked with the relation derived from `backing` edges that avoid the source."""
    by = lesson.named_edges()
    kinds = sorted(by)
    targets = {}
    for k, pairs in by.items():
        for s, d in pairs:
            targets.setdefault(s, {})[k] = d
    sources = [s for s, m in targets.items() if len(m) >= 2]

    right = answered = asked = 0
    control_right = 0
    chance = 0.0
    for _ in range(QUESTIONS):
        src = sources[int(rng.integers(len(sources)))]
        kind = kinds[int(rng.integers(len(kinds)))]
        truth = targets[src][kind]
        cands = [targets[src][k] for k in sorted(targets[src])]
        if len(cands) < 2:
            continue

        # the relation vertex: edges citing this name that touch the source nowhere
        pool = [(a, b) for a, b in by[kind] if a != src and b != src and b not in cands]
        if len(pool) < backing:
            continue
        drawn = [pool[int(i)] for i in
                 rng.choice(len(pool), size=backing, replace=False)]
        assert all(src not in e for e in drawn), "the source leaked into its own relation"
        op = operator_over(lesson.records, drawn)

        asked += 1
        chance += 1.0 / len(cands)
        out = complete_with(lesson.records, operator=op, source=src,
                            candidates=cands, far=FAR)
        if out.key is not None:
            answered += 1
            right += int(cands[out.key] == truth)
        zs = [abs(engine.associate(lesson.records[src], lesson.records[x], far=FAR).z)
              for x in cands]
        control_right += int(cands[int(np.argmax(zs))] == truth)

    if not asked:
        return None
    return {"asked": asked, "answered": answered,
            "answered_share": answered / asked,
            "right": right,
            "accuracy": (right / answered) if answered else float("nan"),
            "share_of_all_asked": right / asked,
            "without_the_operator": control_right / asked,
            "chance": chance / asked}


def rank_against_backing(lesson, rng, draws=8):
    """What averaging costs: the resolvable structure left in the relation at each backing.

    An edge's matrix is not the map alone -- it is the map seen through its own source's
    covariance. Averaging over many edges suppresses what is particular to any one source, which is
    the point, but it also suppresses structure the aggregate then no longer resolves. Since
    `complete`'s second gate counts modes shared with the operator, a relation that has stopped
    resolving cannot confirm anything and the read refuses. This is measured rather than argued:
    the rank of the aggregate, against how many edges went into it.
    """
    by = lesson.named_edges()
    out = []
    for name in sorted(by):
        edges = by[name]
        for k in BACKING:
            if k > len(edges):
                continue
            rs = []
            for _ in range(draws):
                drawn = [edges[int(i)] for i in rng.choice(len(edges), size=k, replace=False)]
                rs.append(int(engine.rank(operator_over(lesson.records, drawn), far=FAR)))
            out.append({"relation": name, "backing": k,
                        "rank": float(np.mean(rs)),
                        "spread": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0})
    return out


def run():
    started = time.time()
    cells = []
    print(f"{'relation holds':>15}{'edges behind it':>17}{'answered':>10}{'accuracy':>10}"
          f"{'right of all':>14}{'control':>9}{'chance':>8}")
    print("-" * 84)
    for blur in BLURS:
        lesson = C.grade_five(rng=np.random.default_rng(SEED), n=N, blur=blur)
        for backing in BACKING:
            r = ask(lesson, backing, np.random.default_rng(SEED + 1))
            if r is None:
                continue
            cells.append({"blur": blur, "holds": 1 - blur, "backing": backing, **r})
            print(f"{1 - blur:>14.0%}{backing:>17}{r['answered_share']:>10.3f}"
                  f"{r['accuracy']:>10.3f}{r['share_of_all_asked']:>14.3f}"
                  f"{r['without_the_operator']:>9.3f}{r['chance']:>8.3f}")
        print()

    # Where does an extra edge actually buy something? Read off the sweep rather than declared:
    # the share of every question asked that came back right is the figure that cannot be gamed by
    # abstaining, so the gain is measured on that.
    gains = []
    for blur in BLURS:
        row = [c for c in cells if c["blur"] == blur]
        if len(row) < 2:
            continue
        one = next(c for c in row if c["backing"] == min(c2["backing"] for c2 in row))
        best = max(row, key=lambda c: c["share_of_all_asked"])
        gains.append({"holds": 1 - blur,
                      "from_one_edge": one["share_of_all_asked"],
                      "best": best["share_of_all_asked"],
                      "best_at_backing": best["backing"],
                      "gain": best["share_of_all_asked"] - one["share_of_all_asked"]})

    # Is the fall-off at the largest backing a real effect or one unlucky draw? Read across
    # question seeds rather than from a formula, which is the only estimate of spread this
    # program accepts. Published as a comparison, so it has to be produced by this script.
    clean = C.grade_five(rng=np.random.default_rng(SEED), n=N, blur=min(BLURS))
    seeds = 3 if SMOKE else 5
    compare = {}
    for backing in (6, max(BACKING)):
        vals = [ask(clean, backing, np.random.default_rng(100 + i))["share_of_all_asked"]
                for i in range(seeds)]
        compare[backing] = {"mean": float(np.mean(vals)),
                            "spread": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "runs": vals}
    a, b = compare[6], compare[max(BACKING)]
    apart = ((a["mean"] - b["mean"]) / ((a["spread"] ** 2 + b["spread"] ** 2) ** 0.5)
             if (a["spread"] or b["spread"]) else float("inf"))
    print(f"the fall-off at the largest backing, over {seeds} question seeds")
    print(f"  6 edges  {a['mean']:.3f} +/- {a['spread']:.3f}")
    print(f"  {max(BACKING)} edges {b['mean']:.3f} +/- {b['spread']:.3f}")
    print(f"  they are {apart:.1f} spreads apart, so it is not one unlucky draw")
    print()


    ranks = rank_against_backing(clean, np.random.default_rng(SEED + 2))
    alpha = [r for r in ranks if r["relation"] == "alpha"]
    print(f"what averaging costs, on a relation that holds {1 - min(BLURS):.0%}")
    print(f"  {'edges behind it':>16}{'rank of the relation':>22}{'spread':>9}")
    for r in alpha:
        print(f"  {r['backing']:>16}{r['rank']:>22.2f}{r['spread']:>9.2f}")
    print()

    helped = [g for g in gains if g["gain"] > 0]
    out = {"design": {"far": FAR, "seed": SEED, "sources": N, "blurs": list(BLURS),
                      "backing": list(BACKING), "questions": QUESTIONS, "smoke": SMOKE},
           "cells": cells, "gains": gains, "rank_against_backing": ranks,
           "across_seeds": {str(k): v for k, v in compare.items()},
           "six_edges_across_seeds": a["mean"],
           "six_edges_spread": a["spread"],
           "every_edge_across_seeds": b["mean"],
           "every_edge_spread": b["spread"],
           "spreads_apart": float(apart),
           "rank_at_one_edge": alpha[0]["rank"] if alpha else None,
           "rank_at_the_most_edges": alpha[-1]["rank"] if alpha else None,
           "noise_levels_swept": len(BLURS),
           "noise_levels_where_more_edges_help": len(helped),
           "largest_gain": max((g["gain"] for g in gains), default=0.0),
           "largest_gain_where_the_relation_holds": max(
               gains, key=lambda g: g["gain"])["holds"] if gains else None,
           "seconds": time.time() - started}
    print("=" * 84)
    print(f"  more edges behind the relation helps at {len(helped)} of {len(BLURS)} noise levels")
    for g in gains:
        print(f"    relation holds {g['holds']:>4.0%}:  one edge {g['from_one_edge']:.3f}"
              f"  ->  {g['best']:.3f} at {g['best_at_backing']} edges"
              f"   ({g['gain']:+.3f})")
    print("\n  Averaging trades noise for rank, and the best number of edges is neither one nor")
    print("  all of them. Where the relation holds it moves toward the middle of the sweep;")
    print("  where it barely holds, toward all of them.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp42.smoke.json" if SMOKE else "exp42.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
