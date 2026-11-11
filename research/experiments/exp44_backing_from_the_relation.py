"""Stage 37 — the backing chosen by the relation itself, without asking the question.

Where this stands
-------------------
`exp42` gave a relation a vertex whose content is derived from the edges citing it, and measured
that averaging trades noise for rank: the best backing is neither one edge nor all of them and it
moves with the corpus. `exp43` tried to let the read's own refusal choose, and that is ruled out
-- re-asking the question at twenty-three backings and keeping what resolved answered a relation
that is not one 0.670 of the time against a single edge's 0.160. A refusal is calibrated for one
question and that rule asked twenty-three.

The distinction that survives
-------------------------------
`exp43` spent the level because it asked the question repeatedly. A rule that inspects only the
relation vertex -- never the source, never the candidates -- asks the question once, so the
level is spent once, and what it costs can be measured against the single-edge rate on the same
corpus.

The aggregate's own rank is such a property. It is what collapses as edges accumulate, it is what
`complete`'s second gate has nothing to confirm with when it reaches zero, and it is computable from
the relation alone:

``resolves``  the largest backing whose aggregate still resolves at all -- as much of the relation's
    history as can still be read, with the collapse marking where to stop.
``widest``    the backing at which the aggregate resolves the most modes.

Both are compared against using one edge, against using every edge, and against the best fixed
backing chosen in hindsight.

The control, as an arm
------------------------
`exp43`'s falsification check -- *accuracy will fall if this is luck* -- runs on a population where
every relation is real, so it cannot fail, and a positive read through it alone is not a result. So
the control is an arm here rather than an afterthought: the same rules asked for `grade three`'s fourth
name, whose edges join unrelated vertices. The number to beat is the single-edge rate on that same
name, and any rule answering it appreciably more often has bought its gains by spending the level.
It is measured over several seeds, because the question is whether the difference is distinguishable
from zero rather than whether two point estimates differ.
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
SMOKE = os.environ.get("EXP44_SMOKE") == "1"
N = 8 if SMOKE else 24
BLURS = (0.0, 0.9) if SMOKE else (0.0, 0.3, 0.6, 0.75, 0.85, 0.9)
QUESTIONS = 20 if SMOKE else 120
CONTROL_SEEDS = 2 if SMOKE else 6


# --- the rules. Each is handed the aggregate's rank at every backing and nothing else. ---

def one_edge(ranks):
    return 1


def every_edge(ranks):
    return max(ranks)


def resolves(ranks):
    """The largest backing whose aggregate still resolves at all."""
    live = [k for k, r in ranks.items() if r > 0]
    return max(live) if live else 1


def widest(ranks):
    """The backing whose aggregate resolves the most modes; ties go to the larger backing."""
    return max(ranks, key=lambda k: (ranks[k], k))


RULES = {"one edge": one_edge, "every edge": every_edge,
         "largest that resolves": resolves, "widest rank": widest}


def aggregates(lesson, edges):
    """Every nested aggregate over `edges`, in the order the lattice holds them, and its rank.

    The order is the lattice's own rather than a fresh shuffle per question: a relation vertex in a
    running system has a history, and "as much of it as still resolves" has to mean something
    reproducible. Nothing here re-orders the edges to suit the answer.
    """
    ops, ranks = {}, {}
    for k in range(1, len(edges) + 1):
        ops[k] = operator_over(lesson.records, edges[:k])
        ranks[k] = int(engine.rank(ops[k], far=FAR))
    return ops, ranks


def ask(lesson, kind_of_interest, rng, questions, score_truth=True):
    """Ask each rule once per question. The rule sees the aggregate; the question is asked once."""
    by = lesson.named_edges()
    kinds = sorted(by)
    targets = {}
    for k, pairs in by.items():
        for s, d in pairs:
            targets.setdefault(s, {})[k] = d
    sources = [s for s, m in targets.items() if len(m) >= 2] or sorted(targets)

    tal = {n: {"answered": 0, "right": 0, "backing": []} for n in RULES}
    per_backing: dict[int, dict] = {}
    asked = 0
    for _ in range(questions):
        src = sources[int(rng.integers(len(sources)))]
        kind = kind_of_interest or kinds[int(rng.integers(len(kinds)))]
        truth = targets[src][kind]
        cands = [targets[src][k] for k in sorted(targets[src])]
        pool = [(a, b) for a, b in by[kind] if a != src and b not in cands]
        if len(cands) < 2 or not pool:
            continue
        ops, ranks = aggregates(lesson, pool)
        asked += 1

        for name, rule in RULES.items():
            k = rule(ranks)
            out = complete_with(lesson.records, operator=ops[k], source=src,
                                candidates=cands, far=FAR)
            tal[name]["answered"] += int(out.key is not None)
            tal[name]["right"] += int(out.key is not None and cands[out.key] == truth)
            tal[name]["backing"].append(k)
        if score_truth:
            for k, op in ops.items():
                cell = per_backing.setdefault(k, {"answered": 0, "right": 0})
                out = complete_with(lesson.records, operator=op, source=src,
                                    candidates=cands, far=FAR)
                cell["answered"] += int(out.key is not None)
                cell["right"] += int(out.key is not None and cands[out.key] == truth)

    scored = {n: {"asked": asked,
                  "answered_share": v["answered"] / asked if asked else 0.0,
                  "accuracy": (v["right"] / v["answered"]) if v["answered"] else float("nan"),
                  "share_of_all_asked": v["right"] / asked if asked else 0.0,
                  "median_backing": float(np.median(v["backing"])) if v["backing"] else float("nan")}
              for n, v in tal.items()}
    fixed = {k: (v["right"] / asked if asked else 0.0) for k, v in per_backing.items()}
    return scored, fixed, asked


def run():
    started = time.time()
    levels = []
    print("ARM A — relations that are real, obscured by noise\n")
    print(f"{'holds':>7}{'rule':>24}{'answered':>10}{'accuracy':>10}{'right of all':>14}"
          f"{'median k':>10}")
    print("-" * 75)
    for blur in BLURS:
        lesson = C.grade_five(rng=np.random.default_rng(SEED), n=N, blur=blur)
        scored, fixed, asked = ask(lesson, None, np.random.default_rng(SEED + 1), QUESTIONS)
        best_k = max(fixed, key=lambda k: fixed[k])
        row = {"blur": blur, "holds": 1 - blur, "asked": asked, "rules": scored,
               "best_fixed_backing_in_hindsight": best_k,
               "best_fixed_share": fixed[best_k]}
        levels.append(row)
        print(f"{1 - blur:>6.0%} {'best fixed (hindsight)':>24}{'':>10}{'':>10}"
              f"{fixed[best_k]:>14.3f}{best_k:>10}")
        for name in RULES:
            c = scored[name]
            print(f"{'':>7}{name:>24}{c['answered_share']:>10.3f}{c['accuracy']:>10.3f}"
                  f"{c['share_of_all_asked']:>14.3f}{c['median_backing']:>10.0f}")
        print()

    # --- ARM B: the same rules, asked for a relation that is not one -------------------
    print("=" * 75)
    print("ARM B — the control: a name whose edges join unrelated vertices\n")
    per_seed = {n: [] for n in RULES}
    control_rows = []
    for s in range(CONTROL_SEEDS):
        # the corpus is redrawn with the questions, so the spread this is read against is
        # corpus-to-corpus and not one lattice's question sampling -- a within-corpus spread
        # would be the narrower number and would call more rules inadmissible than it should
        lesson = C.grade_three(rng=np.random.default_rng(SEED + 50 + s), n=N)
        scored, _, asked = ask(lesson, "hollow",
                               np.random.default_rng(100 + s), QUESTIONS,
                               score_truth=False)
        control_rows.append({"seed": 100 + s, "asked": asked,
                             **{n: scored[n]["answered_share"] for n in RULES}})
        for n in RULES:
            per_seed[n].append(scored[n]["answered_share"])

    # `exp43`'s ruled-out rule, put through this very test on the same lattices. Without it the
    # control states that these rules pass something, and not that the something has the power to
    # fail anything. It asks the question at every backing and keeps the largest that resolved --
    # which is the whole difference: the level is spent once per backing rather than once.
    retried = []
    for s in range(CONTROL_SEEDS):
        lesson = C.grade_three(rng=np.random.default_rng(SEED + 50 + s), n=N)
        by = lesson.named_edges()
        targets = {}
        for k, pairs in by.items():
            for a, b in pairs:
                targets.setdefault(a, {})[k] = b
        sources = sorted(targets)
        rng = np.random.default_rng(100 + s)
        got = asked_here = 0
        for _ in range(QUESTIONS):
            src = sources[int(rng.integers(len(sources)))]
            cands = [targets[src][k] for k in sorted(targets[src])]
            pool = [(a, b) for a, b in by["hollow"] if a != src and b not in cands]
            if len(cands) < 2 or not pool:
                continue
            asked_here += 1
            live = False
            for k in range(1, len(pool) + 1):
                out = complete_with(lesson.records, operator=operator_over(lesson.records, pool[:k]),
                                    source=src, candidates=cands, far=FAR)
                live = live or out.key is not None
            got += int(live)
        retried.append(got / asked_here if asked_here else 0.0)
    per_seed["retry, as exp43 (ruled out)"] = retried

    base = np.array(per_seed["one edge"])
    print(f"{'rule':>28}{'answered':>10}{'spread':>9}{'vs one edge':>13}{'in spreads':>12}")
    control = {}
    for name in per_seed:
        v = np.array(per_seed[name])
        diff = v - base
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        in_spreads = float(np.mean(diff) / sd) if sd > 0 else (
            0.0 if abs(float(np.mean(diff))) < 1e-12 else float("inf"))
        control[name] = {"answered_share": float(np.mean(v)),
                         "spread": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "above_one_edge": float(np.mean(diff)),
                         "above_one_edge_in_spreads": in_spreads}
        print(f"{name:>28}{np.mean(v):>10.3f}{control[name]['spread']:>9.3f}"
              f"{np.mean(diff):>+13.3f}{in_spreads:>12.2f}")

    # A rule is admissible where the extra answers it gives a relation that is not there are not
    # distinguishable from none, read against the seed-to-seed spread of the difference itself.
    probe = np.arange(float(CONTROL_SEEDS * 2)).reshape(CONTROL_SEEDS, 2)
    cutoff = float(engine.associate(probe, probe[::-1].copy(), far=FAR).cutoff)
    admissible = {n: bool(control[n]["above_one_edge_in_spreads"] < cutoff) for n in per_seed}
    # what this control can and cannot see, stated rather than left to the reader
    sd_of_diff = float(np.std(np.array(per_seed["every edge"]) - base, ddof=1))
    detectable = cutoff * sd_of_diff

    best = max((n for n in RULES if admissible[n] and n != "one edge"),
               key=lambda n: min(lv["rules"][n]["share_of_all_asked"] for lv in levels),
               default=None)
    out = {"design": {"far": FAR, "seed": SEED, "sources": N, "blurs": list(BLURS),
                      "questions": QUESTIONS, "control_seeds": CONTROL_SEEDS, "smoke": SMOKE},
           "levels": levels, "control": control, "control_rows": control_rows,
           "control_cutoff_in_spreads": cutoff,
           "admissible": admissible,
           "rules_admissible": sum(1 for n in RULES if admissible[n]),
           "smallest_inflation_this_control_can_see": detectable,
           "inflation_of_the_ruled_out_retry": control["retry, as exp43 (ruled out)"][
               "above_one_edge"],
           "inflation_of_the_ruled_out_retry_in_spreads": control[
               "retry, as exp43 (ruled out)"]["above_one_edge_in_spreads"],
           "the_ruled_out_retry_is_admissible": admissible["retry, as exp43 (ruled out)"],
           "best_admissible_rule": best,
           "worst_level_of_the_best_rule": (
               min(lv["rules"][best]["share_of_all_asked"] for lv in levels) if best else None),
           "worst_level_of_one_edge": min(lv["rules"]["one edge"]["share_of_all_asked"]
                                          for lv in levels),
           "worst_level_of_every_edge": min(lv["rules"]["every edge"]["share_of_all_asked"]
                                            for lv in levels),
           "levels_where_the_best_rule_beats_one_edge": (
               sum(1 for lv in levels
                   if lv["rules"][best]["share_of_all_asked"]
                   > lv["rules"]["one edge"]["share_of_all_asked"]) if best else 0),
           "seconds": time.time() - started}
    print(f"\n  a difference of {cutoff:.2f} seed-spreads is what this level requires;")
    print("  below it the extra answers on a relation that is not there are not distinguishable")
    print("  from none, and the rule has not spent the level to get its gains.\n")
    out_retry = control["retry, as exp43 (ruled out)"]["above_one_edge"]
    print(f"  the smallest inflation this control can see is {detectable:.3f}; the rule "
          f"`exp43` ruled out")
    print(f"  inflates by {out_retry:.3f}, so the test has the power to fail what it was built "
          f"to fail.")
    print()
    for n in per_seed:
        print(f"    {n:>28}  admissible: {admissible[n]}")
    print(f"\n  best admissible rule: {best}")
    if best:
        print(f"    its worst level: {out['worst_level_of_the_best_rule']:.3f}, against one edge's "
              f"{out['worst_level_of_one_edge']:.3f} and every edge's "
              f"{out['worst_level_of_every_edge']:.3f}")
        print(f"    it beats one edge at {out['levels_where_the_best_rule_beats_one_edge']} "
              f"of {len(levels)} noise levels")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp44.smoke.json" if SMOKE else "exp44.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
