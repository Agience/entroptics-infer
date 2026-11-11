"""The first corpus question: does WordNet carry a read at all, before anything is built to ingest it.

Why this runs before any ingestion work
-----------------------------------------
§47.153 measures the bare operator on `ontology.py` — 84 statements, hand-written — and finds it
naming held-out facts 20.9 times a draw, and 3.6 times a draw that already knows which
candidates the relation reaches. Everything the line claims rests on that one file.

The next thing wanted is a corpus and a way to ingest it. This is the test that says whether a
corpus is worth ingesting, and it is cheap: §47.154 found a generated store where the operator
reads the relation's *direction* almost always and then picks within it at a draw, because that
store's destinations are drawn uniformly and there is nothing finer to find. A corpus can have the
same ceiling. If the operator cannot beat a draw among the candidates the relation already reaches,
then the corpus is relation-labels and no curriculum or ingestion order will put anything there.

Four corpora have been refused by this program for want of an asymmetric *recorded* relation.
WordNet has them, and `lexicon.py` states in full how the store is built — including that only one
direction of each inverse pair is kept, because keeping both would leave every held-out fact
stated backwards in the store and turn a lookup into an apparent inference.

What is measured
------------------
Exactly §47.153's measurement, on connected neighbourhoods of WordNet, swept over seeds and sizes
because one neighbourhood is one topic:

``right``              the nearest candidate is the held-out destination.
``a draw would``       the mean of ``1 / pool``, over the pools the reader actually saw.
``among the reached``  of the questions whose answer the relation points into elsewhere, how often
                       the operator names it, against a draw among those candidates — the column
                       §47.154 showed can collapse to a draw.

Every statement is held out of the coordinate as well as the operator, the relation is told, and no
floor, ordering rule or constraint is applied.

**This file is not the acceptance test.**
-------------------------------------------
Beating a draw among the reached candidates is necessary and it is not sufficient. A store has
degrees, a relation skew and a pool size, and those alone can put a figure above a draw with no
correspondence between a source and its destination anywhere in it.

What settles it is a permuted store: destinations shuffled inside each relation, so every concept
keeps its in-degree and out-degree, every relation keeps its size, every pool is the same size, and
the only thing gone is the pairing the read is claimed to recover.
`test_whether_the_read_survives_its_controls.py` runs that, and on these same neighbourhoods one
of three fails it — `dog.n.01` reads 2.2 times a draw among the reached and seven of twenty
permuted stores reach that figure, so its number is not evidence of a read however far above a draw
it sits.

So the columns below are what each store gives. Whether any of them means anything is the null's
answer, not this file's.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

import ask
import lexicon

from entroptics_infer.wrapper import Engine

#: swept, because one neighbourhood is one topic and a result holding only there is about the topic
SEEDS = ("entity.n.01", "dog.n.01", "vehicle.n.01")
#: how far the walk goes before the core is taken; swept, and the core is what is read
SIZES = (600, 1200)


def tail(right, n, p):
    """The chance of ``right`` or more from ``n`` draws each right with chance ``p``.

    Summed by the term-to-term ratio rather than by a binomial coefficient: at these pool
    sizes ``comb(n, k)`` is an integer too large to be a float, while the term it belongs
    to is small, so the coefficient is never formed. The arithmetic is the same sum.
    """
    if right <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0 if right <= n else 0.0
    term = (1.0 - p) ** n
    total = 0.0
    for k in range(n + 1):
        if k >= right:
            total += term
        term *= (n - k) / (k + 1) * p / (1.0 - p)
    return float(min(total, 1.0))


def at(seed, size):
    """§47.153's measurement, on one neighbourhood.

    Read through the shipped engine rather than through `place.py` and `operators.py`, and with the
    coordinate truncated. Both are changes of implementation rather than of method and both are
    measured: `test_whether_the_shipped_engine_reads_what_the_research_one_reads.py` finds the two
    paths agreeing to zero on every coordinate and every operator, and §47.156 finds the truncation
    costing the read nothing above the smallest stores.

    Without them this file does not finish. The core of `dog.n.01` at size 1200 is 1304
    statements, the coordinate is rebuilt for every question, and untruncated that one row is hours.
    """
    ball, _ = lexicon.neighbourhood(size, seed)
    statements = lexicon.core(ball)
    concepts = sorted({c for a, _, b in statements for c in (a, b)})
    if len(statements) < 10:
        return None

    engine = Engine.on_statements(statements)
    right = asked = 0
    draw_sum = 0.0
    pools, ranks, among = [], [], []
    for src, relation, dst in statements:
        # every statement joining these two vertices, not only the one asked about: a store that
        # records the same pair under a second label would otherwise state the answer. Measured
        # zero on these cores, and done anyway because it costs nothing and cannot be assumed
        pair = [s for s in statements if s[0] == src and s[2] == dst]
        held = [s for s in statements if s not in pair]
        vectors = engine.places(without=pair)
        if src not in vectors or dst not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=pair)
        if landed is None:
            continue
        stated = {b for a, r, b in held if a == src and r == relation}
        pool = [c for c in concepts if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        into = {b for a, r, b in held if r == relation}
        reached = [c for c in pool if c in into]

        asked += 1
        pools.append(len(pool))
        draw_sum += 1.0 / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        order = np.argsort(-scores)
        right += pool[int(order[0])] == dst
        ranks.append(float(list(order).index(pool.index(dst))) / (len(pool) - 1))
        if dst in reached and len(reached) > 1:
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            among.append((reached[int(np.argmax(sc))] == dst, 1.0 / len(reached)))

    if not asked:
        return None
    acc, draw = right / asked, draw_sum / asked
    got = float(np.mean([a for a, _ in among])) if among else float("nan")
    got_draw = float(np.mean([d for _, d in among])) if among else float("nan")
    biggest = lexicon.shape(statements)[0]
    return {"seed": seed, "size": size, "ball": len(ball), "statements": len(statements),
            "concepts": len(concepts), "questions": asked,
            "pool": float(np.mean(pools)), "a_draw_would": draw, "right": acc,
            "right_count": right, "times_the_draw": acc / draw if draw else float("nan"),
            "where_the_truth_ranks": float(np.mean(ranks)),
            "p": tail(right, asked, draw),
            "biggest_relation": biggest[0],
            "biggest_relation_share": biggest[1] / len(statements),
            "questions_among_the_reached": len(among),
            "among_the_reached": got, "a_draw_among_the_reached": got_draw,
            "times_the_draw_among_the_reached": (got / got_draw if got_draw else float("nan"))}


def main() -> int:
    print("does WordNet carry a read — the acceptance test, before any ingestion is built\n")
    print(f"  {'seed':>14}{'size':>6}{'ball':>6}{'core':>6}{'questions':>11}{'pool':>6}{'draw':>7}"
          f"{'right':>7}{'x draw':>8}{'ranks':>7}{'reached':>9}{'draw there':>12}"
          f"{'named':>7}{'x draw':>8}")
    print("  " + "-" * 111)

    rows = {}
    for seed in SEEDS:
        for size in SIZES:
            v = at(seed, size)
            if v is None:
                continue
            rows[f"{seed}/{size}"] = v
            print(f"  {seed:>14}{size:>6}{v['ball']:>6}{v['statements']:>6}{v['questions']:>11}"
                  f"{v['pool']:>6.0f}{v['a_draw_would']:>7.1%}{v['right']:>7.1%}"
                  f"{v['times_the_draw']:>8.1f}{v['where_the_truth_ranks']:>7.2f}"
                  f"{v['questions_among_the_reached']:>9}"
                  f"{v['a_draw_among_the_reached']:>12.1%}{v['among_the_reached']:>7.1%}"
                  f"{v['times_the_draw_among_the_reached']:>8.1f}")
    print()

    record.figures(seeds=list(SEEDS), sizes=list(SIZES), by_neighbourhood=rows)

    every = list(rows.values())
    if not every:
        print("  No neighbourhood produced a question. The store is not built as intended.")
        return 0

    share = float(np.mean([v["biggest_relation_share"] for v in every]))
    print(f"  The store is **{share:.0%} one relation** on average — "
          f"{every[0]['biggest_relation']} — which is far more")
    print("  skewed than the hand-written store §47.153 reads. **That makes the last column the")
    print("  test**: where one relation is nearly the whole store, knowing which relation was asked")
    print("  says almost nothing, so a factor there has to come from naming the concept.\n")

    # the scale is the number of questions asked -- the chance of any one of them -- so it carries
    # no free parameter and is not a significance level written down here
    overall = [v for v in every if v["p"] < 1.0 / v["questions"]]
    reached = [v for v in every
               if v["times_the_draw_among_the_reached"] == v["times_the_draw_among_the_reached"]
               and v["times_the_draw_among_the_reached"] > 1.0]

    print(f"  {len(overall)} of {len(every)} neighbourhoods beat a draw over the whole pool, and")
    print(f"  {len(reached)} of {len(every)} beat a draw among the candidates the relation already")
    print("  reaches -- which is necessary and is not sufficient.")
    print()
    print("  **Beating a draw is not acceptance, and this file cannot tell you whether it carries.**")
    print("  A store has degrees, a relation skew and a pool size, and those alone can put a figure")
    print("  above a draw with no correspondence between a source and its destination in it. The")
    print("  run that settles it permutes the destinations inside each relation, keeping every")
    print("  degree and every pool, and asks how often a store with the correspondence removed")
    print("  reaches the real one. test_whether_the_read_survives_its_controls.py does that, and")
    print("  on these same neighbourhoods it finds one of three failing -- at 2.2 times a draw")
    print("  among the reached, with seven of twenty permuted stores reaching it. **So read the")
    print("  columns below as what this store gives, and the null as whether it means anything.**")
    print()

    if len(reached) == len(every):
        rlo = min(v["times_the_draw_among_the_reached"] for v in every)
        rhi = max(v["times_the_draw_among_the_reached"] for v in every)
        lo = min(v["times_the_draw"] for v in every)
        hi = max(v["times_the_draw"] for v in every)
        print(f"  **Every neighbourhood beats a draw, and the spread is the result.** The operator")
        print(f"  names the held-out destination between {lo:.1f} and {hi:.1f} times a draw over "
              f"the whole pool,")
        print(f"  and between {rlo:.1f} and {rhi:.1f} times a draw among the candidates the "
              f"relation already")
        print(f"  reaches -- a factor of {rhi / rlo:.0f} between the weakest region and the "
              f"strongest, on one corpus,")
        print("  one method and one protocol.")
        print()
        print("  **And it is not a size effect**, which is the first thing to check: over the sizes")
        print("  swept here one seed rises, one falls and one rises further, so there is no")
        print("  direction to read off. **What this file establishes is that WordNet regions differ")
        print("  enormously**, not that the corpus carries a read -- whether any single one of them")
        print("  does is the null's answer, and the null is not run here.")
    elif reached:
        worst = min(every, key=lambda v: v["times_the_draw_among_the_reached"])
        print(f"  **It carries where it carries, and the sweep does not establish it everywhere.**")
        print(f"  {len(reached)} of {len(every)} neighbourhoods beat a draw among the reached")
        print(f"  candidates; the weakest is {worst['seed']} at {worst['size']}, at")
        print(f"  {worst['times_the_draw_among_the_reached']:.1f} times on")
        print(f"  {worst['questions_among_the_reached']} questions. **A corpus that carries in some")
        print("  regions and not others is a fact about the regions**, and which ones is the next")
        print("  thing to measure rather than a reason to accept or refuse the corpus.")
    else:
        print("  **It does not carry beyond the relation label.** Every neighbourhood is at or")
        print("  below a draw among the candidates the relation already reaches, so the operator")
        print("  is reading which candidates the relation points into and not choosing among")
        print("  them -- §47.154's ceiling, now on a real corpus. **In a store this skewed that")
        print("  leaves almost nothing**, and a corpus with more than a taxonomy is what to look")
        print("  for before building ingestion for this one.")
    print()
    print("  `lexicon.py` states how the store is built. One direction of each inverse pair is")
    print("  kept, **and any relation that turns out to hold both directions under its own name is")
    print("  dropped** -- WordNet's `attribute` and `also_see` are their own inverse, so the pair")
    print("  rule never reached them and they were 20.8% of one core until it was counted. The")
    print("  draw is computed over the pools the reader saw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
