"""When the read misses, where is the right token?

The question
--------------
`test_whether_more_corpus_closes_the_gap_on_counting.py` at 800 sentences: the derived read names
11.3% of held-out spans, counting with a stated backoff 17.5%, and a draw would name
0.06%. On its own smaller split `test_what_a_trained_model_gets_on_prose.py` puts a fitted TransE
under both. So the read beats a draw by a wide margin and loses to a counter — and every one of
those is a top-1 figure.

(Each of those belongs to the experiment named beside it. Quoting them on one line without saying so
is how a figure ends up attributed to a run that never produced it, which has happened twice in this
program.)

**Top-1 says nothing about how it loses.** Two stores can both name 12% and be in completely
different positions:

* the right token is usually second or third, and what fails is the last discrimination between a
  handful of candidates the read has already found;
* the right token is four hundredth, and the read is not close.

The first is a read worth sharpening. The second is a read that needs a different question. They
are told apart by the rank of the right answer, not by the share at rank 1, and nothing in this
program has measured it.

What is measured
------------------
For every held-out span the same three arms score the same candidates, and this records where the
right token lands, not merely whether it landed first:

``top 1, 5, 10, 50``   the share whose right token is in the first k.
``median rank``        the middle position of the right token over every question.
``reciprocal rank``    the mean of 1/rank — the standard summary, which credits a near miss.

All of it beside the same figures for counting, so the comparison is where each arm puts the answer
rather than how often it is first.

Why counting is scored the same way
-------------------------------------
A backoff counter returns one token, so it has no ranking to read. Scored here it is given the
ranking its own counts imply: candidates ordered by how often each followed this ``(a, b)``, then by
how often each followed ``b``, then by how often each appeared. That is the same backoff, read as an
order instead of as a pick, so the two arms are compared on the same quantity and the counter is not
handicapped by being asked for something it was not built to give.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import os
import pathlib
import random
import sys

import numpy as np

import ask
import record
from checking import rank_summary as summarise

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_same_coordinate_can_be_built_faster import built  # noqa: E402
from test_whether_more_corpus_closes_the_gap_on_counting import bodies, split  # noqa: E402

#: how much prose to read. The size the figures this is pointed at were taken at.
SENTENCES = int(os.environ.get("ENTROPTICS_SENTENCES") or 800)

#: how many held-out spans to ask about
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 2000)

#: the depths reported. Declared rather than chosen: 1 is what every existing figure reports and the
#: rest are the decades above it, so the shape of the curve is visible instead of one point of it.
DEPTHS = (1, 5, 10, 50)

SEED = 0


def counting_order(train, a, b, pool):
    """The backoff counter's own ranking of ``pool`` -- the same backoff, read as an order.

    Ordered by how often each candidate followed this ``(a, b)``, then by how often it followed
    ``b`` alone, then by how often it appeared at all. A candidate no count reaches sorts last, and
    ties break on the candidate's own name so the order does not follow the hash seed.
    """
    pair = collections.Counter(c for x, m, c in train if x == a and m == b)
    middle = collections.Counter(c for _, m, c in train if m == b)
    overall = collections.Counter(c for _, _, c in train)
    return sorted(pool, key=lambda t: (-pair[t], -middle[t], -overall[t], t))


def where(order, right):
    """The 1-based position of the right token in an ordering."""
    return order.index(right) + 1 if right in order else len(order) + 1


def main() -> int:
    names, text = bodies()
    if not text:
        print("  No cached prose bodies found under corpora/. Nothing to measure.")
        return 0

    train, held = split(text, sentences=SENTENCES)
    if len(train) < 50 or len(held) < 20:
        print("  Too little prose to split. Nothing to measure.")
        return 0

    engine = Engine.on_statements(train)
    tokens = sorted({t for a, _, c in train for t in (a, c)})
    vectors = built(engine, only=set(tokens))

    print("when the read misses, where is the right token?\n")
    print(f"  {len(names)} bodies, {SENTENCES} sentences, {len(train)} training spans, "
          f"{len(tokens)} tokens")
    print(f"  one sentence in five held out before anything was built\n")

    rng = random.Random(SEED)
    asking = rng.sample(held, min(QUESTIONS, len(held)))

    read_ranks, count_ranks, offered = [], [], 0
    for a, b, c in asking:
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors)
        if landed is None:
            continue
        pool = [t for t in tokens if t in vectors and t != a]
        if c not in pool or len(pool) < 2:
            continue
        offered = len(pool)
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        order = [pool[int(i)] for i in np.argsort(scores)[::-1]]
        read_ranks.append(where(order, c))
        count_ranks.append(where(counting_order(train, a, b, pool), c))

    if not read_ranks:
        print("  No question survived. Nothing to compare.")
        return 0

    arms = {"the read": summarise(read_ranks, offered, DEPTHS),
            "counting": summarise(count_ranks, offered, DEPTHS)}

    print(f"  {len(read_ranks)} questions, {offered} candidates offered on each\n")
    print(f"  {'arm':<12}" + "".join(f"{f'top {k}':>9}" for k in DEPTHS)
          + f"{'median rank':>14}{'1/rank':>9}")
    print("  " + "-" * 71)
    for label, got in arms.items():
        print(f"  {label:<12}" + "".join(f"{got[f'top_{k}']:>9.1%}" for k in DEPTHS)
              + f"{got['median_rank']:>14.0f}{got['mean_reciprocal_rank']:>9.3f}")
    print()

    # **What an ordering carrying nothing would give**, recorded rather than left to the prose to
    # work out. A number a document computes for itself has no producer, and a reader checking the
    # claim against the results finds it in none of them.
    carried_nothing = (offered + 1) / 2

    record.figures(bodies=names, sentences=SENTENCES, questions=QUESTIONS, seed=SEED,
                   train_spans=len(train), tokens=len(tokens), depths=list(DEPTHS),
                   asked=len(read_ranks), candidates=offered,
                   a_carried_nothing_ordering_would_sit_at=carried_nothing, by_arm=arms)

    read, count = arms["the read"], arms["counting"]
    deep = max(DEPTHS)
    print(f"  The read names **{read['top_1']:.1%}** first and has the right token in its first "
          f"**{deep}** of **{offered}**")
    print(f"  **{read[f'top_{deep}']:.1%}** of the time, at a median rank of "
          f"**{read['median_rank']:.0f}**. Counting: **{count['top_1']:.1%}** and "
          f"**{count[f'top_{deep}']:.1%}**, median **{count['median_rank']:.0f}**.")
    print()
    if read[f"top_{deep}"] > count[f"top_{deep}"]:
        print(f"  **The read finds the answer more often than the counter and orders it worse.** "
              f"It puts the")
        print(f"  right token in the first **{deep}** more often, and first less often, so what "
              f"separates them is")
        print(f"  the last discrimination among candidates the read has already reached. **That is "
              f"a read")
        print(f"  worth sharpening rather than replacing**, and what to sharpen is whatever decides "
              f"between")
        print(f"  the top few.")
    elif read["median_rank"] <= count["median_rank"]:
        print(f"  **The read puts the right token nearer the top overall and wins fewer firsts.** "
              f"Median")
        print(f"  **{read['median_rank']:.0f}** against **{count['median_rank']:.0f}**, and "
              f"reciprocal rank **{read['mean_reciprocal_rank']:.3f}** against "
              f"**{count['mean_reciprocal_rank']:.3f}**. So it carries")
        print(f"  the signal and spends it badly at the top, which is a ranking problem rather than "
              f"a")
        print(f"  representation one.")
    else:
        print(f"  **The read is behind at every depth, not only at the first.** Median "
              f"**{read['median_rank']:.0f}** against")
        print(f"  **{count['median_rank']:.0f}**, reciprocal rank **{read['mean_reciprocal_rank']:.3f}** "
              f"against **{count['mean_reciprocal_rank']:.3f}**. **The gap is not the last**")
        print(f"  **discrimination** — it is where the answer sits at all, so sharpening the top of "
              f"the")
        print(f"  ranking cannot close it and what has to change is what the read is asked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
