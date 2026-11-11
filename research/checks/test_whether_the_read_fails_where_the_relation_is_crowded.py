"""Does the read fail where its relation is crowded?

The question
--------------
`test_where_the_right_token_sits_when_the_read_misses.py` finds the derived read putting the right
token at a **median rank of 218 of 1612** — against a counter's **15**, and against **806.5** for
an ordering that carried nothing. So the read is not narrowly beaten at the top; it is close to
uninformative on most questions and right on a few. **A mean over questions like that hides a
split**, and this conditions on the thing most likely to be causing it.

The mechanism, stated so it can be wrong
------------------------------------------
A read applies the operator for the relation, and for prose the relation is the **middle token** of
the span. The operator is the association over every statement that uses that relation. So the
operator for ``the`` is an average over every ``(first, third)`` pair that ever had ``the`` between
them — thousands of unrelated pairs — and an average over thousands of unrelated directions points
nowhere in particular.

If that is what is happening:

* the read should be **good where the relation is rare**, because few pairs went into the operator
  and they have something in common;
* **near-random where it is crowded**, because the operator is a smear;
* and the crowded relations should be **most of the questions**, because common middle tokens are
  common — which is how a read that works on some questions averages to a median rank near the
  middle.

§47.166 measures the cost side of the same fact: on prose spans, with the candidates held, a read
through a relation used **1250** times does **17.9** times the work of one used **4** times. **This measures the accuracy side.** If the split
is not there, the smear is not the mechanism and the loss is somewhere else.

What is varied, and what is not
---------------------------------
Only **how many statements the question's relation has collected**. The same store, the same
coordinate, the same candidate pool and the same held-out split — the questions are bucketed after
the fact by a property of the relation they go through, and nothing is rebuilt per bucket.

The buckets are **decades of use count**, declared rather than fitted, because the count spans three
orders of magnitude and any equal-width split would put almost every question in one bucket.

What would settle it
----------------------
If rank improves monotonically as the relation gets rarer, the smear is the mechanism and **the way
to improve the read is to write the corpus down so its relations are specific** — which is what the
width sweep (§47.167) and the scale ladder are already about. If rank is flat across the buckets,
the operator's crowding is not what is costing the read and the cause is still to be found.
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

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_same_coordinate_can_be_built_faster import built  # noqa: E402
from test_whether_more_corpus_closes_the_gap_on_counting import bodies, split  # noqa: E402
from test_where_the_right_token_sits_when_the_read_misses import (counting_order,  # noqa: E402
                                                                  where)

SENTENCES = int(os.environ.get("ENTROPTICS_SENTENCES") or 800)
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 2000)

#: the edges of the use-count buckets, as decades. The count runs from a handful to thousands, so
#: an equal-width split would put nearly every question in the lowest bucket and measure nothing.
EDGES = (1, 3, 10, 30, 100, 300, 1000)

SEED = 0


def bucket(count):
    """Which decade of use count a relation falls in, as a label that sorts."""
    low = EDGES[0]
    for edge in EDGES[1:]:
        if count < edge:
            return f"{low}-{edge - 1}"
        low = edge
    return f"{low}+"


def summarise(ranks, offered):
    return {"questions": len(ranks), "candidates": offered,
            "top_1": float(np.mean([r == 1 for r in ranks])),
            "top_10": float(np.mean([r <= 10 for r in ranks])),
            "median_rank": float(np.median(ranks)),
            "mean_reciprocal_rank": float(np.mean([1 / r for r in ranks]))}


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
    uses = collections.Counter(m for _, m, _ in train)

    print("does the read fail where its relation is crowded?\n")
    print(f"  {len(names)} bodies, {SENTENCES} sentences, {len(train)} training spans, "
          f"{len(tokens)} tokens, {len(uses)} relations")
    print(f"  the busiest relation holds {max(uses.values())} statements, "
          f"the median {int(np.median(list(uses.values())))}\n")

    rng = random.Random(SEED)
    asking = rng.sample(held, min(QUESTIONS, len(held)))

    by_bucket = collections.defaultdict(lambda: ([], []))
    read_all, count_all, offered = [], [], 0
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
        r_read = where(order, c)
        r_count = where(counting_order(train, a, b, pool), c)
        read_all.append(r_read)
        count_all.append(r_count)
        got = by_bucket[bucket(uses[b])]
        got[0].append(r_read)
        got[1].append(r_count)

    if not read_all:
        print("  No question survived. Nothing to compare.")
        return 0

    order_of = {b: i for i, b in enumerate(
        [f"{EDGES[i]}-{EDGES[i + 1] - 1}" for i in range(len(EDGES) - 1)] + [f"{EDGES[-1]}+"])}
    rows = {}
    print(f"  {'relation holds':<16}{'questions':>11}{'read top 1':>12}{'read median':>13}"
          f"{'read 1/rank':>13}{'count median':>14}")
    print("  " + "-" * 79)
    for label in sorted(by_bucket, key=lambda b: order_of.get(b, 99)):
        reads, counts = by_bucket[label]
        rows[label] = {"the_read": summarise(reads, offered),
                       "counting": summarise(counts, offered)}
        r, k = rows[label]["the_read"], rows[label]["counting"]
        print(f"  {label:<16}{r['questions']:>11}{r['top_1']:>12.1%}{r['median_rank']:>13.0f}"
              f"{r['mean_reciprocal_rank']:>13.3f}{k['median_rank']:>14.0f}", flush=True)
    print()

    whole = {"the_read": summarise(read_all, offered), "counting": summarise(count_all, offered)}
    record.figures(bodies=names, sentences=SENTENCES, questions=QUESTIONS, seed=SEED,
                   train_spans=len(train), tokens=len(tokens), relations=len(uses),
                   busiest_relation=max(uses.values()), edges=list(EDGES),
                   candidates=offered, by_bucket=rows, over_every_question=whole)

    ordered = sorted(by_bucket, key=lambda b: order_of.get(b, 99))
    medians = [rows[b]["the_read"]["median_rank"] for b in ordered]
    counts = [rows[b]["the_read"]["questions"] for b in ordered]
    rarest, busiest = ordered[0], ordered[-1]
    crowded_share = sum(c for b, c in zip(ordered, counts)
                        if order_of.get(b, 99) >= len(order_of) - 2) / sum(counts)

    print(f"  Over every question the read's median rank is "
          f"**{whole['the_read']['median_rank']:.0f}** of **{offered}**, against counting's")
    print(f"  **{whole['counting']['median_rank']:.0f}**. Split by what the relation holds, the "
          f"read's median runs "
          f"**{' -> '.join(f'{m:.0f}' for m in medians)}**")
    print(f"  from **{rarest}** statements to **{busiest}**.")
    print()
    if medians[0] < medians[-1]:
        print(f"  **The read is sharpest where its relation is rare and worst where it is "
              f"crowded.** The")
        print(f"  operator is the association over every statement using that relation, so a middle "
              f"token")
        print(f"  that sat between thousands of unrelated pairs gives an average over thousands of")
        print(f"  unrelated directions - **a smear, and it points nowhere in particular**. "
              f"**{crowded_share:.0%}** of the")
        print(f"  questions go through the two busiest buckets, which is how a read that works on "
              f"some")
        print(f"  questions averages to a median near the middle.")
        print()
        print(f"  **So the way to improve the read is to write the corpus down so its relations "
              f"are")
        print(f"  specific**, not to sharpen the ranking. §47.167's width sweep and the scale "
              f"ladder are")
        print(f"  both moves in that direction, and this says what they are worth: the difference")
        print(f"  between **{medians[-1]:.0f}** and **{medians[0]:.0f}** is what a relation "
              f"stops being crowded buys.")
    else:
        print(f"  **The read is no better where its relation is rare.** The median runs "
              f"**{medians[0]:.0f}** at "
              f"**{rarest}**")
        print(f"  statements and **{medians[-1]:.0f}** at **{busiest}**, so the operator's crowding "
              f"is not what is costing")
        print(f"  the read and the smear is not the mechanism. **What is costing it is still to be "
              f"found**,")
        print(f"  and the next thing to condition on is named by whatever else separates a question "
              f"the")
        print(f"  read answers from one it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
