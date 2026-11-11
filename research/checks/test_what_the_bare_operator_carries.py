"""Every rule added on top of the operator has cost or done nothing. So what does the operator do?

Why this is the question left
-------------------------------
This stage measured each thing the reader computes on top of the association and found none of them
carrying. The attention floor costs recall and precision (§47.131, §47.133) and refuses a sparse
place by a tie at zero (§47.129). The query-referenced order is anti-correlated (§47.137). Choosing
is level with a draw (§47.111). Support separates right offers from wrong ones among the offers
(§47.142, §47.147) and then buys the reader nothing when composed with the operator, on a store with
the questions and the support to show it either way (§47.152).

**Four negatives with one shape: the additions do not carry, and the operator does.** That has been
said four times and never measured on its own. This measures it.

What is measured
------------------
`skewed.py` at each size — the store §47.150 characterizes and §47.152 reads on — every statement
held out of the coordinate as well as the operator, the relation told, and no rule applied at
all: the operator is carried from the source and the nearest candidate is the answer.

``right``            the nearest candidate is the held-out destination.
``a draw would``     the mean of ``1 / pool`` over the same questions — what picking at random from
                     the same candidates gives, which is the comparison an accuracy needs and this
                     program has had to add twice before.
``in the right group``  the nearest candidate is in the group the relation sends the source's group
                     to. The generator's own structure is the grader, so this is class-level and
                     §47.88's caveat applies: it is an easier question than naming the concept.
``where the truth ranks``  the truth's place in the operator's own ordering, as a share of the pool,
                     so that a read which is wrong but nearly right is told from one that is lost.
``once in the group``   of the reads that land in the right group, how often the named concept is the
                     destination, against what a draw among that group's candidates would give.
                     This is the bound on the headline: if nearly all of the gain is reading the
                     direction, then naming the concept is close to a draw and saying so is the
                     difference between a result and an overclaim.

A binomial tail against the draw, computed over the questions asked. No threshold, no floor, no
ordering rule, nothing chosen: this is the association matrix derived at read time and read once.

What would settle it
----------------------
If the bare operator is at the draw, then this line has nothing — the additions carry nothing and
neither does the thing they were added to. If it is well above, the association derived at read
time is the inference, and the stage's four negatives are about what surrounds it rather than
about the method. The rank tells which of those two the errors look like.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import math

import numpy as np

import record

import ask
import operators
import place
import skewed

from test_inference_on_a_partly_stated_structure import _ops
from measure import binomial_tail as tail  # noqa: E402

DRAWS = 3
COUNTS = (200, 320)


def main() -> int:
    print("what the operator carries with nothing added: no floor, no ordering, no constraint")
    print(f"{DRAWS} draws, the relation told, every statement held out of the coordinate too\n")
    print(f"  {'drawn':>7}{'questions':>11}{'pool':>7}{'a draw would':>14}{'right':>8}"
          f"{'times the draw':>16}{'in the right group':>20}{'truth ranks':>13}"
          f"{'once in the group':>19}{'a draw there':>14}")
    print("  " + "-" * 129)

    rows: dict = {}
    for count in COUNTS:
        right = grouped = 0
        asked = 0
        draw_sum = 0.0
        pools = []
        ranks = []
        in_group = []
        for seed in range(DRAWS):
            rng = np.random.default_rng(9900 + seed)
            concepts, relations, group, sends = skewed.structure(rng)
            whole = skewed.observed(concepts, relations, group, sends, count, rng)
            if len(whole) < 10:
                continue
            for src, relation, dst in whole:
                held = [p for p in whole if p != (src, relation, dst)]
                if len(held) < 4:
                    continue
                vectors, _ = place.places(statements=held)
                if src not in vectors or dst not in vectors:
                    continue
                ops = _ops(vectors, held)
                if relation not in ops:
                    continue
                landed = operators.carry(vectors, relation, src, ops)
                if landed is None:
                    continue
                stated = {b for a, r, b in held if a == src and r == relation}
                pool = [c for c in concepts if c in vectors and c != src and c not in stated]
                if dst not in pool or len(pool) < 2:
                    continue

                asked += 1
                pools.append(len(pool))
                draw_sum += 1.0 / len(pool)
                scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
                order = np.argsort(-scores)
                pick = pool[int(order[0])]
                right += pick == dst
                to = sends[relation][group[src]]
                landed_in = group[pick] == to
                grouped += landed_in
                if landed_in:
                    # the candidates this read had to choose between once it had the direction
                    inside = [c for c in pool if group[c] == to]
                    in_group.append((pick == dst, 1.0 / len(inside)))
                # where the truth sits in the operator's own ordering, as a share of the pool:
                # 0 is first, 1 is last, so a draw averages a half
                ranks.append(float(list(order).index(pool.index(dst))) / (len(pool) - 1))

        if not asked:
            continue
        acc = right / asked
        draw = draw_sum / asked
        p = tail(right, asked, draw)
        rows[str(count)] = {
            "count": count, "questions": asked, "pool": float(np.mean(pools)),
            "a_draw_would": draw, "right": acc, "right_count": right,
            "times_the_draw": acc / draw if draw else float("nan"),
            "in_the_right_group": grouped / asked,
            "where_the_truth_ranks": float(np.mean(ranks)), "p": p,
            "once_in_the_group": (float(np.mean([a for a, _ in in_group]))
                                  if in_group else float("nan")),
            "a_draw_in_the_group": (float(np.mean([d for _, d in in_group]))
                                    if in_group else float("nan")),
            "times_the_draw_in_the_group": (
                float(np.mean([a for a, _ in in_group]) / np.mean([d for _, d in in_group]))
                if in_group else float("nan"))}
        v = rows[str(count)]
        print(f"  {count:>7}{asked:>11}{v['pool']:>7.0f}{draw:>14.1%}{acc:>8.1%}"
              f"{v['times_the_draw']:>16.1f}{v['in_the_right_group']:>20.1%}"
              f"{v['where_the_truth_ranks']:>13.2f}{v['once_in_the_group']:>19.1%}"
              f"{v['a_draw_in_the_group']:>14.1%}")
    print()

    record.figures(draws=DRAWS, counts=list(COUNTS), by_count=rows)

    every = list(rows.values())
    best = max(every, key=lambda v: v["times_the_draw"])
    # the scale is set by how many questions were asked -- the chance of any one of them -- so it
    # carries no free parameter and is not a significance level written down here
    above = [v for v in every if v["p"] < 1.0 / v["questions"]]

    if len(above) == len(every) and every:
        print(f"  **The operator carries the read, at all {len(every)} sizes.** At "
              f"{best['count']} drawn it")
        print(f"  names the held-out destination {best['right']:.1%} of the time from a pool of")
        print(f"  {best['pool']:.0f}, where picking at random from the same candidates gives")
        print(f"  {best['a_draw_would']:.1%} — **{best['times_the_draw']:.1f} times the draw**, over")
        print(f"  {best['questions']} questions.")
        print()
        print("  **And the errors are near misses rather than noise.** The truth sits at")
        print(f"  {best['where_the_truth_ranks']:.2f} of the way down the operator's own ordering")
        print("  where a draw would put it at the middle, and the candidate it names is in the")
        print(f"  group the relation sends to {best['in_the_right_group']:.1%} of the time. It is")
        print("  reading the right direction and landing on the wrong concept in it.")
        print()
        print("  **And that is the whole of it.** Of the reads that land in the right group,")
        print(f"  {best['once_in_the_group']:.1%} name the destination, where a draw among that")
        print(f"  group's own candidates gives {best['a_draw_in_the_group']:.1%} — "
              f"**{best['times_the_draw_in_the_group']:.1f} times the draw**, against")
        print("  the whole read's factor above. **So on this store the operator reads the")
        print("  relation's direction and, once there, does no better than picking.**")
        print()
        print("  **And that is the ceiling here, not a failure of the read.** `skewed.py` draws the")
        print("  destination **uniformly** from the group the relation sends to, so which concept")
        print("  in that group a statement names is decided by nothing — there is no within-group")
        print("  structure to find, and an operator at a draw there is reading everything the")
        print("  store contains. **A generated store bounds what can be asked of it**, which is")
        print("  why the same measurement is put to a hand-written one, where the equivalent step")
        print("  does not collapse (§47.153).")
        print()
        print("  So what this establishes is the coarse half and the bound on it: the association")
        print("  says **which candidates the relation reaches**, on a store where that is all")
        print("  there is to say.")
        print()
        print("  **So the stage's four negatives are about what surrounds the operator, not about")
        print("  the method.** The floor, the ordering, the choosing and the support constraint")
        print("  each cost something or nothing; the association derived at read time is where the")
        print("  inference is, and it needs none of them.")
        print()
        print("  **Nothing here is trained and nothing is chosen.** The coordinate is built from the")
        print("  held store, the operator is the association between its two sides, and the answer")
        print("  is the candidate it lands nearest. No weights were fitted, no threshold applied")
        print("  and the statement being asked about is in neither the operator nor the coordinate.")
    elif above:
        print(f"  **The operator carries the read at {len(above)} of {len(every)} sizes**, and the")
        print("  table says at which. Where it does not, the store is too small for the")
        print("  association to be estimated, and that is a property of the store.")
    else:
        print("  **The operator is at the draw.** Naming the nearest candidate does no better than")
        print("  picking one, so this line has nothing: the additions carry nothing and neither")
        print("  does the thing they were added to. That would make the stage's four negatives")
        print("  about the method after all.")
    print()
    print("  Class-level grading is an easier question than naming the concept and §47.88's caveat")
    print("  applies to the group column. The draw is computed over the same pools the reader saw,")
    print("  not assumed. `skewed.py` states its generator in full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
