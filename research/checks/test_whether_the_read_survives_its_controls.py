"""The result put beside the two runs that would explain it away.

Why this exists
-----------------
§47.153 and the WordNet acceptance test each report one arm: the operator, derived at read time from
a store the asked-about statement has been removed from, names the held-out destination far above a
draw. A single arm cannot distinguish a read from a pipeline artifact. Anyone handed this result
is entitled to ask two questions before believing it, and neither is answerable by argument:

``is it a lookup``      if the asked-about statement were still in the material the coordinate and
                        the operator are built from, the answer would be stated rather than inferred.
                        The protocol says it is removed. This arm puts it back and measures what
                        that is worth. If leaving it in scores the same as holding it out, then
                        holding it out was never doing anything and the protocol is decoration.

``is it the shape``     a store has degrees, a relation skew, and a pool size, and any of those could
                        produce a number above a draw without any correspondence between a source and
                        its destination. This arm destroys the correspondence and keeps everything
                        else: within each relation the destinations are permuted, so every concept
                        keeps its in-degree and its out-degree, every relation keeps its size, the
                        pools are the same size, and the only thing gone is which source goes to
                        which destination. If the read survives that, the read is reading the shape.

What the permutation preserves, stated, because a surrogate that destroys the wrong thing proves
nothing: relation sizes, per-concept in-degree within a relation, per-concept out-degree, the
concept set, and therefore the pool the reader chooses from. What it destroys is exactly the pairing
the claim is about.

**And there is a condition on that, which these stores meet and others do not.** The shuffle is done
inside each relation independently. Where a store records the same pair of concepts under two
relations, that dissolves the doubling — the permuted store ends up with more distinct edges than the
real one, which is a different graph, and a claim compared against it is compared against something
that differs by more than the correspondence. Measured on the WordNet cores read here: every pair
carries exactly one relation, at all three seeds, so an independent shuffle is sound. On a store
where that is not true it is not, and `test_whether_a_real_lattice_carries_a_read.py` permutes whole
label-bundles for exactly that reason. Count the labels per pair before trusting either.

What each arm should do if the claim is true
----------------------------------------------
    held out    above a draw -- the claim.
    stated      at or above the held-out arm. This is a lookup and the number is a ceiling, not a
                result. `hold out of the representation too` records a case where leaving a fact in
                inflated every score 2.4 times; the size of the gap here is what the protocol buys.
    shuffled    at a draw. This is the one that can refute the claim. A shuffled store has no
                correspondence to find, so an operator above a draw on it is reading something about
                the pipeline and every other number here is in doubt.

No level is written down. Each arm is compared to the draw over the pools that arm's reader actually
saw, and the shuffled arm is drawn several times so its spread is measured rather than assumed.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import random
import time

import numpy as np

import record

import ask
import lexicon
import operators
import place

from test_whether_wordnet_carries_a_read import tail

from test_inference_on_a_partly_stated_structure import _ops

#: one topic is not a corpus, so the same sweep the acceptance test uses
SEEDS = ("entity.n.01", "dog.n.01", "vehicle.n.01")
#: one size, because this test is about the arms and not about size
SIZE = 600
#: How many times the destinations are permuted. **This sets the resolution of the whole test**:
#: the claim is compared to how many permuted stores reached it, so with ``d`` draws the smallest
#: share obtainable is ``1 / (d + 1)`` and nothing finer can be said however the numbers fall. At
#: three draws that floor is 0.25, too coarse to support a verdict at all.
DRAWS = 20


def permute_destinations(statements, rng, *, repairs=200):
    """The same store with the correspondence gone and every degree kept exactly.

    Within each relation the destinations are permuted among that relation's statements. Each
    relation keeps its size, each concept keeps how many times it is a destination of that relation
    and how many times it is a source, and the concept set is unchanged. What no longer holds is
    which source reaches which destination -- which is what the read is claimed to recover.

    **Two things a plain shuffle produces are not statements any store makes**: a concept related to
    itself, and the same statement written twice. Dropping them would shrink the store and break the
    degrees the surrogate exists to keep, so instead each one is repaired by swapping its destination
    with another statement of the same relation -- which is another permutation of the same multiset,
    so every degree still holds. A store that cannot be repaired in ``repairs`` swaps is returned as
    it stands and its size is reported, because silently returning a different store is the failure
    this whole file is about.

    **What it cannot remove**, and the caller is told so: where a relation sends many sources to one
    destination, permuting the destinations re-creates some of the original pairs by arithmetic. Those
    survivors are real structure left inside the null, so the null is a conservative bar -- it
    holds more of the store than a clean shuffle would, and the claim has to clear it anyway.
    """
    by_relation = collections.defaultdict(list)
    for i, (_, r, _) in enumerate(statements):
        by_relation[r].append(i)
    out = list(statements)
    for r, idx in by_relation.items():
        dsts = [statements[i][2] for i in idx]
        rng.shuffle(dsts)
        for i, d in zip(idx, dsts):
            out[i] = (statements[i][0], r, d)
        for _ in range(repairs):
            seen = collections.Counter(out[i] for i in idx)
            bad = [i for i in idx if out[i][0] == out[i][2] or seen[out[i]] > 1]
            if not bad:
                break
            i = rng.choice(bad)
            j = rng.choice(idx)
            (a, _, d), (c, _, e) = out[i], out[j]
            out[i], out[j] = (a, r, e), (c, r, d)
    return tuple(out)


def measure(statements, *, hold_out):
    """The acceptance measurement on one store, with the asked-about statement in or out.

    ``hold_out`` is the only difference between the two arms. When it is false the coordinate and
    the operator are built from every statement including the one being asked about, which is a
    lookup; the pool is chosen the same way in both arms so the question put is identical.
    """
    concepts = sorted({c for a, _, b in statements for c in (a, b)})
    right = asked = 0
    draw_sum = 0.0
    pools, among = [], []
    for src, relation, dst in statements:
        held = [s for s in statements if s != (src, relation, dst)]
        source = held if hold_out else list(statements)
        vectors, _ = place.places(statements=source)
        if src not in vectors or dst not in vectors:
            continue
        ops = _ops(vectors, source)
        if relation not in ops:
            continue
        landed = operators.carry(vectors, relation, src, ops)
        if landed is None:
            continue
        # the pool is built from the held store in both arms, so the question is the same question
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
        right += pool[int(np.argmax(scores))] == dst
        if dst in reached and len(reached) > 1:
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            among.append((reached[int(np.argmax(sc))] == dst, 1.0 / len(reached)))

    if not asked:
        return None
    acc, draw = right / asked, draw_sum / asked
    got = float(np.mean([a for a, _ in among])) if among else float("nan")
    got_draw = float(np.mean([d for _, d in among])) if among else float("nan")
    return {"statements": len(statements), "questions": asked, "right_count": right,
            "pool": float(np.mean(pools)), "a_draw_would": draw, "right": acc,
            "times_the_draw": acc / draw if draw else float("nan"),
            "p": tail(right, asked, draw),
            "questions_among_the_reached": len(among),
            "among_the_reached": got, "a_draw_among_the_reached": got_draw,
            "times_the_draw_among_the_reached": (got / got_draw if got_draw else float("nan"))}


def _line(label, v):
    if v is None:
        return f"  {label:>22}{'--':>10}"
    return (f"  {label:>22}{v['statements']:>7}{v['questions']:>11}{v['pool']:>7.0f}"
            f"{v['a_draw_would']:>7.1%}{v['right']:>8.1%}{v['times_the_draw']:>8.1f}"
            f"{v['p']:>11.1e}{v['questions_among_the_reached']:>9}"
            f"{v['a_draw_among_the_reached']:>12.1%}{v['among_the_reached']:>7.1%}"
            f"{v['times_the_draw_among_the_reached']:>8.1f}")


def main() -> int:
    print("the read beside the two runs that would explain it away\n")
    print(f"  {'arm':>22}{'stmts':>7}{'questions':>11}{'pool':>7}{'draw':>7}{'right':>8}"
          f"{'x draw':>8}{'p':>11}{'reached':>9}{'draw there':>12}{'named':>7}{'x draw':>8}")

    rows = {}
    for seed in SEEDS:
        ball, _ = lexicon.neighbourhood(SIZE, seed)
        core = lexicon.core(ball)
        if len(core) < 10:
            continue
        print("  " + "-" * 109)
        # **The condition the null depends on, checked rather than asserted.** Shuffling inside each
        # relation independently is only sound where a pair of concepts carries one relation; where
        # it carries two, the shuffle dissolves the doubling and the permuted store has more distinct
        # edges than the real one, which is a different graph. The docstring says these stores meet
        # that. This is the run finding out, so a store that stops meeting it says so rather than
        # producing a figure against a null that does not match it.
        under = collections.defaultdict(set)
        for a, r, b in core:
            under[(a, b)].add(r)
        doubled = sum(1 for labels in under.values() if len(labels) > 1)
        if doubled:
            print(f"  {seed} -- **{doubled} of {len(under)} pairs carry more than one relation.** "
                  f"The permutation below shuffles")
            print("  each relation independently, which does not preserve that, so the null it "
                  "builds is not a null")
            print("  for this store. Permute label-bundles instead, as "
                  "`test_whether_a_real_lattice_carries_a_read.py` does.")
            continue
        print(f"  {seed}   ({len(under)} pairs, each under exactly one relation)")

        started = time.time()
        held = measure(core, hold_out=True)
        took = time.time() - started
        print(_line("held out (the claim)", held))

        stated = measure(core, hold_out=False)
        print(_line("stated (a lookup)", stated))

        shuffled = []
        for draw in range(DRAWS):
            rng = random.Random(draw)
            v = measure(permute_destinations(core, rng), hold_out=True)
            if v is not None:
                shuffled.append(v)
            print(_line(f"shuffled, draw {draw}", v))

        rows[seed] = {"held_out": held, "stated": stated, "shuffled": shuffled,
                      "seconds_for_one_arm": took}
    print()

    if not rows:
        print("  No neighbourhood produced a question. The store is not built as intended.")
        return 0

    record.figures(seeds=list(SEEDS), size=SIZE, draws=DRAWS, by_neighbourhood=rows)

    # every figure below is read off the arms; nothing here is a level
    claim = [r["held_out"] for r in rows.values() if r["held_out"]]
    look = [(r["held_out"], r["stated"]) for r in rows.values() if r["held_out"] and r["stated"]]
    null = [v for r in rows.values() for v in r["shuffled"]]

    print("  **What the protocol buys.** Leaving the asked-about statement in the material the")
    print("  coordinate and the operator are built from turns the question into a lookup:")
    for seed, r in rows.items():
        if r["held_out"] and r["stated"]:
            h, s = r["held_out"]["right"], r["stated"]["right"]
            times = f"{s / h:>4.1f} times" if h else "the read names none"
            print(f"    {seed:>14}  held out {h:>6.1%}   stated {s:>6.1%}   {times}")
    gaps = [s["right"] / h["right"] for h, s in look if h["right"]]
    if gaps:
        print(f"  The lookup scores {min(gaps):.1f} to {max(gaps):.1f} times the read over "
              f"{len(gaps)} neighbourhoods,")
        print("  which is what holding the statement out of the coordinate is worth. **A protocol")
        print("  that cost nothing would not have moved it.**")
    print()

    print("  **What the shape alone gives, neighbourhood by neighbourhood.** With the")
    print("  destinations permuted inside each relation -- every degree kept, every relation the")
    print("  same size, the same pools -- each real store is put beside its own permuted ones.")
    print()
    print(f"  {'seed':>14}{'the claim':>11}{'null median':>13}{'null worst':>12}"
          f"{'draws >= claim':>16}{'share':>8}")
    print("  " + "-" * 74)

    # **Each neighbourhood against its own null**, not a global minimum against a global maximum:
    # a claim from one neighbourhood and a null from another are not a comparison, and taking the
    # extremes of each makes the test read on whichever pair happened to be furthest apart.
    floor = 1.0 / (DRAWS + 1)
    verdicts = {}
    for seed, r in rows.items():
        c = r["held_out"]
        if not c or c["times_the_draw_among_the_reached"] != c["times_the_draw_among_the_reached"]:
            continue
        mine = [v["times_the_draw_among_the_reached"] for v in r["shuffled"]
                if v["times_the_draw_among_the_reached"] == v["times_the_draw_among_the_reached"]]
        if not mine:
            continue
        got = c["times_the_draw_among_the_reached"]
        over = sum(1 for m in mine if m >= got)
        share = (over + 1) / (len(mine) + 1)
        verdicts[seed] = {"claim": got, "null_median": float(np.median(mine)),
                          "null_worst": max(mine), "draws": len(mine),
                          "reached_or_beat": over, "share": share}
        print(f"  {seed:>14}{got:>11.1f}{float(np.median(mine)):>13.1f}{max(mine):>12.1f}"
              f"{over:>10} of {len(mine):<3}{share:>8.3f}")
    print()

    record.figures(seeds=list(SEEDS), size=SIZE, draws=DRAWS, by_neighbourhood=rows,
                   against_its_own_null=verdicts)

    print(f"  The share is how often a store with **no correspondence in it** reached the real")
    print(f"  store's figure. With {DRAWS} draws the smallest obtainable is {floor:.3f}, so that is")
    print("  the resolution of this test and no neighbourhood can say anything finer than it.")
    print()

    cleared = [k for k, v in verdicts.items() if v["share"] <= floor]
    failed = [k for k, v in verdicts.items() if v["share"] > floor]
    if verdicts and not failed:
        print(f"  **Every neighbourhood clears its own null.** No permuted store reached the real")
        print(f"  store's figure in {DRAWS} draws anywhere, at the test's own floor of {floor:.3f}.")
        print("  **So what is being read is which source goes to which destination**, because that")
        print("  is the only thing the permutation removed.")
    elif cleared:
        worst = max(failed, key=lambda k: verdicts[k]["share"])
        print(f"  **It clears its null where it clears it, and not everywhere.**")
        print(f"  {', '.join(sorted(cleared))} clear at the floor. {', '.join(sorted(failed))} do")
        print(f"  not: on {worst} a permuted store reached the real store's "
              f"{verdicts[worst]['claim']:.1f} times")
        print(f"  a draw on {verdicts[worst]['reached_or_beat']} of {verdicts[worst]['draws']} "
              f"draws, share {verdicts[worst]['share']:.3f}. **On that neighbourhood the")
        print("  measurement does not distinguish the correspondence from the shape**, and its")
        print("  figure is not evidence of a read however far above a draw it sits.")
    elif verdicts:
        print(f"  **No neighbourhood clears its own null.** A permuted store reaches the real")
        print("  store's figure routinely, so **the measurement is reading something other than")
        print("  the correspondence**, and what that is has to be found before any number from")
        print("  this harness is reported again.")
    else:
        print("  No neighbourhood produced both a claim and a null to compare.")
    print()
    print("  The permutation keeps relation sizes, per-concept in-degree within a relation,")
    print("  out-degree, the concept set and the pools; it removes the pairing alone. Each arm's")
    print("  draw is computed over the pools that arm's reader saw. `lexicon.py` states how the")
    print("  store is built and only one direction of each inverse pair is kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
