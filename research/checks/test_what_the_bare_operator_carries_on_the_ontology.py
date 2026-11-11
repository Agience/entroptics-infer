"""The same bare-operator measurement, on the store nobody wrote for this question.

Why it has to be asked twice
------------------------------
The companion measures what the operator carries with nothing added, on `skewed.py`. That store was
built in this stage, by me, days after the questions were framed — and a generated store answers the
question its design allows, which is the failure this line keeps meeting. A result about the
method cannot rest only on a store written alongside the hypothesis.

`research/lib/ontology.py` is a hand-written file. Someone arranged it, so it is not *nothing arranged*;
what is true, and what the contrast rests on, is that it was written as the thing the reader reads,
before any of these questions were put to it, so nothing in it was arranged to have the property
under test. That is what cannot be said of a generator written to a specification.

What is measured
------------------
Exactly the companion's first three columns, on that store: every statement held out of the
coordinate as well as the operator, the relation told, no floor, no ordering rule and no
constraint — the operator carried from the source, the nearest candidate named.

``right``            the nearest candidate is the held-out destination.
``a draw would``     the mean of ``1 / pool`` over the same questions, computed on the pools the
                     reader actually saw rather than assumed.
``where the truth ranks``  the truth's place in the operator's own ordering as a share of the pool,
                     so a read that is wrong but close is told from one that is lost.

**There is no group column and there cannot be.** The generated store has a designed class structure
to grade against; this one has no types and no classes, by construction — which is the whole point of
the representation. So the only question askable here is the hard one: did it name the concept.

**But the bound the group column buys has an analogue here, and it is the one that matters.** On the
generated store the operator reads the relation's *direction* almost always and then picks within the
group no better than a draw — so the whole factor is the coarse step. This store has no groups, but
it has the nearest thing: the candidates the relation points into somewhere else in the store. So
the same decomposition is run against those.

``among the supported``   of the reads whose truth is a candidate the relation points into elsewhere,
                     how often the operator names it, against a draw among those candidates
                     rather than the whole pool. If the factor collapses to one here as it does on
                     the generated store, then the read is a coarse step there too and the headline
                     is about which relation, not which concept.

A binomial tail against the draw. Nothing is chosen, nothing is fitted, and the store is the one
already on the record.

What would settle it
----------------------
If the operator is above the draw here too, the companion's result is not the generator's doing and
the association derived at read time carries a read on a store nobody arranged for it. If it is
at the draw, the companion's factor belongs to a store built in this stage and the claim has to be
made about generated structures only — which would be worth knowing, and is why this runs.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import math

import numpy as np

import record

import ask
import operators
import ontology
import place

from test_inference_on_a_partly_stated_structure import _ops
from measure import binomial_tail as tail  # noqa: E402


def main() -> int:
    statements = [s for s in ontology.STATEMENTS if len(s) == 3]
    concepts = sorted(ontology.concepts())
    print("what the operator carries with nothing added, on the hand-written store")
    print(f"{len(statements)} statements, {len(concepts)} concepts, the relation told,")
    print("every statement held out of the coordinate as well as the operator\n")

    right = asked = 0
    draw_sum = 0.0
    pools, ranks = [], []
    among = []
    for src, relation, dst in statements:
        held = [s for s in statements if s != (src, relation, dst)]
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
        # the candidates this relation points into somewhere else -- the coarse step's analogue
        into = {b for a, r, b in held if r == relation}
        supported = [c for c in pool if c in into]

        asked += 1
        pools.append(len(pool))
        draw_sum += 1.0 / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        order = np.argsort(-scores)
        right += pool[int(order[0])] == dst
        ranks.append(float(list(order).index(pool.index(dst))) / (len(pool) - 1))
        if dst in supported and len(supported) > 1:
            sc = ask._along(np.stack([vectors[c] for c in supported]), landed)
            among.append((supported[int(np.argmax(sc))] == dst, 1.0 / len(supported)))

    if not asked:
        print("  No question on this store survives the held-out protocol.")
        return 0

    acc = right / asked
    draw = draw_sum / asked
    p = tail(right, asked, draw)
    rank = float(np.mean(ranks))
    pool = float(np.mean(pools))

    print(f"  {'questions':>11}{'pool':>7}{'a draw would':>14}{'right':>8}{'times the draw':>16}"
          f"{'truth ranks':>13}{'p':>12}")
    print("  " + "-" * 81)
    print(f"  {asked:>11}{pool:>7.0f}{draw:>14.1%}{acc:>8.1%}{acc / draw:>16.1f}"
          f"{rank:>13.2f}{p:>12.2e}")
    print()

    got = float(np.mean([a for a, _ in among])) if among else float("nan")
    got_draw = float(np.mean([d for _, d in among])) if among else float("nan")
    record.figures(statements=len(statements), concepts=len(concepts), questions=asked,
                   pool=pool, a_draw_would=draw, right=acc, right_count=right,
                   times_the_draw=acc / draw, where_the_truth_ranks=rank, p=p,
                   among_the_supported=got, a_draw_among_the_supported=got_draw,
                   times_the_draw_among_the_supported=(got / got_draw if got_draw else float("nan")),
                   questions_among_the_supported=len(among))

    print(f"  {'among the supported':>21}{'questions':>11}{'a draw there':>14}{'named':>8}"
          f"{'times the draw':>16}")
    print("  " + "-" * 70)
    print(f"  {'':>21}{len(among):>11}{got_draw:>14.1%}{got:>8.1%}"
          f"{(got / got_draw if got_draw else float('nan')):>16.1f}")
    print()

    if acc > draw and p < 1.0 / asked:
        print(f"  **The operator carries a read here too**, naming the held-out destination")
        print(f"  {acc:.1%} of the time from a pool of {pool:.0f} where a draw over the same")
        print(f"  candidates gives {draw:.1%} — **{acc / draw:.1f} times the draw** over {asked}")
        print(f"  questions, and the truth sits at {rank:.2f} of the way down the operator's own")
        print("  ordering where a draw would leave it at the middle.")
        print()
        print("  **So the companion's result is not the generator's doing.** The same measurement")
        print("  on a store written before any of these questions were put to it, with no types")
        print("  and no classes to grade against, puts the operator well above a draw on the hard")
        print("  question — naming the concept. **What the store is asked here it was not arranged")
        print("  to answer**, which is what cannot be said of a store built to a specification.")
        print()
        if got_draw and got / got_draw > 1.0:
            print(f"  **And unlike the generated store, the factor does not collapse.** Of the")
            print(f"  {len(among)} questions whose answer the relation points into elsewhere, the")
            print(f"  operator names it {got:.1%} of the time against {got_draw:.1%} for a draw")
            print(f"  among those same candidates — **{got / got_draw:.1f} times** the draw. On the")
            print("  generated store the equivalent step is at a draw, so there the whole factor is")
            print("  reading which relation; **here something survives the coarse step.**")
        else:
            print(f"  **And the factor collapses, as it does on the generated store.** Of the")
            print(f"  {len(among)} questions whose answer the relation points into elsewhere, the")
            print(f"  operator names it {got:.1%} against {got_draw:.1%} for a draw among those")
            print(f"  candidates — **{got / got_draw:.1f} times**. **So the read is a coarse step")
            print("  here too**: the operator says which candidates the relation reaches and does")
            print("  not choose among them, and the headline factor is about the relation rather")
            print("  than the concept.")
        print()
        print("  **Nothing is trained and nothing is chosen.** The coordinate is rebuilt from the")
        print("  held store for every question, the operator is the association between its two")
        print("  sides, and the answer is the candidate it lands nearest. No weights were fitted,")
        print("  no floor applied, and the statement asked about is in neither the operator nor")
        print("  the coordinate.")
    elif acc > draw:
        print(f"  **Above the draw and this store cannot establish it**: {acc:.1%} against")
        print(f"  {draw:.1%} on {asked} questions gives p {p:.4f}, which is the size of the store")
        print("  rather than the size of the effect — the limit this stage met four times.")
    else:
        print(f"  **At the draw.** {acc:.1%} against {draw:.1%} over {asked} questions: the")
        print("  operator names the destination no better than picking from the same candidates.")
        print("  **So the companion's factor belongs to the store built in this stage**, and what")
        print("  the association carries has to be claimed about generated structures only.")
    print()
    print("  This store has no types and no classes, so there is no easier class-level question to")
    print("  fall back on and the number above is the hard one. `ontology.py` is a hand-written")
    print("  file — someone arranged it — but it was written as the thing the reader reads, before")
    print("  these questions were framed, so nothing in it was arranged to have this property.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
