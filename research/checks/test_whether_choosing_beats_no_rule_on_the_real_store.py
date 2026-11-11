"""Choosing reads below a relation drawn at random. That has only been measured where I built it.

The claim and where it comes from
-----------------------------------
§47.72 is the blocker this line keeps meeting: told which relation acts the read reaches an unstated
pair 77%, 99% and 100% of the time, choosing for itself 7%, 10% and 13%, and
a relation drawn with no rule at all reaches 33%, 38% and 44%. A rule that scores
below no rule is not weak, it is pointing away from the answer, and that is what makes the choosing a
defect rather than a shortfall. §47.83 finds the same on a second generated structure.

**Both of those stores are ones I generated.** §47.7 measures told against choosing on the ontology —
25% to 9% — but nothing else puts the *random* arm to this store, so "worse than no rule"
is otherwise untested where the statements were not arranged. A generated store answers the question
its design allows, and the design here fixes how many relations there are and how alike they are,
which is exactly what a random draw is scored against.

What is measured
------------------
Every statement held out in turn, the coordinate rebuilt without it, the candidates the concepts this
question does not already have an answer among — §47.109's protocol. Four arms on the same questions:

``told``               the question's own relation is applied. The reference every other figure in
                       this line is quoted against.
``choosing``           every relation applied, the highest-scoring offer taken. `ask.walk`'s rule
                       when the query names no relation.
``at random``          one relation drawn with no rule at all, the same draw for every arm's
                       question. The null the *choosing* needs.
``the best on offer``  whether any relation's offer is the answer. An oracle, and the ceiling for
                       any rule that picks among the offers.

Scored on the held-out answer itself, because that is the only right answer this store has. Choosing
against random is paired on the same questions and tested exactly, which is what §47.80 says these
comparisons have to be.
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
import ontology
import operators
import place

from measure import mcnemar


def main() -> int:
    statements = [p for p in ontology.STATEMENTS if len(p) == 3]
    concepts = list(ontology.concepts())
    print(f"the real store: {len(statements)} statements, one held out at a time")
    print("the arms §47.72 compares on a generated structure, on this one\n")

    arms = ("told", "the best on offer", "choosing", "at random")
    got = {a: {} for a in arms}
    chose_true, at_random_true, floors, relations = {}, {}, [], []
    for i, (src, relation, dst) in enumerate(statements):
        drop = [j for j, p in enumerate(ontology.STATEMENTS) if p == (src, relation, dst)]
        held = [p for j, p in enumerate(ontology.STATEMENTS) if j not in set(drop)]
        vectors, _ = place.places(drop=drop)
        if src not in vectors or relation not in vectors or dst not in vectors:
            continue
        ops = operators.all_operators(vectors, statements=held)
        if relation not in ops:
            continue

        stated: dict = {}
        for p in held:
            if len(p) == 3:
                stated.setdefault((p[0], p[1]), set()).add(p[2])
        pool = [c for c in concepts
                if c in vectors and c != src and c not in stated.get((src, relation), ())]
        if dst not in pool:
            continue
        allp = np.stack([vectors[c] for c in pool])

        offers = {}
        for other in ops:
            carried = operators.carry(vectors, other, src, ops)
            if carried is None:
                continue
            scores = ask._along(allp, carried)
            k = int(np.argmax(scores))
            offers[other] = (pool[k], float(scores[k]))
        if relation not in offers or len(offers) < 2:
            continue

        key = (src, relation, dst)
        drawn = str(np.random.default_rng(40000 + i).choice(sorted(offers)))
        best = max(offers, key=lambda r: offers[r][1])

        got["told"][key] = offers[relation][0] == dst
        got["choosing"][key] = offers[best][0] == dst
        got["at random"][key] = offers[drawn][0] == dst
        got["the best on offer"][key] = any(v[0] == dst for v in offers.values())
        chose_true[key] = best == relation
        at_random_true[key] = drawn == relation
        floors.append(1 / len(pool))
        relations.append(len(offers))

    n = len(floors)
    if not n:
        print("  No question could be put this way.")
        return 1

    floor = float(np.mean(floors))
    by_chance = float(np.mean([1 / r for r in relations]))
    rows = {a: {"read": float(np.mean(list(got[a].values())))} for a in arms}
    rows["choosing"]["picks_the_true_relation"] = float(np.mean(list(chose_true.values())))
    rows["at random"]["picks_the_true_relation"] = float(np.mean(list(at_random_true.values())))

    print(f"  {n} questions, a mean of {float(np.mean(relations)):.1f} relations on offer")
    print(f"  the floor is naming one of the candidates at random: {floor:.0%}\n")
    print(f"  {'arm':>20}{'the read':>10}{'picks the true relation':>25}")
    print("  " + "-" * 57)
    for a in arms:
        picks = rows[a].get("picks_the_true_relation")
        shown = f"{picks:.0%}" if picks is not None else ""
        print(f"  {a:>20}{rows[a]['read']:>10.0%}{shown:>25}")
    print()

    w, l, paired_n, p = mcnemar(got["choosing"], got["at random"])
    print(f"  choosing against a relation drawn with no rule, on the same {paired_n} questions:")
    print(f"  choosing right where the draw is wrong {w}, the draw right where choosing is wrong "
          f"{l}, p {p:.4f}\n")

    record.figures(asked=n, floor=floor, relations=float(np.mean(relations)),
                   a_relation_by_chance=by_chance, arms=rows,
                   choosing_against_random={"choosing_only": w, "random_only": l,
                                            "questions": paired_n, "p": p})

    worse = rows["choosing"]["read"] < rows["at random"]["read"]
    picks_worse = (rows["choosing"]["picks_the_true_relation"]
                   < rows["at random"]["picks_the_true_relation"])

    if worse and l > w:
        print("  **It is worse than no rule here too.** On a store not arranged for it, choosing")
        print(f"  reads {rows['choosing']['read']:.0%} against {rows['at random']['read']:.0%} for a")
        print(f"  relation drawn with no rule, losing {l} questions to {w} paired, p {p:.4f}. **So")
        print("  §47.72 is not an artefact of a structure I generated** -- a rule that ranks")
        print("  relations by how well each one lands points away from the answer on a real")
        print("  ontology as well.")
    elif worse:
        print("  **It reads lower than no rule and the pairing does not carry it.** Choosing is")
        print(f"  behind the draw on the shares -- {rows['choosing']['read']:.0%} against")
        print(f"  {rows['at random']['read']:.0%} -- while the paired counts are {w} to {l} at")
        print(f"  p {p:.4f}, so this store can show the direction and not establish it.")
    else:
        print("  **It is not worse than no rule here.** Choosing reads")
        print(f"  {rows['choosing']['read']:.0%} against {rows['at random']['read']:.0%} for a")
        print("  relation drawn with no rule, so what §47.72 measures is the generated structures'")
        print("  and not the method's. **The blocker is still a blocker** -- being told reaches")
        print(f"  {rows['told']['read']:.0%} where choosing reaches {rows['choosing']['read']:.0%}")
        print("  -- but it is a rule that fails to find the answer rather than one that points")
        print("  away from it, and those want different accounts.")
    print()
    if picks_worse:
        print("  It picks the question's own relation")
        print(f"  {rows['choosing']['picks_the_true_relation']:.0%} of the time against")
        print(f"  {rows['at random']['picks_the_true_relation']:.0%} for the draw, so the selection")
        print("  itself is the part that is behind.")
    else:
        print("  It picks the question's own relation")
        print(f"  {rows['choosing']['picks_the_true_relation']:.0%} of the time against")
        print(f"  {rows['at random']['picks_the_true_relation']:.0%} for the draw, so the selection")
        print("  is not what is behind and the difference is in which candidate each arm offers.")
    print()
    print(f"  A correct answer is on the table {rows['the best on offer']['read']:.0%} of the time,")
    print("  so what any rule could reach here is bounded by that and not by the arms above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
