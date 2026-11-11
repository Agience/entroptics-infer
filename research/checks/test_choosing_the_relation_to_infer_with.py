"""Inference when the question does not say which relation to apply, which is the ordinary case.

Why this is the practical question
------------------------------------
Every inference figure in this line hands the reader the relation. `test_inference_on_a_partly_
stated_structure.py` asks "a, r" and applies r; the end-to-end file asks in words but the words
name the relation, because the query is `concept relation` and `ask.acting_in` finds it.

A question that arrives in the ordinary way does not do that. "Where does a salmon live" names no
relation the store holds, so the reader applies every operator and lets the candidates compete.
§47.7 measures what that costs on recovery -- being told reaches the destination 25% of the
time and choosing 9% -- and that is most of the difficulty on that task. Nobody has measured
it on inference, where the answer is a pair the store does not state and the competition is
between operators that all land somewhere plausible.

If choosing costs what it costs on recovery, then the reader infers only for a questioner who
already knows which relation to use, and "in line" is not what it does.

What is measured
------------------
The same structure, the same grader, the same unstated-pairs-only scoring. Two readers:

``told``       the relation is given, as in every other inference file here.
``choosing``   every operator is applied to the concept, each offers its nearest unstated
               candidate, and the one whose offer scores highest against its own landing wins.
               That is what `ask.walk` does when the query names no relation.
``at random``  an operator drawn with no rule at all. The null the *choosing* needs: a re-pairing
               tests the operators, not the choosing, and cannot show a rule to be misleading.
``the best on offer``
               whether any operator's offer is right. An oracle rather than a reader, and the
               ceiling for any rule that picks among the offers -- it settles whether a rule would
               have anything to find, which is a different question from what rule to use.

Scored against the group the true relation leads to, because that is the question that was
asked however the reader went about it. Alongside it, the share of questions where the operator the
reader chose was the one the question was about -- a read can reach the right group through the
wrong relation by luck, and the two are worth separating.

The floor is naming one of the same candidates at random, and the null is a re-pairing of every
operator, so a reader that chooses well by geometry alone is caught in its own row.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

import ask
import operators
import place

from measure import mcnemar
from test_inference_on_a_partly_stated_structure import _ops, observed, structure

DRAWS = 4
SHARES = (0.2, 0.4, 0.6)


def main() -> int:
    print("the same questions, with and without being told which relation acts")
    print("scored only over pairs the store does not state, graded by the structure\n")
    print(f"  {'observed':>9}{'reader':>20}{'asked':>7}{'the read':>10}{'re-paired':>11}"
          f"{'the floor':>11}{'chose the right relation':>26}"
          f"{'lands back home':>18}")
    print("  " + "-" * 104)

    rows: dict = {}
    per_question: dict = {}
    for share in SHARES:
        arms = ("told", "the best on offer", "choosing", "above its own floor", "at random")
        got = {a: {} for a in arms}
        fake_got = {a: {} for a in arms}
        chose_right, floors, chose_home, told_home = [], [], [], []
        clear_right = []
        for seed in range(DRAWS):
            rng = np.random.default_rng(7000 + seed)
            concepts, relations, group, sends = structure(rng)
            statements = observed(concepts, relations, group, sends, share, rng)
            stated: dict = {}
            for a, r, b in statements:
                stated.setdefault((a, r), set()).add(b)

            vectors, _ = place.places(statements=statements)
            real = _ops(vectors, statements)
            fake = _ops(vectors, statements, rng=np.random.default_rng(seed))
            present = [c for c in concepts if c in vectors]
            at = {c: i for i, c in enumerate(present)}
            allp = np.stack([vectors[c] for c in present])

            def offer(ops, a, r, unstated, idx):
                """The nearest unstated candidate, its score, and its score above its own floor.

                The third is the one the comparison needs. A cosine against one operator's landing
                and a cosine against another's are not the same quantity -- §47.73 -- but `_floor`
                already says what a candidate drawn at random would score against that landing,
                so the distance above it is measured in each direction's own terms. Nothing new is
                computed: the reader builds this floor at every step already.
                """
                carried = operators.carry(vectors, r, a, ops)
                if carried is None:
                    return None
                scores = ask._along(allp, carried)[idx]
                i = int(np.argmax(scores))
                floor = float(ask._floor(vectors, carried, n=1))
                return unstated[i], float(scores[i]), float(scores[i]) - floor

            for a in concepts:
                if a not in vectors:
                    continue
                for r in relations:
                    if r not in real or r not in fake:
                        continue
                    unstated = [c for c in present
                                if c not in stated.get((a, r), ()) and c != a]
                    if not unstated:
                        continue
                    want = {c for c in unstated if group[c] == sends[r][group[a]]}
                    if not want or len(want) == len(unstated):
                        continue
                    idx = [at[c] for c in unstated]
                    key = (seed, a, r)

                    for ops, into in ((real, got), (fake, fake_got)):
                        told = offer(ops, a, r, unstated, idx)
                        if told is not None:
                            into["told"][key] = told[0] in want
                            if ops is real:
                                told_home.append(group[told[0]] == group[a])
                        # every operator applied, the best-scoring offer taken -- what the walk
                        # does when the query names no relation
                        best, clearest, any_right = None, None, False
                        for other in ops:
                            got_one = offer(ops, a, other, unstated, idx)
                            if got_one is None:
                                continue
                            any_right = any_right or got_one[0] in want
                            if best is None or got_one[1] > best[1]:
                                best = (got_one[0], got_one[1], other)
                            if clearest is None or got_one[2] > clearest[1]:
                                clearest = (got_one[0], got_one[2], other)
                        if clearest is not None:
                            into["above its own floor"][key] = clearest[0] in want
                            if ops is real:
                                clear_right.append(clearest[2] == r)
                        # the ceiling for any rule that picks among the offers: is one of them
                        # right at all? An oracle is not a rule and nothing here proposes one --
                        # what it settles is whether a rule would have anything to find.
                        into["the best on offer"][key] = any_right
                        if best is not None:
                            into["choosing"][key] = best[0] in want
                            if ops is real:
                                chose_right.append(best[2] == r)
                                # 47.64 found a coordinate that confuses what a concept is like
                                # with what a relation leads to. A selection rule that prefers the
                                # sharpest landing would prefer exactly that confusion, because a
                                # landing back among the source's own neighbours is the sharpest
                                # thing an operator can produce.
                                chose_home.append(group[best[0]] == group[a])
                        # the null the selection rule actually needs: pick an operator with no
                        # rule at all. A rule that scores below this is not weak, it is misleading,
                        # and a re-pairing of the operators cannot show that -- it tests the
                        # operators, not the choosing.
                        drawn = str(rng.choice(sorted(ops)))
                        blind = offer(ops, a, drawn, unstated, idx)
                        if blind is not None:
                            into["at random"][key] = blind[0] in want
                    floors.append(len(want) / len(unstated))

        for reader in arms:
            if not got[reader]:
                continue
            rows[(share, reader)] = {
                "observed": float(share), "asked": len(got[reader]),
                "read": float(np.mean(list(got[reader].values()))),
                "repaired": float(np.mean(list(fake_got[reader].values()))),
                "floor": float(np.mean(floors)),
                "chose_the_right_relation": (
                    float(np.mean(clear_right)) if reader == "above its own floor" and clear_right
                    else float(np.mean(chose_right)) if reader == "choosing" and chose_right
                    else float("nan")),
                "lands_in_the_source's_own_group": (
                    float(np.mean(chose_home)) if reader == "choosing" and chose_home
                    else float(np.mean(told_home)) if reader == "told" and told_home
                    else float("nan")),
                "relations": len(relations),
                "right_relation_by_chance": 1 / len(relations)}
            per_question[(share, reader)] = got[reader]
            v = rows[(share, reader)]
            right = v["chose_the_right_relation"]
            print(f"  {share:>9.0%}{reader:>20}{v['asked']:>7}{v['read']:>10.0%}"
                  f"{v['repaired']:>11.0%}{v['floor']:>11.0%}{right:>26.0%}"
                  f"{v["lands_in_the_source's_own_group"]:>18.0%}")
        print()

    record.figures(readers={f"{s:.1f} observed, {n}": v for (s, n), v in rows.items()})

    print(f"  {'observed':>9}{'told wins':>12}{'choosing wins':>16}{'p':>9}")
    print("  " + "-" * 48)
    paired = {}
    for share in SHARES:
        if (share, "told") not in per_question or (share, "choosing") not in per_question:
            continue
        w, l, n, p = mcnemar(per_question[(share, "told")], per_question[(share, "choosing")])
        paired[f"{share:.1f}"] = {"told": w, "choosing": l, "questions": n, "p": p}
        print(f"  {share:>9.0%}{w:>12}{l:>16}{p:>9.4f}")
    print()
    record.figures(told_against_choosing=paired)

    over_floor = [s for s in SHARES
                  if (s, "choosing") in rows
                  and rows[(s, "choosing")]["read"] > rows[(s, "choosing")]["floor"]]
    over_null = [s for s in SHARES
                 if (s, "choosing") in rows
                 and rows[(s, "choosing")]["read"] > rows[(s, "choosing")]["repaired"]]

    if len(over_floor) == len(SHARES) and len(over_null) == len(SHARES):
        print("  **A question that does not name its relation still infers.** Choosing among every")
        print("  operator beats both the floor and a re-pairing of the same operators at every")
        print("  share, so the reader does not need to be told which relation acts in order to")
        print("  reach a pair the store does not state. What it costs against being told is the")
        print("  paired table above, and it is the price of an ordinary question.")
    elif over_floor:
        print("  **Choosing clears the floor at some shares and not others**, so whether an")
        print("  unguided question can be inferred from depends on how much of the structure the")
        print("  store holds, and the table is the answer.")
    else:
        print("  **A question that does not name its relation cannot be inferred from.** Choosing")
        print("  among every operator scores at the floor, so every inference figure in this line")
        print("  belongs to a questioner who already knows which relation to use -- which is not")
        print("  the ordinary case and should not be read as one.")
    print()
    print("  The structure is synthesized and its generator is stated in the file this imports it")
    print("  from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
