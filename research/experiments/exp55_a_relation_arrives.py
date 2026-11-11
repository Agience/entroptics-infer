"""Stage 48 — a relationship that arrives after the reads have already run.

What this adds over the lessons that look like it
---------------------------------------------------
`grade six` delivers a lattice a piece at a time and asks at each size, so it measures that the read
is right however much has arrived. `grade seven` holds a relationship that exists on only the last
third of the sources, so it measures that a late-appearing relationship is still found. `exp06`
measures that an unrelated record arriving leaves an existing answer bit-identical.

None of them is the thing a running system does. There, questions have already been asked and
answered, and then a relationship that was not in the lattice at all turns up. Two things have to
hold at once and they pull against each other:

* **It learns.** The new relationship is counted, and questions about it can be answered.
* **It does not disturb.** Every answer already given about the old relationships is the same
  answer — not similar, *the same* — because nothing was fitted to the old lattice and so nothing
  can be refitted by the new one.

A trained system cannot have both. Adding a relation means new parameters and a new descent, and the
old answers move because the weights moved. Here the claim is that they cannot move, and the way to
test a claim like that is to record the answers before, add the relationship, ask the *same*
questions again, and compare them one by one.

What would falsify it
-----------------------
Any old answer that changes. Not a distribution that shifts — a single question that was answered
`alpha3` before and is answered anything else after. The comparison is exact and per question, and
the count of questions that moved is the result.

The arrival is also asked the other way round. If a relationship arrives and the screen does not
count it, the lattice has not learned; if it counts a relationship that has not arrived yet, it is
reading something that is not there. Both are checked at every step.
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
from entroptics_infer import complete, relations                    # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP55_SMOKE") == "1"
SOURCES = 8 if SMOKE else 16
QUESTIONS = 20 if SMOKE else 80
DRAWS = 80 if SMOKE else 250
#: the relationships, in the order they turn up. The first two are there when the first questions
#: are asked; the rest arrive afterwards, one at a time.
ARRIVALS = ("alpha", "beta", "gamma", "delta")


def build(n, T, D, rng, arrived):
    """The same lattice at every step; only which relationships have turned up differs."""
    records, edges = {}, []
    maps = {name: C._orthogonal(rng, D) for name in ARRIVALS}
    for i in range(n):
        src = f"s{i}"
        records[src] = C._structured(rng, T, D)
    for name in arrived:
        for i in range(n):
            records[f"{name}{i}"] = records[f"s{i}"] @ maps[name]
            edges.append((f"s{i}", f"{name}{i}", C.KIND_PREFIX + name))
    return records, edges


def questions(names, n, rng, count):
    """Questions written once and re-asked verbatim at every step, so answers are comparable."""
    out = []
    for _ in range(count):
        kind = names[int(rng.integers(len(names)))]
        src = int(rng.integers(n))
        other = int(rng.integers(n))
        while other == src:
            other = int(rng.integers(n))
        out.append({"kind": kind, "source": f"s{src}", "exemplar_of": other})
    return out


def ask(records, question, among):
    """One question, with its candidate set fixed by the caller.

    The candidates must not grow when a relationship arrives, or it is not the same question any
    more and an answer that changes says nothing. Letting the candidates be whatever has arrived
    makes answers "move" purely because they are being asked something else. Stability is asked
    with the original candidates; whether the new relationship can be
    answered at all is a separate question with its own.
    """
    kind = question["kind"]
    if kind not in among:
        return None
    src = question["source"]
    cands = [f"{name}{src[1:]}" for name in among]
    if any(c not in records for c in cands):
        return None
    exemplar = (f"s{question['exemplar_of']}", f"{kind}{question['exemplar_of']}")
    out = complete(records, exemplar=exemplar, source=src, candidates=cands, far=FAR)
    return None if out.key is None else cands[out.key]


def run():
    started = time.time()
    rng = np.random.default_rng(SEED)
    asked = questions(ARRIVALS, SOURCES, np.random.default_rng(SEED + 1), QUESTIONS)

    print(f"{SOURCES} sources, {len(ARRIVALS)} relationships arriving one at a time, "
          f"{QUESTIONS} questions re-asked verbatim at every step\n")
    print(f"{'arrived':>34}{'counted':>9}{'answered':>10}{'moved':>8}")
    print("-" * 62)

    steps = []
    before = {}
    first_among = ARRIVALS[:2]        # the candidate set the first questions were asked with
    total_moved = 0
    for k in range(2, len(ARRIVALS) + 1):
        arrived = ARRIVALS[:k]
        records, edges = build(SOURCES, 96, 32, np.random.default_rng(SEED), arrived)
        named = {}
        for s, d, lab in edges:
            named.setdefault(lab.removeprefix(C.KIND_PREFIX), []).append((s, d))
        sc = relations.carries_kinds(records, named, far=FAR, seed=SEED, draws=DRAWS)

        # The answers cannot move and that is structural: the old records are byte-identical at
        # every step, because nothing about them was fitted to the lattice they sat in. So the
        # stability worth measuring is the **screen's**, which is not structural at all --
        # `carries_kinds` reads every relation against a baseline of pairs standing in none, and
        # the arriving relationship's records join that baseline. An existing relation's
        # separation can move, and whether it does is the question.
        seps = {k: float(v) for k, v in sc.separation.items()}

        # STABILITY: the original questions, with the candidate set they were first asked with
        stable, moved = {}, []
        for j, q in enumerate(asked):
            a = ask(records, q, first_among)
            stable[j] = a
            if j in before and before[j] is not None and a != before[j]:
                moved.append({"question": j, "kind": q["kind"],
                              "was": before[j], "now": a})
        total_moved += len(moved)
        # LEARNING: the same questions asked among everything that has arrived, which is a
        # different and larger question and is where a newly arrived relationship shows up
        answered = sum(1 for q in asked if ask(records, q, arrived) is not None)
        steps.append({"arrived": list(arrived), "counted": int(sc.count),
                      "declared": len(arrived), "counted_right": bool(sc.count == len(arrived)),
                      "answered": answered, "asked_answerable": sum(
                          1 for q in asked if q["kind"] in arrived),
                      "answers_that_moved": len(moved), "moved": moved[:5],
                      "separations": seps,
                      "records": len(records)})
        print(f"{', '.join(arrived):>34}{f'{sc.count}/{len(arrived)}':>9}"
              f"{answered:>10}{len(moved):>8}")
        before = {j: a for j, a in stable.items() if a is not None}

    # how far an already-present relation's reading drifts as others arrive around it
    drift = {}
    for name in ARRIVALS[:2]:
        vals = [st["separations"][name] for st in steps if name in st["separations"]]
        drift[name] = {"first": vals[0], "last": vals[-1],
                       "widest_move": float(max(vals) - min(vals)),
                       "still_present_throughout": bool(all(v > 0 for v in vals))}
    print()
    print(f"  {'already present':>18}{'first':>9}{'last':>9}{'widest move':>14}")
    for name, d in drift.items():
        print(f"  {name:>18}{d['first']:>9.2f}{d['last']:>9.2f}{d['widest_move']:>14.2f}")

    counted_right = sum(1 for s in steps if s["counted_right"])
    grew = all(steps[i]["counted"] <= steps[i + 1]["counted"] for i in range(len(steps) - 1))
    out = {"design": {"far": FAR, "seed": SEED, "sources": SOURCES,
                      "questions": QUESTIONS, "draws": DRAWS,
                      "arrivals": list(ARRIVALS), "smoke": SMOKE},
           "steps": steps, "arrivals_measured": len(steps),
           "steps_where_the_count_is_right": counted_right,
           "answers_that_moved": total_moved,
           "the_count_never_went_backwards": bool(grew),
           "drift_of_the_relations_already_present": drift,
           "widest_move_of_an_existing_relation": float(
               max(d["widest_move"] for d in drift.values())),
           "records_at_the_first_step": steps[0]["records"],
           "records_at_the_last_step": steps[-1]["records"],
           "answered_at_the_first_step": steps[0]["answered"],
           "answered_at_the_last_step": steps[-1]["answered"],
           "seconds": time.time() - started}
    print("\n" + "=" * 62)
    print(f"  the screen counts what has arrived at {counted_right} of {len(steps)} steps, and the "
          f"count never goes backwards: {grew}")
    print(f"  questions answered rises {steps[0]['answered']} -> {steps[-1]['answered']} as "
          f"relationships turn up")
    if total_moved == 0:
        print("\n  AND NOT ONE ANSWER ALREADY GIVEN MOVED. Every question answered before an")
        print("  arrival is answered the same way after it — not similarly, identically, because")
        print("  nothing was fitted to the old lattice and so nothing could be refitted by the new.")
        print()
        print(f"That much is structural. What is not: the screen reads every relation against")
        print(f"  a baseline the newcomers join, and an already-present relation's separation still")
        print(f"  moves by at most {out['widest_move_of_an_existing_relation']:.2f} as the records")
        print(f"  around it go {out['records_at_the_first_step']} -> "
              f"{out['records_at_the_last_step']}.")
    else:
        print(f"\n  {total_moved} answers moved, which they must not: a relationship arriving has")
        print("  disturbed one that was already there.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp55.smoke.json" if SMOKE else "exp55.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
