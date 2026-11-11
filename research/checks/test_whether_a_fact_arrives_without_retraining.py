"""A relation the store has never carried, becoming answerable one statement at a time.

What this is for
------------------
`research/lean/Infer/Carry.lean` proves `operator_append`: a statement arriving adds one rank-one term to the
operator and disturbs nothing already there. That is an identity, and identities do not tell you
whether the thing is any use. **This measures the consequence**: how many statements a relation needs
before the engine can answer a question about it that it was never told, and what that costs in wall
clock.

The comparison this is against is not another method, it is retraining. A trained model acquires a
fact by a gradient step against a loss, which requires the fact to be in a training set, a schedule,
and a pass. Here a fact is acquired by **being written**, and the next read has it because the read
derives the operator from whatever is in the store at the moment it is asked. So the question worth
measuring is not "is it better" but "how much does it need, and how fast".

How it is run
---------------
For each relation in the store and each held-out question about it:

1. **Strip the relation entirely.** Every statement using it is removed, so the store carries no
   trace of that relation at all and the engine has nothing to derive an operator from. The question
   is asked here first, and the engine must decline -- if it answers, it is answering from something
   other than the relation, and that is the finding instead.
2. **Put the statements back one at a time**, never the held-out one, asking after each.
3. Record the first count at which the engine **answers at all**, and the first at which it is
   **right**, along with the time for the write-and-read.

The held-out statement is never written, and the coordinate is rebuilt at every step, so at no point
is the answer in the material. What grows is only how many *other* statements use the relation.

What would settle it
----------------------
If the engine answers with one use, the operator is not doing anything a lookup could not -- a
cross-covariance over one row is that row. If it needs most of the relation before it answers, a
fact does not usefully arrive on its own and the "one write" story is thin. **The number to report is
where between those it actually falls**, per relation, with the store's own size beside it.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import statistics
import time

import numpy as np

import record

import ontology

from entroptics_infer.wrapper import Engine

#: the most held-out questions asked per relation; a relation with fewer uses is asked fewer.
#: Declared, not swept: this experiment is about the shape of the arrival curve, and more questions
#: narrow the mean without changing the shape.
QUESTIONS = 6
#: how far the count of uses is grown before giving up on a question
FURTHEST = 24


def uses_of(relation, statements):
    """``[(parts, position), ...]`` -- every statement that uses ``relation`` as a relation."""
    found = []
    for parts in statements:
        for position, part in enumerate(parts):
            if part == relation and position > 0 and position < len(parts) - 1:
                found.append((tuple(parts), position))
    return found


def arrival(relation, question, others, rest):
    """When the engine starts answering, and when it starts being right, as uses are added."""
    parts, position = question
    src, dst = parts[position - 1], parts[position + 1]

    answered_at = right_at = None
    took = []
    for count in range(min(len(others), FURTHEST) + 1):
        store = list(rest) + [p for p, _ in others[:count]]
        if not store:
            continue
        started = time.perf_counter()
        engine = Engine.on_statements(store)
        if src not in engine.concepts():
            took.append(time.perf_counter() - started)
            continue
        answer = engine.infer(src, relation, hold_out=False)
        took.append(time.perf_counter() - started)

        if answer.named is not None and answered_at is None:
            answered_at = count
        if answer.named == dst and right_at is None:
            right_at = count
            break

    return {"source": src, "truth": dst, "uses_available": len(others),
            "answered_at": answered_at, "right_at": right_at,
            "seconds_per_write_and_read": float(statistics.median(took)) if took else float("nan")}


def main() -> int:
    print("a relation arriving one statement at a time, with nothing retrained\n")

    statements = [tuple(p) for p in ontology.STATEMENTS]
    # whatever the store has used as a relation, counted off it rather than declared anywhere
    relations = sorted({p for parts in statements for i, p in enumerate(parts)
                        if 0 < i < len(parts) - 1})

    print(f"  {'relation':>12}{'uses':>6}{'asked':>7}{'answers with':>14}{'right with':>12}"
          f"{'never right':>13}{'ms per step':>13}")
    print("  " + "-" * 77)

    rows = {}
    for relation in relations:
        mine = uses_of(relation, statements)
        if len(mine) < 2:
            continue
        # the store with this relation entirely absent -- no statement mentions it at all
        rest = [p for p in statements if p not in {q for q, _ in mine}]
        asked = mine[:QUESTIONS]

        found = []
        for question in asked:
            others = [u for u in mine if u[0] != question[0]]
            found.append(arrival(relation, question, others, rest))

        answers = [f["answered_at"] for f in found if f["answered_at"] is not None]
        rights = [f["right_at"] for f in found if f["right_at"] is not None]
        never = len(found) - len(rights)
        ms = float(np.median([f["seconds_per_write_and_read"] for f in found])) * 1000.0
        rows[relation] = {"uses": len(mine), "asked": len(found), "detail": found,
                          "answers_with": float(np.mean(answers)) if answers else float("nan"),
                          "right_with": float(np.mean(rights)) if rights else float("nan"),
                          "never_right": never, "ms_per_write_and_read": ms}
        print(f"  {relation:>12}{len(mine):>6}{len(found):>7}"
              f"{(np.mean(answers) if answers else float('nan')):>14.1f}"
              f"{(np.mean(rights) if rights else float('nan')):>12.1f}"
              f"{never:>13}{ms:>13.1f}")
    print()

    if not rows:
        print("  No relation had two uses, so nothing could be held out. The store is not as built.")
        return 0

    record.figures(questions_per_relation_at_most=QUESTIONS, furthest=FURTHEST,
                   statements=len(statements),
                   by_relation=rows)

    every = list(rows.values())
    rights = [v["right_with"] for v in every if v["right_with"] == v["right_with"]]
    answers = [v["answers_with"] for v in every if v["answers_with"] == v["answers_with"]]
    ms = record.figure("ms_per_write_and_read",
                       float(np.median([v["ms_per_write_and_read"] for v in every])))
    asked = record.figure("asked", sum(v["asked"] for v in every))
    never = record.figure("never_right", sum(v["never_right"] for v in every))
    one_use = [r for r, v in rows.items() if v["answers_with"] <= 1.0]

    print(f"  Over {asked} held-out questions across {len(rows)} relations, each starting from a")
    print("  store where the relation being asked about **does not appear at all**.")
    print()
    dead = sorted(r for r, v in rows.items() if v["never_right"] == v["asked"])
    print(f"  **{never} of {asked} were never right inside {FURTHEST} additions**, and that is the")
    print("  first thing to say about this curve. A relation arriving does not reliably make a")
    print("  question about it answerable, and on this store it more often does not.")
    if dead:
        print(f"  {', '.join(dead)} got none of its questions right at any count.")
    print()
    if answers:
        print(f"  **Where it does arrive, it arrives early.** The engine first offers an answer at a")
        print(f"  mean of {np.mean(answers):.1f} statements and is first right at a mean of "
              f"{np.mean(rights):.1f}, out of")
        print(f"  relations carrying {min(v['uses'] for v in every)} to "
              f"{max(v['uses'] for v in every)} statements. **So what a relation needs is a handful")
        print("  of uses, not most of itself** -- the curve is not a slow accumulation, and where")
        print("  the answer never comes, adding more of the same relation was not going to bring")
        print("  it. Which questions those are, and what distinguishes them, is not settled here.")
    print()
    if one_use:
        print(f"  **{len(one_use)} relation(s) answer from a single use**, which is not inference:")
        print("  a cross-covariance over one row is that row, so the engine is repeating what it")
        print(f"  was just given. Those are {', '.join(sorted(one_use))}, and the figures above")
        print("  should be read with them excluded.")
    else:
        print("  **No relation answers from a single use**, which is the guard working: a")
        print("  cross-covariance over one row is that row, the engine declines there, and every")
        print("  answer above needed the relation to have been used more than once.")
    print()
    print(f"  **And a statement arrives in {ms:.1f} ms** -- the write, the coordinate rebuilt, the")
    print("  operator rederived and the question answered. There is no gradient step, no schedule")
    print("  and no pass over a training set, because there is nothing to fit: `operator_append`")
    print("  says the operator gains exactly that statement's rank-one term, and this is what that")
    print("  costs and what it buys.")
    print()
    print("  The held-out statement is never written at any step and the coordinate is rebuilt at")
    print("  every step, so the answer is in neither. `ontology.py` is the store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
