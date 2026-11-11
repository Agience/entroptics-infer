"""What one question costs, and how that cost moves when the store gets bigger.

Why this is the open item
---------------------------
The README offers this engine to a team, and every result in it is taken on stores of a few hundred
statements. A team cannot plan against that. The first thing anyone will ask is what happens at
ten thousand statements, and "the read is quadratic" is a sentence rather than a measurement --
nobody can size a machine from it.

So this measures the thing directly, on real WordNet cores of increasing size, through the shipped
`Engine` rather than through the research code, because what a team would run is the shipped one.

What is timed, separately, because they do not grow alike
-----------------------------------------------------------
``coordinate``   `Engine.places()` -- every vertex's place, over a basis that grows with the store.
                 This is expected to dominate and the numbers say whether it does.
``operator``     deriving one relation's operator from places already built.
``carry``        applying it -- the step `carry_identity` says never forms the matrix.
``held-out q``   what a caller pays *under the measurement protocol*: an `infer` that rebuilds the
                 coordinate, because holding a statement out of the operator but not out of the
                 coordinate is not holding it out.
``served q``     what a caller pays *answering a question the store does not already hold*. Nothing
                 has to be held out -- the statement is not there -- so the coordinate standing
                 since the store last changed is the one to read, and the question is the carry and
                 the scoring alone.

**Those last two are different regimes, not two estimates of one quantity.** The first is the price
of measuring this method soundly and every published accuracy is paid at it. The second is the price
of running it. Quoting only the first overstates what answering costs by the entire coordinate;
quoting only the second would be quoting a number no published accuracy was taken at.

How growth is reported, and why nothing is fitted
---------------------------------------------------
For each consecutive pair of sizes the table gives the measured ratio of times against the
measured ratio of statements. If a cost were linear those two would match; if quadratic the time
ratio would be the square of the size ratio. No exponent is fitted -- a fitted power law would be
a chosen number dressed as a finding, and the ratios are the data. What the verdict does is compare
each observed time ratio to what linear and to what quadratic would have produced, and say which it
is nearer, per step, without smoothing across steps.

Timing is wall clock on an otherwise quiet machine, one thread, reported as the median of repeats
because a single timing is a measurement of whatever else the machine was doing.
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

import lexicon

from entroptics_infer.wrapper import Engine

#: walked outward from one seed; the core of each ball is what is read, as everywhere else
SEED = "entity.n.01"
#: swept upward until the largest is uncomfortable, because the point is where it stops being cheap
BALLS = (150, 300, 600, 1200, 2400)
#: a single timing measures the machine; the median of this many measures the code
REPEATS = 3


def _median_seconds(fn, repeats=REPEATS):
    """Median wall clock over ``repeats`` calls, and the value the last call returned."""
    took, out = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        out = fn()
        took.append(time.perf_counter() - started)
    return float(statistics.median(took)), out


def at(ball):
    """Every cost at one store size."""
    statements = lexicon.core(lexicon.neighbourhood(ball, SEED)[0])
    if len(statements) < 10:
        return None
    engine = Engine.on_statements(statements)

    coordinate, vectors = _median_seconds(engine.places)
    axes = len(engine.basis())
    relations = engine.relations()
    if not relations:
        return None
    biggest = max(relations, key=lambda r: len(engine.uses(r)))

    operator, matrix = _median_seconds(lambda: engine.operator(biggest, vectors))
    carry, _ = _median_seconds(lambda: engine.carry(
        engine.uses(biggest)[0][0][0], biggest, vectors))

    # what a caller pays under the held-out protocol: the coordinate is rebuilt because the
    # protocol removes the statement from it, so this is not the sum of the rows above
    parts, position = engine.uses(biggest)[0]
    source = parts[position - 1]
    question, answer = _median_seconds(
        lambda: engine.infer(source, biggest), repeats=max(REPEATS - 1, 1))

    # and what a caller pays in service of a question whose answer the store does not hold. There
    # is nothing to hold out -- the statement is not there -- so the coordinate built when the
    # store last changed is the coordinate to read, and the question is the carry and the scoring
    # alone. **These two numbers are different regimes, not two estimates of one thing**, and
    # reporting only the first would overstate what answering costs by the whole coordinate.
    concepts = engine.concepts()
    asked = [c for c in concepts if c != source][:len(relations) + len(relations)]
    served, _ = _median_seconds(
        lambda: [engine.attend(c, biggest, concepts, vectors=vectors) for c in asked])
    per_served = served / len(asked) if asked else float("nan")

    return {"ball": ball, "statements": len(statements), "concepts": len(engine.concepts()),
            "axes": axes, "relations": len(relations),
            "seconds_for_the_coordinate": coordinate,
            "seconds_for_the_operator": operator,
            "seconds_to_carry": carry,
            "seconds_for_one_question": question,
            "seconds_to_serve_one_question": per_served,
            "questions_served": len(asked),
            "operator_is_none": matrix is None,
            "pool": answer.pool, "reached": answer.reached}


def main() -> int:
    print("what one question costs as the store grows\n")
    print(f"  {'ball':>6}{'stmts':>7}{'axes':>7}{'coordinate':>12}{'operator':>10}"
          f"{'carry':>9}{'held-out q':>12}{'served q':>11}")
    print("  " + "-" * 76)

    rows = []
    for ball in BALLS:
        v = at(ball)
        if v is None:
            continue
        rows.append(v)
        print(f"  {v['ball']:>6}{v['statements']:>7}{v['axes']:>7}"
              f"{v['seconds_for_the_coordinate']:>12.3f}{v['seconds_for_the_operator']:>10.3f}"
              f"{v['seconds_to_carry']:>9.4f}{v['seconds_for_one_question']:>12.3f}"
              f"{v['seconds_to_serve_one_question']:>11.4f}")
    print()

    if len(rows) < 2:
        print("  Fewer than two sizes produced a store; there is no growth to report.")
        record.figures(seed=SEED, balls=list(BALLS), repeats=REPEATS, by_size=rows)
        return 0

    record.figures(seed=SEED, balls=list(BALLS), repeats=REPEATS, by_size=rows)

    print(f"  {'from':>7}{'to':>7}{'x statements':>14}{'x coordinate':>14}"
          f"{'x question':>12}{'linear would':>14}{'quadratic would':>17}")
    print("  " + "-" * 85)
    steps = []
    for a, b in zip(rows, rows[1:]):
        grew = b["statements"] / a["statements"]
        coord = (b["seconds_for_the_coordinate"] / a["seconds_for_the_coordinate"]
                 if a["seconds_for_the_coordinate"] else float("nan"))
        ques = (b["seconds_for_one_question"] / a["seconds_for_one_question"]
                if a["seconds_for_one_question"] else float("nan"))
        steps.append({"from": a["statements"], "to": b["statements"], "statements_grew": grew,
                      "coordinate_grew": coord, "question_grew": ques,
                      "linear_would_be": grew, "quadratic_would_be": grew * grew})
        print(f"  {a['statements']:>7}{b['statements']:>7}{grew:>14.2f}{coord:>14.2f}"
              f"{ques:>12.2f}{grew:>14.2f}{grew * grew:>17.2f}")
    print()

    # the two growth ratios are quotients of wall clocks, so they are figures about the box and go
    # under `timings`, which `verify_records.py` does not compare; the size ratios stay figures
    clock = ("coordinate_grew", "question_grew")
    record.figures(seed=SEED, balls=list(BALLS), repeats=REPEATS, by_size=rows,
                   by_step=[{k: v for k, v in s.items() if k not in clock} for s in steps],
                   timings={"by_step": [{k: s[k] for k in clock} for s in steps]})

    # nothing is fitted: each step is compared to what linear and what quadratic would have given,
    # and the nearer of the two is named for that step alone
    nearer = []
    for s in steps:
        lin = abs(s["question_grew"] - s["linear_would_be"])
        quad = abs(s["question_grew"] - s["quadratic_would_be"])
        nearer.append("quadratic" if quad < lin else "linear")

    biggest = rows[-1]
    print(f"  **At {biggest['statements']} statements one held-out question costs "
          f"{biggest['seconds_for_one_question']:.2f} s**, of which")
    print(f"  the coordinate is {biggest['seconds_for_the_coordinate']:.2f} s and applying the")
    print(f"  operator is {biggest['seconds_to_carry']:.4f} s. **The cost is the coordinate**, and")
    print("  it is paid per question because the protocol removes the asked-about statement from")
    print("  it -- a cached coordinate would be answering from material holding the answer.")
    print()
    served = biggest["seconds_to_serve_one_question"]
    print(f"  **But that is the price of measuring it, not of running it.** Asked something the")
    print(f"  store does not already hold there is nothing to hold out, the standing coordinate is")
    print(f"  the one to read, and the question costs {served * 1000:.1f} ms -- "
          f"{biggest['seconds_for_one_question'] / served:.0f} times less, or about")
    print(f"  {1.0 / served:.0f} questions a second at {biggest['statements']} statements. **The")
    print("  inference is cheap and the protocol is expensive**, which is the right way round for")
    print("  a reader but means every accuracy in this program was bought at the higher price.")
    print()
    print(f"  Step by step the question cost is nearer {' then '.join(nearer)}.")
    if all(n == "quadratic" for n in nearer):
        print("  **So the read grows quadratically in the store, measured rather than asserted.**")
        print("  A full held-out evaluation asks one question per statement, so a sweep over a")
        print("  store costs another factor of its size on top of that.")
    elif all(n == "linear" for n in nearer):
        print("  **So the read grows linearly over the range measured**, which is better than the")
        print("  program has been assuming, and the range it was measured over is in the table.")
    else:
        print("  **The steps do not agree with each other**, so no single growth describes the")
        print("  range and the table is the result rather than any summary of it. Where a cost")
        print("  changes character partway, the place it changes is the thing to look at.")
    print()
    print("  Wall clock, one thread, median of repeated calls. `lexicon.py` states how the store is")
    print("  built. No exponent is fitted anywhere here: the ratios are the measurement, and the")
    print("  two reference columns are what linear and quadratic would have produced at the same")
    print("  change in size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
