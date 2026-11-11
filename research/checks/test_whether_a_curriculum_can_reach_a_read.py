"""Can the order a store was assembled in reach a read at all?

The question, and why it is not rhetorical
--------------------------------------------
A curriculum is an ordering of what a model is shown. The paper has carried it as an
unbuilt thing — something still to be designed. Two results together suggest there is nothing to
design.

`research/lean/Infer/Carry.lean`'s ``operator_perm`` proves the operator is invariant under permuting the
statements that use a relation: it is a sum, and a sum does not care about order. So a curriculum,
if it acts at all, acts through the coordinate.

`test_whether_the_order_a_store_was_given_in_reaches_the_read.py` measures the shipped coordinate
under four orders of one store, and it does not move: each vertex contributes once, at the largest
value that reaches it, and that is a shortest path, which does not depend on the order its edges
were offered in.

**So both halves of a read are order-invariant, and a curriculum has nothing to act through.** That is a conclusion drawn from two results rather than measured, and conclusions
drawn that way are often wrong. This measures it directly, on the whole read path
and on more than one store.

What is varied, and what is held
----------------------------------
The arrival order of the statements, and nothing else: the same set, shuffled. Not which
statements are present — a store of a hundred answers differently from a store of a thousand, and
that is store size rather than curriculum. A curriculum claims the *sequence* matters, so the
sequence is what is permuted.

What is compared
------------------
``coordinate``   every cell of every vertex's place, keyed by what survives a shuffle.
``operator``     the matrix itself, for the relation that carries most statements.
``the read``     the held-out answer to every question the store can be asked.

All three, because a curriculum that could not move the coordinate might still move an answer
through a tie, and an answer that does not move might sit on a coordinate that did.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import random
import sys

import numpy as np

import ask
import ontology
import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_order_a_store_was_given_in_reaches_the_read import (  # noqa: E402
    by_name, shipped)

#: how many arrival orders to try. Not a sample the spread is estimated from: either the order
#: reaches the read or it does not, and one that does shows up immediately.
ORDERS = 4

#: how many prose spans to ask about, so a second store is covered without an hour of box time
PROSE_QUESTIONS = 40


def operator_of(engine, statements, walk):
    """The relation carrying most statements, and its matrix keyed by what survives a shuffle.

    A statement's id is its position, so every ``("in", id)`` axis is renamed when the store is
    reordered and the matrix comes back over a permuted basis. Comparing two such matrices cell by
    cell compares different bases and reports a difference that is only the relabelling -- which is
    what the first run of this file did, reporting the operator moving by 3.7 on a store whose
    coordinate had not moved at all, against a Lean proof that it cannot.
    """
    relations = engine.relations()
    if not relations:
        return None, None
    biggest = max(relations, key=lambda r: len(engine.uses(r)))
    vectors = walk(engine)
    matrix = engine.operator(biggest, vectors)
    if matrix is None:
        return biggest, None
    ids = {engine.statement_id(i): parts for i, parts in enumerate(statements)}
    axes = [("in", ids[a[1]]) if a[0] == "in" else a for a in engine.basis()]
    order = sorted(range(len(axes)), key=lambda i: repr(axes[i]))
    return biggest, matrix[np.ix_(order, order)]


def answers(engine, statements, walk, *, asking, candidates):
    """The answer to every asked question, so an answer that moves is visible as an answer."""
    out = {}
    for src, relation, dst in asking:
        pair = [s for s in statements if s[0] == src and s[2] == dst]
        held = [s for s in statements if s not in pair]
        if not any(m == relation for _, m, _ in held):
            continue
        vectors = walk(engine, without=pair)
        if src not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=pair)
        if landed is None:
            continue
        stated = {b for a, m, b in held if a == src and m == relation}
        pool = [c for c in candidates if c in vectors and c != src and c not in stated]
        if len(pool) < 2:
            continue
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        ranked = np.sort(scores)[::-1]
        width = len(next(iter(vectors.values())))
        bound = width * float(np.spacing(float(1))) * abs(float(ranked[0]))
        out[(src, relation, dst)] = (pool[int(np.argmax(scores))],
                                     float(ranked[0] - ranked[1]) <= bound)
    return out


def stores():
    out = {"ontology": (tuple(tuple(s) for s in ontology.STATEMENTS), None)}
    from test_whether_the_read_predicts_the_next_token import spans
    name, store = spans()
    if store:
        out["prose spans"] = (tuple(store), PROSE_QUESTIONS)
    return out


def main() -> int:
    print("can the order a store was assembled in reach a read at all?\n")
    print(f"  {'store':<14}{'walk':<10}{'orders':>8}{'cells move':>12}{'operator moves':>16}"
          f"{'answers move':>14}{'of':>5}")
    print("  " + "-" * 79)

    rows = {}
    for label, (statements, sample) in stores().items():
        engine = Engine.on_statements(statements)
        candidates = engine.concepts()
        asking = list(statements) if sample is None else list(statements)[:sample]
        for walk_name, walk in (("shipped", shipped),):
            base_places = by_name(engine, statements, walk(engine))
            _, base_matrix = operator_of(engine, statements, walk)
            base_answers = answers(engine, statements, walk,
                                   asking=asking, candidates=candidates)

            rng = random.Random(0)
            worst_cell = float(0)
            worst_matrix = float(0)
            moved_answers = moved_inside_the_rounding = 0
            for _ in range(ORDERS):
                other = list(statements)
                rng.shuffle(other)
                other = tuple(other)
                twin = Engine.on_statements(other)
                places = by_name(twin, other, walk(twin))
                for key, row in base_places.items():
                    theirs = places.get(key, {})
                    for axis in set(row) | set(theirs):
                        worst_cell = max(worst_cell,
                                         abs(row.get(axis, float(0))
                                             - theirs.get(axis, float(0))))
                _, matrix = operator_of(twin, other, walk)
                if base_matrix is not None and matrix is not None \
                        and base_matrix.shape == matrix.shape:
                    worst_matrix = max(worst_matrix, float(np.abs(base_matrix - matrix).max()))
                theirs = answers(twin, other, walk,
                                 asking=asking, candidates=candidates)
                for k, (was, _) in base_answers.items():
                    if k in theirs and theirs[k][0] != was:
                        moved_answers += 1
                        if base_answers[k][1]:
                            moved_inside_the_rounding += 1

            rows[f"{label}/{walk_name}"] = {
                "statements": len(statements), "orders": ORDERS,
                "worst_cell_moves": worst_cell, "worst_operator_cell_moves": worst_matrix,
                # what the operator's own arithmetic can resolve: a sum of `width` products,
                # each rounding at `spacing(1)` of the largest entry
                "operator_rounding_bound": (
                    float(np.abs(base_matrix).max()) * base_matrix.shape[0]
                    * float(np.spacing(float(1))) if base_matrix is not None else float(0)),
                "answers_that_move": moved_answers,
                "of_those_decided_inside_the_rounding": moved_inside_the_rounding,
                "answers_asked": len(base_answers)}
            r = rows[f"{label}/{walk_name}"]
            print(f"  {label:<14}{walk_name:<10}{ORDERS:>8}{worst_cell:>12.4g}"
                  f"{worst_matrix:>16.4g}{moved_answers:>14}{len(base_answers):>5}")
    print()

    record.figures(orders=ORDERS, prose_questions=PROSE_QUESTIONS, by_arm=rows)

    settled = {k: r for k, r in rows.items() if k.endswith("shipped")}
    # **An answer that moves on an unmoved coordinate is the arithmetic, not the order.** The
    # columns are numbered by statement position, so a shuffle re-orders the sum without changing
    # a term of it -- and a question decided inside the rounding bound falls whichever way the last
    # bit does. Those are counted apart rather than read as a curriculum reaching the answer.
    # **The operator is compared against its own resolution, not against exact zero.** It is a sum
    # of products over a basis whose columns are numbered by statement position, so a shuffle
    # re-orders the summation without changing a term of it. Demanding an exact 0 of that is
    # demanding the arithmetic be associative: the first run of this condition read 1.78e-15 as
    # movement and called a curriculum real.
    def is_still(r):
        bound = r["operator_rounding_bound"]
        return (r["worst_cell_moves"] == 0
                and r["worst_operator_cell_moves"] <= bound
                and r["answers_that_move"] == r["of_those_decided_inside_the_rounding"])

    still = all(is_still(r) for r in settled.values())

    if still:
        worst = max(settled.values(), key=lambda r: r["answers_that_move"])
        print(f"  **The order reaches nothing but the last bit.** The "
              f"coordinate is")
        print(f"  identical to the cell on {len(settled)} stores and {ORDERS} orders each. The "
              f"operator moves at most")
        print(f"  **{max(r['worst_operator_cell_moves'] for r in settled.values()):.3g}** against a "
              f"rounding bound of "
              f"**{max(r['operator_rounding_bound'] for r in settled.values()):.3g}** -- it is a sum "
              f"and a shuffle")
        print(f"  re-orders the summation without changing a term, which is what `operator_perm` "
              f"proves.")
        print(f"  And **{worst['answers_that_move']}** answers of **{worst['answers_asked']}** "
              f"change, **every one of them** a question already")
        print(f"  decided by a gap the arithmetic cannot resolve.")
        print()
        print(f"  **So there is no curriculum to build.** A curriculum is an ordering of what a "
              f"model is")
        print(f"  shown, and here the ordering reaches nothing: a store is a set and the read is a")
        print(f"  function of the set. What does reach a read is **which** statements are in the "
              f"store,")
        print(f"  which is not an ordering -- a store of a hundred answers differently from a store "
              f"of a")
        print(f"  thousand, and that is size rather than sequence. **The unbuilt thing was not "
              f"missing;")
        print(f"  it was impossible**, and saying so is worth more than leaving it on a list.")
    else:
        moves = max(settled.values(), key=lambda r: r["answers_that_move"])
        print(f"  **The order reaches the shipped read.** Worst cell "
              f"**{moves['worst_cell_moves']:.3g}**,")
        print(f"  worst operator cell **{moves['worst_operator_cell_moves']:.3g}**, "
              f"**{moves['answers_that_move']}** answers of **{moves['answers_asked']}** moved. "
              f"**A curriculum has something")
        print(f"  to act through**, and what it acts through is named above.")
    print()
    shipped_arms = {k: r for k, r in rows.items() if k.endswith("shipped")}
    worst = max(shipped_arms.values(), key=lambda r: r["worst_cell_moves"])
    print(f"  The shipped walk is shown beside it for the contrast: worst cell "
          f"**{worst['worst_cell_moves']:.4g}**, and")
    print(f"  **{sum(r['answers_that_move'] for r in shipped_arms.values())}** answers moved across "
          f"both stores. That is the input §47.162 measures, and it is")
    print(f"  the reason this question had an answer worth checking rather than an obvious one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
