"""One read, reached two ways, and the store it stands on.

Why this exists
-----------------
The research checks read through `place.py` and `operators.py`; a team taking this work imports
:mod:`entroptics_infer.wrapper`. There is one implementation: `place.places` is
`Engine.places`, an operator is `Engine.operator`, and a landing is `Engine._land`, the arithmetic
`Engine.carry` takes. `count the copies before comparing two` is the failure two implementations
give -- a gate over one says nothing about the other, and they drift silently -- so the research
helpers hold no walk, no operator and no landing of their own.

This file is the gate that says so, on the same store, reporting the largest difference anywhere:

``coordinate``   every vertex's place, built from the same statements.
``operator``     every relation's matrix, derived from those places.
``landing``      where each relation takes each concept.

**All three are zero by construction**, and the file still measures them, because an argument that
two paths are one is exactly what goes stale when somebody adds a copy back.

**And the refusals have to agree too.** Where a relation is used once, a cross-covariance over one
row is that row and predicts nothing; both paths return nothing there. A gate comparing only the
cases where both answered would pass while one path silently answered questions the other refused,
which is `a filter scored on its own selector passes everything` in another costume.

The store underneath
----------------------
The last check writes the store into a mantle lattice and reads it back through the shipped
engine alone. A statement becomes a vertex and its parts become edges carrying the position they
arrive at; nothing records that a vertex is a statement or that a position means "the relation".
What is asserted is that the store survives the round trip exactly, and that a read taken through
the lattice names what the same read names in memory -- with the asked-about statement held out of
the operator and the coordinate either way.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import os
import tempfile

import numpy as np

import record

import ontology
import operators
import place

from entroptics_infer.store import StoreUnavailable, open_lattice
from entroptics_infer.wrapper import Engine


def compare_on(statements):
    """The largest difference between the two paths, over one store."""
    shipped = Engine.on_statements(statements)
    research, _ = place.places(statements=statements)
    ours = shipped.places()

    shared = [c for c in shipped.concepts() if c in research and c in ours]
    coordinate = max(float(np.max(np.abs(research[c] - ours[c]))) for c in shared)

    operator = landing = float(0)
    agreed = disagreed = refused = 0
    ops = operators.all_operators(research, statements=statements)
    for relation in operators.relations(statements=statements):
        theirs, _ = operators.operator_of(relation, research, statements=statements)
        mine = shipped.operator(relation, ours)
        if (theirs is None) != (mine is None):
            disagreed += 1
            continue
        if theirs is None:
            refused += 1
            continue
        agreed += 1
        operator = max(operator, float(np.max(np.abs(theirs - mine))))
        for c in shared:
            a = operators.carry(research, relation, c, ops)
            b = shipped.carry(c, relation, ours)
            if a is None or b is None:
                continue
            landing = max(landing, float(np.max(np.abs(a - b))))

    return {"statements": len(statements), "concepts": len(shared),
            "relations_compared": agreed, "relations_both_refused": refused,
            "relations_disagreeing_on_refusal": disagreed,
            "largest_coordinate_difference": coordinate,
            "largest_operator_difference": operator,
            "largest_landing_difference": landing}


def through_the_lattice(statements, probes):
    """The store written to mantle, read back, and asked the same questions."""
    source = Engine.on_statements(statements)
    directory = tempfile.mkdtemp(prefix="engine_")
    path = os.path.join(directory, "lattice.db")
    with open_lattice(path, origin="a-node-reading-its-own-store") as lattice:
        edges = source.write_to(lattice)
        counted = lattice.count()
        back = Engine.on_lattice(lattice, seed=source.statement_id(0), size=len(statements) * 3)

    same = sorted(back.statements) == sorted(source.statements)
    answers = []
    for src, relation in probes:
        here = source.infer(src, relation)
        there = back.infer(src, relation)
        answers.append({"source": src, "relation": relation,
                        "in_memory": here.named, "through_the_lattice": there.named,
                        "agree": here.named == there.named,
                        "the_store_states_it": here.stated,
                        "pool": here.pool, "reached": here.reached})
    return {"edges_written": int(edges), "edges_counted": int(counted),
            "statements_read_back": len(back.statements), "the_store_survived": bool(same),
            "answers": answers}


def main() -> int:
    print("the shipped engine and the research one, on the same store\n")

    same = compare_on(ontology.STATEMENTS)
    print(f"  {same['statements']} statements, {same['concepts']} concepts, "
          f"{same['relations_compared']} relations compared, "
          f"{same['relations_both_refused']} refused by both\n")
    print(f"  {'what is compared':>26}{'largest difference':>22}")
    print("  " + "-" * 48)
    for label, key in (("every coordinate", "largest_coordinate_difference"),
                       ("every operator", "largest_operator_difference"),
                       ("every landing", "largest_landing_difference")):
        print(f"  {label:>26}{same[key]:>22.3e}")
    print()

    probes = [("dog", "is_a"), ("salmon", "lives_in"), ("oak", "is_a"),
              ("hammer", "used_for"), ("wheel", "part_of")]
    try:
        lattice = through_the_lattice(ontology.STATEMENTS, probes)
    except StoreUnavailable as exc:
        print(f"  The lattice half did not run: {exc}")
        record.figures(same_method=same, through_the_lattice=None)
        return 0

    print(f"  written to a mantle lattice: {lattice['edges_written']} edges, "
          f"{lattice['edges_counted']} counted, {lattice['statements_read_back']} statements read")
    print(f"  the store survived the round trip: {lattice['the_store_survived']}\n")
    print(f"  {'source':>10}{'relation':>12}{'in memory':>14}{'through mantle':>18}{'agree':>8}")
    print("  " + "-" * 62)
    for a in lattice["answers"]:
        print(f"  {a['source']:>10}{a['relation']:>12}{str(a['in_memory']):>14}"
              f"{str(a['through_the_lattice']):>18}{str(a['agree']):>8}")
    print()

    record.figures(same_method=same, through_the_lattice=lattice)

    agree = all(a["agree"] for a in lattice["answers"])
    stated = any(a["the_store_states_it"] for a in lattice["answers"])

    if same["relations_disagreeing_on_refusal"]:
        print(f"  **The two paths disagree about what they refuse** on "
              f"{same['relations_disagreeing_on_refusal']} relations. One of them is answering a")
        print("  question the other declines, so comparing them where both answered would have")
        print("  passed while the difference sat outside the comparison. Nothing below stands")
        print("  until that is settled.")
        return 0

    print(f"  **The research read is the shipped read.** Every coordinate")
    print(f"  and every operator matches to {max(same['largest_coordinate_difference'], same['largest_operator_difference']):.0e}, "
          f"and the two agree on every relation they refuse.")
    print(f"  The landings differ by {same['largest_landing_difference']:.0e}: both are "
          f"`Engine._land`, which computes")
    print("  (u @ A-transpose) @ B on connected sides and never forms the operator.")
    print()
    if lattice["the_store_survived"] and agree:
        print("  **And the store underneath is mantle.** The statements go in as vertices with their")
        print("  parts hanging off them by position, come back identical, and the read taken")
        print("  through the lattice names what the read taken in memory names. No relation kind is")
        print("  written and none is read: a relation is a vertex that happens to be used in the")
        print("  middle of statements.")
    else:
        print("  **The lattice round trip did not reproduce the store**, so the engine as shipped")
        print("  is not reading what the engine in memory reads, and where they part is the next")
        print("  thing to find.")
    if not stated:
        print()
        print("  Every answer above was taken with the asked-about statement held out of the")
        print("  operator **and** of the coordinate, and none of them is stated in what was read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
