"""What the aperture moves in the read, store by store.

What is compared
------------------
`Engine.places` settles each vertex's neighbourhood -- one contribution per vertex, at the largest
value that reaches it, so nothing depends on the order the statements were written down in -- and
stops each walk at its own aperture. The same walk read to its end settles every vertex it reaches.
This reads both on every store whose figures it would move, under the same held-out protocol: every
statement joining the asked-about pair removed from the operator and the coordinate, the same
questions, the same candidates.

``read to its end``   `place.read_to_its_end`: the shipped `_settle` and `_row` with the aperture's
                      stop left out.
``shipped``           `Engine.places`.

The ``moved`` column is the shipped read minus the walk read to its end.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import os
import pathlib
import sys

import numpy as np

import ask
import ontology
import place
import record
from checking import shipped

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402


#: the truncation the prose arms use; the ontology is read untruncated, as its figures are

#: how many prose spans to ask about. The ontology is asked in full, being small enough that
#: sampling it would be the only reason the two arms differed.
PROSE_QUESTIONS = 60

#: how many lattice statements to ask about. Both arms see the same ones, so a sample costs
#: precision on the difference and cannot invent one.
LATTICE_QUESTIONS = 120


def read_with(walk, statements, *, asking, candidates):
    """Held-out accuracy under one walk, on one set of questions."""
    engine = Engine.on_statements(statements)
    right = among_right = asked = among_asked = 0
    draw = float(0)
    for src, relation, dst in asking:
        pair = [s for s in statements if s[0] == src and s[2] == dst]
        held = [s for s in statements if s not in pair]
        if not any(m == relation for _, m, _ in held):
            continue
        vectors = walk(engine, without=pair)
        if src not in vectors or dst not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=pair)
        if landed is None:
            continue
        stated = {b for a, m, b in held if a == src and m == relation}
        pool = [c for c in candidates if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        into = {b for a, m, b in held if m == relation}
        reached = [c for c in pool if c in into]
        asked += 1
        draw += float(1) / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        if pool[int(np.argmax(scores))] == dst:
            right += 1
        if dst in reached and len(reached) > 1:
            among_asked += 1
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            if reached[int(np.argmax(sc))] == dst:
                among_right += 1
    if not asked:
        return None
    return {"asked": asked, "names": right / asked, "a_draw_would": draw / asked,
            "among_asked": among_asked,
            "among": among_right / among_asked if among_asked else float("nan")}


def to_its_end(engine, *, without=()):
    return place.read_to_its_end(engine, without=without)


def stores():
    """The stores whose published figures the swap would move."""
    out = {"ontology": (tuple(tuple(s) for s in ontology.STATEMENTS), None)}
    from test_whether_the_read_predicts_the_next_token import spans
    name, store = spans()
    if store:
        out["prose spans"] = (tuple(store), PROSE_QUESTIONS)

    # **The lattice is the store the shipped walk is for**, so the swap is not priced
    # until it is priced here. It needs `ENTROPTICS_LATTICE`; without it this arm is absent and the
    # verdict says so rather than reading two stores and calling the question settled.
    lattice = os.environ.get("ENTROPTICS_LATTICE")
    if lattice and os.path.exists(lattice):
        from heldout import core
        from test_whether_a_real_lattice_carries_a_read import MOST, a_connected_part
        from test_whether_the_authored_lattice_carries_a_read import halves
        imported, authored, how = halves(lattice)
        store = core(a_connected_part(core(authored), MOST))
        if len(store) >= 10:
            out["lattice, authored"] = (tuple(store), LATTICE_QUESTIONS)
    return out


def main() -> int:
    print("what the aperture moves in the read, store by store" + chr(10))
    print(f"  {'store':<16}{'asked':>7}{'draw':>8}{'to its end':>12}{'shipped':>10}{'moved':>9}"
          f"{'reached':>9}{'to its end':>12}{'shipped':>10}{'moved':>9}")
    print("  " + "-" * 102)

    rows = {}
    for label, (statements, sample) in stores().items():
        engine = Engine.on_statements(statements)
        candidates = engine.concepts()
        asking = list(statements) if sample is None else list(statements)[:sample]
        whole = read_with(to_its_end, statements, asking=asking, candidates=candidates)
        now = read_with(shipped, statements, asking=asking, candidates=candidates)
        if whole is None or now is None:
            continue
        rows[label] = {"statements": len(statements), "read_to_its_end": whole, "shipped": now,
                       "names_moved": now["names"] - whole["names"],
                       "among_moved": now["among"] - whole["among"]}
        r = rows[label]
        print(f"  {label:<16}{now['asked']:>7}{now['a_draw_would']:>8.2%}{whole['names']:>12.1%}"
              f"{now['names']:>10.1%}{r['names_moved']:>+9.1%}{now['among_asked']:>9}"
              f"{whole['among']:>12.1%}{now['among']:>10.1%}{r['among_moved']:>+9.1%}")
    print()
    if not rows:
        print("  No store could be read on both coordinates. Nothing to compare.")
        return 0
    record.figures(prose_questions=PROSE_QUESTIONS, by_store=rows)
    for label, r in rows.items():
        print(f"  {label}: the shipped read names {r['shipped']['names']:.1%} against "
              f"{r['read_to_its_end']['names']:.1%} read to its end ({r['names_moved']:+.1%}), and "
              f"{r['shipped']['among']:.1%} against {r['read_to_its_end']['among']:.1%} among the "
              f"reached.")
    if "lattice, authored" not in rows:
        print("  The lattice is not in this run; set ENTROPTICS_LATTICE to read it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
