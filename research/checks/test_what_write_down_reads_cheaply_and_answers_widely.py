"""The trade a corpus's write-down makes: a cheap read against a question it can answer.

What is being traded
----------------------
`test_whether_many_small_relations_read_cheaply.py` establishes both halves of it. A read costs what
its relation has collected — 43.8 times less through a relation used 3 times than through one
used 2435, on one store with everything else held. And a relation used once has no operator at
all, because a cross-covariance over one row is that row, so on prose written as three-token spans
only 308 of 680 relations can be asked, carrying 85.9% of the statements.

So fine relations read cheaply and answer narrowly; coarse relations read expensively and answer
widely. Neither end is the engine's choice — it is how the corpus was written down, which is
what a model builder controls. This measures the curve between them.

The write-down, and why there is no knob
------------------------------------------
Prose is written down by the shipped ingestion, `entroptics_infer.ingest`: each position is cut at
the extent its own middle earns and the middle stands as the relation, because role comes from
position. There is no width to sweep -- the extent of every span is read off the corpus -- so
what this measures is the store the ingestion writes, and the same store with its sentences placed.

The second scale adds one relation used by every token of a sentence, so it is the coarsest
relation a prose store can have, alongside the fine ones the spans already give.

What is measured at each point
--------------------------------
``relations``     how many distinct middles the write-down produces.
``askable``       how many are used at least twice, and what share of the statements they carry.
                  A store that reads cheaply and can be asked nothing is not a cheaper engine.
``work``          the steps a read costs, through the biggest relation and through the median
                  askable one, with the candidates held so the relation is what differs.

What is *not* measured here is whether the read is any good on each store. That needs held-out
questions and is the next arm; this one says what the write-down costs and what it leaves
askable, which are the two quantities a scale trades between.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import sys

import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_read_predicts_the_next_token import SENTENCES  # noqa: E402
from test_whether_many_small_relations_read_cheaply import (ASKABLE, HELD,    # noqa: E402
                                                            a_read_through)

#: What a relation needs to be askable **under the hold-out protocol**, which is how every figure in
#: this program is taken. `ASKABLE` is what the store admits: two uses, because a cross-covariance
#: over one row is that row. The protocol then removes the statement being asked about, which costs
#: its relation one use -- so a relation needs one more than that to survive being asked. Derived
#: from the protocol, not chosen: change the protocol and this follows it.
ASKABLE_HELD_OUT = ASKABLE + 1
from entroptics_infer import ingest                             # noqa: E402

from test_whether_the_read_predicts_the_next_token import body  # noqa: E402


def stores():
    """``(body name, {"spans": ..., "spans + sentences": ...})`` -- the shipped ingestion's stores."""
    name, text = body()
    if text is None:
        return None, {}
    spans = tuple(ingest.spans(text, sentences=SENTENCES))
    placed, _ = ingest.corpus(text, scales=("sentence",), sentences=SENTENCES)
    return name, {"spans": spans, "spans + sentences": tuple(placed)}


def at(label, statements):
    """What this write-down costs and what it leaves askable."""
    engine = Engine.on_statements(statements)
    relations = engine.relations()
    if not relations:
        return None
    sizes = {r: len(engine.uses(r)) for r in relations}
    askable = [r for r in relations if sizes[r] >= ASKABLE]
    survives = [r for r in relations if sizes[r] >= ASKABLE_HELD_OUT]
    if not askable:
        return None
    concepts = engine.concepts()
    held = list(concepts[:HELD])

    biggest = max(relations, key=lambda r: sizes[r])
    middling = sorted(askable, key=lambda r: sizes[r])[len(askable) // 2]
    return {"statements": len(statements), "concepts": len(concepts),
            "vertices": len(engine.vertices()), "relations": len(relations),
            "relations_askable": len(askable),
            "share_of_relations_askable": len(askable) / len(relations),
            "share_of_statements_askable": sum(sizes[r] for r in askable) / len(statements),
            "relations_askable_held_out": len(survives),
            "share_of_statements_askable_held_out": (sum(sizes[r] for r in survives)
                                                     / len(statements)),
            "biggest_relation": sizes[biggest], "median_askable_relation": sizes[middling],
            "through_the_biggest": a_read_through(engine, biggest, held, statements),
            "through_a_median": a_read_through(engine, middling, held, statements)}


def main() -> int:
    print("what a write-down costs to read, and what it leaves askable\n")

    name, written = stores()
    if not written:
        print("  No cached prose body found under corpora/. Nothing to measure.")
        return 0
    rows = {}
    for label, store in written.items():
        got = at(label, store)
        if got is not None:
            rows[label] = got
    if len(rows) < 2:
        print("  Fewer than two write-downs could be measured. Nothing to compare.")
        return 0

    print(f"  {'write-down':<22}{'stmts':>7}{'rels':>7}{'askable':>9}{'of stmts':>10}"
          f"{'held out':>10}{'of stmts':>10}{'big rel':>9}{'work big':>11}{'work med':>11}")
    print("  " + "-" * 108)
    for label, r in rows.items():
        big, med = r["through_the_biggest"], r["through_a_median"]
        print(f"  {label:<22}{r['statements']:>7}{r['relations']:>7}{r['relations_askable']:>9}"
              f"{r['share_of_statements_askable']:>10.1%}"
              f"{r['relations_askable_held_out']:>10}"
              f"{r['share_of_statements_askable_held_out']:>10.1%}"
              f"{r['biggest_relation']:>9}{big['work']:>11}{med['work']:>11}")
    print()

    # what the second scale buys and what it costs, both recorded: a claim about a trade quotes
    # both sides, and a difference worked out in prose is a figure with no producer
    plain, wider = rows["spans"], rows["spans + sentences"]
    scale = {"askable_share_gained": (wider["share_of_statements_askable"]
                                      - plain["share_of_statements_askable"]),
             # the gain that matters, because it is the one the hold-out protocol leaves
             "askable_share_gained_held_out": (wider["share_of_statements_askable_held_out"]
                                               - plain["share_of_statements_askable_held_out"]),
             "times_the_read_costs": (wider["through_a_median"]["work"]
                                      / max(plain["through_a_median"]["work"], 1))}
    record.figures(body=name, sentences=SENTENCES, askable_at=ASKABLE,
                   askable_at_held_out=ASKABLE_HELD_OUT, candidates_held=HELD,
                   by_write_down=rows, the_second_scale=scale)

    print(f"  **Askable is two numbers, and the second is the one the protocol allows.** A "
          f"relation needs")
    print(f"  {ASKABLE} uses to have an operator at all, and {ASKABLE_HELD_OUT} to still have one "
          f"when the statement being asked")
    print(f"  about is removed. The shipped write-down leaves "
          f"{plain['share_of_statements_askable']:.1%} and "
          f"**{plain['share_of_statements_askable_held_out']:.1%}** of its statements askable;")
    print(f"  placing its sentences takes that to {wider['share_of_statements_askable']:.1%} and "
          f"**{wider['share_of_statements_askable_held_out']:.1%}**, for "
          f"**{scale['times_the_read_costs']:.2f}** times the read through a median relation.")
    print()
    print("  **What is not measured here is whether the read is any good on either store.** That")
    print("  needs held-out questions, and this arm says only what the write-down costs and what")
    print("  it leaves askable -- the two quantities a scale trades between.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
