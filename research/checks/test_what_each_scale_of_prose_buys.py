"""Text is a vertex at every scale: what each rung of the ladder buys, and what it costs.

The model this measures
-------------------------
A token is a vertex. The text a token sits in is **also** a vertex, and so is the text that sits in.
Tokens are colimits of longer prose and it continues in both directions: span, sentence, paragraph,
document. Each is a vertex by the same rule and ``(token, placed_in_X, X)`` is a statement like any
other — no new shape, no new code, and the membership token earns an operator the way every middle
token does, from the statements that use it.

`test_what_write_down_reads_cheaply_and_answers_widely.py` establishes what one extra scale does:
adding the sentence takes prose from **97.7%** of its statements askable under hold-out to
**98.8%**, and the read through a median relation costs **0.83** times what it costs without it.
**So on that store one more scale buys little and costs little**, and the question is what each
further rung of the ladder is worth.

Why the answer is not obvious
-------------------------------
A coarser scale relates more tokens to each other, which is what buys askability. It also collects
more statements under one relation, and §47.166 measures a read costing what its relation has
collected. The **document** scale is the limit of both: one relation naming every token in the
corpus, which relates everything to everything and is the most expensive relation a store can have.

So the ladder should show askability rising and cost rising with it, and where the curve stops being
worth climbing is a measurement rather than a preference.

What is measured at each rung
-------------------------------
``relations``    how many distinct middles the store now has.
``askable``      at **2** uses, which is what a store admits, and at **3**, which is what survives
                 the hold-out protocol removing the statement being asked about.
``work``         the steps a read costs through the biggest relation and through the median askable
                 one, with the candidates held so the relation is what differs.

The membership labels contain an underscore, which the word reader cannot produce -- it matches
``[a-z']+`` -- so a scale's relation is its own and cannot merge with a word in the text. That is
checked here rather than assumed: a label such as `in` is an ordinary English word, already
standing as the middle of spans in the text.
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

from entroptics_infer import ingest                             # noqa: E402

from test_whether_the_read_predicts_the_next_token import SENTENCES, body  # noqa: E402
from test_what_write_down_reads_cheaply_and_answers_widely import (ASKABLE,  # noqa: E402
                                                                  ASKABLE_HELD_OUT, at)

#: the rungs, coarsest last -- the shipped ingestion's own labels
IN_SENTENCE = ingest.IN_SENTENCE
IN_PARAGRAPH = ingest.IN_PARAGRAPH
IN_DOCUMENT = ingest.IN_DOCUMENT


def ladder(text):
    """``({rung: statements}, how)`` -- the shipped ingestion at each rung of its ladder."""
    spans = tuple(ingest.spans(text, sentences=SENTENCES))
    out = {"spans": spans}
    for k, rung in enumerate(("+ sentences", "+ paragraphs", "+ the document"), start=1):
        statements, how = ingest.corpus(text, scales=tuple(ingest.LABELS)[:k], sentences=SENTENCES)
        out[rung] = tuple(statements)
    return out, how


def main() -> int:
    name, text = body()
    if text is None:
        print("  No cached prose body found under corpora/. Nothing to measure.")
        return 0
    ladder_of, how = ladder(text)
    spans = ladder_of["spans"]

    print("what each scale of prose buys, and what it costs\n")
    print(f"  {len(spans)} spans written by the shipped ingestion from {name}, over "
          f"{how['sentences']} sentences and {how['paragraphs']} paragraphs\n")

    # **No membership label may already be a word in the text.** Counted by the ingestion itself.
    collides = how["label_collisions"]
    if any(collides.values()):
        print(f"  **A membership label is already a word in this text**: {collides}. Every figure")
        print(f"  below pools two meanings into one relation.")
    else:
        print(f"  no membership label is a word in this text, so each scale's relation is its own")
    print()

    rows = {}
    for label, statements in ladder_of.items():
        got = at(label, statements)
        if got is not None:
            rows[label] = got

    if len(rows) < 2:
        print("  Fewer than two rungs could be measured. Nothing to compare.")
        return 0

    print(f"  {'rung':<16}{'stmts':>7}{'rels':>6}{'askable':>9}{'of stmts':>10}"
          f"{'held out':>10}{'of stmts':>10}{'big rel':>9}{'work big':>11}{'work med':>11}")
    print("  " + "-" * 99)
    for label, r in rows.items():
        big, med = r["through_the_biggest"], r["through_a_median"]
        print(f"  {label:<16}{r['statements']:>7}{r['relations']:>6}{r['relations_askable']:>9}"
              f"{r['share_of_statements_askable']:>10.1%}"
              f"{r['relations_askable_held_out']:>10}"
              f"{r['share_of_statements_askable_held_out']:>10.1%}"
              f"{r['biggest_relation']:>9}{big['work']:>11}{med['work']:>11}")
    print()

    rungs = list(rows)
    steps = []
    for a, b in zip(rungs, rungs[1:]):
        before, after = rows[a], rows[b]
        steps.append({
            "from": a, "to": b,
            "askable_gained": (after["share_of_statements_askable_held_out"]
                               - before["share_of_statements_askable_held_out"]),
            "times_the_read": (after["through_a_median"]["work"]
                               / max(before["through_a_median"]["work"], 1))})
    # **The quantity a builder decides on**, recorded rather than worked out in prose: points of
    # askable share bought per unit of read cost. A rung that buys less per unit than the one below
    # it is where the ladder starts flattening, and that is the whole shape of the result.
    for s in steps:
        s["points_per_times_the_read"] = (s["askable_gained"] * 100
                                          / max(s["times_the_read"], 1e-9))
    record.figures(sentences=SENTENCES, body=name,
                   spans=len(spans), sentences_read=how["sentences"],
                   paragraphs_read=how["paragraphs"],
                   # the two thresholds a claim about askability has to name, since the difference
                   # between them is the hold-out protocol removing the statement being asked about
                   askable_at=ASKABLE, askable_at_held_out=ASKABLE_HELD_OUT,
                   label_collisions=collides, by_rung=rows, by_step=steps)

    print(f"  {'step':<32}{'askable gained':>16}{'times the read':>16}{'points per':>13}")
    print("  " + "-" * 77)
    for s in steps:
        print(f"  {s['from'] + ' to ' + s['to']:<32}{s['askable_gained']:>+16.1%}"
              f"{s['times_the_read']:>16.2f}{s['points_per_times_the_read']:>13.2f}")
    print()

    worth = [s for s in steps if s["askable_gained"] > float(0)]
    best = max(steps, key=lambda s: s["points_per_times_the_read"])
    falling = all(a["points_per_times_the_read"] > b["points_per_times_the_read"]
                  for a, b in zip(steps, steps[1:]))
    rates = ", ".join(f"{s['points_per_times_the_read']:.2f}" for s in steps)
    if falling:
        print(f"  **All {len(steps)} rungs buy something, and each buys less per unit of read than "
              f"the one below it.**")
    else:
        print(f"  **{len(worth)} of the {len(steps)} rungs buy something**, and the return does not "
              f"fall all the way up.")
    print(f"  In points of askable share per unit of read cost, the rungs run {rates}. The best is")
    print(f"  `{best['from']} to {best['to']}` at **{best['askable_gained']:+.1%}** for "
          f"**{best['times_the_read']:.2f}** times the read.")
    print()
    last = steps[-1]
    print(f"  **The document is the limit of the ladder and shows what the limit costs**: one "
          f"relation")
    print(f"  naming every token, so it relates everything to everything. That step buys "
          f"**{last['askable_gained']:+.1%}**")
    print(f"  for **{last['times_the_read']:.2f}** times the read. **A scale is worth climbing "
          f"while it groups tokens that")
    print(f"  belong together and stops when it groups all of them**, and where that is, is this")
    print(f"  table rather than a preference.")
    print()
    print(f"  What is not measured here is whether the read is any **better** at each rung. That")
    print(f"  needs held-out questions per rung, which is what "
          f"`test_whether_prose_at_two_scales_changes_the_read.py` does for one of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
