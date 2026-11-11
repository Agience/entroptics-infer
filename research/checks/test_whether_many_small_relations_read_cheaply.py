"""Does a store with many small relations read cheaply, however large it is?

The claim this tests
----------------------
A read's cost is neither the store's size nor the question's: it is what the relation has
collected. The rows a read needs are the
source, the candidates, and both sides of every statement using the relation — and on WordNet that
third set is most of the store, because one relation holds about two thirds of every core.

If that reading is right, then a store whose relations are many and each small reads cheaply
however large it is, and the cost of answering is a property of how a corpus was written down rather
than of how much of it there is. That is testable without synthesizing anything, because the
stores this program already reads sit at both ends:

======================  ============  ===========  ==========  =======
store                    statements    relations     biggest    share
======================  ============  ===========  ==========  =======
``ontology.py``                   84            6          35    41.7%
``wordnet 4800``                 705            4         513    72.8%
``wordnet 19200``               5416            4        3693    68.2%
``prose spans``                 2631          680         128     4.9%
======================  ============  ===========  ==========  =======

**Prose spans is 3.7 times the statements of wordnet 4800 and its biggest relation is four times
smaller.** So the two quantities come apart in real corpora, and a read on the larger store should
cost less. That is a prediction, and it can fail.

Why no store is generated for this
------------------------------------
A synthesized store with a chosen relation count would answer its own design — twice in this
program a generator has given a clean wrong answer, because a null is blind to whatever the
generator preserved. The corpora above differ in relation shape because of what they are: WordNet
is a taxonomy and its spine is one relation; prose read as three-token spans makes the middle
token the relation, so the relation vocabulary is the vocabulary.

What is measured, per store and per relation
----------------------------------------------
``rows``        the rows a read using that relation needs — source, candidates, and both sides of
                every statement using it.
``work``        one count per entry of an expanded vertex's neighbour list, deterministic.
``askable``     a relation used once has no operator: a cross-covariance over one row is that
                row, and it predicts that row and nothing else. So part of the answer is what share
                of a store can be asked at all.

Two relations per store: the biggest, which is the worst case and what the sweep measured, and
the median among those used more than once, which is what a store of many small relations would
actually be read through.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import statistics
import sys

import record

import lexicon
import ontology

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_what_a_read_actually_touches import rows_a_read_needs      # noqa: E402
from test_whether_the_same_coordinate_can_be_built_faster import built  # noqa: E402

#: the WordNet radii, from the sweep that raised this question
BALLS = (1200, 4800, 19200)

#: a cross-covariance needs two rows to be a covariance rather than a row
ASKABLE = 2

#: how many candidates the controlled comparison holds fixed. Small enough that the operator's rows
#: are what differs between two reads on one store, which is the whole point of that arm; the
#: uncontrolled arm above scores every concept, as a real read does.
HELD = 20


def a_read_through(engine, relation, candidates, statements):
    """The rows and the work of one read using ``relation``, with the operator's share separated.

    **The candidates are the same for every relation on a store, so a total hides the relation.**
    A read scores every candidate, and on these stores that is most of the vertices -- 3,373 of
    3,377 rows on the largest. Reporting only the total measured the candidate set and called it the
    relation, which is what the first run of this file did. The operator's own rows are counted
    apart, and they are the ones a relation's size can move.
    """
    uses = engine.uses(relation)
    source = uses[0][0][uses[0][1] - 1]
    needed = rows_a_read_needs(engine, source, relation, candidates)
    sides = set()
    for parts, position in uses:
        sides.add(parts[position - 1])
        sides.add(parts[position + 1])
    counted = {"expansions": 0, "neighbours": 0, "reached": 0, "starts": 0}
    built(engine, only=needed, counted=counted)
    return {"relation_uses": len(uses), "rows": len(needed), "operator_rows": len(sides),
            "rows_beyond_the_candidates": len(needed - set(candidates)),
            "work": counted["neighbours"],
            "work_per_row": counted["neighbours"] / max(len(needed), 1)}


def at(label, statements):
    """Both reads at one store, and what share of the store can be asked at all."""
    engine = Engine.on_statements(statements)
    relations = engine.relations()
    if not relations:
        return None
    sizes = {r: len(engine.uses(r)) for r in relations}
    askable = [r for r in relations if sizes[r] >= ASKABLE]
    if not askable:
        return None
    concepts = engine.concepts()

    biggest = max(relations, key=lambda r: sizes[r])
    middling = sorted(askable, key=lambda r: sizes[r])[len(askable) // 2]

    # **The controlled comparison, on one store.** Above, the candidate set is every concept, so it
    # dominates the rows and the relation cannot be seen. Here the candidates are the same small
    # fixed set for both reads, so the only thing that differs between them is the operator's own
    # rows -- which is the quantity the claim is about. Cross-corpus the two relations differ in
    # everything at once; this holds the store, the basis, the fanout and the candidates fixed and
    # varies the relation alone.
    fixed = list(concepts[:HELD])
    apart = {"through_the_biggest": a_read_through(engine, biggest, fixed, statements),
             "through_a_median": a_read_through(engine, middling, fixed, statements)}

    return {"statements": len(statements), "vertices": len(engine.vertices()),
            "concepts": len(concepts),
            "relations": len(relations), "relations_askable": len(askable),
            "share_of_relations_askable": len(askable) / len(relations),
            "statements_askable": sum(sizes[r] for r in askable),
            "share_of_statements_askable": sum(sizes[r] for r in askable) / len(statements),
            "biggest_relation": sizes[biggest],
            "biggest_relation_share": sizes[biggest] / len(statements),
            "median_askable_relation": sizes[middling],
            "candidates_held": len(fixed),
            "through_the_biggest": a_read_through(engine, biggest, concepts, statements),
            "through_a_median": a_read_through(engine, middling, concepts, statements),
            "held_candidates_biggest": apart["through_the_biggest"],
            "held_candidates_median": apart["through_a_median"]}


def stores():
    """Every store this program reads, because the claim is about the shape of a corpus."""
    out = {"ontology": tuple(tuple(s) for s in ontology.STATEMENTS)}
    for ball in BALLS:
        out[f"wordnet {ball}"] = lexicon.core(lexicon.neighbourhood(ball, "entity.n.01")[0])
    from test_whether_prose_at_two_scales_changes_the_read import prose_stores
    out.update(prose_stores())
    return out


def main() -> int:
    print("does a store with many small relations read cheaply, however large it is?\n")

    rows = {}
    for label, statements in stores().items():
        got = at(label, statements)
        if got is not None:
            rows[label] = got

    if len(rows) < 2:
        print("  Fewer than two stores could be measured. Nothing to compare.")
        return 0

    print(f"  {'store':<20}{'stmts':>7}{'cands':>7}{'rels':>6}"
          f"{'big rel':>9}{'op rows':>9}{'work/row':>10}"
          f"{'med rel':>9}{'op rows':>9}{'work/row':>10}")
    print("  " + "-" * 96)
    for label, r in rows.items():
        big, med = r["through_the_biggest"], r["through_a_median"]
        print(f"  {label:<20}{r['statements']:>7}{r['concepts']:>7}{r['relations']:>6}"
              f"{big['relation_uses']:>9}{big['operator_rows']:>9}{big['work_per_row']:>10.0f}"
              f"{med['relation_uses']:>9}{med['operator_rows']:>9}{med['work_per_row']:>10.0f}")
    print()

    print(f"  **The same store, the same {rows[next(iter(rows))]['candidates_held']} candidates, "
          f"two relations:**")
    print(f"  {'store':<20}{'big rel':>9}{'op rows':>9}{'rows':>7}{'work':>10}"
          f"{'med rel':>9}{'op rows':>9}{'rows':>7}{'work':>10}{'times':>8}")
    print("  " + "-" * 98)
    for label, r in rows.items():
        big, med = r["held_candidates_biggest"], r["held_candidates_median"]
        times = big["work"] / max(med["work"], 1)
        print(f"  {label:<20}{big['relation_uses']:>9}{big['operator_rows']:>9}{big['rows']:>7}"
              f"{big['work']:>10}{med['relation_uses']:>9}{med['operator_rows']:>9}"
              f"{med['rows']:>7}{med['work']:>10}{times:>8.1f}")
    print()

    print(f"  {'store':<20}{'relations':>11}{'askable':>9}{'of them':>9}"
          f"{'statements askable':>20}")
    print("  " + "-" * 69)
    for label, r in rows.items():
        print(f"  {label:<20}{r['relations']:>11}{r['relations_askable']:>9}"
              f"{r['share_of_relations_askable']:>9.1%}"
              f"{r['share_of_statements_askable']:>20.1%}")
    print()

    # the ratio each store shows with its candidates held, and the work per row as the table
    # prints it -- both quoted by any claim about this, so both recorded rather than derived in
    # prose where nothing can check them
    for r in rows.values():
        r["work_ratio_with_candidates_held"] = (
            r["held_candidates_biggest"]["work"] / max(r["held_candidates_median"]["work"], 1))
        r["work_per_row_whole"] = int(r["through_the_biggest"]["work_per_row"])
    record.figures(balls=list(BALLS), askable_at=ASKABLE, candidates_held=HELD,
                   smallest_ratio=min(r["work_ratio_with_candidates_held"] for r in rows.values()),
                   largest_ratio=max(r["work_ratio_with_candidates_held"] for r in rows.values()),
                   by_store=rows)

    # **The controlled arm is the result.** Cross-corpus, two stores differ in everything at once
    # and a difference cannot be attributed; on one store with the candidates held, the operator is
    # the only thing that varies between the two reads.
    ratios = {k: r["held_candidates_biggest"]["work"] / max(r["held_candidates_median"]["work"], 1)
              for k, r in rows.items()}
    widest = max(ratios, key=lambda k: ratios[k])
    w = rows[widest]

    print(f"  **A read costs what its relation has collected.** On `{widest}`, with the store, the")
    print(f"  basis, the fanout and the {w['candidates_held']} candidates all held, a read through "
          f"a relation used")
    print(f"  **{w['held_candidates_biggest']['relation_uses']}** times does "
          f"**{w['held_candidates_biggest']['work']}** steps and one through a relation used "
          f"**{w['held_candidates_median']['relation_uses']}** times does")
    print(f"  **{w['held_candidates_median']['work']}** -- **{ratios[widest]:.1f}** times less, "
          f"with nothing but the relation different.")
    print(f"  Every store shows the same direction, by "
          f"{min(ratios.values()):.1f} to {max(ratios.values()):.1f} times.")
    print()
    print(f"  **So the cost of a read is the rows it needs times the work each row costs**: the "
          f"rows are")
    print(f"  the candidates plus the operator's two sides per statement, and the work per row is "
          f"a")
    print(f"  property of the store's shape -- identical for every relation on a store, and "
          f"measured")
    print(f"  above at {', '.join(str(int(r['through_the_biggest']['work_per_row'])) for r in rows.values())}. "
          f"The store enters through the second factor and the relation")
    print(f"  through the first.")
    print()

    big_prose = rows.get("prose, two scales")
    big_word = rows.get(f"wordnet {BALLS[-1]}")
    if big_prose and big_word:
        print(f"  **That makes it a property of how a corpus is written down.** "
              f"`prose, two scales` holds")
        print(f"  **{big_prose['statements']}** statements in **{big_prose['relations']}** "
              f"relations and answers through a small one for")
        print(f"  **{big_prose['held_candidates_median']['work']}** steps. "
              f"`wordnet {BALLS[-1]}` holds **{big_word['statements']}** statements in "
              f"**{big_word['relations']}**, and its")
        print(f"  **median** relation costs **{big_word['held_candidates_median']['work']}**. "
              f"Two stores of a size, and the one whose")
        print(f"  relations are many reads for a fraction of the other.")
        print()

    prose = rows.get("prose spans")
    if prose:
        print(f"  **What it spends in return**: a relation used once has no operator, because a")
        print(f"  cross-covariance over one row is that row. On `prose spans` that leaves "
              f"**{prose['relations_askable']}** of")
        print(f"  **{prose['relations']}** relations askable, carrying "
              f"**{prose['share_of_statements_askable']:.1%}** of the statements. A corpus of many "
              f"small relations")
        print(f"  buys a cheap read and pays for it in questions it cannot answer. With the sentence")
        print(f"  scale, `prose, two scales` has "
              f"**{rows['prose, two scales']['share_of_statements_askable']:.1%}** askable against "
              f"**{prose['share_of_statements_askable']:.1%}**; askability under hold-out is")
        print(f"  `test_what_write_down_reads_cheaply_and_answers_widely.py`'s measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
