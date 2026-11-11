"""Does the shipped ingestion write the store the measurements were taken on?

Why this has to be checked rather than assumed
------------------------------------------------
`entroptics_infer.ingest` is the step a model builder uses: prose in, statements out. It has no
width. Each position is cut at the extent its own middle earns (`ingest.at_their_own_width`), so
the extent of every span is read off the corpus, and `corpus()` reports the extents it wrote.

**So the figures about prose are about a store, and the ingestion has to write that store.** An
engine whose ingestion quietly reads a different corpus than the research measuring it is worse
than an engine with no ingestion: the numbers still come out, and nothing says they are about
something else. Every prose check in ``research/`` builds its store through three helpers, and each
is held here against `ingest.corpus` on the same text and the same cap:

``spans``      `test_whether_the_read_predicts_the_next_token.written`, the spans with the
               sentence each came from.
``placings``   `test_whether_prose_at_two_scales_changes_the_read.placed`, the sentence
               memberships.
``the ladder`` `test_what_each_scale_of_prose_buys.ladder`, the store at every rung.

What else is checked
----------------------
``the labels``   that no membership label is already a word in the corpus, counted rather than
                 trusted -- the first one was `in`, which the corpus already used as the middle of
                 thirty-six spans, and the merge was silent because the relation count simply did
                 not rise.
``the extents``  the extent of every span the ingestion wrote, counted off its own output against
                 what `corpus()` reports.
``it reads``     that `Engine` can take what comes out and answer from it, because a statement list
                 that does not load is not an ingestion.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import pathlib
import sys

import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import ingest                             # noqa: E402
from entroptics_infer.wrapper import Engine                     # noqa: E402

from test_whether_the_read_predicts_the_next_token import SENTENCES, body, written  # noqa: E402
from test_whether_prose_at_two_scales_changes_the_read import placed  # noqa: E402
from test_what_each_scale_of_prose_buys import ladder           # noqa: E402


def extent(middle: str) -> int:
    """DERIVED: a span's extent is its two ends plus every token of its middle."""
    return len(middle.split()) + ingest.A_PAIR if middle else ingest.A_PAIR


def main() -> int:
    name, text = body()
    if text is None:
        print("  No cached prose body found under corpora/. Nothing to check.")
        return 0

    print("does the shipped ingestion write the store the measurements were taken on?\n")

    got, how = ingest.corpus(text, scales=("sentence",), sentences=SENTENCES)
    got_spans = [s for s in got if s[1] != ingest.IN_SENTENCE]
    got_places = [s for s in got if s[1] == ingest.IN_SENTENCE]

    _, research = written()
    research_spans = [span for _, span in research]
    research_places = placed(text)
    rungs, ladder_how = ladder(text)
    every_rung = all(
        list(rungs[label]) == list(ingest.corpus(text, scales=tuple(ingest.LABELS)[:k],
                                                 sentences=SENTENCES)[0])
        for k, label in enumerate(("+ sentences", "+ paragraphs", "+ the document"), start=1))

    same_spans = research_spans == got_spans
    same_places = sorted(research_places) == sorted(got_places)
    same_sentences = len({i for i, _ in research}) == how["sentences"]

    print(f"  {'':<28}{'research':>14}{'the ingestion':>16}{'same':>7}")
    print("  " + "-" * 65)
    print(f"  {'spans':<28}{len(research_spans):>14}{len(got_spans):>16}"
          f"{('yes' if same_spans else 'NO'):>7}")
    print(f"  {'sentence placings':<28}{len(research_places):>14}{len(got_places):>16}"
          f"{('yes' if same_places else 'NO'):>7}")
    print(f"  {'sentences read':<28}{len({i for i, _ in research}):>14}{how['sentences']:>16}"
          f"{('yes' if same_sentences else 'NO'):>7}")
    print(f"  {'every rung of the ladder':<28}{len(rungs):>14}{len(rungs):>16}"
          f"{('yes' if every_rung else 'NO'):>7}")
    print()

    wrote = collections.Counter(extent(m) for _, m, _ in got_spans)
    extents_reported = sorted(wrote) == list(how["extents"])
    print("  extents the ingestion wrote: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(wrote.items()))
          + f"  (reported by corpus(): {how['extents']}, "
          + f"{'the same' if extents_reported else 'NOT the same'})")

    # every label at every rung, because a collision is silent and the ladder has three
    every = ladder_how["label_collisions"]
    print(f"  label collisions across the whole ladder: {every}")

    engine = Engine.on_statements(got)
    relations = engine.relations()
    reads = len(relations) > 0 and len(engine.concepts()) > 0
    print(f"  the store loads: {len(engine.vertices())} vertices, {len(relations)} relations, "
          f"{len(engine.concepts())} concepts")
    print()

    record.figures(body=name, sentences=SENTENCES,
                   ingestion_spans=len(got_spans), ingestion_placings=len(got_places),
                   ingestion_sentences=how["sentences"], ingestion_paragraphs=how["paragraphs"],
                   research_spans=len(research_spans), research_placings=len(research_places),
                   spans_identical=same_spans, placings_identical=same_places,
                   sentences_match=same_sentences, every_rung_identical=every_rung,
                   ingestion_extents={str(k): v for k, v in sorted(wrote.items())},
                   extents_match_what_corpus_reports=extents_reported,
                   collisions_across_the_ladder=every,
                   vertices=len(engine.vertices()), relations=len(relations),
                   concepts=len(engine.concepts()))

    if (same_spans and same_places and same_sentences and every_rung and extents_reported
            and not any(every.values()) and reads):
        print("  **The shipped ingestion writes the store the prose figures are about**, statement")
        print("  for statement and rung for rung, with no membership label colliding with a word,")
        print("  and the result loads into `Engine` and has relations to read through.")
    else:
        print("  **The research store and the shipped ingestion differ.** Spans identical: "
              f"{same_spans};")
        print(f"  placings: {same_places}; every rung: {every_rung}; collisions: {every}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
