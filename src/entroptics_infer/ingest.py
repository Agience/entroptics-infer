"""Prose in, statements out — the shape a corpus has to be written down in to be read.

Why this exists
-----------------
:class:`~entroptics_infer.wrapper.Engine` reads statements. ``on_statements`` takes triples and
``on_lattice`` reads a mantle store, and someone swapping this engine into a model has neither —
they have text. This is the missing step, and what shape it should take is not a design opinion:
every choice in it is a figure from ``research/``.

What a corpus is written down as
----------------------------------
**A token is a vertex. The text a token sits in is also a vertex.** A span is a statement whose
middle is the relation, because role comes from position and nothing declares a vocabulary — so
naming the last token from the first and the middle is next-token prediction. Membership in a
larger piece of text is a statement of the same shape.

Where the measurements put each choice:

``extents``
    There is no width argument. Each position is cut at the extent its own middle earns
    (:func:`at_their_own_width`, scored by :func:`narrowed_by`), and :func:`corpus` reports the
    extents it wrote under ``how["extents"]``. A fixed width is one zoom imposed on a whole corpus;
    the fixed widths were measured and are the reason one is not imposed: prose written one sentence
    per statement shares 0.1% of its coordinate, at three-token spans 41.8%, against
    hand-written triples' 54.7% (``research/checks/test_what_shape_prose_has_to_be_ingested_in.py``),
    and ``research/checks/test_what_write_down_reads_cheaply_and_answers_widely.py`` prices widths 3, 4
    and 5 at 76.4%, 19.7% and 2.5% askable under hold-out.
    ``research/checks/test_whether_a_corpus_cut_at_its_own_extents_reads_better.py`` compares the
    fixed three-wide cut with this one on the same text, and
    ``research/checks/test_whether_the_shipped_ingestion_writes_the_measured_store.py`` checks that
    :func:`corpus` writes the extents arm's store span for span.

``scales``
    Each larger piece of text a token sits in is another vertex, and
    ``research/checks/test_what_each_scale_of_prose_buys.py`` prices the ladder, and
    ``research/results/test_what_each_scale_of_prose_buys.json`` holds the figures: under hold-out,
    sentences take the share of statements askable from 97.7% to 98.8%, paragraphs to
    99.1%, the whole document to 99.2% — gains of 1.29, 0.33 and 0.08 points, each divided by the
    ratio of its read cost to the rung below's. Each rung buys less than the one below it, and the
    document is the limit: one relation naming every token, relating everything to everything. The
    default is sentences, the rung with the largest askable share for its read cost;
    ``scales=()`` writes spans alone, which reads more accurately on the measured prose.

Nothing here is fitted and nothing is a threshold. The defaults are where the measurements put the
elbow, they are arguments rather than constants, and the figures that chose them are named above so
a caller can disagree with the evidence rather than with a number.

The membership labels
-----------------------
A relation is whatever token stands in the middle, and the model has no reserved vocabulary — so a
membership label that is an ordinary word merges with that word's own statements. With ``in``
as the label, on a corpus that already uses it as the middle of thirty-six spans, 2399 membership
statements pool with them into one relation whose operator runs 98.5% membership and 1.5%
preposition -- and nothing reports it, because the relation count simply does not rise.

So a label here carries a character :data:`WORD` cannot produce, and :func:`collisions` counts what
a label would merge with rather than trusting that it merges with nothing.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

#: What counts as a token. Letters and the apostrophe, which is what the measured arms read, and
#: deliberately *not* the underscore -- that is what keeps a membership label out of the vocabulary.
WORD = re.compile(r"[a-z']+")

#: Where one piece of text ends and the next begins, at each scale a corpus has.
SENTENCE_END = re.compile(r"[.!?]+")
PARAGRAPH_END = re.compile(r"\n\s*\n")

#: The membership relation at each scale. Each carries an underscore `WORD` cannot produce, so it
#: cannot merge with a word in the text -- and `collisions` checks that rather than assuming it.
IN_SENTENCE = "placed_in_sentence"
IN_PARAGRAPH = "placed_in_paragraph"
IN_DOCUMENT = "placed_in_document"

#: The ladder, coarsest last, so `scales=("sentence",)` means what it looks like.
LABELS = {"sentence": IN_SENTENCE, "paragraph": IN_PARAGRAPH, "document": IN_DOCUMENT}

#: **The relation two tokens stand in when they stand next to each other.** Carries an underscore
#: `WORD` cannot produce, for the same reason the membership labels do, and `collisions` checks it
#: rather than assuming it.
#:
#: Why this exists: `at_their_own_width` writes a span as `(first, middle, last)` meaning
#: `first --middle--> last`, which puts a **token** in the operator slot. The program's own
#: correction says a token is content -- `cat` is not an arrow -- and a store written that way
#: cannot be read without supplying one, because the pair a reader actually has in hand, two
#: adjacent tokens, is not an edge in it. Measured on 120 sentences: of 400 held-out spans `(a, c)`
#: is a recorded end-pair 84 times and `(a, b)` 17, those incidental. `adjacencies` writes the
#: other way round, and then every token is content and the arrow is a position.
ADJACENT = "stands_next_to"


#: The places a statement has: what stands first, and what stands last. A span cannot be shorter
#: than the two things a relation relates, and **nothing here says how far apart they are** -- what
#: stands between them may be many tokens, one, or none at all, and which it is, is read off the
#: corpus by :func:`narrowed_by`.
ENDS = ("what stands first", "what stands last")

#: A span of two: the two ends, with nothing between them. The shortest span there is, and the one
#: whose relation is "these stand next to each other".
A_PAIR = len(ENDS)


def what_follows_each_middle(read: list) -> tuple:
    """``({width: {middle: {last: how often}}}, the whole corpus's last-token entropy)``.

    One pass over every piece at every extent it admits. What is counted is not how often a middle
    occurs but what stands after it, because that is the thing an extent is supposed to narrow.
    """
    counts: dict = {}
    overall: dict = {}
    for _paragraph, _index, got in read:
        for start in range(len(got)):
            for width in range(A_PAIR, len(got) - start + 1):
                middle = " ".join(got[start + 1:start + width - 1])
                last = got[start + width - 1]
                after = counts.setdefault(width, {}).setdefault(middle, {})
                after[last] = after.get(last, 0) + 1
                overall[last] = overall.get(last, 0) + 1
    return counts, _entropy(overall)


def _entropy(counts: dict) -> float:
    """Shannon entropy, in bits, of a mapping from outcome to how often it occurred."""
    total = sum(counts.values())
    if total <= 0:
        return float(0)
    out = float(0)
    for n in counts.values():
        if n > 0:
            p = n / total
            out -= p * math.log2(p)
    return out


def narrowed_by(after: dict, overall: float) -> float:
    """How much knowing this middle narrows what stands last, over the whole corpus.

    An extent is worth what it tells you about the token it points at. A middle seen ``k`` times is
    followed by some distribution of last tokens; the bits it saves per occurrence are what that
    distribution is narrower by than the corpus as a whole, and it saves them ``k`` times.

    **The reference is what ``k`` observations could show, not what the corpus holds.** A middle seen
    once is followed by exactly one token and looks perfectly predictive -- entropy zero, the whole
    corpus narrowed away -- and it has told you nothing, because one draw from any distribution looks
    like certainty. ``k`` draws cannot exhibit more than ``log2(k)`` bits of spread whatever they are
    drawn from, so that is what they are credited against.

    The two degenerate cases fall out rather than being forbidden:

    * a middle seen once: ``log2(1)`` is 0 and its observed entropy is 0, so it narrows nothing.
    * a middle standing between every pair -- the empty one, adjacency -- is followed by the
      corpus's own distribution, so it narrows nothing either.

    Between them it is largest where a middle is seen often enough to have shown its spread and is
    narrower than the corpus anyway. Total surprisal is the wrong comparison: it collapses the
    store from 571 relations to 14, because a middle standing between almost every pair accounts for
    a great deal of surprisal while saying nothing about any particular pair.
    """
    times = sum(after.values())
    if times <= 0:
        return float(0)
    # **Miller-Madow**, the plug-in bias the instrument itself corrects entropy with
    # (`PAPER-1-THE-INSTRUMENT` §8.1): an entropy estimated from few samples reads too low by about
    # `(distinct - 1) / (2 k ln2)`, because outcomes that were never drawn contribute nothing to the
    # sum and the spread looks narrower than it is. Without it a middle seen twice, followed by two
    # different tokens, reads as one perfectly resolved bit.
    distinct = sum(1 for n in after.values() if n > 0)
    observed = _entropy(after) + (distinct - 1) / (2 * times * math.log(2))
    could_show = min(math.log2(times), overall) if times > 1 else float(0)
    return times * max(could_show - observed, float(0))


def at_their_own_width(read: list, counts: dict, overall: float) -> list:
    """``[(first, middle, last), ...]`` -- every span cut at the extent its own middle earns.

    **There is no width here, and that is the point.** A width is a zoom, and one width is one zoom
    imposed on a whole corpus: it says a phrase in this text and a phrase in that one stand at the
    same distance from the reader, which nothing about the text supports. So each position takes the
    extent whose middle narrows what follows it most, by :func:`narrowed_by`, and ties go to the
    shorter span because it cuts more from the same text and nothing else distinguishes them.

    Zoomed out there is a forest, zoomed in a tree, then a leaf, and a corpus is all three at once in
    different places. This is that, written down: the text cut at the resolution each part of it
    actually has, rather than at one resolution chosen for all of it.
    """
    out = []
    for _paragraph, _index, got in read:
        for start in range(len(got)):
            best, at = float(-1), None
            for width in range(A_PAIR, len(got) - start + 1):
                middle = " ".join(got[start + 1:start + width - 1])
                after = counts.get(width, {}).get(middle, {})
                narrows = narrowed_by(after, overall)
                if narrows > best:
                    best, at = narrows, (width, middle)
            if at is None:
                continue
            width, middle = at
            out.append((got[start], middle, got[start + width - 1]))
    return out


def adjacencies(read: list) -> list:
    """``[(token, ADJACENT, the token after it), ...]`` -- prose written so no token is an arrow.

    The alternative to :func:`at_their_own_width`, and the difference is which slot a token can
    occupy. There, a span becomes ``first --middle--> last`` and the middle is a word; here every
    token is content, standing at both ends of statements and never in the operator slot, and
    the only relation is the one position itself supplies.

    **What this gives up, and why it is not a loss of derivation.** The derived width chose how far
    apart the two ends stand by asking which middle narrowed what followed it most. With adjacency
    there is no middle to choose, so that reading does not set the statement's shape. What sets a
    coordinate's reach is then the walk and its aperture, which is where the program already puts
    that decision, and the scales ladder still carries the coarser groupings.

    **What it buys.** ``carry(token, ADJACENT)`` is next-token prediction stated in the engine's own
    terms, with nothing supplied: one operator derived from every adjacent pair the corpus wrote
    down. And the pair a reader actually holds becomes an edge, so ``between`` and ``ahead`` can be
    asked about it at all.
    """
    out = []
    for _paragraph, _index, got in read:
        for i in range(len(got) - 1):
            out.append((got[i], ADJACENT, got[i + 1]))
    return out


def tokens(text: str) -> list:
    """The tokens of one piece of text, lowered, in order."""
    return WORD.findall(text.lower())


def pieces(text: str, *, sentences: int | None = None) -> list:
    """``[(paragraph index, sentence index, tokens), ...]`` in reading order.

    One pass, so a sentence's paragraph is the one it came from rather than a regrouping guessed
    afterwards.

    **Only sentences with something to stand between two ends are counted** -- more than
    :data:`A_PAIR` tokens -- and a paragraph that holds none takes no index, so ``sentences=120``
    counts sentences that carry statements and a cap reads the same store wherever it is applied.
    """
    out, sentence, paragraph = [], 0, 0
    for block in PARAGRAPH_END.split(text):
        used = False
        for piece in SENTENCE_END.split(block):
            got = tokens(piece)
            if len(got) <= A_PAIR:
                continue
            out.append((paragraph, sentence, got))
            sentence += 1
            used = True
            if sentences is not None and sentence >= sentences:
                break
        if used:
            paragraph += 1
        if sentences is not None and sentence >= sentences:
            break
    return out


def spans(text: str, *, sentences: int | None = None) -> list:
    """``[(first, middle, last), ...]`` -- every span, each at the extent its own middle earns.

    Duplicates are kept because a statement said twice is said twice: the operator is a sum over the
    statements using a relation, so repetition is evidence and removing it would be a choice about
    the data.
    """
    read = pieces(text, sentences=sentences)
    counts, overall = what_follows_each_middle(read)
    return at_their_own_width(read, counts, overall)


def memberships(text: str, *, scales: Sequence[str] = ("sentence",),
                sentences: int | None = None) -> list:
    """``[(token, label, container), ...]`` -- one statement per distinct token and container.

    Distinct, because a token appearing four times in a sentence is in it once; repeating the
    statement would weight the relation by how wordy the piece was rather than by what it holds.
    """
    unknown = [s for s in scales if s not in LABELS]
    if unknown:
        raise ValueError(f"no such scale: {unknown}; the ladder is {sorted(LABELS)}")
    out = set()
    for paragraph, index, got in pieces(text, sentences=sentences):
        where = {"sentence": f"sentence {index}", "paragraph": f"paragraph {paragraph}",
                 "document": "the document"}
        for token in got:
            for scale in scales:
                out.add((token, LABELS[scale], where[scale]))
    return sorted(out)


def collisions(statements: Iterable[Sequence[str]],
               scales: Sequence[str] = ("sentence",)) -> dict:
    """``{label: how many statements already stand on it}`` -- counted, never assumed.

    A membership label that is already a word in the text merges with that word's own statements and
    the merge is silent: the relation count does not rise, and nothing else says so either. This
    is what noticing it looks like.
    """
    labels = [LABELS[s] for s in scales if s in LABELS]
    found = {label: 0 for label in labels}
    for parts in statements:
        if len(parts) > 2 and parts[1] in found:
            found[parts[1]] += 1
    return found


def corpus(text: str, *, scales: Sequence[str] = ("sentence",),
           sentences: int | None = None) -> tuple:
    """``(statements, how)`` -- a corpus written down ready to read.

    ``how`` reports what was written and what it would merge with, because an ingestion that quietly
    pools two meanings into one relation is worse than one that refuses: the figures come out and
    nothing says they are about a store that does that.
    """
    written = spans(text, sentences=sentences)
    placed = memberships(text, scales=scales, sentences=sentences)
    statements = tuple(written) + tuple(placed)
    # what was *ingested*, not what the file holds: reporting the document's totals beside a
    # capped read describes a store nobody built
    read = pieces(text, sentences=sentences)
    extents = sorted({len(m.split()) + A_PAIR if m else A_PAIR for _, m, _ in written})
    how = {"extents": extents, "scales": tuple(scales),
           "spans": len(written), "memberships": len(placed),
           "sentences": len({i for _, i, _ in read}),
           "paragraphs": len({p for p, _, _ in read}),
           "tokens": len({t for span in written for t in (span[0], span[2])}),
           "label_collisions": collisions(written, scales)}
    return statements, how
