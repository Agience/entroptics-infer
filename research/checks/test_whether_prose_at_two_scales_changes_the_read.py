"""Next-token prediction when the store holds the sentence as a vertex too, not only the spans.

What this is asking
---------------------
`test_whether_the_read_predicts_the_next_token.py` writes prose as three-token spans -- a token is a
vertex, a span is a statement, and naming the third token from the first two is next-token
prediction. On that store the read names 7.8% of 90 held-out spans against counting's 16.7%,
and paired it loses 8 to 0.

That store holds one scale. A token meets another token only when the two stand in spans whose
remaining two tokens are identical, which is why the sharing measured 41.8% and no higher. The
text itself is not in it. The sentence those spans came out of is a thing that exists, tokens are
what it is made of, and nothing in the store says so.

So this adds it. A sentence is a vertex like any other and ``(token, in, sentence)`` is a statement
like any other -- no new shape, no new code in the engine, and ``in`` is a token used in the middle
position exactly as ``is_a`` is. Two tokens in one sentence then meet on the axis that sentence
opens, which is a kind of sharing the span store cannot produce at any width.

**Whether that helps is the question.** It might not: a sentence of twenty tokens divides what it
carries by twenty, so a large scale can dilute rather than inform, and the growth sweep says the
coordinate is quadratic in what is in it.

The arms
----------
``spans``                 the existing store, unchanged, so the comparison has a floor that is
                          already in the ledger.
``spans + sentences``     the same spans, plus one statement per distinct ``(token, sentence)``.
``sentences permuted``    the same statements with the sentence each token is placed in shuffled.
                          The same vertices, each sentence named as often as it was and each token
                          placed as often as it was; only which tokens a sentence puts together is
                          destroyed. Anything the second arm gains over this is the text rather than
                          the extra axes.
``in the sentence``       the second arm's read again, scored on only the tokens that share a
                          sentence with the one being asked from -- the candidates a counter told
                          the same thing would be choosing between.

Counting runs on every arm, from the spans alone and with its backoff stated in full as the
companion file states it, because a derived read given more material has to be compared with a
counter given the same material rather than with the counter it already beat or lost to.

The hold-out
--------------
Every span joining the two tokens is removed, as in the companion file. The ``(token, sentence)``
statements are not removed, because they are not the fact being asked -- but they do put the
asked token and its answer in one sentence, which is a two-hop path to the answer and is exactly
what the permuted arm is there to price. A gain over the permutation is a read; a gain over the
span store alone is not, because the extra axes alone could produce it.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import os
import random
from concurrent import futures

import numpy as np

import ask
import record

from entroptics_infer import ingest
from entroptics_infer.wrapper import Engine

from test_whether_the_read_predicts_the_next_token import (QUESTIONS, SEED, SENTENCES, body,
                                                           counted, written)
# `Engine.places`; `test_whether_the_same_coordinate_can_be_built_faster` checks it cell for cell
from test_whether_the_same_coordinate_can_be_built_faster import built

#: The token used in the middle position to place a token in a sentence. A vertex, not a keyword:
#: nothing in the engine knows this string, and it earns an operator the same way every other middle
#: token does -- from the statements that use it.
#:
#: **It has to be a token the corpus does not use, and that is checked rather than assumed.** This
#: was `"in"`, which is an ordinary English word: on the body read here it already stood as the
#: middle of **36** spans -- `adventures in wonderland`, `sends in a` -- so the 2399 membership
#: statements were pooled with them into one relation of 2435, whose operator was then 98.5%
#: membership and 1.5% preposition. The model has no reserved vocabulary by design, a relation being
#: whatever token stands in the middle, so the collision was the design working on a carelessly
#: chosen label. `placed_in_sentence` contains a character the word reader cannot produce -- it
#: matches `[a-z']+`, so no underscore -- which makes the distinctness a property of the reader
#: rather than a hope about the corpus.
IN = ingest.IN_SENTENCE

#: a third of the box, because it is shared. Overridable, never a majority by default.
WORKERS = int(os.environ.get("ENTROPTICS_WORKERS") or max(2, (os.cpu_count() or 4) // 3))

#: how many spans to ask about. The companion file's **120** is what a per-question rebuild of the
#: shipped walk could afford; the sampling is uniform over the distinct spans either way, so raising
#: this changes the precision of the figure and nothing about the store or the arms. Declared here
#: rather than swept, because the sweep that matters is over the arms.
ASKING = int(os.environ.get("ENTROPTICS_QUESTIONS") or QUESTIONS)


def placed(text):
    """``[(token, IN, "sentence N"), ...]`` -- the shipped ingestion's sentence memberships."""
    return ingest.memberships(text, scales=("sentence",), sentences=SENTENCES)


def by_sentence(places):
    """``{token: {sentence, ...}}`` -- what the counting arm is told when it is told the same."""
    out = collections.defaultdict(set)
    for token, _, container in places:
        out[token].add(container)
    return out


def prose_stores():
    """``{"prose spans": ..., "prose, two scales": ...}`` -- the shipped ingestion's two stores."""
    name, text = body()
    if text is None:
        return {}
    _, got = written()
    store = tuple(span for _, span in got)
    return {"prose spans": store, "prose, two scales": store + tuple(placed(text))}


def permute(statements, rng):
    """The same statements with the sentence each token is placed in shuffled.

    The multiset of sentence names is permuted, so each sentence is named as often as it was and
    each token is placed as often as it was. What is destroyed is which tokens a sentence puts
    together, which is the only thing the arm above claims to use.

    A shuffle can land the same token in the same sentence twice; those duplicates are dropped, so
    the permuted arm is at most as large as the arm it is a null for and never larger. How many were
    dropped is returned, because a null that lost a tenth of its statements is a different store and
    the reader should be able to see that it did not.
    """
    places = [s for s in statements if s[1] == IN]
    rest = [s for s in statements if s[1] != IN]
    names = [s[2] for s in places]
    rng.shuffle(names)
    moved = sorted({(a, IN, name) for (a, _, _), name in zip(places, names)})
    return rest + moved, len(places) - len(moved)


def ask_one(engine, tokens, store, a, b, c, *, among=None):
    """One held-out question on one store. ``None`` when it cannot be asked.

    ``tokens`` is the candidate universe and is the same on every arm: the tokens the prose is
    made of, never the sentence vertices. A next token is a token, and letting a store that has more
    vertices in it also offer more candidates would change the draw between arms rather than the
    read.

    ``among`` narrows it further, to score the read on exactly the candidates a counter told about
    the sentence would be choosing between.
    """
    pair = [s for s in store if s[1] != IN and s[0] == a and s[2] == c]
    held = [s for s in store if s not in pair]
    vectors = built(engine, without=pair)
    if a not in vectors or c not in vectors:
        return None
    landed = engine.carry(a, b, vectors, without=pair)
    if landed is None:
        return None
    stated = {z for x, y, z in held if y == b and x == a}
    pool = [t for t in tokens if t in vectors and t != a and t not in stated]
    if among is not None:
        pool = [t for t in pool if t in among]
    if c not in pool or len(pool) < 2:
        return None
    scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
    spans_held = [s for s in held if s[1] != IN]
    return {"pool": len(pool), "read": pool[int(np.argmax(scores))],
            "counted": counted(spans_held, a, b, pool), "truth": c}


#: the arms, named once so the pool and the table agree on what came back
ARMS = ("spans", "spans + sentences", "sentences permuted", "in the sentence")


def _one_question(job):
    """Every arm on one held-out question, in its own process.

    **The question is the unit of work and the questions are independent.** Each one rebuilds four
    coordinates from scratch, because the protocol removes the asked-about spans from the coordinate
    as well as from the operator, and that rebuild is the whole of the cost.

    Top level, and taking plain data, because a pool cannot pickle an ``Engine``.
    """
    arms, tokens, a, b, c, near = job
    got = {label: ask_one(Engine.on_statements(statements), tokens, statements, a, b, c)
           for label, statements in arms.items()}
    got["in the sentence"] = ask_one(Engine.on_statements(arms["spans + sentences"]), tokens,
                                     arms["spans + sentences"], a, b, c, among=set(near))
    return None if any(v is None for v in got.values()) else got


def main() -> int:
    name, got = written()
    if not got:
        print("  No cached prose body found under corpora/. Nothing to measure.")
        return 0

    store = [span for _, span in got]
    sentences = {index for index, _ in got}
    places = placed(body()[1])
    where = by_sentence(places)
    rng = random.Random(SEED)

    # **The membership label must be a token the corpus cannot produce, and that is checked.**
    # With `"in"`, an ordinary English word, as the label, on this body it already stands as the
    # middle of 36 spans -- so 2399 membership statements pool with them into one relation whose
    # operator is 98.5% membership and 1.5% preposition. Nothing reports it: the relation count
    # simply does not go up, and a count that does not change is not an error anybody sees. This
    # counts the collision instead of trusting the choice.
    collides = sum(1 for a, b, c in store if b == IN)

    permuted, collided = permute(list(store) + places, random.Random(SEED))
    arms = {
        "spans": list(store),
        "spans + sentences": list(store) + places,
        "sentences permuted": permuted,
    }

    print(f"next token when the sentence is a vertex too -- {len(store)} spans written by the shipped "
          f"ingestion from {name}\n")
    print(f"  {len(sentences)} sentences, {len(places)} statements placing a token in one, "
          f"{len({t for a, _, c in store for t in (a, c)})} tokens")
    print(f"  the permuted arm loses {collided} of them to a token landing twice in one "
          f"sentence, so it is that much smaller and never larger")
    if collides:
        print(f"  **{collides} spans already use `{IN}` as their middle token**, so the membership "
              f"statements are")
        print(f"  pooled with them into one relation and its operator mixes two meanings. Every "
              f"figure below")
        print(f"  is taken on a store that does that.")
    else:
        print(f"  no span uses `{IN}` as its middle token, so the membership relation is its own")

    tokens = sorted({t for a, _, c in store for t in (a, c)})
    members = collections.defaultdict(set)
    for token, _, container in places:
        members[container].add(token)
    asking = rng.sample(sorted(set(store)), min(ASKING, len(set(store))))
    jobs = [(arms, tokens, a, b, c,
             sorted({t for s in where[a] for t in members[s]}))
            for a, b, c in asking]
    print(f"  {len(jobs)} questions over {WORKERS} workers; each rebuilds four coordinates")
    print()
    with futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        answers = [g for g in pool.map(_one_question, jobs, chunksize=1) if g is not None]

    rows = {label: [g[label] for g in answers] for label in ARMS}
    asked = len(rows["spans"])
    if not asked:
        print("  No question survived on every arm. Nothing to compare.")
        return 0

    def rate(label, key):
        return sum(1 for r in rows[label] if r[key] == r["truth"]) / len(rows[label])

    def draw_on(label):
        return sum(1.0 / r["pool"] for r in rows[label]) / len(rows[label])

    figures = {
        "spans, the read": rate("spans", "read"),
        "spans, counting": rate("spans", "counted"),
        "spans + sentences, the read": rate("spans + sentences", "read"),
        "sentences permuted, the read": rate("sentences permuted", "read"),
        "in the sentence, the read": rate("in the sentence", "read"),
        "in the sentence, counting": rate("in the sentence", "counted"),
    }
    draws = {label: draw_on(label.split(",")[0]) for label in figures}
    draw = draws["spans, the read"]

    print(f"  {'arm':<30}{'questions':>11}{'candidates':>12}{'names':>8}{'times a draw':>15}")
    print("  " + "-" * 76)
    for label, value in figures.items():
        pool = np.mean([r["pool"] for r in rows[label.split(",")[0]]])
        print(f"  {label:<30}{asked:>11}{pool:>12.0f}{value:>8.1%}"
              f"{value / draws[label] if draws[label] else 0:>15.1f}")
    print()

    gain = figures["spans + sentences, the read"] - figures["spans, the read"]
    over_null = figures["spans + sentences, the read"] - figures["sentences permuted, the read"]
    record.figures(body=name, sentences=len(sentences), seed=SEED,
                   membership_label=IN, spans_the_label_collides_with=collides,
                   asked_for=ASKING,
                   spans=len(store), placings=len(places), collided=collided,
                   questions=asked, a_draw_would=draw,
                   by_arm=figures, draws=draws, gain_over_spans=gain,
                   gain_over_the_permutation=over_null)

    print(f"  Adding the sentence moves the read from **{figures['spans, the read']:.1%}** to "
          f"**{figures['spans + sentences, the read']:.1%}** on {asked} held-out")
    print(f"  spans, a change of **{gain:+.1%}**. Against the arm where the sentences are permuted "
          f"it is")
    print(f"  **{over_null:+.1%}**, and that is the figure that says whether the text did it or the "
          f"extra axes did.")
    print()
    inside = figures["in the sentence, the read"] - figures["in the sentence, counting"]
    print(f"  On the same candidates a counter told about the sentence chooses between, the read "
          f"names")
    print(f"  **{figures['in the sentence, the read']:.1%}** and that counter names "
          f"**{figures['in the sentence, counting']:.1%}** -- a difference of **{inside:+.1%}**.")
    print()
    # **Two numbers, and the headline is the one against the arm without the scale.** Keying the
    # verdict on the permutation alone said "the second scale helps" while the read had gone *down*:
    # the extra axes dilute, the real grouping recovers some of that, and whether the pair is worth
    # having is the difference against plain spans -- not the difference against a shuffle of
    # something that should not have been added.
    helps_at_all = gain > float(0)
    if over_null > float(0) and inside > float(0) and helps_at_all:
        print("  **The second scale is carrying something the span store could not.** It beats both")
        print("  the permutation and the counter given the same material, which is the first time "
              "in")
        print("  this program a derived read has beaten counting on prose.")
    elif over_null > float(0) and helps_at_all:
        print(f"  **The second scale helps, and counting told the same thing still wins.**")
        print("  So the sentence is information the read can use and not information only the read")
        print("  can use, which is a weaker result than the one this was written to look for.")
    elif over_null > float(0):
        print(f"  **The second scale costs more than it brings.** Against plain spans the read "
              f"moves")
        print(f"  **{gain:+.1%}**, so adding the sentence made it worse. Against the permutation it "
              f"is **{over_null:+.1%}**,")
        print(f"  so the grouping *is* information -- the sentences being the real ones beats them "
              f"being")
        print(f"  shuffled. **Both are true and the first is the headline**: the extra axes dilute "
              f"by more")
        print(f"  than the real structure recovers. The scale earns its place on what it makes "
              f"askable")
        print(f"  rather than on what it names.")
    else:
        print("  **The second scale does not carry.** The permuted arm does as well or better, so")
        print("  whatever the extra statements bought, they did not buy it by putting the right")
        print("  tokens together. A sentence divides what it carries by how many tokens are in it,")
        print("  and on this corpus that dilution is at least as large as anything it adds.")
    print()
    print("  Nothing in the engine was changed to run this. A sentence is a vertex, `in` is a token")
    print("  used in the middle position, and the operator for it is derived from the statements")
    print("  that use it exactly as every other operator here is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
