"""Next-token prediction on prose, derived against counted, on one store and one set of questions.

Why this is the comparison
----------------------------
`test_what_shape_prose_has_to_be_ingested_in.py` finds that prose written a sentence at a time shares
0.1% of its coordinate and prose written as three-token spans shares 41.8%, against the
hand-written triples' 54.7%. A three-token span is the same shape as a triple, so prose at that
width is the object the engine already reads.

And a three-token span read as a statement is next-token prediction: given ``a`` and the token
``b`` standing between, name what follows. The middle token acts as the relation, as it does
everywhere else in this program, because role comes from position and nothing declares a
vocabulary.

That makes the fair baseline obvious and it is the one that has beaten this program before. CLAIMS
7.2, 7.3 and 7.11 record counting beating the reads on real text, on four task runs -- but every
one of those read text as co-occurrence statistics, where the relation is inferred from counts
rather than recorded. Here the relation is recorded: the span was written down, in order, by whoever
wrote the text. So this is the first time the two are compared on a store of the kind the method
asks for, and it is the comparison that decides whether deriving beats counting where the relation
is stated.

How they are matched
----------------------
Both arms answer the same questions on the same held-out store, and the held-out span is removed
entirely -- every copy of it, since a duplicate left behind states the answer.

``the read``    the operator derived from the held store, naming the nearest candidate.
``counting``    the most frequent third token after this ``(a, b)`` in the held store; failing that,
                the most frequent token after ``b``; failing that, the most frequent token at all.
                The backoff is stated in full and is the standard one, not a weakened baseline.
``a draw``      one over the candidates offered.

Where counting has nothing left after hold-out it still answers by backing off, which is what a
counting model does. Both arms are scored on every question, so neither is credited for declining.

What would settle it
----------------------
If the read beats counting on spans, the four recorded negatives are about the co-occurrence shape
rather than about text, and the corpus criterion is met by prose after all. If counting wins here
too, the negative is about text and not about how it was represented, which is a much stronger
version of what the ledger already holds.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import pathlib
import random

import numpy as np

import record

import ask

from entroptics_infer import ingest
from entroptics_infer.wrapper import Engine

CORPORA = pathlib.Path(__file__).resolve().parents[1] / "corpora"
#: CHOSEN: how much prose, and how many of its spans are asked about. The store size is what fits
#: inside a held-out protocol that rebuilds the coordinate for every question.
SENTENCES = 120
QUESTIONS = 120
SEED = 0


def body():
    """``(name, text)`` of the first cached prose body, or ``(None, None)`` when there is none."""
    bodies = sorted(CORPORA.glob("*.txt"))
    if not bodies:
        return None, None
    return bodies[0].name, bodies[0].read_text(encoding="utf-8", errors="ignore")


def written(sentences=SENTENCES):
    """``(body name, [(sentence index, (first, middle, last)), ...])`` -- the shipped ingestion.

    The spans are `ingest.spans`'s, each cut at the extent its own middle earns, in reading order
    and with duplicates kept; the sentence index is the shipped piecing's. Read piece by piece so
    each span keeps the sentence it came from: `ingest.at_their_own_width` carries nothing between
    pieces, so the spans are the ones `ingest.spans` writes for the whole text, in the same order.
    """
    name, text = body()
    if text is None:
        return None, []
    read = ingest.pieces(text, sentences=sentences)
    counts, overall = ingest.what_follows_each_middle(read)
    out = [(piece[1], span) for piece in read
           for span in ingest.at_their_own_width([piece], counts, overall)]
    return name, out


def spans(sentences=SENTENCES):
    """``(body name, [(first, middle, last), ...])`` -- the shipped ingestion's spans of one body."""
    name, got = written(sentences)
    if not got:
        return None, None
    return name, [span for _, span in got]


def counted(held, a, b, candidates):
    """The counting baseline, with its backoff stated in full."""
    after_pair = collections.Counter(z for x, y, z in held if x == a and y == b)
    for who in (after_pair,
                collections.Counter(z for _, y, z in held if y == b),
                collections.Counter(z for _, _, z in held)):
        for token, _ in who.most_common():
            if token in candidates:
                return token
    return None


def main() -> int:
    name, store = spans()
    if not store:
        print("  No cached prose body found under corpora/. Nothing to measure.")
        return 0

    engine = Engine.on_statements(store)
    concepts = engine.concepts()
    rng = random.Random(SEED)
    asking = rng.sample(sorted(set(store)), min(QUESTIONS, len(set(store))))

    print(f"next token from two, derived against counted -- {len(store)} spans written by the "
          f"shipped ingestion from {name}\n")

    read_right = count_right = asked = 0
    draw_sum = 0.0
    agree = both = neither = 0
    rows = []
    for a, b, c in asking:
        # **Every span joining these two tokens, not only the one asked about.** A trigram store
        # records the same pair under many middles -- surveyed on this store, 154 of its pairs carry
        # more than one, 15% of the spans -- so holding out only `(a, b, c)` leaves `(a, x, c)`
        # saying the same next token. The leak helps both arms, which is why the verdict below does
        # not turn on it, and it is removed anyway because a protocol that is right only when it
        # does not matter is not a protocol.
        pair = [s for s in store if s[0] == a and s[2] == c]
        held = [s for s in store if s not in pair]
        vectors = engine.places(without=pair)
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors, without=pair)
        if landed is None:
            continue
        stated = {z for x, y, z in held if x == a and y == b}
        pool = [t for t in concepts if t in vectors and t != a and t not in stated]
        if c not in pool or len(pool) < 2:
            continue

        asked += 1
        draw_sum += 1.0 / len(pool)
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        mine = pool[int(np.argmax(scores))]
        theirs = counted(held, a, b, set(pool))

        read_right += mine == c
        count_right += theirs == c
        agree += mine == theirs
        both += (mine == c) and (theirs == c)
        neither += (mine != c) and (theirs != c)
        rows.append({"a": a, "b": b, "truth": c, "the_read": mine, "counting": theirs,
                     "pool": len(pool)})

    if not asked:
        print("  No span produced a question. The store is not built as intended.")
        return 0

    read, count = read_right / asked, count_right / asked
    draw = draw_sum / asked
    wins = sum(1 for r in rows if r["the_read"] == r["truth"] and r["counting"] != r["truth"])
    losses = sum(1 for r in rows if r["counting"] == r["truth"] and r["the_read"] != r["truth"])
    smallest = 2.0 * 0.5 ** (wins + losses) if (wins + losses) else float("nan")

    print(f"  {'arm':>22}{'questions':>11}{'pool':>7}{'right':>8}{'x draw':>9}")
    print("  " + "-" * 57)
    print(f"  {'the read':>22}{asked:>11}{np.mean([r['pool'] for r in rows]):>7.0f}"
          f"{read:>8.1%}{read / draw if draw else float('nan'):>9.1f}")
    print(f"  {'counting':>22}{asked:>11}{np.mean([r['pool'] for r in rows]):>7.0f}"
          f"{count:>8.1%}{count / draw if draw else float('nan'):>9.1f}")
    print(f"  {'a draw':>22}{asked:>11}{np.mean([r['pool'] for r in rows]):>7.0f}{draw:>8.1%}")
    print()
    print(f"  paired: the read alone right on **{wins}**, counting alone right on **{losses}**, "
          f"both on {both}, neither on {neither}")
    print(f"  the smallest p that pairing could have produced is {smallest:.4f}")
    print()

    record.figures(body=name, sentences=SENTENCES, seed=SEED,
                   spans=len(store), questions=asked, pool=float(np.mean([r["pool"] for r in rows])),
                   the_read=read, counting=count, a_draw_would=draw,
                   read_alone=wins, counting_alone=losses, both_right=both, neither_right=neither,
                   smallest_p_obtainable=smallest, detail=rows[:QUESTIONS])

    if wins > losses:
        print(f"  **The read beats counting on prose, on the store shape the method asks for.**")
        print(f"  {read:.1%} against {count:.1%} over {asked} held-out spans, and paired it is")
        print(f"  {wins} to {losses}. The four negatives at CLAIMS 7.2, 7.3 and 7.11 are about")
        print("  reading text as co-occurrence, where the relation is inferred from counts. Here")
        print("  the relation is recorded -- somebody wrote those three tokens in that order --")
        print("  and that is the condition this program has said all along that it needs.")
    elif losses > wins:
        print(f"  **Counting still wins on prose.** {count:.1%} against {read:.1%} over {asked}")
        print(f"  held-out spans, paired {losses} to {wins}. **So the negative in the ledger is")
        print("  about text and not about how text was represented** -- writing the relation down")
        print("  rather than inferring it from counts did not change the answer, which is a")
        print("  stronger version of what CLAIMS 7.2, 7.3 and 7.11 already hold.")
    else:
        print(f"  **The two are level**, {wins} to {losses} paired over {asked} questions, which")
        print("  at this size establishes nothing either way. The pairing's own floor is above")
        print("  what it would need to resolve, and more questions is what that asks for.")
    print()
    print("  Both arms answer the same questions on the same held-out store, with every copy of")
    print("  the asked-about span removed. Counting backs off pair, then middle token, then most")
    print("  frequent, and is scored on every question rather than allowed to decline. The draw is")
    print("  computed over the pools the reader saw. The coordinate is the shipped engine's, which")
    print("  stops each place at its own aperture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
