"""How a corpus cut at its own extents reads, against a counter on the same store.

What the write-down is
------------------------
`entroptics_infer.ingest` cuts each position at the extent whose middle carries the most over the
corpus -- how much knowing the middle narrows what stands last, credited against what that many
observations could show. A middle standing between nearly every pair carries little per occurrence;
a middle occurring once carries little in total. Between them it is largest where a middle is common
enough to relate things and rare enough to say which. No width is supplied: the extent of every
statement is read off the corpus.

What is measured
------------------
The store that cut writes, with whole sentences held out before anything is built and the extents
read off the training sentences alone, so the write-down never sees the sentences the questions come
from. Every question is scored as an ordering, the read and a counter with the stated backoff
alike, so both are compared on where the answer lands; the counter is re-counted on this store
because the write-down decides what it can memorize as much as what the read can derive.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import os
import pathlib
import random
import sys

import numpy as np

import ask
import record
from checking import rank_summary as summarise

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import ingest                            # noqa: E402
from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_more_corpus_closes_the_gap_on_counting import bodies, HELD_OUT  # noqa: E402
from test_where_the_right_token_sits_when_the_read_misses import (counting_order,  # noqa: E402
                                                                  where)

SENTENCES = int(os.environ.get("ENTROPTICS_SENTENCES") or 120)
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 400)

DEPTHS = (1, 5, 10, 50)

SEED = 0


def at_extents(read, held=()):
    """``[(sentence index, (first, middle, last)), ...]`` -- the shipped cut, piece by piece.

    The extents are read off the pieces not in ``held``, so a held-out sentence is cut with what the
    training corpus says, which is what a reader would have on the day.
    """
    counts, overall = ingest.what_follows_each_middle([p for p in read if p[1] not in held])
    return [(piece[1], span) for piece in read
            for span in ingest.at_their_own_width([piece], counts, overall)]


def split(written, held):
    train = tuple(s for index, s in written if index not in held)
    ask_about = [s for index, s in written if index in held]
    return train, ask_about


def measure(written, held):
    train, ask_about = split(written, held)
    if len(train) < 50 or len(ask_about) < 20:
        return None
    engine = Engine.on_statements(train)
    tokens = sorted({t for a, _, c in train for t in (a, c)})
    vectors = engine.places(only=set(tokens))

    rng = random.Random(SEED)
    asking = rng.sample(ask_about, min(QUESTIONS, len(ask_about)))
    read_ranks, count_ranks, offered = [], [], 0
    for a, b, c in asking:
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors)
        if landed is None:
            continue
        pool = [t for t in tokens if t in vectors and t != a]
        if c not in pool or len(pool) < 2:
            continue
        offered = len(pool)
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        read_ranks.append(where([pool[int(i)] for i in np.argsort(scores)[::-1]], c))
        count_ranks.append(where(counting_order(train, a, b, pool), c))
    if not read_ranks:
        return None
    extents = collections.Counter(len(m.split()) + ingest.A_PAIR if m else ingest.A_PAIR
                                 for _, m, _ in train)
    return {"statements": len(train), "held_out": len(ask_about), "tokens": len(tokens),
            "relations": len({m for _, m, _ in train}), "asked": len(read_ranks),
            "extents": {str(k): v for k, v in sorted(extents.items())},
            "the_read": summarise(read_ranks, offered, DEPTHS),
            "counting": summarise(count_ranks, offered, DEPTHS)}


def main() -> int:
    names, text = bodies()
    if not text:
        print("  No cached prose bodies found under corpora/. Nothing to measure.")
        return 0

    read = ingest.pieces(text, sentences=SENTENCES)
    rng = random.Random(SEED)
    held = {i for i in range(len(read)) if rng.randrange(HELD_OUT) == 0}

    print("how a corpus cut at its own extents reads, against a counter on the same store\n")
    print(f"  {len(names)} bodies, {SENTENCES} sentences, one in {HELD_OUT} held out before "
          f"anything was built\n")

    got = measure(at_extents(read, held), held)
    if got is None:
        print("  The write-down did not produce a readable store. Nothing to measure.")
        return 0
    print(f"  {'':<16}{'statements':>11}{'relations':>10}{'asked':>7}"
          + "".join(f"{f'top {k}':>9}" for k in DEPTHS) + f"{'median':>9}{'1/rank':>9}")
    print("  " + "-" * 89)
    for label, r in (("the read", got["the_read"]), ("counting", got["counting"])):
        print(f"  {label:<16}{got['statements']:>11}{got['relations']:>10}{got['asked']:>7}"
              + "".join(f"{r[f'top_{k}']:>9.1%}" for k in DEPTHS)
              + f"{r['median_rank']:>9.0f}{r['mean_reciprocal_rank']:>9.3f}")
    print(f"\n  extents written: {got['extents']}\n")

    over = (got["the_read"]["mean_reciprocal_rank"] / got["counting"]["mean_reciprocal_rank"]
            if got["counting"]["mean_reciprocal_rank"] else float("nan"))
    record.figures(bodies=names, sentences=SENTENCES, questions=QUESTIONS, seed=SEED,
                   held_out_one_in=HELD_OUT, depths=list(DEPTHS),
                   by_write_down={"at its extents": got}, the_read_over_counting=over)
    print(f"  The read names **{got['the_read']['top_1']:.1%}** first at a median rank of "
          f"**{got['the_read']['median_rank']:.0f}**; the counter")
    print(f"  **{got['counting']['top_1']:.1%}** at **{got['counting']['median_rank']:.0f}**. "
          f"In reciprocal rank the read stands at **{over:.3f}** of the counter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
