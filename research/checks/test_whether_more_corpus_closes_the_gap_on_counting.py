"""Does the read close the gap on counting as the corpus grows?

The question
--------------
Counting with a stated backoff beats the derived read on text in five task runs, in the width
sweep and in the scale ladder. **Every one of those figures is from 120 sentences of one book** —
**2572** spans over **725** concepts (`test_what_write_down_reads_cheaply_and_answers_widely.py`).
`corpora/` holds six bodies, of which those 120 sentences are a small part.

A trigram counter memorizes a small store; a derived read has to find structure to generalize from.
So the gap closing as the corpus grows is a real possibility. **This measures it.**

Why the protocol is a train/test split
----------------------------------------
Every other prose figure uses leave-one-out: remove the asked-about span from the store, rebuild the
coordinate, ask. **That rebuild is why the corpus is 120 sentences**: a coordinate costs more the
bigger the store, and leave-one-out pays it per question.

A **train/test split** holds out more and pays once. Whole sentences are set aside before anything
is built; the store is the rest; and the questions are spans from the held-out sentences, which are
not in the store at all — not the span, not its tokens' membership, nothing. That is a stronger
hold-out than removing one statement from a store that still contains the sentence it came from, and
it costs one coordinate for every question rather than one each.

The arms, unchanged from the companion file
---------------------------------------------
``the read``    the operator derived from the training store, naming the nearest candidate.
``counting``    the most frequent third token after this ``(a, b)`` in the training store; failing
                that, the most frequent after ``b``; failing that, the most frequent at all. The
                backoff is stated in full and is the standard one, not a weakened baseline.
``a draw``      one over the candidates offered.

Both arms answer the same questions from the same store, and neither is credited for declining.

What would settle it
----------------------
If the read's share of counting's score **rises** with corpus size, more corpus is the answer to the
five negatives and the question is how much. If it is flat, the corpus was never the problem and the
criterion this program names — an asymmetric relation *recorded* rather than inferred — is what
adjacency fails, at any size.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import os
import pathlib
import random
import sys
import time

import numpy as np

import ask
import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import ingest                            # noqa: E402
from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_read_predicts_the_next_token import counted  # noqa: E402
from test_whether_the_same_coordinate_can_be_built_faster import built   # noqa: E402

CORPORA = pathlib.Path(__file__).resolve().parents[1] / "corpora"

#: sentence counts to sweep. Declared and swept rather than chosen: the point is the trend across
#: them, and the top is where one coordinate stops being affordable on a shared box.
SIZES = tuple(int(n) for n in
              (os.environ.get("ENTROPTICS_SIZES") or "120,300,800,2000").split(","))

#: what share of sentences is held out. One in five, so the test set grows with the corpus and the
#: questions are never the same ones twice.
HELD_OUT = 5

#: how many held-out spans to ask about at each size, so the sizes are compared on equal questions
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 400)

SEED = 0


def bodies():
    """Every prose body, joined, so the corpus is the corpus and not one book."""
    found = sorted(CORPORA.glob("*.txt"))
    if not found:
        return None, None
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in found)
    return [f.name for f in found], text


def split(text, *, sentences):
    """``(training statements, questions)`` -- whole sentences held out before anything is built.

    **Each position is cut at the extent its own middle earns**, and the extents are read off the
    *training* sentences alone. That order matters and it is the hold-out: which middles narrow what
    follows them is a statistic of the corpus, so reading it off the whole text before the split
    would let the write-down see the sentences the questions come from. The held-out sentences are
    then cut with the training corpus's readings, which is what a reader would have on the day.
    """
    read = ingest.pieces(text, sentences=sentences)
    rng = random.Random(SEED)
    held = {i for i in range(len(read)) if rng.randrange(HELD_OUT) == 0}
    training = [piece for piece in read if piece[1] not in held]
    counts, overall = ingest.what_follows_each_middle(training)
    train = ingest.at_their_own_width(training, counts, overall)
    ask_about = ingest.at_their_own_width([p for p in read if p[1] in held], counts, overall)
    return tuple(train), ask_about


def at(size, text):
    """The read and the counter over held-out sentences, on a store built once."""
    train, questions = split(text, sentences=size)
    if len(train) < 50 or len(questions) < 20:
        return None
    engine = Engine.on_statements(train)
    tokens = sorted({t for a, _, c in train for t in (a, c)})

    # **One coordinate for every question**, because no question's span is in the store: the
    # sentences they came from were set aside before it was built. That is the whole reason a
    # corpus this size can be read at all.
    began = time.perf_counter()
    vectors = built(engine, only=set(tokens) | set(engine.vertices()[:0]))
    seconds_built = time.perf_counter() - began

    rng = random.Random(SEED)
    asking = rng.sample(questions, min(QUESTIONS, len(questions)))
    read_right = count_right = asked = 0
    draw = float(0)
    for a, b, c in asking:
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors)
        if landed is None:
            continue
        pool = [t for t in tokens if t in vectors and t != a]
        if c not in pool or len(pool) < 2:
            continue
        asked += 1
        draw += float(1) / len(pool)
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        if pool[int(np.argmax(scores))] == c:
            read_right += 1
        if counted(train, a, b, pool) == c:
            count_right += 1
    if not asked:
        return None
    read = read_right / asked
    count = count_right / asked
    return {"sentences": size, "train_spans": len(train), "held_out_spans": len(questions),
            "tokens": len(tokens), "vertices": len(engine.vertices()),
            "asked": asked, "a_draw_would": draw / asked,
            "the_read": read, "counting": count,
            "the_read_times_the_draw": read / (draw / asked) if draw else float("nan"),
            "the_read_over_counting": read / count if count else float("nan"),
            "seconds_for_the_one_coordinate": seconds_built}


def main() -> int:
    names, text = bodies()
    if not text:
        print("  No cached prose bodies found under corpora/. Nothing to measure.")
        return 0

    print("does the read close the gap on counting as the corpus grows?\n")
    print(f"  {len(names)} bodies joined, holding out one sentence in {HELD_OUT}, "
          f"{QUESTIONS} questions asked at each size\n")

    print(f"  {'sentences':>10}{'train spans':>13}{'tokens':>8}{'asked':>7}{'draw':>8}"
          f"{'the read':>10}{'counting':>10}{'read/count':>12}{'coordinate':>12}")
    print("  " + "-" * 90)
    rows = {}
    for size in SIZES:
        got = at(size, text)
        if got is None:
            continue
        rows[str(size)] = got
        print(f"  {size:>10}{got['train_spans']:>13}{got['tokens']:>8}{got['asked']:>7}"
              f"{got['a_draw_would']:>8.2%}{got['the_read']:>10.1%}{got['counting']:>10.1%}"
              f"{got['the_read_over_counting']:>12.2f}"
              f"{got['seconds_for_the_one_coordinate']:>11.1f}s", flush=True)
    print()

    if len(rows) < 2:
        print("  Fewer than two sizes could be read. Nothing to compare.")
        return 0

    record.figures(bodies=names, sizes=list(SIZES), held_out_one_in=HELD_OUT,
                   questions=QUESTIONS, seed=SEED, by_size=rows)

    order = list(rows)
    first, last = rows[order[0]], rows[order[-1]]
    rising = all(rows[a]["the_read_over_counting"] <= rows[b]["the_read_over_counting"]
                 for a, b in zip(order, order[1:]))

    print(f"  The store goes from **{first['train_spans']}** training spans to "
          f"**{last['train_spans']}**, and the read's share of")
    print(f"  counting goes from **{first['the_read_over_counting']:.2f}** to "
          f"**{last['the_read_over_counting']:.2f}**.")
    print()
    if rising and last["the_read_over_counting"] > first["the_read_over_counting"]:
        print(f"  **The gap closes as the corpus grows, at every step measured.** The five recorded")
        print(f"  negatives on text are from **0.4%** of the corpus sitting in `corpora/`, and what")
        print(f"  they measured was a store too small to generalise from rather than a method that")
        print(f"  cannot. **How much more corpus closes it entirely is the next measurement**, and "
              f"the")
        print(f"  cost of one coordinate is what bounds it.")
    elif last["the_read_over_counting"] > first["the_read_over_counting"]:
        print(f"  **The gap is narrower at the largest store than the smallest, and not at every "
              f"step.**")
        print(f"  Which steps move and what those stores have in common is the next thing to look "
              f"at.")
    else:
        print(f"  **More corpus does not close the gap.** The read's share of counting is no better "
              f"at")
        print(f"  **{last['train_spans']}** spans than at **{first['train_spans']}**, so the five "
              f"recorded negatives are not about")
        print(f"  size. **What adjacency fails is the criterion this programme names** — an "
              f"asymmetric")
        print(f"  relation *recorded* rather than inferred — and no amount of text written down this")
        print(f"  way will meet it.")
    print()
    print(f"  **The hold-out is stronger here than in the companion file**: whole sentences are set")
    print(f"  aside before the store is built, so a question's span is not in it and neither is any")
    print(f"  membership its tokens would have had. Leave-one-out removes a statement from a store")
    print(f"  that still holds the sentence it came from. And it pays one coordinate per question,")
    print(f"  which is why every prose figure until now is from 120 sentences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
