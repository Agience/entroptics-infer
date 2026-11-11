"""A fitted model on prose, beside the derived read and the counter, on one set of questions.

Why this has to be run before the text negative is stated
-----------------------------------------------------------
`test_whether_more_corpus_closes_the_gap_on_counting.py` finds the derived read at 0.83 of
counting's score at 120 sentences and 0.65 at 800 and 2000 — the gap widens with corpus and then
holds. The reading offered for that is the corpus criterion this program names: adjacency records
*order*, not an asymmetric relation, so there is nothing for an operator to find at any size.

**That reading is only sound if prose is hard for the derived read in particular.** If a fitted model
also loses to a trigram counter on the same questions, then prose at three-token spans is hard for
everything that is not counting, and the negative is about the corpus rather than about deriving.

TransE has been run on `ontology.py` and on the lattice, and beat the derived read on both. This
puts it on the third store, where the derived read is weakest.

What is matched
-----------------
Everything. The same train/test split — whole sentences held out before anything is built — the same
training store, the same held-out spans, the same candidate pool. Three arms answer the identical
questions:

``the read``    the operator derived from the training store, nothing fitted.
``counting``    the most frequent third token after this ``(a, b)``; failing that after ``b``;
                failing that the most frequent at all. The standard backoff, stated in full.
``TransE``      ``h + r ≈ t`` with margin loss, fitted on the training store, swept over
                dimension and margin, each setting under several seeds, its best reported.

The baseline is written out in the companion file and imported rather than reimplemented, so it is
the same model that won on the other two stores.

One difference from the companion, and it favours the baseline
----------------------------------------------------------------
There, TransE is retrained from scratch for every held-out question, so it never sees the answer.
Here the split does that once: the held-out sentences are absent from the training store, so a single
fit never sees any of the answers. That gives the fitted model more training signal per question,
not less — it fits on the whole training store rather than on that store minus one statement.
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

from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_the_read_predicts_the_next_token import counted     # noqa: E402
from test_whether_the_same_coordinate_can_be_built_faster import built      # noqa: E402
from test_whether_more_corpus_closes_the_gap_on_counting import (bodies, split)  # noqa: E402
from test_what_a_trained_model_gets_on_the_same_store import (DIMENSIONS, EPOCHS,  # noqa: E402
                                                              MARGINS, RATE, SEEDS, train)

#: how much prose to read. One size rather than a sweep: the sweep is the companion file's, and what
#: this adds is the third arm on a store big enough for the gap to have settled.
SENTENCES = int(os.environ.get("ENTROPTICS_SENTENCES") or 800)

#: how many held-out spans to ask about
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 1200)

SEED = 0


def main() -> int:
    names, text = bodies()
    if not text:
        print("  No cached prose bodies found under corpora/. Nothing to measure.")
        return 0

    train_spans, held = split(text, sentences=SENTENCES)
    if len(train_spans) < 50 or len(held) < 20:
        print("  Too little prose to split. Nothing to measure.")
        return 0

    engine = Engine.on_statements(train_spans)
    tokens = sorted({t for a, _, c in train_spans for t in (a, c)})
    relations = sorted({m for _, m, _ in train_spans})

    print("a fitted model on prose, beside the derived read and the counter\n")
    print(f"  {len(names)} bodies, {SENTENCES} sentences, {len(train_spans)} training spans, "
          f"{len(tokens)} tokens, {len(relations)} relations")
    print(f"  one sentence in five held out before anything was built, "
          f"{QUESTIONS} of its spans asked\n")

    began = time.perf_counter()
    vectors = built(engine, only=set(tokens))
    print(f"  the coordinate: {time.perf_counter() - began:.1f}s for {len(vectors)} rows",
          flush=True)

    rng = random.Random(SEED)
    asking = rng.sample(held, min(QUESTIONS, len(held)))

    # the questions both arms can answer, settled once so every arm sees exactly the same ones
    questions = []
    for a, b, c in asking:
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors)
        if landed is None:
            continue
        pool = [t for t in tokens if t in vectors and t != a]
        if c not in pool or len(pool) < 2:
            continue
        questions.append((a, b, c, landed, pool))
    if not questions:
        print("  No question survived. Nothing to compare.")
        return 0

    read_right = count_right = 0
    draw = float(0)
    for a, b, c, landed, pool in questions:
        draw += float(1) / len(pool)
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        if pool[int(np.argmax(scores))] == c:
            read_right += 1
        if counted(train_spans, a, b, pool) == c:
            count_right += 1
    asked = len(questions)
    draw /= asked

    print(f"  {asked} questions survived, a draw would name {draw:.3%}\n")
    print(f"  {'dim':>5}{'margin':>8}{'TransE':>9}{'over seeds':>18}{'the read':>10}"
          f"{'counting':>10}{'fitted':>9}")
    print("  " + "-" * 69)

    by_setting = {}
    for dim in DIMENSIONS:
        for margin in MARGINS:
            shares = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                E, R, vi, ri = train(train_spans, tokens, relations, dim=dim, margin=margin,
                                     epochs=EPOCHS, rate=RATE, rng=rng)
                right = 0
                for a, b, c, _, pool in questions:
                    if a not in vi or b not in ri:
                        continue
                    want = E[vi[a]] + R[ri[b]]
                    far = np.linalg.norm(np.stack([E[vi[t]] for t in pool]) - want, axis=1)
                    if pool[int(np.argmin(far))] == c:
                        right += 1
                shares.append(right / asked)
            key = f"{dim}/{margin}"
            by_setting[key] = {"dim": dim, "margin": margin,
                               "trained": float(np.mean(shares)),
                               "trained_lowest": min(shares), "trained_highest": max(shares)}
            v = by_setting[key]
            print(f"  {dim:>5}{margin:>8}{v['trained']:>9.1%}"
                  f"{f'{min(shares):.1%}-{max(shares):.1%}':>18}"
                  f"{read_right / asked:>10.1%}{count_right / asked:>10.1%}"
                  f"{v['trained'] / (count_right / asked) if count_right else 0:>9.2f}", flush=True)
    print()

    read = read_right / asked
    count = count_right / asked
    best = max(by_setting.values(), key=lambda v: v["trained"])
    record.figures(bodies=names, sentences=SENTENCES, questions=QUESTIONS, seed=SEED,
                   train_spans=len(train_spans), tokens=len(tokens), relations=len(relations),
                   asked=asked, a_draw_would=draw, the_read=read, counting=count,
                   the_read_over_counting=read / count if count else float("nan"),
                   trained_best=best["trained"],
                   trained_best_over_counting=best["trained"] / count if count else float("nan"),
                   dimensions=list(DIMENSIONS), margins=list(MARGINS), epochs=EPOCHS,
                   rate=RATE, seeds=list(SEEDS), by_setting=by_setting)

    print(f"  Counting names **{count:.1%}** of **{asked}** held-out spans. The derived read names "
          f"**{read:.1%}**,")
    print(f"  **{read / count:.2f}** times the counter. TransE at its best setting — dim "
          f"**{best['dim']}**, margin **{best['margin']}** —")
    print(f"  names **{best['trained']:.1%}**, **{best['trained'] / count:.2f}** times it.")
    print()
    if best["trained"] < count:
        print(f"  **The fitted model loses to counting here too.** So prose written as three-token")
        print(f"  spans is hard for a fitted model and for a derived one alike, and what separates")
        print(f"  them from the counter is not that one is fitted. **The negative on text is about")
        print(f"  the corpus, not about deriving** — adjacency records order, and neither a fitted")
        print(f"  relation vector nor a derived operator can find a relation that was never "
              f"written")
        print(f"  down. On the two stores where one *is* recorded, both arms beat a draw by a wide")
        print(f"  margin and the fitted one wins.")
    else:
        print(f"  **The fitted model beats counting where the derived read does not.** So prose is")
        print(f"  not hard for everything: there is signal in these spans that fitting reaches and")
        print(f"  deriving does not. **That makes the text negative a fact about this method** "
              f"rather")
        print(f"  than about adjacency, and what the operator is failing to capture is the next")
        print(f"  thing to name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
