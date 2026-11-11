"""Small readings the reader's checks share. None of this is the method.

``ask.py`` does not import this module, so nothing here is part of what the reader does: these are
the ways several checks score, label or split what the reader returns, defined once so that two
checks said to score the same way do. A reading that depends on a check's own design -- the depths
it ranks to, the sentences it reads -- takes it as an argument, and each check passes its own.
"""
from __future__ import annotations

import random

import numpy as np

import ontology                         # puts src/ on the path, so it comes first
from entroptics_infer import ingest     # noqa: E402


def reached(path, dst):
    """Whether any step of a walked ``path`` reached ``dst``."""
    return any(r["reached"] == dst for step in path for r in step["reached"])


def joined(a, b):
    """Does any statement mention both, in either order?"""
    return any(a in p and b in p for p in ontology.STATEMENTS)


def shipped(engine, *, without=()):
    """The places the shipped engine builds, with ``without`` left out."""
    return engine.places(without=without)


def depth_label(depth):
    """A spread depth as a label: its number, or ``"run out"`` for a spread that is not stopped."""
    return "run out" if depth is None else str(depth)


def rank_summary(ranks, offered, depths):
    """Where the right token sits among ``offered`` candidates: median, reciprocal rank and the
    share inside each of ``depths``."""
    got = {"questions": len(ranks), "candidates": offered,
           "median_rank": float(np.median(ranks)),
           "mean_reciprocal_rank": float(np.mean([1 / r for r in ranks]))}
    for k in depths:
        got[f"top_{k}"] = float(np.mean([r <= k for r in ranks]))
    return got


def pooled_rank_summary(ranks, pools, depths):
    """:func:`rank_summary` where each question has its own pool, reported by the pools' mean."""
    got = {"questions": len(ranks), "pool_mean": float(np.mean(pools)),
           "median_rank": float(np.median(ranks)),
           "mean_reciprocal_rank": float(np.mean([1 / r for r in ranks]))}
    for k in depths:
        got[f"top_{k}"] = float(np.mean([r <= k for r in ranks]))
    return got


def read_prose(text, sentences, seed, held_out):
    """``(training spans, held-out spans, training sentences)`` -- one split, three stores.

    One sentence in ``held_out`` is held out, drawn with ``seed`` over the first ``sentences``. The
    sentences come back as well as the spans, because the ladder is built from them and the hold-out
    has to remove a sentence from both or a question's answer survives in a membership statement
    while its span is gone.
    """
    pieces = ingest.pieces(text, sentences=sentences)
    rng = random.Random(seed)
    held = {i for i in range(len(pieces)) if rng.randrange(held_out) == 0}
    # the extents are read off the **training** sentences alone: which middles narrow what follows
    # them is a statistic of the corpus, and reading it off the whole text before the split would
    # let the write-down see the sentences the questions come from
    training = [p for p in pieces if p[1] not in held]
    counts, overall = ingest.what_follows_each_middle(training)
    spans = ingest.at_their_own_width(training, counts, overall)
    ask_about = ingest.at_their_own_width([p for p in pieces if p[1] in held], counts, overall)
    groups = {f"sentence {index}": sorted(set(got)) for _, index, got in training}
    return tuple(spans), ask_about, groups
