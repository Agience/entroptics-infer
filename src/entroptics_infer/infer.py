"""The forward pass: given two of a triple, infer the third — with nothing trained and nothing named.

The operation
-------------
`(context, content, operator)` is the platform's unit and it is a three-terminal device: the operator
modulates what passes between the other two. Inference is that device run forward — supply two
terminals and read the third.

The difficulty is the operator. Naming it would need a vocabulary, and there is none: the kinds of
relation a corpus contains are directions it turns out to occupy, not members of a list. So the
operator is supplied by example. Given a pair `(c, d)` that stands in some relation, and a source
`a`, the question is which candidate `b` stands in *that same relation* to `a`.

That is an analogy, and it is the only form the question can take in a system with no relation
names. It is also the form that makes the operator's role checkable: a method that ignores the
exemplar cannot do better than chance among `a`'s neighbours, because every one of them is related
to `a` — just differently.

How it is computed
------------------
The exemplar's relation is the cross-covariance `M(c, d)`, a matrix. Each candidate's relation to
the source is `M(a, b)`, also a matrix. The two are compared the same way two records are: their
alignment standardized by the exact permutation null. Nothing is trained, no relation is named, and
the only supplied number is the reader's false-alarm level.

Why this is the whole forward pass
-----------------------------------
A transformer's layer is `out_i = Σ_j softmax(q_i·k_j/√d) v_j` — score every candidate against the
query, then combine. Here the score is a measured association against a null instead of a
temperature-scaled dot product, the operator steers which candidate is wanted rather than being
folded into trained projections, and the row is not normalized, so nothing is returned when nothing
resolves. The pieces that a transformer trains — the embedding table, `W_Q`/`W_K`/`W_V`, the
relation vocabulary, the layer count — are each replaced by something the corpus is measured for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bind import association_matrix
from .engine import Association, Unreadable, associate, rank

__all__ = ["Completion", "BOUND", "NOTHING_RESOLVED", "INDISTINGUISHABLE", "complete",
           "complete_with", "operator_of", "operator_over", "relation_over"]

#: Why a completion returned nothing. Strings, so a caller branches on a reason rather than a code.
NOTHING_RESOLVED = "nothing resolved"       # no candidate's relation resolves against the operator
INDISTINGUISHABLE = "indistinguishable"     # two or more candidates are equally good
BOUND = "bound"


@dataclass(frozen=True)
class Completion:
    """Which candidate completes the triple, or why none does.

    ``key``        index into ``candidates``, or ``None``.
    ``why``        which of the outcomes above.
    ``scores``     the association of each candidate's relation with the operator, in order. The
                   row is not normalized — it may be all zeros, which is refusal.
    ``modes``      shared resolved modes between each candidate's relation and the operator.
    """

    key: int | None
    why: str
    scores: tuple
    modes: tuple


def operator_of(records, src, dst) -> np.ndarray:
    """The relation an exemplar pair stands in — the operator, as the matrix it is.

    This is what a trained system holds as `R_k` or `W_r` and obtains by descent over a named
    relation index. Here it is read off one example, and it has no name because there is nothing to
    name it from.
    """
    return association_matrix(records[src], records[dst])


def complete(records, *, exemplar, source, candidates, far: float) -> Completion:
    """Given `(c, d)` exemplifying a relation and a source `a`, which candidate stands in it to `a`?

    Two reads decide, as in :func:`entroptics_infer.bind.decide`, and for the same reason: an
    alignment always has a maximum, so a method resting on one alone always answers.

    1. **Alignment proposes** — the candidate whose relation aligns most strongly with the operator,
       among those that resolve against it at all.
    2. **Breadth confirms** — that proposal is returned only if it is the *unique* candidate sharing
       the most resolved modes with the operator.

    Where nothing resolves, or where two candidates are equally good, nothing is returned. That is
    the answer: the corpus does not distinguish them, and choosing anyway would be the system
    asserting a relation it cannot see.
    """
    if len(exemplar) != 2:
        raise Unreadable(
            f"an operator is exemplified by a pair; got {len(exemplar)} vertices. One record does "
            f"not stand in a relation to anything, and three do not name one relation."
        )
    return complete_with(records, operator=operator_of(records, exemplar[0], exemplar[1]),
                         source=source, candidates=candidates, far=far)


def operator_over(records, edges) -> np.ndarray:
    """The relation a set of pairs stands in — what their examples agree on.

    :func:`operator_of` reads a relation off one pair, which is all an analogy gives it. When the
    lattice already holds several edges under one name, the relation those edges stand in is the
    operator they have in common, and averaging their matrices is what "in common" means here: the
    part that recurs survives and the part particular to any one pair does not.

    This is the whole of how a relation acquires content in this system. It is derived from the
    edges that cite it, at the moment it is asked for, and there is nothing to train and nothing to
    unlearn -- an edge added or withdrawn changes it by being included in or absent from this mean.

    This averages every edge given. How many to give it is :func:`relation_over`.
    """
    ms = [association_matrix(records[s], records[d]) for s, d in edges]
    if not ms:
        raise Unreadable(
            "a relation with no edges behind it has no content to read. Name it on a pair first."
        )
    return np.mean(ms, axis=0)


def relation_over(records, edges, *, far: float) -> np.ndarray:
    """The relation those edges stand in, over as many of them as still resolve.

    Averaging trades noise for rank. It suppresses what is particular to any one pair, which is the
    point, and suppresses resolvable structure with it: past some number of edges the mean stops
    resolving at all, and a relation that resolves nothing cannot confirm anything in
    :func:`complete_with`'s second gate. So the mean is taken over the longest run of edges whose
    aggregate still resolves at the reader's own level, and the collapse marks where to stop.

    **Why this is allowed and re-asking the question is not.** The choice reads only the relation --
    never the source, never the candidates, never the answer -- so the question is asked once and
    the reader's level is spent once. A rule that instead re-asks at every backing and keeps
    whatever resolved answers a relation with nothing behind it 0.492 more often than a single
    edge does, 5.69 corpus-to-corpus spreads out; this one answers it 0.038 more often, 0.25
    spreads, which is not distinguishable from none at the power of that control.

    The run is taken in the order the edges arrive, because a relation in a running lattice has a
    history and nothing here may re-order it to suit an answer.
    """
    ms = [association_matrix(records[s], records[d]) for s, d in edges]
    if not ms:
        raise Unreadable(
            "a relation with no edges behind it has no content to read. Name it on a pair first."
        )
    running, best = None, np.asarray(ms[0], dtype=float)
    for i, m in enumerate(ms, start=1):
        running = m if running is None else running + m
        mean = running / i
        if rank(mean, far=far) > 0:
            best = mean
    return best


def complete_with(records, *, operator, source, candidates, far: float) -> Completion:
    """:func:`complete`, with the relation supplied as a matrix rather than read off a pair.

    Both gates are the same and the arithmetic is the same; only where the operator came from
    differs. :func:`complete` obtains it from one exemplar, and a relation vertex obtains it from
    every edge that cites it via :func:`operator_over`.
    """
    op = np.asarray(operator, dtype=float)

    scores: list[Association] = []
    modes: list[int] = []
    for c in candidates:
        m = association_matrix(records[source], records[c])
        scores.append(associate(op, m, far=far))
        # How much of the operator this candidate's relation actually carries. Both sides are
        # matrices between the same coordinates, so the concatenation is a field the floor is
        # derived for, exactly as in `bind.shared_modes`.
        modes.append(
            int(rank(op, far=far) + rank(m, far=far)
                - rank(np.concatenate([op, m], axis=1), far=far))
        )

    live = [i for i, a in enumerate(scores) if a.resolved]
    if not live:
        return Completion(key=None, why=NOTHING_RESOLVED, scores=tuple(scores), modes=tuple(modes))

    proposal = max(live, key=lambda i: scores[i].z)
    widest = max(modes[i] for i in live)
    at_widest = [i for i in live if modes[i] == widest]
    if widest <= 0 or len(at_widest) != 1 or at_widest[0] != proposal:
        return Completion(key=None, why=INDISTINGUISHABLE, scores=tuple(scores),
                          modes=tuple(modes))

    return Completion(key=proposal, why=BOUND, scores=tuple(scores), modes=tuple(modes))
