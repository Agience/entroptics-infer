"""Relation type, read from the corpus. No vocabulary, no enumeration, no declared kinds.

What a relation is here
-----------------------
Not a label and not a member of a set. Between the two endpoint records of an edge there is a
cross-covariance ``M = Ã ᵀ B̃``, and that matrix is the relation. Two edges are of the same kind
when their matrices point the same way; the kinds a corpus contains are the directions its edge
matrices occupy; and how many there are is a rank, measured.

This is RESCAL's ``X_k ≈ A R_k Aᵀ`` with the fitting removed. There, ``R_k`` is a trained matrix per
relation and ``k`` ranges over a vocabulary supplied with the dataset. Here ``M`` is observed rather
than fitted, and there is no ``k`` — an edge's kind is *where its matrix sits*, which is a position
in a continuous space rather than a choice from a list.

Why a type cannot be a label
----------------------------
A fixed set of kinds is a decision made in advance about which relationships are possible, and it
fails in both directions. Relations that do not fit are flattened into the nearest member at write
time, where nothing downstream can recover the distinction. A vocabulary larger than the corpus
asserts structure that is not there. Nothing in a trained network has an edge-type table: what such
a system calls a relation is a direction that emerged, and the number of them is bounded by nothing
declared beforehand.

So the count here is unbounded by construction. :func:`relation_modes` returns whatever the
corpus resolves; no ceiling is imposed, none is configurable, and a corpus with more kinds than the
last one simply reads a larger number.

What this replaces, precisely
-----------------------------
R-GCN carries one weight matrix ``W_r`` per relation and, because that explodes, factorizes it into
``B`` shared basis matrices — where ``B`` is chosen by hand and then selected on a held-out split.
``B`` is an estimate of how many independent modes the relational structure has.
:func:`relation_modes` is that quantity, measured from the edges before any task is scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bind import association_matrix
from .engine import Unreadable, associate, span_rank

__all__ = [
    "Kinds",
    "Screen",
    "signature",
    "signatures",
    "relation_modes",
    "relation_basis",
    "relation_coordinates",
    "carries_kinds",
]


def signature(records, src, dst) -> np.ndarray:
    """One edge's relation, flattened — the association matrix of its two endpoint records.

    Flattening is not a loss here: the object of interest is *which* matrix this is, and two
    matrices are the same relation exactly when their flattened forms are parallel. The ``D × D``
    shape matters for reading a single relation's structure and not for comparing relations to each
    other, which is what this is for.
    """
    return np.asarray(association_matrix(records[src], records[dst])).ravel()


def signatures(records, edges) -> np.ndarray:
    """``(n_edges, D²)`` — every edge's relation, stacked in the order given.

    This stack is the corpus's relational structure with nothing said about it. Every read below is
    taken on this and on nothing else, so no read can be influenced by a type that was written down,
    because none was.
    """
    if not len(edges):
        raise Unreadable(
            "no edges, so there is no relational structure to read. This raises rather than "
            "returning an empty stack, because a rank of 0 taken on nothing would report as 'this "
            "corpus has no relations' — a claim about the corpus, made where no observation exists."
        )
    rows = [signature(records, s, d) for s, d in edges]
    widths = {r.shape[0] for r in rows}
    if len(widths) != 1:
        raise Unreadable(
            f"the edges' records do not meet on one basis: association matrices of {sorted(widths)} "
            f"entries. Relations can only be compared where they live in the same space."
        )
    return np.asarray(rows)


def relation_modes(records, edges, *, far: float) -> int:
    """**How many kinds of relation this corpus contains** — counted, never declared.

    The resolved rank of the stacked edge signatures: the number of independent directions the
    corpus's association matrices occupy above the derived floor, counted from the origin
    (:func:`entroptics_infer.engine.span_rank`). The zero relation is no relation, so the span is
    the linear one: ``R`` independent relations occupy ``R`` directions, which is what
    ``RelationCount.lean`` proves, where a count about the corpus's mean relation would read
    ``R - 1``. This is the counted form of R-GCN's ``B``, and of RESCAL's implicit relation count.

    It is unbounded. No maximum is imposed here or configurable anywhere, so a corpus carrying
    more kinds than the last one reads a larger number rather than saturating against a ceiling
    someone chose. What does bound it is the geometry the reading is taken at and the records'
    signal-to-noise, and that limit is measured and reported rather than assumed away.

    Zero is a legitimate answer: it says the edges share no common structure, which is a finding
    about the corpus and not a failure of the read.
    """
    return int(span_rank(signatures(records, edges), far=far))


@dataclass(frozen=True)
class Kinds:
    """The kinds a corpus's edges turn out to occupy.

    ``stack``     ``(n_edges, D²)`` the observed relations, as measured.
    ``centred``   the same, connected — the zero removed is the corpus's own mean relation.
    ``count``     how many kinds, counted by the instrument. Nothing else decides this.
    ``directions````(count, D²)`` the kinds themselves; a row reshapes to the matrix it is.
    """

    stack: np.ndarray
    centred: np.ndarray
    count: int
    directions: np.ndarray


def _kinds(records, edges, *, far: float) -> Kinds:
    """The stack, its counted number of kinds, and the directions they occupy.

    The count is the instrument's and nothing else decides it. The directions are then the
    leading right-singular directions of the centred stack, taken in the signatures' own ``D²``
    coordinates so that a kind can be reshaped back to the ``D × D`` matrix it is.

    They are computed here rather than read from :func:`entroptics_infer.engine.basis` for a
    concrete reason: the instrument folds a wide feature axis before reading it, so the basis it
    returns lives in the folded coordinates. That is correct for the instrument's own purposes and
    wrong for this one, where a direction has to be interpretable as a matrix between the endpoint
    records' coordinates. The truncation point is still entirely the instrument's — no rank is
    chosen here, and no tolerance enters.
    """
    S = signatures(records, edges)
    k = int(span_rank(S, far=far))
    Sc = S - S.mean(axis=0, keepdims=True)
    if k <= 0:
        return Kinds(stack=S, centred=Sc, count=k, directions=np.zeros((0, S.shape[1])))
    _, _, Vt = np.linalg.svd(Sc, full_matrices=False)
    return Kinds(stack=S, centred=Sc, count=k, directions=Vt[:k])


def relation_basis(records, edges, *, far: float) -> np.ndarray:
    """``(K, D²)`` — the kinds themselves, as the directions the corpus's relations occupy.

    Each row is one independent relation mode. These are what a trained system holds as its
    per-relation weight matrices, read off the edges instead. Reshaping a row to ``D × D`` gives
    that kind as the matrix it is.
    """
    return _kinds(records, edges, far=far).directions


def relation_coordinates(records, edges, *, far: float) -> np.ndarray:
    """``(n_edges, K)`` — where each edge sits among the kinds.

    **This is an edge's type**, and it is a position rather than a name. It is continuous, so an
    edge that is partly one kind and partly another is represented as exactly that rather than being
    forced to whichever label is nearest. Two edges are of the same kind when their coordinates are
    parallel, and how strongly an edge expresses a kind is the size of its component along it.

    Returns an ``(n_edges, 0)`` array where the corpus resolves no kinds, so a caller that sums over
    types gets an empty sum — nothing asserted where nothing was measured.
    """
    k = _kinds(records, edges, far=far)
    return k.centred @ k.directions.T

@dataclass(frozen=True)
class Screen:
    """What a corpus offers a read that needs a relation supplied by example.

    ``present``     per name, whether that relation is in the signal at all: its edges associate
                    more than pairs standing in no named relation, by more than the instrument's
                    own cutoff at the caller's level.
    ``separation``  per name, by how much, in units of the unrelated pairs' own spread.
    ``alignment``   ``(n, n)`` how far each name's mean relation points along each other's. The
                    diagonal is a name against an independent sample of itself, so it is the scale
                    the off-diagonal has to be read against rather than 1 by construction.
    ``distinct``    per name, whether it points somewhere the others do not.
    ``count``       how many names are both present and distinct.
    ``exemplifiable`` whether a relation can be supplied by example here at all, which needs two.

    And the other question, which the four above do not answer:

    ``carried``     per name, how far its edges' mean stands above the mean of the same number
                    of unrelated pairs, in spreads of that drawn null.
    ``carries``     per name, whether that mean exceeds every draw of the null.

    The difference matters and is not a refinement. ``separation`` divides by the spread of a
    single unrelated pair, so it asks whether *one* edge is distinguishable — the right question
    for :func:`entroptics_infer.complete`, which is handed one exemplar pair and must name the
    relation from it. Being a ratio of two per-pair quantities it does not move with edge count:
    measured flat across a 64x range, 6.95 to 7.86, while the z of the mean over the same edges went
    16 to 141.

    ``carried`` asks whether the lattice holds the relation at all, which is a question about the
    mean and therefore about how many edges there are.
    :meth:`entroptics_infer.Lattice.operator` aggregates every edge citing a relation, so this is
    the question an engine needs and the one a corpus should be refused on. A road network reads
    0.45 to 1.42 on ``separation`` — refused in all 30 cells swept — and 7.9 to 32.6 on ``carried``
    against a control at −1.2 to 1.3.
    """

    names: tuple
    present: dict
    separation: dict
    alignment: np.ndarray
    distinct: dict
    count: int
    exemplifiable: bool
    carried: dict
    carries: dict


def carries_kinds(records, named_edges, *, far: float, seed: int, draws: int,
                  unrelated=None) -> Screen:
    """**Can this corpus exemplify a relation?** The question to settle before scoring a task on it.

    :func:`entroptics_infer.infer.complete` supplies a relation by example: given a pair standing in
    one, which candidate stands in it to a third record? That question needs two kinds present
    and telling apart, because with one there is nothing for "which relation is being asked" to
    distinguish. Scoring the task on a corpus that cannot carry it measures the corpus and reports
    it as the read.

    Two corpora have been screened this way and both were refused, for different reasons worth
    keeping apart. An inertial corpus carried exactly one kind under either reading of what a record
    is. A word stream's *continues k later* relations were each consistent — an operator read off
    one pair pointed the same way as one read off another — and identical to each other, which is
    a different failure and equally fatal: four names, one relation.

    So both are measured. A name is present when its edges associate more than pairs standing in
    no named relation; a name is distinct when its mean relation points somewhere the others do
    not. Nothing is supplied for either test: the comparison level is the caller's ``far`` and the
    cutoff is read from the instrument, exactly as :func:`entroptics_infer.admission.axis_carries`
    reads it.

    ``named_edges`` maps a name to its edges as ``(src, dst)`` pairs.

    The null, and the assumption it makes
    ---------------------------------------
    Unless ``unrelated`` is given, the null is drawn from pairs appearing in no named set —
    ``draws`` of them — which is what "unrelated" can mean when nothing is labelled.

    That assumes ``named_edges`` lists a relation's edges completely. Where a caller passes a
    *sample* of a larger family, the rest of that family stays in the null and raises it toward the
    named sets. Measured on a corpus with two real relations and a third of each named, that
    contamination does not change the verdict — both are still found — so the drawn null is
    sound at that level and this is a caveat rather than a defect.

    ``unrelated`` takes the assumption out of the instrument's hands: an explicit sequence of
    ``(src, dst)`` pairs standing in no named relation. Supply it where the named sets are a small
    sample of a large family, or where some pairs are related in ways no name covers.
    """
    names = tuple(named_edges)
    if len(names) < 1:
        raise Unreadable(
            "no named relations, so there is nothing to screen. This raises rather than returning a "
            "verdict, because 'this corpus carries no kinds' is a claim and no observation was made."
        )
    keys = sorted(records)
    if len(keys) < 2:
        raise Unreadable(
            f"a relation needs two records to stand between and this corpus has {len(keys)}."
        )
    rng = np.random.default_rng(seed)
    claimed = {(s, d) for edges in named_edges.values() for s, d in edges}

    if unrelated is not None:
        pairs = tuple(unrelated)
        if not pairs:
            raise Unreadable(
                "an empty `unrelated` was given, so there is no null to read the named relations "
                "against. Omit it to draw one, or supply pairs that stand in no named relation."
            )
        sample = [pairs[int(i)] for i in rng.integers(len(pairs), size=draws)]
        unrelated = [abs(associate(records[a], records[b], far=far).z) for a, b in sample]
    else:
        unrelated = []
        for _ in range(draws):
            a = keys[int(rng.integers(len(keys)))]
            b = keys[int(rng.integers(len(keys)))]
            if a != b and (a, b) not in claimed and (b, a) not in claimed:
                unrelated.append(abs(associate(records[a], records[b], far=far).z))
    if len(unrelated) < 2:
        raise Unreadable(
            "fewer than two pairs stand in no named relation, so there is no null to read the named "
            "ones against. Name fewer relations, or bring a corpus with more records."
        )
    base = float(np.mean(unrelated))
    spread = float(np.std(unrelated, ddof=1))

    probe = np.arange(float(len(unrelated) * 2)).reshape(len(unrelated), 2)
    cutoff = associate(probe, probe[::-1].copy(), far=far).cutoff

    pool = np.asarray(unrelated, dtype=float)
    # Its own generator, deliberately not `rng`. `rng` is drawn from again below to split each
    # relation's edges for the alignment, so consuming it here would shift that stream and move
    # every `distinct` verdict and every figure downstream of one -- a silent change to results
    # that have already been published, caused by adding a reading that is supposed to be additive.
    #
    # It is spawned from the caller's `seed` and nothing else. A second seed written down here
    # would be a number chosen in this file, which is what `test_no_constants` refuses.
    aggregate_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])

    present, separation, halves, carried, carries = {}, {}, {}, {}, {}
    for name in names:
        edges = tuple(named_edges[name])
        if not edges:
            raise Unreadable(f"the relation {name!r} was named with no edges to read it from.")
        zs = [abs(associate(records[s], records[d], far=far).z) for s, d in edges]
        sep = (float(np.mean(zs)) - base) / spread if spread > 0 else float(0)
        separation[name] = sep
        present[name] = bool(sep >= cutoff)

        # The same pairs, asked the other question: does the lattice carry this relation at all?
        # The null is the distribution of a mean over **as many pairs as this relation has edges**,
        # resampled from the unrelated pool that was drawn above -- so it costs no further reads,
        # and the edge count enters where it belongs instead of being divided out. The verdict is
        # the observed mean's rank against that null, which is the same permutation-style reading
        # used elsewhere here and carries no cutoff of its own.
        observed = float(np.mean(zs))
        null_means = np.array([pool[aggregate_rng.integers(pool.size, size=len(edges))].mean()
                               for _ in range(draws)])
        sd = float(np.std(null_means, ddof=1))
        carried[name] = ((observed - float(np.mean(null_means))) / sd if sd > 0 else float(0))
        carries[name] = bool(observed > float(np.max(null_means)))
        # split the edges so a name can be compared against an independent sample of itself, which
        # is the only fair scale for "points somewhere the others do not"
        S = signatures(records, edges)
        order = rng.permutation(len(edges))
        cut = max(1, len(edges) // 2)
        halves[name] = (S[order[:cut]].mean(axis=0), S[order[cut:]].mean(axis=0)
                        if len(edges) > 1 else S[order[:cut]].mean(axis=0))

    def _along(u, v):
        d = float(np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.dot(u, v)) / d if d > 0 else float(0)

    n = len(names)
    alignment = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            alignment[i, j] = _along(halves[a][0], halves[b][1])

    # The scale for "points somewhere the others do not" is how far *different* names point along
    # each other across the whole matrix. That is a quantity the corpus supplies; a fixed divisor
    # would be a number chosen here, which this package does not do.
    off = [alignment[i, j] for i in range(n) for j in range(n) if i != j]
    sd = float(np.std(off, ddof=1)) if len(off) > 1 else float(0)
    distinct = {}
    for i, name in enumerate(names):
        others = [alignment[i, j] for j in range(n) if j != i]
        if not others:
            # one name cannot be distinct from others there are none of, so presence is the whole
            # of what can be said about it
            distinct[name] = bool(present[name])
            continue
        gap = alignment[i, i] - float(np.mean(others))
        distinct[name] = bool(present[name] and sd > 0 and gap / sd >= cutoff)

    count = sum(1 for name in names if present[name] and distinct[name])
    return Screen(
        names=names,
        present=present,
        separation=separation,
        alignment=alignment,
        distinct=distinct,
        count=count,
        exemplifiable=bool(count >= 2),
        carried=carried,
        carries=carries,
    )
