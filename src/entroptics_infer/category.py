"""Colimits and limits over records: gluing several observations, and what they all admit.

Why a categorical object rather than an aggregation
----------------------------------------------------
A query is often not one record but several — a multi-word context, a set of observations of one
concept, the artifacts a walk reached. Scoring a candidate against such a set forces a choice: sum
the pairwise scores, take the maximum, take the mean. None of those is given by the data, and
whichever is picked is a constant wearing the shape of a method. The three disagree, and in the one
place they have been measured against each other here they disagree by more than the effect under
study.

A colimit takes no such choice. It is the *universal* object the whole diagram maps into — canonical
by its defining property, not selected — and a candidate is then scored against one object rather
than against a set plus a rule for combining.

The two constructions, and which question each answers
-------------------------------------------------------
Records live as subspaces of one shared feature basis: a record's resolved directions are what the
instrument says it occupies. On that lattice the two universal constructions are concrete.

**Colimit — the join.** The span of the members' subspaces: the smallest thing every member maps
into. It answers *"what does this set of observations, taken together, cover?"* — the topic a context
spans, the concept several records are records of. `paper-2` §72 takes colimits in this sense over
concept artifacts. (`paper-2` is *Knowledge without weights*,
``research/paper-2-knowledge-without-weights.md`` in https://github.com/Agience/agience-pharos.)

**Limit — the meet.** The intersection: the directions every member admits. It answers *"what do
these all agree on?"* — the common part of several frames, the objects two constraints both admit.
`paper-2` §789 names this as the open piece: *"Everything categorical today is a co-limit … The
target is the dual … It buys intersection as a first-class operation."* This module is that.

Degrees of freedom, counted
---------------------------
The dimension of each is a measurement, not a setting. The join's is the resolved rank of the
members' stacked bases; the meet's follows from it by inclusion–exclusion over counted ranks. A
coherent context joins to few directions and meets in several; a scattered one joins to many and
meets in none. That spread is the degrees of freedom the set actually has, and it is read rather
than declared — which is the whole reason for preferring a categorical object to an aggregation.

Where zero is an answer
-----------------------
A meet of dimension zero says the members agree on nothing, and that is a finding about the set
rather than a failure. Nothing is returned in that case and nothing is substituted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Unreadable, basis, frame

__all__ = ["Diagram", "diagram", "colimit", "limit", "inside"]


@dataclass(frozen=True)
class Diagram:
    """A set of records taken together, with what they cover and what they agree on.

    ``members``       the record keys, in the order given.
    ``dimensions``    each member's own resolved rank — how much structure it has alone.
    ``join``          ``(J, D)`` the colimit's directions: what the set covers together.
    ``join_dim``      ``J``, counted.
    ``meet_dim``      how many directions every member admits, by inclusion–exclusion on
                      counted ranks. Zero is a legitimate answer.
    """

    members: tuple
    dimensions: tuple
    join: np.ndarray
    join_dim: int
    meet_dim: int


def _bases(records, members, *, far: float):
    out = []
    for m in members:
        b = basis(records[m], far=far)
        out.append(b)
    widths = {b.shape[1] for b in out}
    if len(widths) != 1:
        raise Unreadable(
            f"the members do not meet on one basis: {sorted(widths)} coordinates. A colimit is "
            f"taken in a category whose objects share a home, and these do not."
        )
    return out


def diagram(records, members, *, far: float) -> Diagram:
    """Read a set of records as one diagram: what it covers, and what it agrees on."""
    members = tuple(members)
    if not members:
        raise Unreadable(
            "an empty diagram has no colimit to take. This raises rather than returning a "
            "zero-dimensional object, because 'these records cover nothing' is a claim about "
            "records and there are none."
        )
    parts = _bases(records, members, far=far)
    dims = tuple(int(p.shape[0]) for p in parts)

    stacked = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0))
    if stacked.shape[0] == 0:
        width = frame(records[members[0]]).shape[1]
        return Diagram(members=members, dimensions=dims, join=np.zeros((0, width)),
                       join_dim=0, meet_dim=0)

    # The join is taken at machine precision, not at the reader's level, and the distinction is
    # the point. Each member's own dimension was already counted by the instrument against a
    # derived noise floor — that is where the statistics happened, and it happened once. What
    # remains is a question about a set of directions that have already been separated from noise:
    # how many of them are independent. That is linear algebra, and asking the instrument for a
    # statistical rank of a handful of denoised unit vectors is asking it about a frame outside the
    # family its floor is derived for — a stack of three directions in 192 coordinates reads as
    # nothing, which is the correct answer to a question that should not have been put.
    #
    # `numpy.linalg.matrix_rank`'s default tolerance is `eps * max(shape) * sigma_max`, a quantity
    # derived from the machine and the matrix rather than fitted to data. `paper-2` §73 sets the
    # same precedent, computing its tolerance as `eps * ||F|| * d` from the frame rather than
    # typing one.
    _, _, Vt = np.linalg.svd(stacked, full_matrices=False)
    j = int(np.linalg.matrix_rank(stacked))
    join = Vt[:j] if j > 0 else np.zeros((0, stacked.shape[1]))

    # Inclusion-exclusion over counted ranks: what every member admits is what the sum of their
    # dimensions carries beyond what the join needs. Clamped at zero because a count cannot be
    # negative, and a negative value would mean the members overlap less than not at all.
    meet = max(0, sum(dims) - j) if len(dims) > 1 else (dims[0] if dims else 0)

    return Diagram(members=members, dimensions=dims, join=join, join_dim=j, meet_dim=meet)


def colimit(records, members, *, far: float) -> np.ndarray:
    """``(J, D)`` — the directions the set covers together. The universal object it maps into."""
    return diagram(records, members, far=far).join


def limit(records, members, *, far: float) -> int:
    """How many directions every member admits. Zero means they agree on nothing."""
    return diagram(records, members, far=far).meet_dim


def inside(records, candidate, members, *, far: float) -> float:
    """How much of a candidate lies within what the set covers, in ``[0, 1]``.

    The candidate's own resolved directions are projected onto the colimit and the retained energy
    is reported as a fraction. Nothing here chooses how to combine the members: they were combined
    once, canonically, into the object being projected onto.

    Returns zero where the set covers nothing or the candidate resolves nothing — in both cases
    there is no overlap to report, and reporting a number anyway would invent one.
    """
    d = diagram(records, members, far=far)
    own = basis(records[candidate], far=far)
    if d.join_dim == 0 or own.shape[0] == 0:
        return float(0)
    projected = own @ d.join.T
    kept = float(np.sum(projected * projected))
    total = float(np.sum(own * own))
    return kept / total if total > 0 else float(0)
