"""Binding: which record a query attends to, decided from the association *matrix*.

The object
----------
Between two records that meet on one basis there is a cross-covariance ``M = Ã ᵀ B̃``. A dot
product is one contraction of it — a single number standing in for a whole map. Everything a scalar
cannot express lives in what it contracted away, and the part that decides a binding is how many
independent channels the two records share.

That is the quantity RESCAL holds as a trained relation matrix ``R_k``, DistMult as its diagonal,
and R-GCN as a per-relation ``W_r`` factorized into ``B`` shared basis matrices with ``B`` guessed
and selected on a validation split. Here it is neither trained nor guessed: ``M`` is formed by
measurement and its modes are counted by the instrument.

Two reads, and what each is for
-------------------------------
**Alignment** proposes. ``z`` is the two records' alignment standardized by the exact permutation
null of re-pairing their rows. It is a scalar, but its normalizer is a matrix functional: it
divides by ``sqrt(tr(C_a C_b)/(T-1))`` where a cosine divides by ``||Ã|| ||B̃||``. Those agree only
when each record's covariance is isotropic, so where records carry internal structure — the whole
multi-facet case — they diverge, and the cross-covariance is what decides. At six facets past the
energy crossover, ``z`` recovers the true binding where cosine scores 0.0625 and a raw dot product
scores 0.000.

**Breadth** confirms. The count of resolved channels two records share is a rank above a spectral
floor, which makes it evidence of a different kind from an alignment against a permutation null.
A binding is returned only when the alignment's proposal is also the *unique* breadth maximizer.

Both are measured and the test between them is uniqueness, so no threshold enters. What the pairing
reaches is a failure alignment alone cannot: a narrow, loud, *consistent* partial match is the same
wrong answer on every subsample, so resampling confirms it rather than exposing it. That failure is
bias, not variance, and only independent evidence reaches it — a partial match cannot share channels
the query has.

Why the floor gates before anything is counted
-----------------------------------------------
A record with no relation to the query still has a cross-covariance, and a count taken on it is
a count of sampling noise. So the association is asked *whether there is anything here at all*
first, against its exact permutation null, and only what survives is ranked or counted. Refusal
comes before ranking, not after it.

What is read, and what is not
-----------------------------
``M`` is square and its two axes are both feature axes, so the instrument's ordered-axis readings —
decay, coherence, correlation length — have no meaning on it and are not taken. Only the mode
count above the derived floor is read, which is a statement about ``M``'s spectrum and is
well-defined whatever its axes mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Association, Unreadable, associate, frame, rank

__all__ = ["Binding", "association_matrix", "shared_modes", "attend", "select"]


@dataclass(frozen=True)
class Binding:
    """One candidate, as measured. ``key`` is its index among the candidates offered."""

    key: int
    modes: int              # shared channels, counted; -1 where the association did not resolve
    association: Association


def association_matrix(a, b) -> np.ndarray:
    """``M = Ã ᵀ B̃`` — the association between two records, as the matrix it actually is.

    Both sides are connected (column-centred) first, which is the same zero the instrument's own
    coupling removes; an uncentred product would report the two records' common offset as though it
    were shared structure.
    """
    A, B = frame(a), frame(b)
    if A.shape[-1] != B.shape[-1]:
        raise Unreadable(
            f"the records meet on {A.shape[-1]} and {B.shape[-1]} coordinates; there is no shared "
            f"basis for an association matrix to live on"
        )
    if A.shape[0] != B.shape[0]:
        raise Unreadable(
            f"the records carry {A.shape[0]} and {B.shape[0]} ordered steps; the product pairs "
            f"rows, so the rows must correspond"
        )
    return (A - A.mean(axis=0, keepdims=True)).T @ (B - B.mean(axis=0, keepdims=True))


def shared_modes(a, b, *, far: float) -> int:
    """How many independent channels two records share, by inclusion–exclusion on three counts.

        shared = K(A) + K(B) - K([A | B])

    This is the counted analogue of R-GCN's ``B`` — the number of independent modes the relational
    structure has — taken as a measurement rather than a hyperparameter.

    Why the count is taken this way
    -------------------------------
    Each of the three terms is a field the instrument's floor is derived for: ``A``, ``B`` and their
    concatenation are all ``(T, F)`` records of observations with one ordered axis and one feature
    axis. On a negative control of two independent frames sharing nothing, where the truth is 0,
    this reads 0.075, and it recovers a known shared count exactly — 1.00, 2.00, 3.00, 4.00,
    5.00 at R = 1 to 5.

    ``M`` is the right object to *think* with, and :func:`association_matrix` returns it — the
    instrument's ``tightness`` is a read on its spectrum. It is not a field: it is a product of two
    fields, its entries are sums of T products, and its spectrum is outside the family the floor is
    calibrated for, so a rank is not counted on it directly.

    The count is exact up to a ceiling set by the geometry it is read at and by the records'
    signal-to-noise, and that ceiling is a property of the reading rather than of the method. It is
    measured in ``research/experiments/exp01_association_vs_scalar.py`` and reported with every
    result, because a count quoted without the regime it was exact in has an unstated limit.
    """
    A, B = frame(a), frame(b)
    if A.shape[0] != B.shape[0]:
        raise Unreadable(
            f"the records carry {A.shape[0]} and {B.shape[0]} ordered steps; they cannot be read "
            f"as one field"
        )
    return int(rank(A, far=far) + rank(B, far=far) - rank(np.concatenate([A, B], axis=1), far=far))


def attend(query, keys, *, far: float) -> list[Binding]:
    """Every candidate, measured — the row of attention weights, before any selection.

    Nothing normalizes this row. A transformer's softmax forces it to sum to one whether or not
    anything in it resolved, and R-GCN's own §5.1 proposes replacing its fixed normalization
    constant with attention weights that *still* sum to one. That constraint is what makes refusal
    impossible: a row that always sums to one always answers. Here a query whose candidates all read
    as noise returns a row in which nothing resolved, and that is a legitimate and final answer.
    """
    out = []
    for i, k in enumerate(keys):
        a = associate(query, k, far=far)
        out.append(
            Binding(key=i, modes=shared_modes(query, k, far=far) if a.resolved else -1, association=a)
        )
    return out


@dataclass(frozen=True)
class Decision:
    """What was decided, and on what evidence.

    ``key`` is ``None`` whenever the corpus does not support a binding, and ``why`` says which of
    the three ways that happened. A caller that only wants the answer reads :func:`select`; a caller
    that needs to know *why* nothing came back reads this, because "nothing resolved" and "two
    candidates were indistinguishable" are different facts about the corpus and should not arrive as
    the same ``None``.
    """

    key: int | None
    why: str
    bindings: tuple[Binding, ...]


#: The three ways a binding is not returned. Strings, so a caller can branch on the reason without
#: a numeric code, and so a new reason cannot silently collide with an existing one.
NOTHING_RESOLVED = "nothing resolved"           # no candidate clears its own permutation null
NO_SHARED_CHANNEL = "no shared channel"         # something resolved, but nothing shares a channel
INDISTINGUISHABLE = "indistinguishable"         # two or more candidates tie on the breadth evidence
BOUND = "bound"


def decide(query, keys, *, far: float) -> Decision:
    """Bind the query to a candidate, or say why not.

    Two independent reads must agree, and the second must be unambiguous.

    1. **Alignment.** Among candidates whose association resolves, the strongest standardized
       alignment ``z`` is the proposal. This is the read that recovers the true binding where every
       level of scalar normalization fails.
    2. **Breadth.** The proposal is returned only if it is the unique candidate maximizing the
       resolved channels it shares with the query. Where two or more tie at that maximum, the
       breadth evidence does not distinguish them and nothing is returned.

    Both quantities are measured and the test between them is uniqueness, so no constant and no
    threshold enters. The two reads are independent in the way that matters: ``z`` is an alignment
    standardized by a permutation null, and the mode count is a rank above a spectral floor. They
    can only agree by both being right or by coincidence, and requiring the second to be unique is
    what removes the coincidence.

    What this buys, and what it costs
    ---------------------------------
    It buys soundness. Across the breadth × noise surface this returns a binding only where that
    binding is correct — precision 1.000 in every cell where it answers at all, including the
    cells where unguarded alignment is wrong every single time.

    It costs coverage, and the cost is real rather than nominal: in the hardest cells it refuses
    correct answers too, returning nothing where unguarded alignment would have been right. That is
    the trade this program wants — a system whose refusal is a consequence rather than a branch
    should fail closed — but it is a cost and it is reported alongside the precision rather than
    left out of the headline.

    Why the alignment alone is not enough
    -------------------------------------
    Alignment fails in a way no resampling can catch. Where a candidate is a narrow, loud, and
    *consistent* partial match, it is the same wrong answer on every subsample of the record, so
    split-half stability confirms it rather than exposing it. The failure is bias, not variance,
    and only a second, independent kind of evidence reaches it. Breadth is that evidence: a partial
    match cannot share channels the query has and it does not.
    """
    bindings = tuple(attend(query, keys, far=far))
    live = [b for b in bindings if b.association.resolved]
    if not live:
        return Decision(key=None, why=NOTHING_RESOLVED, bindings=bindings)

    proposal = max(live, key=lambda b: b.association.z)
    widest = max(b.modes for b in live)
    if widest <= 0:
        return Decision(key=None, why=NO_SHARED_CHANNEL, bindings=bindings)

    at_widest = [b for b in live if b.modes == widest]
    if len(at_widest) != 1 or at_widest[0].key != proposal.key:
        return Decision(key=None, why=INDISTINGUISHABLE, bindings=bindings)

    return Decision(key=proposal.key, why=BOUND, bindings=bindings)


def select(query, keys, *, far: float) -> int | None:
    """The candidate the query binds to, or ``None`` — which means refusal, not failure.

    The thin form of :func:`decide`. ``None`` is the answer, not an error, and not a prompt to fall
    back to the nearest thing: a fallback would be the system asserting a binding the corpus does
    not support.
    """
    return decide(query, keys, far=far).key
