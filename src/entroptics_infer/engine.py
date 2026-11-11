"""The one seam onto ``entroptics``. Nothing else in this package imports the instrument.

Why a seam at all
-----------------
The program's first rule is that entroptics must be load-bearing — that it *decides*,
rather than decorating a decision made elsewhere. A rule like that is only worth stating if it
can be checked, and it can only be checked if there is exactly one place where the instrument is
reached. That place is this module. ``tests/test_entroptics_is_loadbearing.py`` neuters the
functions here and asserts that every downstream read changes; a read that survives a neutered
instrument was never resting on it.

Why nothing here has a default ``far``
--------------------------------------
``far`` is the false-alarm level — by Neyman–Pearson it belongs to the reader, and it is the one
number this program does not derive. Giving it a default here would let a caller take a reading
without ever naming the level it was taken at, and that reading would then travel as though it
were parameter-free. So every function in this module requires it explicitly, and every published
claim in ``research/`` is a sweep across it.

What the instrument decides, in this program's terms
------------------------------------------------------
``rank``      how many directions a record resolves — the counted analogue of a model dimension.
``basis``     the directions themselves — the counted analogue of a learned projection.
``floor``     the level below which there is nothing to report, which is what makes refusal a
              consequence rather than a branch.
``associate`` whether and how two records co-resolve on a shared basis — the counted analogue of
              an attention weight, standardized by an exact permutation null rather than divided
              by a temperature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from entroptics import Projection as _Projection
from entroptics.reads import coupling as _coupling

__all__ = ["Unreadable", "Association", "associate", "rank", "span_rank", "basis", "floor", "frame"]


class Unreadable(RuntimeError):
    """The instrument declined to take this reading, and the caller must not substitute one.

    Raised where a frame cannot be read at all — a shape the instrument has no reading for, a
    non-finite coordinate, two sides that do not meet on one basis. It is deliberately not
    raised when the instrument reads *nothing*: a rank of zero and an unresolved association are
    answers, and the difference between "I cannot resolve this" and "there is one thing here"
    must never collapse into the same return value.
    """


@dataclass(frozen=True)
class Association:
    """What was measured between two records that meet on one basis.

    The split between decision and evidence is the instrument's, and it is kept here because
    inference needs both. ``strength`` is zeroed below the level, so an unresolved association
    reports as *no* association rather than as a small one — which is what stops a long tail of
    noise from summing into an answer. ``z`` is never zeroed, so the evidence remains legible
    even where the decision says nothing.

    ``z``          signed standardized alignment, against the exact permutation null.
    ``sign``       ``+1`` co-resolving, ``-1`` anti-resolving, ``0`` unresolved.
    ``strength``   signed cosine of the two centred frames, zeroed below the level.
    ``tightness``  dominant-mode share of the cross spectrum: ``→1`` locked through one mode,
                   ``→0`` shared variance spread over many. This is the facet count showing
                   itself, and it is the read that a scalar dot product cannot produce at all.
    ``resolved``   the decision at the level the caller named.
    ``cutoff``     the ``|z|`` that level required.
    ``n``          rows compared.
    """

    z: float
    sign: int
    strength: float
    tightness: float
    resolved: bool
    cutoff: float
    n: int


def frame(record: Any) -> np.ndarray:
    """A record as the ``(T, D)`` frame the instrument reads, or ``Unreadable``.

    The instrument reads a two-dimensional field with one ordered axis and one feature axis. A
    record that is not already that shape has no reading here — this function refuses rather
    than reshaping, because a reshape chooses which axis is ordered and that choice would be a
    forcing made silently.
    """
    W = np.asarray(record, dtype=float)
    if W.ndim != 2:
        raise Unreadable(
            f"a record is read as a (T, D) frame with one ordered axis and one feature axis; "
            f"got shape {W.shape}. Reshaping it here would choose which axis is ordered, and "
            f"that choice belongs to whoever knows what the record is."
        )
    if not np.isfinite(W).all():
        raise Unreadable("the record carries a non-finite coordinate; there is no reading of it")
    return W


def rank(record: Any, *, far: float) -> int:
    """How many directions this record resolves above its own floor.

    This is the counted analogue of a model dimension. It is not chosen, not swept for a best
    value, and not bounded by a ceiling: it is what the record's own entropy admits. Zero is a
    legitimate answer and means the record reads as noise does — which is the instrument
    working, not failing.
    """
    return int(_Projection(frame(record), far=far).read().K_signal)


def span_rank(record: Any, *, far: float) -> int:
    """How many directions this record's linear span resolves above its own floor: :func:`rank`
    with the origin kept.

    :func:`rank` reads the record about its own mean, so it counts how the rows spread around
    their centre. Where the record's zero means something -- a stack of association matrices,
    whose zero is no relation -- the centre is itself a direction, and the count wanted is the
    dimension of the span the rows occupy from the origin. ``R`` linearly independent rows read
    ``R`` here and ``R - 1`` about their mean. The instrument reads it with
    ``Projection(..., disconnected=None)``, whose floor holds its level on noise symmetric about 0.
    """
    return int(_Projection(frame(record), far=far, disconnected=None).read().K_signal)


def basis(record: Any, *, far: float) -> np.ndarray:
    """The ``(K, D)`` directions this record resolves — the counted analogue of a projection.

    Where a transformer holds ``W_Q``/``W_K``/``W_V`` as matrices fitted by descent, this is the
    record's own resolved directions, truncated at the rank the instrument counted. Nothing is
    fitted, and the number of rows is a measurement.

    **The directions are in the record's own ``D`` coordinates.** The instrument folds a wide
    feature axis before reading it, so the basis it holds internally lives in folded coordinates
    whose width varies from record to record — correct for its own purposes, and unusable for
    comparing two records or for reading a direction as a map between their coordinates. The
    truncation point is still entirely the instrument's: no rank is chosen here, and no tolerance
    enters.

    Returns an empty ``(0, D)`` array where the record resolves nothing, so a caller that sums
    over the directions gets an empty sum — a refusal by arithmetic rather than by a branch.
    """
    W = frame(record)
    k = rank(W, far=far)
    if k <= 0:
        return np.zeros((0, W.shape[1]))
    _, _, Vt = np.linalg.svd(W - W.mean(axis=0, keepdims=True), full_matrices=False)
    return np.asarray(Vt)[:k]


def floor(record: Any, *, far: float) -> float:
    """The level below which this record has nothing to report.

    Refusal in this program is this number and nothing else: no fallback branch, no minimum
    confidence, no "if we found nothing, return the closest thing". Where the signal does not
    clear this, the answer is empty.
    """
    return float(_Projection(frame(record), far=far).read().noise_floor)


def associate(a: Any, b: Any, *, far: float) -> Association:
    """The measured association between two records meeting on one shared basis.

    This is the attention weight, and what it replaces is worth being exact about. A transformer
    scores a query against a key with ``q·k/√d`` and normalizes the row with a softmax. Both
    halves carry a constant: ``√d`` is a temperature, and the softmax imposes a normalization
    that forces the row to sum to one whether or not anything in it was resolved. Here the score
    is the two frames' alignment standardized by the exact permutation null of re-pairing
    their rows — closed-form moments, no sampling, no RNG, and no scale supplied by the caller.
    Nothing forces the row to sum to anything; a query whose keys all read as noise gets a row of
    zeros, which is refusal.

    Both sides must meet on one basis, and the instrument raises without one. That is a real
    constraint, not a formality: the sign of an association is a property of the two sides being
    in the *same* coordinates, and across two different bases the only available statistic is
    non-negative by construction and has no exact null. A caller who cannot put two records in
    one basis does not have an association to measure.
    """
    A, B = frame(a), frame(b)
    if A.shape[-1] != B.shape[-1]:
        raise Unreadable(
            f"the two records meet on {A.shape[-1]} and {B.shape[-1]} coordinates. An "
            f"association is signed only on a shared basis; across two bases the statistic is "
            f"non-negative by construction and has no exact null, so there is nothing to report."
        )
    if A.shape[0] != B.shape[0]:
        raise Unreadable(
            f"the two records carry {A.shape[0]} and {B.shape[0]} ordered steps. The null is a "
            f"re-pairing of rows, so the rows must correspond."
        )
    c = _coupling(A, B, far=far)
    return Association(
        z=float(c.z),
        sign=int(c.sign),
        strength=float(c.strength),
        tightness=float(c.tightness),
        resolved=bool(c.resolved),
        cutoff=float(c.cutoff),
        n=int(c.n),
    )
