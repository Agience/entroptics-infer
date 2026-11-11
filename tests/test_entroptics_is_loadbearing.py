"""Rule 1: the instrument must *decide*, and a claim like that is only worth making if it fails.

What load-bearing means here
----------------------------
Not "the package imports entroptics". Not "entroptics appears in the call graph". Load-bearing
means the instrument decides, without a supplied constant — so if the instrument is replaced by
something that always returns the same answer, every read that rested on it must change. A read
that survives a neutered instrument was resting on something else, and whatever that something else
was is the thing actually deciding.

This is the test that would catch the failure mode the program is most exposed to: an
implementation that calls the instrument, ignores what comes back, and computes the answer from
arithmetic of its own. That implementation passes an import check, passes a call-graph check, and
is a lie. It does not pass this one.

Why the neutering is done by substitution rather than by a flag
---------------------------------------------------------------
A flag inside the package would be a branch the package could be written around — and it would be a
constant. Substituting the module-level names ``engine`` binds is the same thing the real dependency
does, so a function that has genuinely routed its decision through the instrument cannot avoid the
substitute.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import engine

FAR = 0.05  # the level these tests read at; the package never supplies one, so a caller must


def _structured(seed):
    """A frame with real structure: a few outer products under noise.

    Written here rather than imported so the test does not depend on the corpus machinery it is
    meant to be independent of.
    """
    rng = np.random.default_rng(seed)
    T, D, modes = 80, 12, 3
    W = sum(np.outer(rng.normal(size=T), rng.normal(size=D)) for _ in range(modes))
    return W + rng.normal(size=(T, D)) * 0.25


class _Blind:
    """A projection that resolves a fixed amount whatever it is shown."""

    class _Read:
        K_signal = 1
        noise_floor = 1.0

    def __init__(self, W, *, far):
        self._D = W.shape[1]

    def read(self):
        return self._Read()

    @property
    def Vt(self):
        return np.eye(self._D)


def _blind_coupling(a, b, *, far):
    """A coupling that reports the same association for every pair."""

    class _C:
        z, sign, strength, tightness, resolved, cutoff, n = 1.0, 1, 1.0, 1.0, True, 1.0, 1

    return _C()


def test_rank_is_decided_by_the_instrument(monkeypatch):
    """A structured frame and a noise frame must not read the same rank — and must stop differing
    the moment the instrument stops deciding."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(80, 12))
    structured = _structured(0)

    live = (engine.rank(structured, far=FAR), engine.rank(noise, far=FAR))
    assert live[0] != live[1], (
        "structure and noise read the same rank, so rank is not discriminating at all and the "
        "rest of this test cannot mean anything"
    )

    monkeypatch.setattr(engine, "_Projection", _Blind)
    blinded = (engine.rank(structured, far=FAR), engine.rank(noise, far=FAR))
    assert blinded != live, (
        "rank did not change when the instrument was neutered: something other than entroptics "
        "is deciding it"
    )


def test_the_floor_is_decided_by_the_instrument(monkeypatch):
    """Refusal in this program is the floor and nothing else, so the floor must move with the
    data and must collapse when the instrument is neutered."""
    a, b = _structured(1), _structured(2) * 10.0
    live = (engine.floor(a, far=FAR), engine.floor(b, far=FAR))
    assert live[0] != pytest.approx(live[1]), "the floor does not respond to the data"

    monkeypatch.setattr(engine, "_Projection", _Blind)
    assert engine.floor(a, far=FAR) == engine.floor(b, far=FAR), (
        "the floor still varied with a neutered instrument, so it is not being read from it"
    )


def test_the_basis_is_decided_by_the_instrument(monkeypatch):
    """The basis is what replaces a trained projection, so it must come from the record."""
    W = _structured(3)
    live = engine.basis(W, far=FAR)
    assert live.shape[0] > 0

    monkeypatch.setattr(engine, "_Projection", _Blind)
    blinded = engine.basis(W, far=FAR)
    assert blinded.shape != live.shape or not np.allclose(blinded, live), (
        "the basis is unchanged by a neutered instrument: it is not the record's resolved basis"
    )


def test_association_is_decided_by_the_instrument(monkeypatch):
    """The association is the attention weight. Two records that share structure and two that do
    not must read differently, and that difference must be the instrument's."""
    rng = np.random.default_rng(4)
    A = _structured(4)
    shared = A * 0.5 + rng.normal(size=A.shape) * 0.5
    independent = rng.normal(size=A.shape)

    live_shared = engine.associate(A, shared, far=FAR)
    live_indep = engine.associate(A, independent, far=FAR)
    assert live_shared.resolved and not live_indep.resolved, (
        "the association does not separate a shared record from an independent one, so there is "
        "nothing here for the instrument to be load-bearing for"
    )

    monkeypatch.setattr(engine, "_coupling", _blind_coupling)
    assert engine.associate(A, independent, far=FAR).resolved, (
        "the association still refused an independent pair with a neutered coupling, so the "
        "refusal is coming from somewhere other than the instrument"
    )


def test_refusal_is_a_reading_and_not_an_error():
    """Nothing resolved and nothing readable must stay distinguishable.

    ``paper-2`` (*Knowledge without weights*, in github.com/Agience/agience-pharos) §65 states the
    reason: *"I cannot resolve this"* and *"there is one thing here"*
    must never be the same answer. The same applies one level up — a frame the instrument
    *declines to read at all* is a third state, and collapsing it into "resolved nothing" would
    silently convert a broken input into a confident refusal.
    """
    rng = np.random.default_rng(5)
    assert engine.rank(rng.normal(size=(80, 12)), far=FAR) == 0        # read, resolved nothing
    assert engine.basis(rng.normal(size=(80, 12)), far=FAR).shape[0] == 0

    with pytest.raises(engine.Unreadable):
        engine.frame(np.zeros((2, 2, 2)))                              # not a frame at all
    with pytest.raises(engine.Unreadable):
        engine.frame(np.array([[np.nan, 1.0], [1.0, 1.0]]))            # not finite


def test_an_association_across_two_bases_is_refused_not_approximated():
    """Two records on different bases have no signed association, and the seam must say so.

    The temptation is to pad, truncate or project one side so a number comes back. Every one of
    those choices is a forcing made silently, and the resulting number has no null.
    """
    rng = np.random.default_rng(6)
    with pytest.raises(engine.Unreadable):
        engine.associate(rng.normal(size=(80, 12)), rng.normal(size=(80, 8)), far=FAR)
    with pytest.raises(engine.Unreadable):
        engine.associate(rng.normal(size=(80, 12)), rng.normal(size=(40, 12)), far=FAR)
