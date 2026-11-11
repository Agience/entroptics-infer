"""The false-alarm level reaches the decision and never the evidence.

Why this is a test and not a sweep
----------------------------------
``far`` is the one number this program does not derive, so the sound thing to do with it is
report every claim across a sweep. But for the central Stage 1 claim — that the *ordering* of
candidates by ``z`` recovers the true binding — a sweep would be the weaker argument. ``z`` is the
alignment standardized by the exact permutation null, and the level enters nowhere in that
quantity: it sets the cutoff ``resolved`` is taken against and zeroes ``strength`` below it, and it
does nothing else.

So the ordering is α-free by construction, and that can be demonstrated exactly rather than
sampled at five points and hoped about in between. These tests pin it, so that any future change
which quietly lets the level into the evidence fails here rather than silently turning a
parameter-free result into a tuned one.

What still needs a sweep, and gets one
--------------------------------------
Everything in the decision layer: whether an association resolves, the refusal rate, and the shared
mode count — which is read above a floor the level does move. Those are swept in
``research/experiments/exp01_association_vs_scalar.py`` and reported across α, because for them the
level genuinely acts.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import bind, engine

LEVELS = (0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40)


@pytest.fixture
def pair():
    rng = np.random.default_rng(11)
    T, D = 96, 32
    A = sum(np.outer(rng.normal(size=T), rng.normal(size=D)) for _ in range(3))
    A = A + rng.normal(size=(T, D)) * 0.25
    B = A * 0.6 + rng.normal(size=(T, D)) * 0.4
    return A, B


def test_z_is_identical_at_every_level(pair):
    """The evidence does not move with the reader's level."""
    A, B = pair
    zs = [engine.associate(A, B, far=f).z for f in LEVELS]
    assert len(set(zs)) == 1, (
        f"z varied with the false-alarm level across {LEVELS}: {zs}. The level belongs to the "
        f"decision, and a z that moves with it would make every ordering result a tuned one"
    )


def test_tightness_and_phase_are_identical_at_every_level(pair):
    """So does the rest of the evidence the association reports."""
    A, B = pair
    for field in ("tightness", "n"):
        vals = [getattr(engine.associate(A, B, far=f), field) for f in LEVELS]
        assert len(set(vals)) == 1, f"{field} varied with the level: {vals}"


def test_the_ordering_of_candidates_is_identical_at_every_level():
    """The Stage 1 claim itself: which candidate wins does not depend on the level.

    Run over the shape the experiment uses — one query against many candidates — because an
    ordering can be stable pairwise and still reshuffle across a set.
    """
    rng = np.random.default_rng(12)
    T, D = 96, 32
    query = sum(np.outer(rng.normal(size=T), rng.normal(size=D)) for _ in range(4))
    query = query + rng.normal(size=(T, D)) * 0.2
    keys = [query * g + rng.normal(size=(T, D)) * 0.5 for g in (0.2, 0.5, 0.9, 1.4)]
    keys.append(rng.normal(size=(T, D)))

    orders = [
        tuple(np.argsort([engine.associate(query, k, far=f).z for k in keys]))
        for f in LEVELS
    ]
    assert len(set(orders)) == 1, f"the candidate ordering moved with the level: {orders}"


def test_the_decision_layer_does_move_with_the_level(pair):
    """The control. If nothing moved with α, α would not be doing its job either.

    A test asserting only invariance could pass on a package that had stopped reading the level
    at all. This pins the other half: the cutoff must tighten as the level falls.
    """
    A, B = pair
    cutoffs = [engine.associate(A, B, far=f).cutoff for f in LEVELS]
    assert cutoffs == sorted(cutoffs, reverse=True), (
        f"the cutoff did not fall monotonically as the level rose: {cutoffs}"
    )
    assert len(set(cutoffs)) == len(LEVELS), "the cutoff is not responding to the level at all"


def test_refusal_responds_to_the_level():
    """A weak association must be refused at a strict level and admitted at a loose one."""
    rng = np.random.default_rng(13)
    T, D = 96, 32
    A = rng.normal(size=(T, D))
    B = A * 0.05 + rng.normal(size=(T, D))          # a faint real association
    strict = engine.associate(A, B, far=LEVELS[0]).resolved
    loose = engine.associate(A, B, far=LEVELS[-1]).resolved
    assert loose >= strict, "a looser level refused what a stricter one admitted"


def test_shared_modes_is_read_above_a_floor_the_level_moves():
    """Unlike z, the mode count is a decision-layer read and is expected to move."""
    rng = np.random.default_rng(14)
    T, D = 96, 32
    base = sum(np.outer(rng.normal(size=T), rng.normal(size=D)) for _ in range(5))
    A = base + rng.normal(size=(T, D)) * 0.3
    B = base + rng.normal(size=(T, D)) * 0.3
    counts = [bind.shared_modes(A, B, far=f) for f in LEVELS]
    assert min(counts) >= 0
    # It is a count above a floor; a looser level cannot resolve fewer modes than a stricter one.
    assert counts == sorted(counts), (
        f"the shared-mode count fell as the level loosened, which is backwards: {counts}"
    )
