"""Propagation: depth is measured, and there is nowhere to put a cap.

These pin properties that must hold on any input, not the sweep the experiment measures. The most
important one is negative: no function here takes a depth, so the walk cannot be configured to stop
early or told to keep going.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from entroptics_infer import propagate

FAR = 0.05
T, D, CH = 96, 32, 4


def _structured(rng):
    A = rng.normal(size=(CH, D))
    Q, _ = np.linalg.qr(A.T)
    dirs = Q.T[:CH]
    X = np.zeros((T, D))
    for k in range(CH):
        X += np.outer(rng.normal(size=T), dirs[k]) / np.sqrt(CH)
    return X


def _chain(seed, length, transmission, noise=0.05):
    rng = np.random.default_rng(seed)
    carried = _structured(rng)
    keep = transmission
    fresh = float(np.sqrt(max(0.0, 1.0 - keep * keep)))
    order, records = [], {}
    for t in range(length):
        if t:
            carried = keep * carried + fresh * _structured(rng)
        name = f"c{t}"
        order.append(name)
        records[name] = carried + rng.normal(size=(T, D)) * noise
    return records, tuple(order)


def _neighbours(order):
    index = {v: i for i, v in enumerate(order)}

    def nbrs(v):
        i = index[v]
        return [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]

    return nbrs


def test_no_function_here_accepts_a_depth():
    """The property the module exists for, asserted rather than assumed.

    A maximum depth, hop limit, beam width or budget would undo the whole point, and it is the kind
    of parameter that gets added back for a good local reason. This fails if one appears.
    """
    banned = {"depth", "max_depth", "hops", "max_hops", "cap", "limit", "budget", "beam",
              "radius", "steps", "max_steps", "rounds", "layers"}
    for name, fn in vars(propagate).items():
        if not inspect.isfunction(fn) or fn.__module__ != propagate.__name__:
            continue
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), f"{name} accepts {sorted(params & banned)}"


def test_nothing_transmitted_reaches_nothing():
    """The control. A walk that reaches into a chain carrying nothing is following noise."""
    records, order = _chain(0, length=12, transmission=0.0)
    assert propagate.horizon(records, _neighbours(order), order[0], far=FAR) == 0


def test_the_horizon_grows_with_what_is_transmitted():
    """More of the record surviving each step means the origin is recognizable further along."""
    depths = []
    for a in (0.3, 0.5, 0.7, 0.9):
        records, order = _chain(1, length=20, transmission=a)
        depths.append(propagate.horizon(records, _neighbours(order), order[0], far=FAR))
    assert depths == sorted(depths), f"horizon did not grow with transmission: {depths}"
    assert depths[0] < depths[-1]


def test_the_horizon_is_a_property_of_the_query():
    """Different origins in one corpus return different depths.

    A fixed layer count cannot do this — it returns the same number for every query — and this is
    the difference that makes depth worth measuring rather than configuring.
    """
    records, order = _chain(2, length=24, transmission=0.7)
    nbrs = _neighbours(order)
    depths = {v: propagate.horizon(records, nbrs, v, far=FAR) for v in order[::4]}
    assert len(set(depths.values())) > 1, f"every origin returned the same depth: {depths}"


def test_the_walk_terminates_on_a_fully_connected_graph():
    """Termination does not rest on the graph being sparse.

    With every vertex adjacent to every other, a walk that did not stop on its own would reach the
    whole corpus in one step and then keep expanding. The floor is what ends it.
    """
    records, order = _chain(3, length=14, transmission=0.3)

    def all_others(v):
        return [w for w in order if w != v]

    r = propagate.reach(records, all_others, order[0], far=FAR)
    assert r.depth >= 0
    assert len(r.reached) <= len(order)


def test_what_was_ruled_out_is_reported():
    """A walk's edge is a measurement too, and reporting only what was reached hides it."""
    records, order = _chain(4, length=20, transmission=0.5)
    r = propagate.reach(records, _neighbours(order), order[0], far=FAR)
    assert r.examined >= len(r.reached) - 1
    assert len(r.frontier) == r.examined - (len(r.reached) - 1)


def test_the_origin_is_reached_at_depth_zero():
    """The walk starts where it starts, and that is not a hop."""
    records, order = _chain(5, length=8, transmission=0.9)
    r = propagate.reach(records, _neighbours(order), order[0], far=FAR)
    assert r.reached[order[0]] == 0
    assert r.origin == order[0]


def test_every_reached_vertex_resolves_against_the_origin():
    """Not against its parent. A chain of resolvable steps must not carry reach it cannot justify."""
    from entroptics_infer import engine

    records, order = _chain(6, length=20, transmission=0.6)
    r = propagate.reach(records, _neighbours(order), order[0], far=FAR)
    for v, depth in r.reached.items():
        if depth == 0:
            continue
        assert engine.associate(records[order[0]], records[v], far=FAR).resolved, (
            f"{v} was reached at depth {depth} but does not resolve against the origin"
        )


def test_an_unknown_origin_is_an_error_not_an_empty_reach():
    """Asking about a vertex the corpus does not have is a mistake, not a finding of nothing."""
    records, order = _chain(7, length=6, transmission=0.8)
    with pytest.raises(KeyError):
        propagate.reach(records, _neighbours(order), "not-a-vertex", far=FAR)
