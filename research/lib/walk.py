"""The walk that carries an edge weight and can stop at a floor, and the bases it is read on.

What this is
--------------
**Not the shipped walk, and named so.** `entroptics_infer.wrapper.Engine` places a vertex by a walk
that carries no edge weight and stops at its own aperture (`Engine._resolved`, `Engine._row`). Asking
what an edge weight, or a floor at the uniform share, does to a walk needs a walk that has both.
This is it, and it is the only one in `research/`.

:func:`spread` is the walk itself: from each start it carries ``1.0``, adds ``carried * w`` into
every axis an edge touches, and passes ``carried * w / sum(w)`` on while that beats the best share
already found for the vertex and the floor. ``floor=0.0`` is no floor. :func:`carried_places` builds
the adjacency for one of three bases and walks it.

The bases
-----------
``typed``   one axis per ``(label, direction, concept)``; concepts only.
``vertex``  one axis per ``(position, vertex)``: the unbound basis.
``bound``   the shipped engine's own basis and adjacency.
"""
from __future__ import annotations

import numpy as np

from ontology import Engine                                    # the shipped engine


def triples(statements):
    """The three-part statements, as ``(source, relation, destination)``."""
    return [(a, r, b) for a, r, b in (p for p in statements if len(p) == 3)]


def typed_basis(statements):
    """``(starts, adjacency, axes)`` -- one axis per ``(label, direction, concept)``, concepts only."""
    labels = sorted({r for _, r, _ in triples(statements)})
    concepts = sorted({c for a, r, b in triples(statements) for c in (a, b)})
    axes = [(lab, d, c) for lab in labels for d in ("out", "in") for c in concepts]
    adj: dict = {c: [] for c in concepts}
    for a, r, b in triples(statements):
        adj[a].append((b, (r, "out", b)))
        adj[b].append((a, (r, "in", a)))
    return concepts, adj, axes


def vertex_basis(statements):
    """``(starts, adjacency, axes)`` -- one axis per ``(position, vertex)``: the unbound basis."""
    verts = sorted({c for parts in statements for c in parts})
    sids = [f"v{i:03d}" for i in range(len(statements))]
    everything = verts + sids
    arity = max(len(p) for p in statements)
    axes = [(p, v) for p in range(arity) for v in everything]
    adj: dict = {v: [] for v in everything}
    for i, parts in enumerate(statements):
        for position, c in enumerate(parts):
            adj[sids[i]].append((c, (position, c)))
            adj[c].append((sids[i], (position, sids[i])))
    return everything, adj, axes


def spread(starts, adj, axes, *, floor=0.0):
    """``{start: place}`` -- the carried walk over ``adj``, ``{vertex: [(axis, other, w), ...]}``.

    ``floor`` is the share below which a step is not passed on; ``0.0`` walks until no step beats
    the best share already found.
    """
    index = {a: i for i, a in enumerate(axes)}
    out = {}
    for start in starts:
        v = np.zeros(len(axes))
        seen = {start: 1.0}
        frontier = [(start, 1.0)]
        while frontier:
            nxt = []
            for here, carried in frontier:
                attached = adj[here]
                tot = sum(w for *_, w in attached) or 1.0
                for axis, other, w in attached:
                    v[index[axis]] += carried * w
                    step = carried * w / tot
                    if step > seen.get(other, 0.0) and step > floor:
                        seen[other] = step
                        nxt.append((other, step))
            frontier = nxt
        out[start] = v
    return out


def carried_places(basis, statements, *, weighted, floored):
    """``(places, axes)`` on the carried walk, over ``basis`` -- ``"typed"``, ``"vertex"`` or ``"bound"``.

    ``weighted`` weighs each edge by the surprise of arriving at its concept end, divided through by
    the largest; ``weighted=False`` gives every edge weight one, so a study compares the weight
    against no weight on the same walk. ``floored`` stops the walk at the uniform share
    ``1/|starts|``.
    """
    if basis == "typed":
        concepts, _plain, axes = typed_basis(statements)
        arrivals: dict = {}
        for a, r, b in triples(statements):
            arrivals[(r, b)] = arrivals.get((r, b), 0) + 1
        total = sum(arrivals.values()) or 1
        adj = {c: [] for c in concepts}
        for a, r, b in triples(statements):
            w = -np.log(max(arrivals[(r, b)] / total, 1 / total)) if weighted else 1.0
            adj[a].append(((r, "out", b), b, w))
            adj[b].append(((r, "in", a), a, w))
        starts = concepts
    else:
        if basis == "vertex":
            starts, plain, axes = vertex_basis(statements)
            name = {f"v{i:03d}": i for i in range(len(statements))}
        else:
            engine = Engine(statements)
            starts, plain, axes = engine.vertices(), engine._adjacency(()), list(engine.basis())
            name = {engine.statement_id(i): i for i in range(len(statements))}
        degree: dict = {}
        for i, parts in enumerate(statements):
            for c in parts:
                degree[c] = degree.get(c, 0) + 1
        # each part is counted at both its ends, the concept and the statement it is in
        total = sum(degree.values()) * 2 or 1
        adj = {}
        for v, ends in plain.items():
            # an edge is weighted by the surprise of arriving at the concept on its statement end
            adj[v] = [(axis, o, (-np.log(max(degree[o if v in name else v] / total, 1 / total))
                                 if weighted else 1.0)) for o, axis in ends]
    if weighted:
        top = max((w for ends in adj.values() for *_, w in ends), default=1.0)
        adj = {k: [(a, o, w / top) for a, o, w in v] for k, v in adj.items()}
    floor = 1.0 / len(starts) if floored else 0.0
    return spread(starts, adj, axes, floor=floor), axes
