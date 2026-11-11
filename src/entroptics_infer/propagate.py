"""Propagation: how far a signal reaches, measured rather than capped.

Depth is not a layer count
--------------------------
A transformer has as many rounds of message passing as it has layers, and a graph learner has as
many hops as its architecture declares. Both numbers are decided in advance, on other data, and are
the same kind of object as a threshold: a constant standing where a measurement belongs.

Here a walk continues while the signal still clears the floor and stops when it does not. Depth is
therefore a property of the query, different for different origins in the same corpus, and it is
read off rather than configured. Nothing in this module accepts a maximum depth, a hop limit, or a
beam width, and there is nowhere to put one.

Why it terminates
-----------------
Two reasons, and both are needed.

The graph is finite and no vertex is visited twice, so the walk cannot run forever even in the
worst case. That alone is not the interesting guarantee — it would let a walk reach the whole corpus
before stopping, which is what a hop cap exists to prevent and what removing the cap is supposed to
be safe without.

The interesting guarantee is that a finite aperture band-limits, so the correlation it can carry
decays at a positive rate — the reading of Yang–Mills in `entroptics-mass-gap`, where the mass gap
is exactly the statement that ``‖C(τ)‖ ≤ M e^{−Δτ}`` with ``Δ > 0``. A signal that decays
geometrically against a floor that does not crosses it after finitely many steps, and where it
crosses is the horizon. The walk stops there because there is nothing left above the floor, not
because someone counted.

What is measured, and what it means
-----------------------------------
Each reached vertex is scored against the origin, not against whatever it was reached through.
Scoring against the parent would let a chain of individually-resolvable steps carry a signal
arbitrarily far while the origin's own association with the far end was long gone — which is how a
propagation acquires reach it cannot justify. The question at every vertex is the same one: *does
this record still resolve against the record the walk started from?*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .engine import associate

__all__ = ["Reach", "reach", "horizon"]


@dataclass(frozen=True)
class Reach:
    """What a signal reached, and how far.

    ``origin``    the vertex the walk started from.
    ``depth``     the measured horizon: the greatest depth at which anything still resolved.
    ``reached``   every vertex that resolved against the origin, mapped to the depth it was
                  reached at.
    ``frontier``  the vertices examined and not reached — where the signal fell below the floor.
                  Kept because a walk's edge is a measurement too, and reporting only what was
                  reached hides how much of the corpus was ruled out.
    ``examined``  how many vertices were scored in total.
    """

    origin: str
    depth: int
    reached: dict = field(default_factory=dict)
    frontier: tuple = ()
    examined: int = 0


def reach(records, neighbours: Callable[[str], Iterable[str]], origin: str, *, far: float) -> Reach:
    """Everything the origin's signal reaches before it falls below the floor.

    ``neighbours`` is a callable from a vertex to its adjacent vertices, so the same read works over
    a lattice, an in-memory graph, or anything else that can answer that question. The walk holds no
    opinion about where the edges come from.

    There is no depth argument, and adding one would be the whole point of this module undone.
    The walk expands only from vertices that resolved, so a region the signal cannot reach is never
    explored — the horizon prunes the search as a side effect of being a measurement rather than a
    budget.
    """
    seen = {origin}
    reached: dict = {origin: 0}
    frontier: list = []
    examined = 0
    live = [origin]
    depth = 0

    while live:
        depth += 1
        nxt = []
        for v in live:
            for w in neighbours(v):
                if w in seen:
                    continue
                seen.add(w)
                examined += 1
                if associate(records[origin], records[w], far=far).resolved:
                    reached[w] = depth
                    nxt.append(w)
                else:
                    frontier.append(w)
        live = nxt

    return Reach(
        origin=origin,
        depth=max(reached.values()) if reached else 0,
        reached=reached,
        frontier=tuple(frontier),
        examined=examined,
    )


def horizon(records, neighbours: Callable[[str], Iterable[str]], origin: str, *, far: float) -> int:
    """The measured depth alone — how many hops the origin's signal survives.

    This is the number a layer count stands in for, and the difference is that this one is a
    property of the corpus and the query rather than of the architecture. Two origins in one corpus
    routinely return different values, which is the behaviour a fixed depth cannot have.
    """
    return reach(records, neighbours, origin, far=far).depth
