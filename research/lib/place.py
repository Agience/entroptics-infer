"""Where a vertex sits, computed from the edges it carries and stored nowhere.

What a coordinate is here
---------------------------
A vertex's place is what it is connected to, and where the connection arrives — read off the
edge list when it is asked for. Nothing is stored on a vertex, nothing is trained, and no text is in
it: two things are close because the ontology puts them close, not because their glosses read alike.

The basis, and why it is the rest of the statement
----------------------------------------------------
One axis per (position, the rest of the statement). A concept at position 0 of "dog is_a
mammal" occupies the axis ``(0, ((1, is_a), (2, mammal)))`` — a single coordinate saying *I stand
first in something that relates by `is_a` to `mammal`*.

The obvious basis is one axis per ``(position, vertex)``, and it is weaker. It gives ``dog`` the
axes ``(1, is_a)`` and ``(2, mammal)`` separately, and a concept in many statements sums them all
together, so which relation went with which target is lost. Summing over statements is where the
pairing goes.

Binding the position to the rest of the statement keeps it, and keeps it without a vocabulary:
the pairs are whatever the store happens to contain, nothing is declared, a new relation arrives
carrying its own pairs, and the number of distinct pairs is bounded by the number of statements so
the basis stays finite without anyone choosing a size.

Measured, with the places rebuilt for each question so that nothing about the held-out statement
survives anywhere: the bound coordinate wins every question it disagrees with the unbound one about,
7-0, and is not separable from an axis per declared ``(relation, direction, target)`` — 3-2 over
the same 68 questions, all three placed by the shipped walk. `test_binding_the_coordinate.py` and `test_is_the_gap_real.py` run it.

**Position is what carries the asymmetry**, and it is a place rather than a name. ``dog is_a
mammal`` and ``mammal is_a dog`` are different statements because their parts arrive in a different
order; nothing here says position 1 means the relation, and no read branches on the number. How many
positions there are is read off the widest statement rather than written down.

A statement's own axis
------------------------
A statement is a vertex too, and it gets one axis of its own: reaching it is reaching that whole
larger concept rather than any of its parts. Its parts are its decomposition and both levels are
traversable, so the walk is free to arrive at either and the path decides which was right.

Why the neighbours' places are folded in
------------------------------------------
Immediate edges alone leave every concept nearly orthogonal to every other — ``salmon`` and ``shark``
would share only the statements they both appear in, which is none. The ontology says more than
that, and it is carried by their neighbours: both are in statements with ``fish``.

So a vertex's place is its own row plus its neighbours' rows, decayed by the branching it passed
through. A statement with three parts divides by three; a concept in twenty statements divides by
twenty, so a hub does not flood the places of everything hanging off it. That is the ontology's own
structure setting the decay, not a number written here.
"""
from __future__ import annotations

import numpy as np

import ontology
from ontology import Engine
from entroptics_infer.wrapper import _rest_of as rest_of  # noqa: E402,F401  (the shipped one)


def _engine(statements=None):
    """The shipped engine over ``statements``, or over the ontology when none are given.

    **Every place in ``research/`` is the shipped engine's place.** This module names the research
    helpers by the names the tests use, and each one is the engine's own method: there is one walk,
    in ``entroptics_infer.wrapper.Engine``, and nothing here computes a coordinate of its own.
    """
    return Engine(ontology.STATEMENTS if statements is None else statements)


def _without(drop, statements=None) -> list:
    """The statements named by position in ``drop``, in the form the engine holds out."""
    source = ontology.STATEMENTS if statements is None else statements
    return [source[i] for i in sorted(set(drop or ()))]


def kept(drop=None, statements=None) -> tuple:
    """``((statement id, parts), ...)`` — the store, minus the statements named in ``drop``.

    ``statements`` replaces the store itself, for asking what a *different* store would place
    where. Ids are positional either way, so a replacement store names its statements the same way
    the real one does.

    Every read here takes ``drop``, because holding a statement out of a model but not out of the
    coordinates is not holding it out. The engine holds out a statement by what it says, so a store
    that says one thing twice loses both copies when that thing is held out -- otherwise the copy
    left behind would state the answer.
    """
    return _engine(statements)._kept(_without(drop, statements))


def vertices(drop=None, statements=None) -> tuple:
    """Everything still in play: the concepts some surviving statement mentions, and those."""
    return _engine(statements).vertices(without=_without(drop, statements))


def arity(drop=None, statements=None) -> int:
    """How many positions a statement can have. Read off the widest one, never declared."""
    return max(len(parts) for _, parts in kept(drop, statements))


def contexts(drop=None, statements=None) -> tuple:
    """Every ``(position, rest-of-statement)`` any vertex occupies. Counted off the store."""
    return tuple(a for a in basis(drop, statements) if a[0] != "in")


def basis(drop=None, statements=None) -> tuple:
    """The contexts, plus one axis per statement for the statement itself."""
    return _engine(statements).basis(without=_without(drop, statements))


def adjacency(drop=None, statements=None):
    """``{vertex: [(other, axis), ...]}`` — the edges, both ways, each carrying where it arrives."""
    return _engine(statements)._adjacency(_without(drop, statements))


def neighbours(adj, vertex) -> tuple[str, ...]:
    """Everything one edge away. A concept's neighbours are statements, and the other way round."""
    return tuple(sorted({o for o, _ in adj[vertex]}))


def places(drop=None, statements=None):
    """``({vertex: vector}, basis)`` — the shipped engine's places, every concept and statement."""
    engine = _engine(statements)
    without = _without(drop, statements)
    return engine.places(without=without), engine.basis(without=without)


def walked(engine, *, without=(), only=None, aperture=True):
    """``(vectors, resolved)`` -- the shipped walk, stopped at its aperture or read to its end.

    Both arms are the shipped code: `Engine._resolved` (stopped at the aperture) or `Engine._settle`
    (every vertex the walk reaches) for which vertices a place keeps, and `Engine._row` for the place
    they make. Nothing here walks or adds on its own. ``resolved`` is how many vertices each
    place kept; ``only`` limits the starts.
    """
    axes = engine.basis(without=without)
    index = {a: i for i, a in enumerate(axes)}
    adj = engine._adjacency(without)
    wanted = engine.vertices(without=without)
    if only is not None:
        keep = set(only)
        wanted = tuple(v for v in wanted if v in keep)
    vectors, resolved = {}, {}
    for start in wanted:
        kept = engine._resolved(adj, start) if aperture else list(engine._settle(adj, start))
        vectors[start] = engine._row(adj, index, kept)
        resolved[start] = len(kept)
    return vectors, resolved


def read_to_its_end(engine, *, without=(), only=None):
    """``{vertex: vector}`` -- the shipped walk read to its end: the untruncated coordinate.

    `Engine.places` stops each walk at its own aperture. This is :func:`walked` with that stop not
    applied, so a check about the untruncated or dense coordinate measures exactly that and nothing
    else. It is a measurement of the walk, never a read anything ships.
    """
    return walked(engine, without=without, only=only, aperture=False)[0]


def nearest(vectors, target, *, count, among=None):
    """The vertices whose place is most like ``target``'s. For checking by eye, not for reads."""
    keys = [k for k in (among or vectors) if not np.array_equal(vectors[k], target)]

    def along(a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b)) / d if d > 0 else 0.0

    return sorted(((along(vectors[k], target), k) for k in keys), reverse=True)[:count]


if __name__ == "__main__":
    vectors, axes = places()
    concepts = ontology.concepts()
    print(f"{len(vectors)} vertices placed on {len(axes)} axes "
          f"({len(contexts())} contexts + {len(kept())} statements)")
    filled = int(np.mean([np.sum(v > 0) for v in vectors.values()]))
    print(f"a place touches {filled} axes on average\n")
    print("nearest CONCEPTS, to be judged by eye:")
    for probe in ("dog", "salmon", "hammer", "wheel", "oak", "is_a"):
        near = ", ".join(f"{k} {s:.2f}"
                         for s, k in nearest(vectors, vectors[probe], among=concepts, count=4))
        print(f"  {probe:<8} {near}")
