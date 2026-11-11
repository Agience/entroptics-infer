"""The store you can add to and ask: what it must do, and what it must refuse.

`Lattice` is the reads in the shape a running system needs — a store things arrive into and a
question about one of them, rather than a function handed an exemplar. Five things have to hold, and
each is tested against the case that would break it.

1. **It answers.** With the candidates the source actually stands among, it is right.
2. **It learns.** A relation arriving is carried, and can be asked about immediately.
3. **It does not disturb.** An answer already given is the *same* answer afterwards, which
   `Infer/Arrival.lean` proves and this checks is true of the code as well as the model.
4. **A relation is derived, not stored.** Withdrawing an edge changes the operator, because there
   was never a fitted thing to leave behind.
5. **It refuses.** No vocabulary, no relation without edges, no record that is not there.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import Lattice, Unreadable

FAR = 0.05
SEED = 5
T, D = 96, 32


def _structured(rng, facets=3, noise=0.1):
    loadings = np.linalg.qr(rng.normal(size=(D, facets)))[0][:, :facets]
    return rng.normal(size=(T, facets)) @ loadings.T + noise * rng.normal(size=(T, D))


def _orthogonal(rng):
    return np.linalg.qr(rng.normal(size=(D, D)))[0]


def _lattice(n=12, relations=("alpha", "beta")):
    rng = np.random.default_rng(SEED)
    maps = {name: _orthogonal(rng) for name in ("alpha", "beta", "gamma")}
    lat = Lattice(far=FAR)
    for i in range(n):
        src = _structured(rng)
        lat.add(f"s{i}", src)
        for name in relations:
            lat.add(f"{name}{i}", src @ maps[name])
    for i in range(n):
        for name in relations:
            lat.relate(f"s{i}", f"{name}{i}", name)
    return lat, maps, n


def test_it_answers_among_the_candidates_a_source_stands_among():
    lat, _, n = _lattice()
    right = sum(lat.answer(f"s{i}", "alpha", candidates=[f"alpha{i}", f"beta{i}"]) == f"alpha{i}"
                for i in range(n))
    assert right == n, f"{right}/{n} — the read cannot tell its own relations apart"


def test_it_carries_what_it_holds_and_not_what_it_does_not():
    lat, _, _ = _lattice()
    verdict = lat.carries(seed=SEED, draws=250)
    assert set(verdict.carrying) == {"alpha", "beta"}
    assert "gamma" not in verdict, "a relation with no edges is being counted"


def test_a_relation_arriving_is_carried_and_can_be_asked_about():
    lat, maps, n = _lattice()
    assert "gamma" not in lat.carries(seed=SEED, draws=250)
    for i in range(n):
        lat.add(f"gamma{i}", lat.records[f"s{i}"] @ maps["gamma"])
        lat.relate(f"s{i}", f"gamma{i}", "gamma")
    assert "gamma" in lat.carries(seed=SEED, draws=250), "the arrival was not learned"
    got = lat.answer("s4", "gamma", candidates=["alpha4", "beta4", "gamma4"])
    assert got == "gamma4", f"asked about the newcomer and got {got!r}"


def test_an_arrival_does_not_move_an_answer_already_given():
    """`Infer/Arrival.lean`'s `answer_arrive`, checked against the code rather than the model.

    Not "close" and not "usually" — the same question with the same candidates must return the
    same name, because a completion reads only the records it was handed.
    """
    lat, maps, n = _lattice()
    asked = [(f"s{i}", [f"alpha{i}", f"beta{i}"]) for i in range(n)]
    before = {s: lat.answer(s, "alpha", candidates=c) for s, c in asked}
    for i in range(n):
        lat.add(f"gamma{i}", lat.records[f"s{i}"] @ maps["gamma"])
        lat.relate(f"s{i}", f"gamma{i}", "gamma")
    after = {s: lat.answer(s, "alpha", candidates=c) for s, c in asked}
    moved = [s for s in before if before[s] != after[s]]
    assert not moved, f"{len(moved)} answers moved when a relation arrived: {moved[:3]}"


def test_a_relation_is_derived_from_its_edges_rather_than_stored():
    """Withdraw an edge and the operator changes, because nothing was ever kept."""
    lat, _, _ = _lattice()
    before = lat.operator("alpha").copy()
    lat.forget("s0", "alpha0", "alpha")
    after = lat.operator("alpha")
    assert not np.array_equal(before, after), (
        "withdrawing an edge left the relation untouched, so it is being held rather than derived"
    )
    # and withdrawing the last edge leaves nothing to derive at all
    for i in range(1, 12):
        lat.forget(f"s{i}", f"alpha{i}", "alpha")
    with pytest.raises(Unreadable):
        lat.operator("alpha")


def test_it_refuses_what_it_should():
    lat, _, _ = _lattice()
    with pytest.raises(Unreadable):
        lat.operator("never-named")            # a relation with nothing behind it
    with pytest.raises(Unreadable):
        lat.ask("not-a-record", "alpha")       # a source that is not here
    with pytest.raises(Unreadable):
        lat.relate("s0", "not-a-record", "alpha")   # an edge to nothing
    # a name never seen before needs no registration: the store holds no vocabulary
    lat.relate("s0", "beta1", "a-relation-nobody-declared")
    assert "a-relation-nobody-declared" in lat.edges


def test_it_holds_a_relation_it_could_not_exemplify():
    """The two verdicts, and the gap between them where real corpora live.

    `carrying` asks whether a single exemplar pair could name the relation, which is what
    `complete` needs. `holding` asks whether the lattice has it at all, which is what
    `Lattice.operator` needs because it derives the relation from every edge citing it.

    A relation carried faintly is held and not exemplifiable, and that is not a corner case: a road
    network reads 0.45 to 1.42 on separation, outside `carrying` in all 30 cells swept, and 7.9 to
    32.6 on strength. A store reporting only `carrying` would say it holds nothing while holding it.
    """
    rng = np.random.default_rng(SEED)
    keep = _orthogonal(rng)
    lat = Lattice(far=FAR)
    # Measured on these records at 120 edges: carry 0.0 is neither held nor carried (strength 0.3),
    # 0.12 is neither (2.6), 0.3 is held and not carried (1.70 / 19.2), 0.4 is both (2.76 / 30.2).
    # 0.3 sits inside that band with both assertions far from their edges. The test asserts both
    # sides, so a carry too faint or too strong fails it rather than passing quietly.
    n, carry = 120, 0.3
    for i in range(n):
        src = _structured(rng)
        lat.add(f"s{i}", src)
        lat.add(f"t{i}", carry * (src @ keep) + np.sqrt(1 - carry ** 2) * _structured(rng))
    for i in range(n):
        lat.relate(f"s{i}", f"t{i}", "faint")

    verdict = lat.carries(seed=SEED, draws=250)

    assert "faint" in verdict, (
        f"the lattice holds this relation on {n} edges and said it did not: "
        f"strength {verdict.strength['faint']:.1f}"
    )
    assert "faint" not in verdict.carrying, (
        "this relation reads from a single pair, so it does not separate the two verdicts and the "
        f"test proves nothing: separation {verdict.separation['faint']:.2f}"
    )


def test_a_relation_that_is_not_there_is_not_held():
    """The control: `holding` grows with edges by design, so it must refuse an empty relation."""
    rng = np.random.default_rng(SEED)
    lat = Lattice(far=FAR)
    n = 120
    for i in range(n):
        lat.add(f"s{i}", _structured(rng))
        lat.add(f"t{i}", _structured(rng))          # an independent draw: no relation at all
    for i in range(n):
        lat.relate(f"s{i}", f"t{i}", "nothing")

    verdict = lat.carries(seed=SEED, draws=250)
    assert "nothing" not in verdict, (
        f"a relation that is not there was held on {n} edges: "
        f"strength {verdict.strength['nothing']:.1f}"
    )


def test_more_edges_of_the_asked_relation_change_its_operator():
    """The scope of the arrival guarantee, pinned from the other side.

    `test_an_arrival_does_not_move_an_answer_already_given` adds a *different* relation, and nothing
    moves — which is what `Infer/Arrival.lean` proves and what a store must do. This pins the
    complement: `Lattice.answer` derives its operator from every edge citing the relation, so
    more edges of *that* relation change the read. Measured on a conversation, an unrelated arrival
    of 1,492 records and 840 edges moved 0 of 60 answers; growing the asked relation's own edges
    27-fold moved 37 and more than doubled its strength.

    That second behaviour is the learning, and a test that let it be silently removed would let the
    store stop learning while every other test stayed green.
    """
    lat, maps, n = _lattice()
    half = n // 2
    lean = Lattice(far=FAR)
    for name, rec in lat.records.items():
        lean.add(name, rec)
    for i in range(half):
        lean.relate(f"s{i}", f"alpha{i}", "alpha")
    before = lean.operator("alpha").copy()

    # an arrival that does not touch alpha's edges: its operator may not move at all
    for i in range(n):
        lean.relate(f"s{i}", f"beta{i}", "beta")
    assert np.array_equal(lean.operator("alpha"), before), (
        "an arrival of a different relation changed alpha's operator, so an answer could move "
        "where `Infer/Arrival.lean` says it cannot"
    )

    # more edges of alpha itself: derived from more, so it must move
    for i in range(half, n):
        lean.relate(f"s{i}", f"alpha{i}", "alpha")
    after = lean.operator("alpha")
    assert not np.array_equal(after, before), (
        "the relation's own edges grew and its operator did not change, so the store is not "
        "deriving the relation from its edges and cannot be learning from them"
    )
