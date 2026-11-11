"""The corpus screen: can a relation be supplied by example here at all?

`complete` needs two relation kinds present and telling apart. Two real corpora are refused by this
check, for reasons worth keeping apart: an inertial corpus carries one kind, and a word stream
carries four names that are one relation — each consistent with itself and identical to the
others.

So the screen has two jobs and both are tested here, each against a corpus built to fail it. A screen
that only ever says yes would pass a corpus that cannot carry the task, and the task would then
measure the corpus and report it as the read.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import Unreadable, relations

FAR = 0.05
SEED = 5
DRAWS = 60
T, D = 96, 8


def _corpus(rng, n, relation_maps):
    """Each name gets its own sources, carried to targets by that name's map.

    Sharing one set of sources across names would make two names applying the same map produce
    *identical target records*, and the null -- which draws pairs standing in no named relation --
    would then sample pairs of identical frames and be inflated by them. The names would read as
    absent for a reason that has nothing to do with what is being tested.
    """
    records, edges = {}, {name: [] for name in relation_maps}
    for name, M in relation_maps.items():
        for i in range(n):
            a = rng.normal(size=(T, D))
            records[f"{name}_src{i}"] = a
            records[f"{name}_dst{i}"] = a @ M
            edges[name].append((f"{name}_src{i}", f"{name}_dst{i}"))
    return records, edges


def _orthogonal(rng, d):
    return np.linalg.qr(rng.normal(size=(d, d)))[0]


def test_two_genuinely_different_relations_are_both_carried():
    """The positive case: two kinds, built to be different maps, and the screen sees both."""
    rng = np.random.default_rng(SEED)
    maps = {"alpha": _orthogonal(rng, D), "beta": _orthogonal(rng, D)}
    records, edges = _corpus(rng, 14, maps)

    screen = relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS)

    assert screen.count == 2, (
        f"two relations were built as different maps and the screen counted {screen.count}: "
        f"present {screen.present}, distinct {screen.distinct}, "
        f"separation {screen.separation}"
    )
    assert screen.exemplifiable, "two kinds present and distinct is exactly what complete needs"


def test_one_relation_wearing_two_names_is_refused():
    """The failure that text produced: several names, one relation.

    Both names are given the same map, so every edge stands in the same relation and there is
    nothing for "which relation is being asked" to distinguish. Both are present in the signal --
    that is what makes this the hard case -- and the screen has to refuse them anyway.
    """
    rng = np.random.default_rng(SEED)
    one = _orthogonal(rng, D)
    maps = {"alpha": one, "beta": one.copy()}
    records, edges = _corpus(rng, 14, maps)

    screen = relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS)

    assert all(screen.present.values()), (
        "both names should be present in the signal -- they are real relations. If they are not, "
        "this test is passing for the wrong reason and proves nothing about distinctness."
    )
    assert not screen.exemplifiable, (
        f"two names for one relation were accepted as exemplifiable. alignment\n"
        f"{np.round(screen.alignment, 3)}\ndistinct {screen.distinct}"
    )


def test_a_relation_that_is_not_in_the_signal_is_refused():
    """The failure the inertial corpus produced: a name with nothing behind it."""
    rng = np.random.default_rng(SEED)
    maps = {"alpha": _orthogonal(rng, D)}
    records, edges = _corpus(rng, 14, maps)
    # a second name whose edges pair records that stand in no relation at all
    keys = sorted(k for k in records if k.endswith(tuple(str(i) for i in range(10)))
                  and "_src" in k)
    edges["ghost"] = [(keys[i], keys[(i + 5) % len(keys)]) for i in range(len(keys))]

    screen = relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS)

    assert not screen.present["ghost"], (
        f"a relation with nothing behind it read as present, separation "
        f"{screen.separation['ghost']:.2f}"
    )
    assert not screen.exemplifiable, "one real kind and one empty name cannot exemplify anything"


def test_the_diagonal_is_a_sample_against_itself_and_not_one_by_construction():
    """The scale the off-diagonal is read against has to be earned, not assumed.

    A name is compared against an independent half of its own edges. If the diagonal were a
    vector against itself it would be 1.000 whatever the corpus did, and every name would look
    distinct from every other.
    """
    rng = np.random.default_rng(SEED)
    maps = {"alpha": _orthogonal(rng, D), "beta": _orthogonal(rng, D)}
    records, edges = _corpus(rng, 14, maps)

    screen = relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS)

    diag = np.diag(screen.alignment)
    assert np.all(diag < 0.999), (
        f"the diagonal reads {np.round(diag, 4)} -- at 1.000 it is a vector against itself and "
        f"carries no information about whether the relation is consistent"
    )


def test_naming_nothing_raises_rather_than_answering():
    rng = np.random.default_rng(SEED)
    records, _ = _corpus(rng, 6, {"alpha": _orthogonal(rng, D)})
    with pytest.raises(Unreadable):
        relations.carries_kinds(records, {}, far=FAR, seed=SEED, draws=DRAWS)


@pytest.mark.parametrize("n_names", [2, 3])
def test_every_name_needs_edges_to_be_read_from(n_names):
    rng = np.random.default_rng(SEED)
    maps = {f"r{i}": _orthogonal(rng, D) for i in range(n_names)}
    records, edges = _corpus(rng, 10, maps)
    edges["empty"] = []
    with pytest.raises(Unreadable):
        relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS)


def test_an_explicit_null_is_accepted_and_agrees_with_a_drawn_one():
    """The null can be supplied, and where both are valid they say the same thing.

    Unless told otherwise the null is drawn from pairs in no named set, which assumes the named sets
    are complete. A caller passing a sample of a larger family leaves the rest of that family in
    the null. Measured here, that contamination is not on its own enough to change the verdict --
    a third of each relation's edges named, the rest left in the null, and both relations are still
    found. So the parameter exists to make the assumption explicit and controllable, not because the
    drawn null fails at this contamination.
    """
    rng = np.random.default_rng(SEED)
    maps = {"alpha": _orthogonal(rng, D), "beta": _orthogonal(rng, D)}
    records, edges = _corpus(rng, 24, maps)
    sampled = {name: es[: len(es) // 3] for name, es in edges.items()}

    drawn = relations.carries_kinds(records, sampled, far=FAR, seed=SEED, draws=DRAWS)

    # pairs that genuinely stand in no named relation: one name's source against the other's target
    clean = [(f"alpha_src{i}", f"beta_dst{(i + 7) % 24}") for i in range(24)]
    told = relations.carries_kinds(records, sampled, far=FAR, seed=SEED, draws=DRAWS,
                                   unrelated=clean)

    assert told.exemplifiable, (
        f"with the null supplied the sampled relations should be found: present {told.present}, "
        f"distinct {told.distinct}, separation {told.separation}"
    )
    assert drawn.exemplifiable == told.exemplifiable, (
        f"the two nulls disagree on a corpus where both are valid: drawn {drawn.exemplifiable}, "
        f"supplied {told.exemplifiable}"
    )


def test_an_empty_explicit_null_raises_rather_than_answering():
    rng = np.random.default_rng(SEED)
    maps = {"alpha": _orthogonal(rng, D), "beta": _orthogonal(rng, D)}
    records, edges = _corpus(rng, 10, maps)
    with pytest.raises(Unreadable):
        relations.carries_kinds(records, edges, far=FAR, seed=SEED, draws=DRAWS, unrelated=[])


def test_the_lattice_carries_a_relation_one_exemplar_cannot_name():
    """The screen's two questions, and the case where they part company.

    `separation` divides by the spread of a single unrelated pair, so it asks whether one edge
    is distinguishable — the right question for `complete`, which is handed one exemplar pair.
    `carried` asks whether the relation's *mean* stands above the mean of as many unrelated pairs,
    which is the question `Lattice.operator` needs because it aggregates every edge.

    A relation carried faintly enough that no single edge reads is the case that separates them,
    and it is not a corner case: a road network reads 0.45 to 1.42 on `separation` and 7.9 to 32.6
    on `carried`. Refusing that corpus on the first number is refusing it for failing a test its
    reader never takes.
    """
    rng = np.random.default_rng(SEED)
    keep = _orthogonal(rng, D)
    records, edges = {}, []
    for i in range(120):
        src = rng.normal(size=(T, D))
        records[f"s{i}"] = src
        # a twelfth of the destination is the relation and the rest is unrelated structure
        carry = 0.12
        records[f"t{i}"] = (carry * (src @ keep)
                            + np.sqrt(1 - carry ** 2) * rng.normal(size=(T, D)))
        edges.append((f"s{i}", f"t{i}"))

    screen = relations.carries_kinds(records, {"faint": edges}, far=FAR, seed=SEED, draws=DRAWS)

    assert not screen.present["faint"], (
        "the per-pair screen read a relation this faint as present, so this corpus does not "
        f"separate the two questions and the test proves nothing: separation "
        f"{screen.separation['faint']:.2f}"
    )
    assert screen.carries["faint"], (
        f"the lattice holds this relation on {len(edges)} edges and the aggregate reading missed "
        f"it: carried {screen.carried['faint']:.1f}"
    )


def test_a_relation_that_is_not_there_is_refused_however_many_edges_it_has():
    """The control that makes the reading above worth anything.

    `carried` grows with edge count by construction — that is the whole point of it — so a reading
    that grew on anything would be counting edges rather than finding a relation. Here the
    destinations stand in no relation to their sources at all, at the same edge count.
    """
    rng = np.random.default_rng(SEED)
    records, edges = {}, []
    for i in range(120):
        records[f"s{i}"] = rng.normal(size=(T, D))
        records[f"t{i}"] = rng.normal(size=(T, D))          # an independent draw: no relation
        edges.append((f"s{i}", f"t{i}"))

    screen = relations.carries_kinds(records, {"nothing": edges}, far=FAR, seed=SEED, draws=DRAWS)

    assert not screen.carries["nothing"], (
        f"a relation that is not there was read as carried on {len(edges)} edges: "
        f"carried {screen.carried['nothing']:.1f}"
    )
