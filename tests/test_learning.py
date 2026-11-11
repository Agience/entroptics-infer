"""Learning: the corpus grows, and the properties that follow from having no weights.

The distinctive claims here are the ones a trained system cannot make. Order independence and
locality are exact rather than approximate, because a read is a pure function of the records it is
given — so these are asserted with `==`, not with a tolerance.
"""
from __future__ import annotations

import numpy as np

from entroptics_infer import bind, engine, relations

FAR = 0.05
T, D = 96, 24


def _world(rng, channels=8):
    """One set of latent channels and directions, shared by every record drawn from it.

    Records must share *latents*, not merely directions: two frames built on the same directions
    with independent time-courses have no association at all, which is the correct answer and not
    the situation these tests mean to set up.
    """
    A = rng.normal(size=(channels, D))
    Q, _ = np.linalg.qr(A.T)
    return rng.normal(size=(channels, T)), Q.T[:channels]


def _facets(rng, world, which, noise=0.10):
    lat, dirs = world
    X = np.zeros((T, D))
    for f in which:
        X += np.outer(lat[f], dirs[f]) / np.sqrt(len(which))
    return X + rng.normal(size=(T, D)) * noise


def _relational(rng, n_relations, per_relation, noise=0.05):
    A = rng.normal(size=(2 * n_relations, D))
    Q, _ = np.linalg.qr(A.T)
    roles = Q.T[: 2 * n_relations].reshape(n_relations, 2, D)
    records, edges = {}, []
    for k in range(n_relations):
        for e in range(per_relation):
            s = rng.normal(size=T)
            a, b = f"s{k}_{e}", f"t{k}_{e}"
            records[a] = np.outer(s, roles[k, 0]) + rng.normal(size=(T, D)) * noise
            records[b] = np.outer(s, roles[k, 1]) + rng.normal(size=(T, D)) * noise
            edges.append((a, b))
    return records, tuple(edges)


def test_a_new_kind_raises_the_count_and_more_of_an_old_one_does_not():
    """Learning a relation, against merely seeing more of one already known.

    Both corpora grow by the same number of edges. Only one of them carries structure that was not
    there before, and only that one should change the count. A counter that rises either way is
    measuring volume.
    """
    rng = np.random.default_rng(0)
    base_records, base_edges = _relational(rng, 4, per_relation=10)
    base = relations.relation_modes(base_records, base_edges, far=FAR)

    rng = np.random.default_rng(0)
    new_records, new_edges = _relational(rng, 6, per_relation=10)
    with_new = relations.relation_modes(new_records, new_edges, far=FAR)

    rng = np.random.default_rng(0)
    more_records, more_edges = _relational(rng, 4, per_relation=15)
    with_more = relations.relation_modes(more_records, more_edges, far=FAR)

    assert with_new > base, "new relational structure did not raise the count"
    assert with_more == base, (
        f"more edges of kinds already present raised the count {base} -> {with_more}; the count is "
        f"tracking volume rather than structure"
    )


def test_candidate_order_does_not_change_the_answer():
    """Exact, not approximate. A pure function of the records cannot depend on their order."""
    rng = np.random.default_rng(1)
    w = _world(rng)
    query = _facets(rng, w, (0, 1, 2))
    keys = [_facets(rng, w, (0, 1, 2)), _facets(rng, w, (3,)),
            _facets(rng, w, (4, 5)), _facets(rng, w, (6,))]

    first = bind.decide(query, keys, far=FAR)
    order = [2, 0, 3, 1]
    second = bind.decide(query, [keys[i] for i in order], far=FAR)

    assert first.why == second.why
    if first.key is None:
        assert second.key is None
    else:
        assert order[second.key] == first.key


def test_an_unrelated_arrival_leaves_an_answer_bit_identical():
    """The property a trained system cannot have.

    Fine-tuning moves every weight, so answers to questions nobody asked about change. Here an
    answer depends only on the records it was computed from, and the scores of the original
    candidates must be unchanged to the last bit — not close, identical.
    """
    rng = np.random.default_rng(2)
    w = _world(rng)
    query = _facets(rng, w, (0, 1))
    keys = [_facets(rng, w, (0, 1)), _facets(rng, w, (2,)), _facets(rng, w, (3, 4))]

    before = bind.decide(query, keys, far=FAR)
    newcomers = [rng.normal(size=(T, D)) for _ in range(4)]
    after = bind.decide(query, keys + newcomers, far=FAR)

    assert before.key == after.key and before.why == after.why
    for b0, b1 in zip(before.bindings, after.bindings[: len(keys)]):
        assert b0.association.z == b1.association.z
        assert b0.modes == b1.modes


def test_coverage_rises_as_observations_arrive():
    """Evidence arriving changes the answer, with no fitting step anywhere in between.

    The property is about the trend, not about one pair of row counts: how many observations it
    takes to answer depends on how hard the particular question is, so a test pinned to a single
    threshold would be pinning the fixture rather than the behaviour. What must hold is that the
    system answers *more often* with more evidence, and is not wrong when it does.
    """
    scarce_answered = plenty_answered = wrong = 0
    draws = 24
    for d in range(draws):
        rng = np.random.default_rng(100 + d)
        w = _world(rng)
        query = _facets(rng, w, (0, 1, 2), noise=0.4)
        keys = [_facets(rng, w, (0, 1, 2), noise=0.4), _facets(rng, w, (3,), noise=0.4),
                _facets(rng, w, (4, 5), noise=0.4), _facets(rng, w, (6,), noise=0.4)]
        scarce = bind.decide(query[:14], [k[:14] for k in keys], far=FAR)
        plenty = bind.decide(query, keys, far=FAR)
        scarce_answered += int(scarce.key is not None)
        plenty_answered += int(plenty.key is not None)
        wrong += int(plenty.key is not None and plenty.key != 0)

    assert plenty_answered > scarce_answered, (
        f"more evidence did not produce more answers: {scarce_answered} -> {plenty_answered} "
        f"of {draws}"
    )
    assert wrong == 0, f"answered wrongly {wrong} times with full evidence"


def test_nothing_in_the_package_holds_state_between_reads():
    """No caches, no accumulators, no fitted state. Two identical calls must agree exactly."""
    rng = np.random.default_rng(4)
    w = _world(rng)
    a, b = _facets(rng, w, (0, 1)), _facets(rng, w, (0, 1))
    first = engine.associate(a, b, far=FAR)
    for _ in range(3):
        again = engine.associate(a, b, far=FAR)
        assert again == first, "a repeated read disagreed with itself; something is carrying state"
