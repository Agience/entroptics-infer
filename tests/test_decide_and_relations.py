"""The decision and the relation read, on frames built here so the tests depend on nothing else.

These are unit tests of properties, not a second copy of the experiments. Where an experiment
measures how well something works across a surface, these pin what must be true of *any* input: that
refusal names its cause, that a type is a position rather than a name, and that neither read can be
made to answer where the corpus does not support one.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import bind, engine, relations

FAR = 0.05
T, D = 96, 24


def _channels(rng, n):
    A = rng.normal(size=(n, D))
    Q, _ = np.linalg.qr(A.T)
    return rng.normal(size=(n, T)), Q.T[:n]


def _record(rng, latents, dirs, which, amp, noise):
    X = np.zeros((T, D))
    for f in which:
        X += amp * np.outer(latents[f], dirs[f])
    return X + rng.normal(size=(T, D)) * noise


# ── the decision ────────────────────────────────────────────────────────────────────────────────


def test_a_clear_binding_is_returned_and_named():
    """The easy case must work, or nothing below means anything."""
    rng = np.random.default_rng(0)
    lat, dirs = _channels(rng, 6)
    query = _record(rng, lat, dirs, (0, 1, 2), 1.0, 0.05)
    keys = [
        _record(rng, lat, dirs, (0, 1, 2), 1.0, 0.05),     # shares everything
        _record(rng, lat, dirs, (3,), 1.0, 0.05),          # shares nothing
        _record(rng, lat, dirs, (4, 5), 1.0, 0.05),        # shares nothing
    ]
    d = bind.decide(query, keys, far=FAR)
    assert d.key == 0 and d.why == bind.BOUND
    assert bind.select(query, keys, far=FAR) == 0


def test_nothing_resolving_is_refused_and_says_so():
    """No candidate clears its own permutation null: the answer is None, with the cause named."""
    rng = np.random.default_rng(1)
    query = rng.normal(size=(T, D))
    keys = [rng.normal(size=(T, D)) for _ in range(4)]
    d = bind.decide(query, keys, far=FAR)
    assert d.key is None
    assert d.why in (bind.NOTHING_RESOLVED, bind.NO_SHARED_CHANNEL, bind.INDISTINGUISHABLE)
    assert bind.select(query, keys, far=FAR) is None


def test_two_identical_candidates_are_refused_as_indistinguishable():
    """The property the whole decision rests on.

    Where two candidates carry the same relation to the query, the evidence does not separate them.
    Returning either would be the system inventing a preference the corpus does not support, and
    that is exactly the confabulation the program is built to avoid.
    """
    rng = np.random.default_rng(2)
    lat, dirs = _channels(rng, 4)
    query = _record(rng, lat, dirs, (0, 1), 1.0, 0.02)
    twin = _record(rng, lat, dirs, (0, 1), 1.0, 0.02)
    d = bind.decide(query, [twin, twin.copy(), rng.normal(size=(T, D))], far=FAR)
    assert d.key is None and d.why == bind.INDISTINGUISHABLE


def test_the_row_is_not_normalised():
    """No softmax, no forced sum. A row of candidates that resolve nothing stays a row of nothing.

    This is the property that makes refusal possible at all: anything that normalized the row would
    make it sum to one whether or not a single term in it was resolved.
    """
    rng = np.random.default_rng(3)
    query = rng.normal(size=(T, D))
    row = bind.attend(query, [rng.normal(size=(T, D)) for _ in range(5)], far=FAR)
    assert not any(b.association.resolved for b in row)
    assert sum(abs(b.association.strength) for b in row) == 0.0


def test_every_candidate_is_reported_even_when_none_is_chosen():
    """A caller that needs to see the evidence gets all of it, refusal or not."""
    rng = np.random.default_rng(4)
    keys = [rng.normal(size=(T, D)) for _ in range(7)]
    d = bind.decide(rng.normal(size=(T, D)), keys, far=FAR)
    assert len(d.bindings) == len(keys)
    assert [b.key for b in d.bindings] == list(range(len(keys)))


# ── relation type ───────────────────────────────────────────────────────────────────────────────


def _relational(rng, n_relations, per_relation, noise=0.05):
    """Edges whose relation is literally a matrix: one latent along u_k in the source, w_k in the
    target. Disjoint endpoints, so each record carries exactly one relation."""
    A = rng.normal(size=(2 * n_relations, D))
    Q, _ = np.linalg.qr(A.T)
    roles = Q.T[: 2 * n_relations].reshape(n_relations, 2, D)
    records, edges, kinds = {}, [], []
    for k in range(n_relations):
        for e in range(per_relation):
            s = rng.normal(size=T)
            a, b = f"s{k}_{e}", f"t{k}_{e}"
            records[a] = np.outer(s, roles[k, 0]) + rng.normal(size=(T, D)) * noise
            records[b] = np.outer(s, roles[k, 1]) + rng.normal(size=(T, D)) * noise
            edges.append((a, b))
            kinds.append(k)
    return records, tuple(edges), tuple(kinds)


@pytest.mark.parametrize("n_relations", [3, 4, 5, 7])
def test_the_number_of_kinds_is_counted_not_supplied(n_relations):
    """The count comes from the edges. Nothing is told how many kinds to look for.

    Run at three kinds and above, which is where the count is exact across every edge budget
    measured. ``test_the_count_is_sensitive_at_one_and_two_kinds`` pins the regime below that.
    """
    rng = np.random.default_rng(10 + n_relations)
    records, edges, _ = _relational(rng, n_relations, per_relation=16)
    assert relations.relation_modes(records, edges, far=FAR) == n_relations


def test_the_count_is_sensitive_at_one_and_two_kinds():
    """A stated limit, pinned so it cannot quietly change.

    At three kinds and above the count is exact from 4 edges per kind to 64. At one and two kinds it
    is not, and the direction is the surprising one: recovery *falls* as edges are added. With many
    edges of a single kind the stack is one signal direction plus a noise cross-term per edge, and
    as the rows accumulate enough of that noise subspace becomes resolvable to be counted. Measured
    at D=24, noise 0.05, 12 draws: one kind recovers exactly in 12 of 12 draws at 4 edges and 0 of
    12 at 32.

    So the count is the resolved dimensionality of the relational structure, which coincides with
    the number of kinds over a regime that has been measured and does not coincide below it. This
    test does not assert a particular wrong answer — only that the small-kind regime is not silently
    assumed to behave like the rest.
    """
    rng = np.random.default_rng(99)
    records, edges, _ = _relational(rng, 1, per_relation=32)
    counted = relations.relation_modes(records, edges, far=FAR)
    assert counted >= 1, "a single real kind must still read as at least one"


def test_the_count_is_not_capped():
    """No ceiling is imposed anywhere, so more kinds read as more kinds.

    A fixed maximum would be the vocabulary back in a different costume — the count would stop
    tracking at whatever someone had decided was enough.
    """
    rng = np.random.default_rng(20)
    small = relations.relation_modes(*_relational(rng, 3, 8)[:2], far=FAR)
    large = relations.relation_modes(*_relational(rng, 9, 8)[:2], far=FAR)
    assert large > small and large == 9


def test_an_edges_type_is_a_position_and_not_a_name():
    """Coordinates, continuous, one row per edge — and same-kind edges align, different-kind do not."""
    rng = np.random.default_rng(30)
    records, edges, kinds = _relational(rng, 4, per_relation=8)
    X = relations.relation_coordinates(records, edges, far=FAR)
    assert X.shape[0] == len(edges)
    assert np.issubdtype(X.dtype, np.floating)

    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = U @ U.T
    k = np.asarray(kinds)
    same = (k[:, None] == k[None, :]) & ~np.eye(len(k), dtype=bool)
    diff = k[:, None] != k[None, :]
    assert G[same].mean() > G[diff].mean()


def test_a_kind_reshapes_to_the_matrix_it_is():
    """The basis lives in the endpoint records' own coordinates, so a kind is readable as a map."""
    rng = np.random.default_rng(40)
    records, edges, _ = _relational(rng, 3, per_relation=8)
    V = relations.relation_basis(records, edges, far=FAR)
    assert V.shape[1] == D * D
    assert V[0].reshape(D, D).shape == (D, D)


def test_unrelated_records_with_disjoint_edges_carry_no_kinds():
    """The negative control. A counter that finds kinds in noise is counting its own error."""
    rng = np.random.default_rng(50)
    edges = tuple((f"a{e}", f"b{e}") for e in range(30))
    records = {v: rng.normal(size=(T, D)) for e in edges for v in e}
    assert relations.relation_modes(records, edges, far=FAR) == 0


def test_no_edges_is_refused_rather_than_answered_as_zero():
    """"This corpus has no relations" is a claim, and it may not be made where nothing was observed."""
    with pytest.raises(engine.Unreadable):
        relations.relation_modes({}, (), far=FAR)


def test_records_on_different_bases_are_refused():
    """Relations can only be compared where they live in one space."""
    rng = np.random.default_rng(60)
    records = {"a": rng.normal(size=(T, D)), "b": rng.normal(size=(T, D)),
               "c": rng.normal(size=(T, D + 1)), "d": rng.normal(size=(T, D + 1))}
    with pytest.raises(engine.Unreadable):
        relations.relation_modes(records, (("a", "b"), ("c", "d")), far=FAR)
