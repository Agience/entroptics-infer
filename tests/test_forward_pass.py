"""The forward pass: two of a triple in, the third out, with the operator given by example.

Property tests, not a second copy of the experiment. The one that matters most is negative: the
operator must be *load-bearing*, so removing it has to collapse the method to chance. A forward pass
that works without its operator is answering a different question from the one it was asked.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import bind, engine, infer

FAR = 0.05
T, D = 96, 32


def _triples(rng, n_relations, n_sources, noise=0.05):
    """Every source related to every one of its targets, each by a different kind."""
    A = rng.normal(size=(2 * n_relations, D))
    Q, _ = np.linalg.qr(A.T)
    roles = Q.T[: 2 * n_relations].reshape(n_relations, 2, D)
    records, targets = {}, {}
    sources = [f"a{i}" for i in range(n_sources)]
    for a in sources:
        records.setdefault(a, np.zeros((T, D)))
        for k in range(n_relations):
            b = f"{a}_k{k}"
            records.setdefault(b, np.zeros((T, D)))
            s = rng.normal(size=T)
            records[a] = records[a] + np.outer(s, roles[k, 0])
            records[b] = records[b] + np.outer(s, roles[k, 1])
            targets[(a, k)] = b
    for v in records:
        records[v] = records[v] + rng.normal(size=(T, D)) * noise
    return records, sources, targets


def _ask(records, sources, targets, n_relations, source_i, kind):
    a = sources[source_i]
    other = sources[(source_i + 1) % len(sources)]
    cands = [targets[(a, k)] for k in range(n_relations)]
    return a, cands, (other, targets[(other, kind)])


def test_the_operator_decides_the_answer():
    """The forward pass returns the target the exemplar's relation points at."""
    rng = np.random.default_rng(0)
    records, sources, targets = _triples(rng, 4, 5)
    right = total = 0
    for i in range(len(sources)):
        for k in range(4):
            a, cands, ex = _ask(records, sources, targets, 4, i, k)
            op = infer.operator_of(records, *ex)
            z = [engine.associate(op, bind.association_matrix(records[a], records[b]), far=FAR).z
                 for b in cands]
            right += int(int(np.argmax(z)) == k)
            total += 1
    assert right / total > 0.75, f"only {right}/{total} with the operator supplied"


def test_without_the_operator_the_method_is_at_chance():
    """The control that makes the claim mean anything.

    Every candidate is genuinely related to the source — just differently — so ranking by raw
    association strength cannot beat chance. If it could, the corpus would be leaking the answer
    through something other than the relation, and the accuracy above would not be the operator's
    doing.
    """
    rng = np.random.default_rng(1)
    n_rel = 4
    records, sources, targets = _triples(rng, n_rel, 5)
    right = total = 0
    for i, a in enumerate(sources):
        cands = [targets[(a, k)] for k in range(n_rel)]
        strength = [engine.associate(records[a], records[b], far=FAR).z for b in cands]
        best = int(np.argmax(strength))
        for k in range(n_rel):
            right += int(best == k)
            total += 1
    assert abs(right / total - 1.0 / n_rel) < 1e-9, (
        f"ranking without the operator scored {right / total:.3f}, not chance {1 / n_rel:.3f}"
    )


def test_the_wrong_operator_returns_the_wrong_target():
    """Supplying an exemplar of a different kind should return that kind's target instead."""
    rng = np.random.default_rng(2)
    n_rel = 4
    records, sources, targets = _triples(rng, n_rel, 5)
    asked_for = total = 0
    for i, a in enumerate(sources):
        other = sources[(i + 1) % len(sources)]
        cands = [targets[(a, k)] for k in range(n_rel)]
        for k in range(n_rel):
            wrong_kind = (k + 1) % n_rel
            op = infer.operator_of(records, other, targets[(other, wrong_kind)])
            z = [engine.associate(op, bind.association_matrix(records[a], records[b]), far=FAR).z
                 for b in cands]
            asked_for += int(int(np.argmax(z)) == k)
            total += 1
    assert asked_for / total < 1.0 / n_rel, (
        f"a wrong-kind exemplar returned the asked-for target {asked_for}/{total} times, at or "
        f"above chance — the operator is not deciding"
    )


def test_an_exemplar_must_be_a_pair():
    """One record stands in no relation; three do not name one."""
    rng = np.random.default_rng(3)
    records, sources, targets = _triples(rng, 2, 3)
    for bad in (("a0",), ("a0", "a0_k0", "a1")):
        with pytest.raises(engine.Unreadable):
            infer.complete(records, exemplar=bad, source="a0",
                           candidates=["a0_k0", "a0_k1"], far=FAR)


def test_the_row_is_not_normalised():
    """Nothing forces the candidate scores to sum to anything, which is what allows refusal."""
    rng = np.random.default_rng(4)
    records = {v: rng.normal(size=(T, D)) for v in ("a", "b", "c", "x", "y")}
    res = infer.complete(records, exemplar=("x", "y"), source="a",
                         candidates=["b", "c"], far=FAR)
    assert res.key is None and res.why == infer.NOTHING_RESOLVED
    assert all(s.strength == 0.0 for s in res.scores)


def test_every_candidate_is_scored_even_when_none_is_returned():
    rng = np.random.default_rng(5)
    records = {v: rng.normal(size=(T, D)) for v in ("a", "b", "c", "d", "x", "y")}
    res = infer.complete(records, exemplar=("x", "y"), source="a",
                         candidates=["b", "c", "d"], far=FAR)
    assert len(res.scores) == 3 and len(res.modes) == 3


def test_a_refusal_names_which_kind_it_is():
    """"Nothing resolved" and "two candidates were equally good" are different facts."""
    assert infer.NOTHING_RESOLVED != infer.INDISTINGUISHABLE != infer.BOUND
    rng = np.random.default_rng(6)
    records, sources, targets = _triples(rng, 3, 4)
    seen = set()
    for i, a in enumerate(sources):
        other = sources[(i + 1) % len(sources)]
        cands = [targets[(a, k)] for k in range(3)]
        for k in range(3):
            res = infer.complete(records, exemplar=(other, targets[(other, k)]), source=a,
                                 candidates=cands, far=FAR)
            seen.add(res.why)
    assert seen <= {infer.BOUND, infer.INDISTINGUISHABLE, infer.NOTHING_RESOLVED}
    assert infer.BOUND in seen, "the forward pass never bound anything on a clean corpus"
