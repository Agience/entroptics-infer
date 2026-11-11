"""A relation derived from the edges that cite it, rather than supplied by one example.

`complete` reads the relation off a single pair, which is all an analogy gives it. A lattice holds a
name cited by many edges, and `operator_over` derives the relation those edges stand in. Three
things have to hold for that to be worth having, and each is tested against the case that would
break it:

1. **The split changes nothing.** `complete` is the case of `complete_with` where the operator
   comes from one pair, so it must give the same answers, or the published baseline moves with it.
2. **More edges resolve more.** A relation half-buried in noise should be read better from several
   of its edges than from one — otherwise the vertex buys nothing and relations should keep being
   supplied by example.
3. **Averaging nothing gives nothing.** Edges that stand in no relation must not average into one.
   A derivation that manufactures a relation from unrelated pairs would make every screen useless.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import (Unreadable, complete, complete_with, engine,
                              operator_of, operator_over, relation_over)

FAR = 0.05
SEED = 5
T, D = 96, 24


def _structured(rng, facets=3, noise=0.1):
    loadings = np.linalg.qr(rng.normal(size=(D, facets)))[0][:, :facets]
    return rng.normal(size=(T, facets)) @ loadings.T + noise * rng.normal(size=(T, D))


def _orthogonal(rng):
    return np.linalg.qr(rng.normal(size=(D, D)))[0]


def _corpus(rng, n, blur):
    """`n` sources, each with a target under two relations, each holding `1 - blur` of the way."""
    M = {"alpha": _orthogonal(rng), "beta": _orthogonal(rng)}
    records, edges = {}, {"alpha": [], "beta": []}
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng)
        for kind, m in M.items():
            records[f"{kind}{i}"] = (1 - blur) * (records[src] @ m) + blur * _structured(rng)
            edges[kind].append((src, f"{kind}{i}"))
    return records, edges


def test_the_split_left_complete_alone():
    """`complete_with`, given the operator `complete` would have read, must answer identically.

    This checks the delegation and only that: `complete` hands its work to `complete_with`,
    so a change inside `complete_with` moves both arms of this comparison together and cancels.
    Sabotaging the shared arithmetic leaves this test green and fails five others, in
    `test_forward_pass` and `test_cli`, which exercise `complete` against fixed expectations. What
    this catches is the delegation passing the wrong operator, source or candidates -- which is the
    way a refactor of this shape actually goes wrong.
    """
    rng = np.random.default_rng(SEED)
    records, edges = _corpus(rng, 8, blur=0.5)
    moved = 0
    for i in range(1, 8):
        src, cands = f"s{i}", [f"alpha{i}", f"beta{i}"]
        by_pair = complete(records, exemplar=edges["alpha"][0], source=src,
                           candidates=cands, far=FAR)
        by_matrix = complete_with(records, operator=operator_of(records, *edges["alpha"][0]),
                                  source=src, candidates=cands, far=FAR)
        moved += int((by_pair.key, by_pair.why) != (by_matrix.key, by_matrix.why))
        assert by_pair.modes == by_matrix.modes
    assert moved == 0, f"{moved} answers moved when the operator was passed rather than read"


def test_more_edges_read_a_buried_relation_that_one_edge_cannot():
    """The claim the vertex exists for. Measured as answers, not as a similarity going up."""
    rng = np.random.default_rng(SEED)
    records, edges = _corpus(rng, 16, blur=0.85)

    def answered(backing):
        got = 0
        for i in range(16):
            src, cands = f"s{i}", [f"alpha{i}", f"beta{i}"]
            pool = [e for e in edges["alpha"] if e[0] != src]
            op = operator_over(records, pool[:backing])
            out = complete_with(records, operator=op, source=src, candidates=cands, far=FAR)
            got += int(out.key is not None and cands[out.key] == f"alpha{i}")
        return got

    one, many = answered(1), answered(12)
    assert many > one, (
        f"a relation buried at 85% was read from 1 edge {one}/16 times and from 12 edges "
        f"{many}/16 — the aggregate bought nothing, so deriving it is not worth doing"
    )


def test_unrelated_edges_do_not_average_into_a_relation():
    """The negative: a name with nothing behind it must not acquire content by being cited often.

    Asserted on what the relation lets the read *do*, not on the aggregate's rank. Rank is measured
    against a matrix's own floor, so it does not separate these two: at a backing of 16 a real
    relation's aggregate reads rank 0 while an unrelated one reads 1, and a test on rank would have
    called the working case the broken one. What separates them is whether questions get answered,
    which is what a relation vertex is for.
    """
    rng = np.random.default_rng(SEED)
    n = 24
    Ma, Mb = _orthogonal(rng), _orthogonal(rng)
    records, alpha, hollow = {}, [], []
    for i in range(n):
        records[f"s{i}"] = _structured(rng)
        records[f"a{i}"] = records[f"s{i}"] @ Ma
        records[f"b{i}"] = records[f"s{i}"] @ Mb
        records[f"x{i}"] = _structured(rng)          # stands in no relation to anything
        alpha.append((f"s{i}", f"a{i}"))
        hollow.append((f"s{i}", f"x{(i + 7) % n}"))

    def answered(edges, backing):
        got = 0
        for i in range(n):
            src, cands = f"s{i}", [f"a{i}", f"b{i}"]
            pool = [e for e in edges if e[0] != src and e[1] not in cands]
            out = complete_with(records, operator=operator_over(records, pool[:backing]),
                                source=src, candidates=cands, far=FAR)
            got += int(out.key is not None)
        return got

    for backing in (1, 4, 12):
        real, empty = answered(alpha, backing), answered(hollow, backing)
        assert empty * 4 < real, (
            f"from {backing} edges a real relation answered {real}/{n} and a name with nothing "
            f"behind it answered {empty}/{n} — too close to tell apart, so any set of edges could "
            f"manufacture a relation by being numerous"
        )
    # and citing it more often must not help it
    assert answered(hollow, 12) <= answered(hollow, 1) + 2, (
        "a hollow relation answered more questions as edges accumulated, which is the failure "
        "`carries_kinds` exists to prevent arriving by another route"
    )


def test_a_relation_with_no_edges_is_refused_rather_than_invented():
    rng = np.random.default_rng(SEED)
    records, _ = _corpus(rng, 4, blur=0.0)
    with pytest.raises(Unreadable):
        operator_over(records, [])


def test_trimming_the_backing_reads_only_the_relation():
    """Given `far`, `operator_over` stops at the longest run of edges that still resolves.

    The choice must not consult the source, the candidates or the answer -- that is the whole
    reason it is allowed. Asserted by construction: the same call with the same edges returns the
    same matrix whatever question is later asked of it.
    """
    rng = np.random.default_rng(SEED)
    records, edges = _corpus(rng, 20, blur=0.0)
    whole = operator_over(records, edges["alpha"])
    trimmed = relation_over(records, edges["alpha"], far=FAR)
    assert engine.rank(whole, far=FAR) == 0, (
        "this corpus was chosen because its full aggregate collapses; if it no longer does, the "
        "test is not exercising the trim"
    )
    assert engine.rank(trimmed, far=FAR) > 0, "the trim returned an aggregate that resolves nothing"
    # and it is a pure function of the relation's own edges
    again = relation_over(records, edges["alpha"], far=FAR)
    assert np.array_equal(trimmed, again)


def test_trimming_helps_a_real_relation_and_not_a_hollow_one():
    """The admissibility control, in miniature.

    Trimming buys answers on a relation that is there. On a name whose edges join unrelated
    vertices it must buy approximately none -- otherwise it has bought its gains by eroding the
    level rather than by finding structure.
    """
    rng = np.random.default_rng(SEED)
    n = 20
    Ma, Mb = _orthogonal(rng), _orthogonal(rng)
    records, alpha, hollow = {}, [], []
    for i in range(n):
        records[f"s{i}"] = _structured(rng)
        records[f"a{i}"] = records[f"s{i}"] @ Ma
        records[f"b{i}"] = records[f"s{i}"] @ Mb
        records[f"x{i}"] = _structured(rng)
        alpha.append((f"s{i}", f"a{i}"))
        hollow.append((f"s{i}", f"x{(i + 7) % n}"))

    def answered(edges, trim):
        got = 0
        for i in range(n):
            src, cands = f"s{i}", [f"a{i}", f"b{i}"]
            pool = [e for e in edges if e[0] != src and e[1] not in cands]
            op = (relation_over(records, pool, far=FAR) if trim
                  else operator_over(records, pool))
            out = complete_with(records, operator=op, source=src, candidates=cands, far=FAR)
            got += int(out.key is not None)
        return got

    real_gain = answered(alpha, True) - answered(alpha, False)
    hollow_gain = answered(hollow, True) - answered(hollow, False)
    assert real_gain > 0, (
        f"trimming bought nothing on a relation that is there ({real_gain:+d} of {n}), so there "
        f"is no reason to do it"
    )
    assert hollow_gain <= real_gain / 2, (
        f"trimming bought {hollow_gain:+d} of {n} on a name with nothing behind it against "
        f"{real_gain:+d} on a real one — too much of the gain is the level being spent"
    )
