"""The proved second moment, checked against brute force and against the instrument.

`research/lean/Infer/SecondMoment.lean` proves that on an axis of `n + 1` rows, with both records
centred,

    n * sum over permutations of S(pi)^2  =  card(Perm) * (sum a^2)(sum b^2)

A proof about a statistic is worth little if the code standardizes by something else, so this file
closes that gap from both ends. It enumerates every permutation for a small axis and checks the
closed form exactly, and then checks that `entroptics`'s own coupling divides by that same quantity.

The second is the one that matters. The first says the algebra is right; the second says the
algebra is the algebra the reads actually use.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from entroptics_infer import engine

FAR = 0.05
SEED = 20260915


def _centred(rng, n):
    x = rng.normal(size=n)
    return x - x.mean()


def _S(a, b, perm):
    """The statistic: one record's rows re-paired against the other's."""
    return float(np.sum(a[list(perm)] * b))


@pytest.mark.parametrize("n", [4, 5, 6])
def test_second_moment_matches_the_closed_form(n):
    """Every permutation enumerated, against `n - 1` times the sum equalling the closed form.

    This is the proved statement at `card(iota) = n`, with nothing sampled: for `n = 6` that is all
    720 permutations, so the equality is exact rather than statistical.
    """
    rng = np.random.default_rng(SEED + n)
    for _ in range(20):
        a, b = _centred(rng, n), _centred(rng, n)
        perms = list(itertools.permutations(range(n)))
        total = sum(_S(a, b, p) ** 2 for p in perms)
        closed = math.factorial(n) * float(np.sum(a ** 2) * np.sum(b ** 2))
        assert (n - 1) * total == pytest.approx(closed, rel=1e-9), (
            f"(n-1) * sum_pi S^2 = {(n - 1) * total} against card(Perm) * (sum a^2)(sum b^2) "
            f"= {closed}; the Lean statement and the arithmetic disagree"
        )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_the_first_moment_is_zero_too(n):
    """The other half of the null, enumerated the same way.

    Proved in `Permutation.lean` and checked here for the same reason: a variance is only the right
    denominator for a score whose mean is what the proof says it is.
    """
    rng = np.random.default_rng(SEED + n)
    for _ in range(20):
        a, b = _centred(rng, n), _centred(rng, n)
        total = sum(_S(a, b, p) for p in itertools.permutations(range(n)))
        assert total == pytest.approx(0.0, abs=1e-8)


def test_centring_is_what_makes_the_closed_form_hold():
    """Set the hypothesis deliberately wrong and the identity must fail.

    A test that only ever passes on the intended input does not show the hypothesis is doing
    anything. Both centrings are load-bearing in the proof and they do different work, so both are
    broken here separately.
    """
    n = 5
    rng = np.random.default_rng(SEED)
    perms = list(itertools.permutations(range(n)))

    def gap(a, b):
        total = sum(_S(a, b, p) ** 2 for p in perms)
        closed = math.factorial(n) * float(np.sum(a ** 2) * np.sum(b ** 2))
        return abs((n - 1) * total - closed)

    a, b = _centred(rng, n), _centred(rng, n)
    assert gap(a, b) < 1e-6, "the identity should hold where both records are centred"

    # `a` off-centre: the off-diagonal's product term no longer vanishes
    assert gap(a + 1.0, b) > 1e-3, (
        "with `a` un-centred the closed form still held — the centring hypothesis is not being "
        "exercised and this test proves nothing"
    )
    # `b` off-centre: the step that collapses the off-diagonal onto the diagonal is gone
    assert gap(a, b + 1.0) > 1e-3, (
        "with `b` un-centred the closed form still held — the second centring hypothesis is not "
        "being exercised"
    )


@pytest.mark.parametrize("n", [8, 12, 24])
def test_the_instrument_divides_by_the_proved_quantity(n):
    """**The one that matters.** `entroptics` standardizes by exactly this variance.

    A proof about the null is decoration unless the instrument uses it. The coupling's `z` should be
    the alignment over the square root of `(sum a^2)(sum b^2) / (n - 1)`, which is the proved second
    moment divided by the number of permutations. Checked at a single feature coordinate, where the
    matrix form reduces to the scalar one the theorem is stated for.
    """
    rng = np.random.default_rng(SEED + n)
    for _ in range(10):
        a, b = _centred(rng, n), _centred(rng, n)
        A, B = a.reshape(n, 1), b.reshape(n, 1)

        alignment = float(np.sum(a * b))
        variance = float(np.sum(a ** 2) * np.sum(b ** 2)) / (n - 1)
        expected = alignment / math.sqrt(variance)

        got = engine.associate(A, B, far=FAR).z
        assert got == pytest.approx(expected, rel=1e-6), (
            f"the instrument's z is {got} where the proved null gives {expected}. Either the "
            f"implementation standardizes by something other than the second moment proved in "
            f"`SecondMoment.lean`, or the proof is about a different statistic."
        )


def _centred_frame(rng, n, d):
    X = rng.normal(size=(n, d))
    return X - X.mean(axis=0, keepdims=True)


@pytest.mark.parametrize("n,d", [(5, 2), (5, 3), (6, 2)])
def test_the_matrix_second_moment_matches_the_closed_form(n, d):
    """`SecondMoment.sum_perm_alignment_sq`, enumerated over every permutation.

    The matrix statistic's null has variance `tr(Ca Cb) / (n - 1)` where `Ca = A^T A`. Where the
    scalar case has a product of two magnitudes, this has a matrix functional — which is the
    whole reason the association can tell apart candidates a cosine cannot.
    """
    rng = np.random.default_rng(SEED + n * d)
    for _ in range(15):
        A, B = _centred_frame(rng, n, d), _centred_frame(rng, n, d)
        total = float(sum(np.sum(A[list(p)] * B) ** 2
                          for p in itertools.permutations(range(n))))
        closed = math.factorial(n) * float(np.trace((A.T @ A) @ (B.T @ B)))
        assert (n - 1) * total == pytest.approx(closed, rel=1e-9), (
            f"(n-1) * sum_pi S^2 = {(n - 1) * total} against card(Perm) * tr(Ca Cb) = {closed}"
        )


@pytest.mark.parametrize("n,d", [(12, 3), (24, 5), (32, 8)])
def test_the_instrument_divides_by_the_matrix_second_moment(n, d):
    """The matrix case of the link: `associate` standardizes by `tr(Ca Cb) / (n - 1)`.

    This is the one the paper's central mechanism rests on. A cosine's normalizer is
    `||A|| ||B||` — a product of magnitudes — and this is a trace of a product of covariances. They
    agree only when both covariances are isotropic, and where they differ is exactly where the
    association reads something a cosine cannot.
    """
    rng = np.random.default_rng(SEED + n + d)
    for _ in range(10):
        A, B = _centred_frame(rng, n, d), _centred_frame(rng, n, d)
        alignment = float(np.sum(A * B))
        variance = float(np.trace((A.T @ A) @ (B.T @ B))) / (n - 1)
        expected = alignment / math.sqrt(variance)

        got = engine.associate(A, B, far=FAR).z
        assert got == pytest.approx(expected, rel=1e-6), (
            f"the instrument's z is {got} where the proved matrix null gives {expected}"
        )
