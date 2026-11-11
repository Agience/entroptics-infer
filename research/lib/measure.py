"""The arithmetic the research scripts share: directions, the exact tests they decide with, and the
statistics of a record.

One copy of each, so a check that scores a place, settles a paired comparison or compares two
records does it the same way as every other. Nothing here is a level or a constant: each test is
exact for the counts it is given, and a verdict compares its p to nothing chosen here.
"""
from __future__ import annotations

import math

import numpy as np


def unit(v):
    """``v`` scaled to length one; a zero vector is returned as it is."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def along(screen, V):
    """How far each row of ``screen`` lies along the direction ``V``: the cosine, zero rows at zero."""
    norms = np.linalg.norm(screen, axis=1)
    norms[norms == 0] = 1.0
    n = float(np.linalg.norm(V))
    return (screen @ V) / (norms * (n if n > 0 else 1.0))


def sign_test(wins, losses):
    """Two-sided exact binomial on the questions where the two differ; ``nan`` when none do."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def mcnemar(a, b):
    """``(wins, losses, shared, p)`` -- the exact two-sided p on the questions where two arms disagree.

    ``a`` and ``b`` map a question to whether that arm got it right. Under equivalence each
    discordant question falls either way with equal chance, so the p is the sign test on the
    discordant pairs; with none, nothing separates the arms and the p is one.
    """
    shared = sorted(set(a) & set(b), key=repr)
    wins = sum(1 for k in shared if a[k] and not b[k])
    losses = sum(1 for k in shared if b[k] and not a[k])
    n = wins + losses
    if n == 0:
        return wins, losses, len(shared), 1.0
    tail = sum(math.comb(n, j) for j in range(0, min(wins, losses) + 1)) / 2 ** n
    return wins, losses, len(shared), min(1.0, 2 * tail)


def binomial_tail(right, n, p):
    """The chance of ``right`` or more from ``n`` independent draws each right with chance ``p``."""
    return float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                     for k in range(right, n + 1)))


def centred(X):
    """``X`` with each column's mean taken off."""
    return X - X.mean(axis=0, keepdims=True)


def centred_dot(A, B):
    """The inner product of two records after each is centred."""
    return float(np.sum(centred(A) * centred(B)))


def centred_cos(A, B):
    """The cosine between two records after each is centred; zero when either is constant."""
    a, b = centred(A), centred(B)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.sum(a * b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def correlation(a, b):
    """Pearson's correlation of two series; zero when either is constant."""
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def autocorrelation(X, kmax):
    """A record's own lag-``k`` row autocorrelation for ``k = 1 .. kmax``, averaged over its channels."""
    Z = X - X.mean(axis=0, keepdims=True)
    v = (Z * Z).sum(axis=0)
    safe = np.where(v > 0, v, 1.0)
    return np.array([float(np.mean((Z[:-k] * Z[k:]).sum(axis=0) / safe))
                     for k in range(1, kmax + 1)])


def reciprocal_rank(scores, truth):
    """1 / the true item's position in the ordering. Ties take the average position they span."""
    s = scores[truth]
    better = int(np.sum(scores > s))
    tied = int(np.sum(scores == s))
    return 1.0 / (better + (tied + 1) / 2.0)


def fisher_upper(hits, drawn, marked, total):
    """One-sided Fisher exact: the chance of ``hits`` or more marked among ``drawn`` of ``total``.

    ``marked`` of the ``total`` carry the mark. ``nan`` when nothing or everything is drawn, where
    the draw cannot say anything.
    """
    if not drawn or drawn == total:
        return float("nan")
    top = min(marked, drawn)
    return float(sum(math.comb(marked, x) * math.comb(total - marked, drawn - x)
                     for x in range(hits, top + 1)) / math.comb(total, drawn))
