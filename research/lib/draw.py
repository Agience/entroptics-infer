"""Records and stores drawn from a stated process, for the scripts that draw the same one.

Each draw here is used by more than one script. It is defined once, so that two scripts said to draw
the same thing do draw the same thing, from the same generator calls in the same order. A draw that
depends on a shape takes the shape as an argument, and each caller passes its own.
"""
from __future__ import annotations

import numpy as np


def structured(rng, T, D, facets=3, noise=0.1):
    """A few latent channels seen through a fixed basis -- something there is to read."""
    loadings = np.linalg.qr(rng.normal(size=(D, facets)))[0][:, :facets]
    return rng.normal(size=(T, facets)) @ loadings.T + noise * rng.normal(size=(T, D))


def unit_directions(rng, n, d):
    """``n`` orthonormal directions in ``d`` dimensions, as the rows of an ``(n, d)`` array."""
    A = rng.normal(size=(n, d))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:n]


def ar1(rng, rho, T, D):
    """A ``(T, D)`` record whose rows carry autocorrelation `rho` and which shares nothing with any
    other."""
    x = np.zeros((T, D))
    x[0] = rng.normal(size=D)
    step = rng.normal(size=(T, D)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + step[t]
    return x


def observed(concepts, relations, group, sends, share, rng):
    """The statements the store is given: a random share of the true edges, and nothing else."""
    true_edges = [(a, r, b) for a in concepts for r in relations
                  for b in concepts if group[b] == sends[r][group[a]] and b != a]
    keep = rng.random(len(true_edges)) < share
    return tuple(e for e, k in zip(true_edges, keep) if k)
