"""Is the remaining gap between the structures bigger than the noise in 68 questions?

Why ask
---------
Scored with the places rebuilt per question, the three structures land a few answers apart over the
askable statements. Before any of that is explained, it has to be shown to exist.

Two things make the table above read stronger than it is:

* **A few dozen questions.** The standard error on a single proportion near a quarter is about a
  twentieth at this size, so two structures a few answers apart overlap comfortably if they are
  read as independent samples.
* **A best configuration.** Reporting each structure at its best configuration lifts the reported
  value whether or not anything real is there, and lifts it more for the structure with more
  variance.

How it is settled
-------------------
The questions are the same statements for every structure, so the comparison is paired and does
not need the independent-sample error at all. What matters is only the questions the two disagree
on: if the structures were equivalent, a disagreement would fall either way with equal chance, so
the count of wins on one side is binomial and an exact two-sided test reads off it directly. That is
McNemar's test, and nothing in it is fitted or chosen.

The configuration is fixed in advance rather than maximized: every structure is placed by the
shipped walk on its own basis, unblanked, and scored once -- nothing is picked after the fact.

What each outcome means
-------------------------
A structure that wins its discordant pairs well beyond chance is really better. One that does not
is not distinguishable at this store size, and any explanation of "the gap" is an explanation of
noise -- in which case the next move is more statements, not more hypotheses.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import itertools
import math

import numpy as np

from ontology import Engine                                    # the shipped engine

import record

import ontology
import test_binding_the_coordinate as binding
import test_both_structures_fairly as fair
from measure import mcnemar  # noqa: E402,F401


def outcomes(build, *, weighted, candidates, repaired=None):
    """``{(relation, source, destination): was it top-1}`` -- per question, not averaged.

    ``repaired`` is a generator. Given one, each relation's pairs are re-paired at random before the
    operator is formed -- same coordinate, same statements, same number of pairs, only which source
    went with which destination destroyed. That is the null for "the operator carries the relation"
    rather than for "the coordinate carries the concepts", and without it a score says only that
    something in the geometry is non-random.
    """
    from entroptics_infer.bind import association_matrix

    all_statements = list(ontology.STATEMENTS)
    by_relation: dict = {}
    for i, (a, r, b) in enumerate(fair.triples(all_statements)):
        by_relation.setdefault(r, []).append((i, a, b))

    got: dict = {}
    for relation, items in by_relation.items():
        if len(items) < 3:
            continue
        for si, src, dst in items:
            held = [p for j, p in enumerate(all_statements) if j != si]
            pairs = [(a, b) for a, r, b in fair.triples(held) if r == relation]
            if len(pairs) < 2:
                continue
            vectors, _ = build(held, weighted=weighted, floored=False)
            if src not in vectors or dst not in vectors:
                continue
            present = [c for c in candidates if c in vectors]
            index = {c: i for i, c in enumerate(present)}
            if dst not in index:
                continue
            allp = np.stack([vectors[c] for c in present])
            A = np.stack([vectors[a] for a, _ in pairs])
            order = (repaired.permutation(len(pairs)) if repaired is not None
                     else np.arange(len(pairs)))
            B = np.stack([vectors[pairs[int(j)][1]] for j in order])
            v = vectors[src]
            scores = fair._along(allp, Engine._land(v, A, B))
            if src in index:
                scores[index[src]] = -np.inf
            rank = int(np.where(np.argsort(-scores) == index[dst])[0][0]) + 1
            got[(relation, src, dst)] = rank == 1
    return got


def main() -> int:
    candidates = fair.answerable(ontology.STATEMENTS)
    builds = (("typed", fair.typed_places),
              ("vertex", fair.vertex_places),
              ("vertex, bound", binding.bound_places))

    print(f"{len(ontology.STATEMENTS)} statements, chance 1 in {len(candidates)}, "
          "places rebuilt per question")
    print("configuration fixed in advance, not maximized over\n")

    kept: dict = {}
    repaired_scores: list = []
    got = {name: outcomes(build, weighted=False, candidates=candidates) for name, build in builds}
    kept["unweighted"] = got
    print(f"    {'structure':<16}{'top-1':>9}{'asked':>7}")
    for name, _ in builds:
        r = got[name]
        print(f"    {name:<16}{np.mean(list(r.values())):>9.2f}{len(r):>7}")
    print()
    # the operator's own null, on the coordinate the live path uses
    seeds = [len(ontology.STATEMENTS) + k for k in range(5)]
    scores = [np.mean(list(outcomes(binding.bound_places, weighted=False,
                                    candidates=candidates,
                                    repaired=np.random.default_rng(s)).values()))
              for s in seeds]
    repaired_scores = [float(x) for x in scores]
    print(f"    {'vertex, bound, re-paired':<16}{np.mean(scores):>9.2f}"
          f"        over {len(seeds)} draws, "
          f"{min(scores):.2f} to {max(scores):.2f}")
    print()

    print(f"    {'pair':<32}{'wins':>6}{'losses':>8}{'shared':>8}{'p':>10}")
    print("    " + "-" * 64)
    for (na, _), (nb, _) in itertools.combinations(builds, 2):
        w, l, n, p = mcnemar(got[na], got[nb])
        verdict = "distinguishable" if p < 0.05 else "not distinguishable"
        print(f"    {na + ' vs ' + nb:<32}{w:>6}{l:>8}{n:>8}{p:>10.3f}   {verdict}")
    print()

    # filed under the walk it was read on, which is the shipped one and carries no weight
    record.figures(
        top1={arm: {k: float(np.mean(list(v.values()))) for k, v in g.items()}
              for arm, g in kept.items()},
        paired={arm: {f"{a} vs {b}": dict(zip(("wins", "losses", "shared", "p"),
                                              mcnemar(g[a], g[b])))
                      for (a, _), (b, _) in itertools.combinations(builds, 2)}
                for arm, g in kept.items()},
        repaired=repaired_scores,
        repaired_mean=float(np.mean(repaired_scores)) if repaired_scores else None,
        questions=len(next(iter(kept["unweighted"].values()))),
        candidates=len(candidates),
    )

    print("  A pair that disagrees on only a handful of questions cannot be separated by them, and")
    print("  at this store size that is the expected outcome. What it would take to separate them")
    print("  is more statements -- the discordant count grows with the store, the chance level does")
    print("  not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
