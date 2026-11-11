"""`ontology.py` justifies its whole shape with a count that cannot be re-derived here.

The figure and why it cannot be re-derived
--------------------------------------------
`ontology.py` says edges measured against six relation names occupy more independent directions
than there are names -- so a name is a coarse label for a region. That is the reason the store has
no relation vocabulary, and the design of everything downstream follows from it.

Measuring that count needs an `ontology.labels()`, which the store the count argues for does not
have. The count is not recoverable here, so no figure is quoted for it. What can be asked is the
same question of this store.

Why it matters beyond the docstring
-------------------------------------
§47.111 leaves the blocker as relation selection and §47.112 finds the store holds little that would
let a reader choose. There is a prior possibility neither of them tests: that the relations are not
distinguishable as operators at all. If `is_a` and `part_of` occupy nearly the same direction, no
rule can separate them and the failure is the store's rather than the rule's — which is what §47.87
suspects from a different side, and what §47.83 assumes away by building a structure where the
relations plainly differ.

What is measured
------------------
Each relation's operator is a matrix. Scaled to unit length so a relation with more uses does not
count for more, and stacked, the singular values say how many independent directions the set
occupies.

Because every row is scaled to unit length, the squared singular values sum to the number of
relations however they are arranged, so what says how many directions they occupy is the shape of
the spectrum and not its size. ``(sum of squares)^2 / (sum of fourth powers)`` reads that shape
directly: it is the number of relations when the operators are orthogonal and one when they all point
the same way, and it takes no level and no cut.

**Comparing singular values index by index does not work and the first run of this file did it.** A
set that is more spread has a *smaller* leading value, so the real store came out below its null at
the first index and above it at the rest, which reads as a missing direction and is the opposite of
what it means.

The null is not a level either: every statement keeps its pair and is given a relation drawn from the
store's own, which destroys which relation a pair belongs to and keeps everything else — how many
relations there are, how many uses each has, and the pairs themselves. Under it the operators are
samples of one map and occupy fewer directions.

Alongside it, how aligned each pair of operators is, which names *which* relations a reader could not
tell apart rather than only how many it could.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import itertools

import numpy as np

import record

import ontology
import operators
import place

DRAWS = 50


def stack(vectors, statements, relations):
    """Every relation's operator, flattened and scaled to unit length, as rows."""
    from entroptics_infer.bind import association_matrix

    rows, names = [], []
    for r in relations:
        pairs = [(a, b) for a, rel, b in statements
                 if rel == r and a in vectors and b in vectors]
        if len(pairs) < 2:
            continue
        A = np.stack([vectors[a] for a, _ in pairs])
        B = np.stack([vectors[b] for _, b in pairs])
        M = np.asarray(association_matrix(A, B)).ravel()
        n = float(np.linalg.norm(M))
        if n == 0:
            continue
        rows.append(M / n)
        names.append(r)
    return np.stack(rows), names


def main() -> int:
    triples = [p for p in ontology.STATEMENTS if len(p) == 3]
    vectors, _ = place.places()
    relations = sorted(operators.relations())
    real, names = stack(vectors, triples, relations)
    k = len(names)
    left_out = [r for r in relations if r not in names]
    print(f"the store's {k} relations as operators, scaled to unit length and stacked")
    if left_out:
        print(f"  {len(left_out)} of {len(relations)} left out, having fewer than two pairs or an "
              f"operator of length zero: {', '.join(left_out)}")
    print(f"{len(triples)} statements, {DRAWS} draws of the null\n")

    got = np.linalg.svd(real, compute_uv=False)

    null = np.zeros((DRAWS, k))
    for d in range(DRAWS):
        rng = np.random.default_rng(52000 + d)
        # a permutation of the relation column, not a fresh draw: the use counts are part of
        # what the null keeps, and drawing with replacement gives each relation a new count
        # every draw -- which widens the null and flatters the store by about a third
        column = [r for _, r, _ in triples]
        order = rng.permutation(len(column))
        shuffled = [(a, column[int(order[i])], b) for i, (a, _, b) in enumerate(triples)]
        rows, got_names = stack(vectors, shuffled, names)
        s = np.linalg.svd(rows, compute_uv=False)
        null[d, :len(s)] = s

    def directions(s):
        """How many directions a spectrum occupies: k when orthogonal, 1 when all one way."""
        sq = float(np.sum(s ** 2))
        return sq * sq / float(np.sum(s ** 4))

    kinds = directions(got)
    drawn = np.array([directions(null[d]) for d in range(DRAWS)])
    beats = int(np.sum(drawn >= kinds))
    p = (beats + 1) / (DRAWS + 1)

    print(f"  {'direction':>11}{'the store':>12}{'the null, its widest':>23}"
          f"{'its narrowest':>16}")
    print("  " + "-" * 62)
    for i in range(k):
        print(f"  {i + 1:>11}{got[i]:>12.4f}{null[:, i].max():>23.4f}{null[:, i].min():>16.4f}")
    print()
    print(f"  directions occupied: the store {kinds:.2f} of {k}, the null "
          f"{drawn.mean():.2f} on average and {drawn.max():.2f} at its widest, p {p:.4f}")
    print()

    pairs = {}
    for a, b in itertools.combinations(range(k), 2):
        pairs[f"{names[a]} and {names[b]}"] = float(real[a] @ real[b])
    closest = max(pairs, key=pairs.get)

    print(f"  {'two relations':>28}{'how aligned':>14}")
    print("  " + "-" * 42)
    for key in sorted(pairs, key=pairs.get, reverse=True):
        print(f"  {key:>28}{pairs[key]:>14.3f}")
    print()

    record.figures(relations=k, left_out=len(left_out), kinds=kinds, statements=len(triples), draws=DRAWS,
                   the_null=float(drawn.mean()), the_null_at_its_widest=float(drawn.max()),
                   p=float(p),
                   singular_values={str(i + 1): float(got[i]) for i in range(k)},
                   the_null_widest_at={str(i + 1): float(null[:, i].max()) for i in range(k)},
                   how_aligned=pairs, closest_pair=pairs[closest])

    if kinds > drawn.max():
        print(f"  **The store's relations are all but independent as operators.** They occupy")
        print(f"  {kinds:.2f} of a possible {k} directions, above every draw of a null that keeps")
        print("  each pair and each relation's count and destroys only which relation a pair")
        print("  belongs to. **So a reader that cannot choose between them is not failing because")
        print("  there is nothing to choose between** -- §47.111's gap is not this, and §47.83's")
        print("  generated structure, built so the relations plainly differ, is not assuming away")
        print("  something the ontology lacks.")
    elif kinds > drawn.mean():
        print(f"  **Wider than the null and not clear of it.** The store occupies {kinds:.2f} of")
        print(f"  {k} directions against {drawn.mean():.2f} on average and {drawn.max():.2f} at the")
        print(f"  null's widest, p {p:.4f}, so its relations are more separate than relabelling")
        print("  would make them without this store being able to establish by how much.")
    else:
        print(f"  **The store carries fewer kinds than names.** It occupies {kinds:.2f} of {k}")
        print("  directions, no more than relabelling gives, so part of what a reader is asked to")
        print("  choose between is not there to be chosen. **That is a property of the store and a")
        print("  bound on any rule**, in the same family as §47.112 and from a different side.")
    print()
    if pairs[closest] > 0.5:
        print(f"  The closest two are **{closest}** at **{pairs[closest]:.3f}**, which is close")
        print("  enough that a reader would confuse them, and names which two.")
    else:
        print(f"  The closest two are **{closest}** at **{pairs[closest]:.3f}** -- nearer each")
        print("  other than any other pair and still nearly at right angles, so this names the")
        print("  **least** separated pair and not a confusable one.")
    print()
    print("  The figure `ontology.py` quotes is a measurement of the structure this one replaced and")
    print("  its script cannot run against the store that exists now. This is not that figure and")
    print("  does not restate it: it is the same question asked of what is here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
