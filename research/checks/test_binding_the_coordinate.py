"""Can the vertex model have the typed model's expressiveness without a vocabulary?

The remaining gap
-------------------
Scored like for like on the same facts and the same 31 candidates -- with the places rebuilt per
question, so nothing about the held statement survives anywhere -- typed edges stay ahead of the
vertex model, and win every question the two disagree about. Four explanations have been measured;
three were refused by their own controls.

What is left is expressiveness per axis, and it is visible in the two bases:

    typed   one axis per (relation, direction, target)   -- "I have an is_a to a mammal"
    vertex  one axis per (position, vertex)              -- "I am at position 0 of s000"
                                                            "is_a is at position 1 of something"
                                                            "mammal is at position 2 of something"

The typed axis carries the relation and the target together. The vertex place carries them
apart and has to recombine them through the statement, and summing over statements is where the
pairing is lost. That was tested once as a *completion* -- ask the statement for its missing part --
and refused at 0.16. This tests it the other way: leave the read alone and bind the coordinate.

Binding without a vocabulary
------------------------------
For a concept at position ``p`` of a statement, the rest of that statement is a tuple: ``dog`` at 0
has ``(is_a@1, mammal@2)``. Make that one axis.

It is exactly as expressive as the typed axis and it is read off the statements -- the pairs are
whatever the store contains, no list is written down, and a new relation brings its own pairs with
it. The number of distinct pairs is bounded by the number of statements, so the basis stays finite
without anyone choosing a size.

If this reaches the typed score, the vertex structure was never what cost the accuracy; the
coordinate was. If it does not, expressiveness is not the remaining cause either.

One walk, three bases
-----------------------
All three arms are placed by the shipped walk -- `Engine._resolved` and `Engine._row` -- and differ
only in their basis. The bound arm is `Engine.places` itself; typed and unbound are the same two
steps on their own adjacency (`test_both_structures_fairly.shipped_walk`). So a difference between
arms is the basis, and nothing walks here that the engine does not.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pathlib
import sys

import numpy as np

import record

import ontology

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from ontology import Engine                                    # noqa: E402  the shipped engine

import test_both_structures_fairly as fair                    # noqa: E402


def bound_places(statements, *, weighted=False, floored=False):
    """The bound ``(position, rest-of-statement)`` basis: the shipped coordinate itself.

    ``dog`` in "dog is_a mammal" contributes to the axis ``(0, ((1, 'is_a'), (2, 'mammal')))`` --
    one coordinate carrying the relation and the target together, the way the typed basis did, with
    the pair taken from the statement rather than from a declared set of labels. That is the basis
    `Engine.basis` counts off the store, so this arm is `Engine.places` and nothing else; the two
    other arms are the same walk (`test_both_structures_fairly.shipped_walk`) on another basis.
    """
    fair._only_the_shipped_walk(weighted, floored)
    engine = Engine(statements)
    return engine.places(), list(engine.basis())


def main() -> int:
    candidates = fair.answerable(ontology.STATEMENTS)
    print(f"{len(ontology.STATEMENTS)} statements, {len(candidates)} candidates, "
          f"chance 1 in {len(candidates)}\n")

    builds = (("typed", fair.typed_places),
              ("vertex", fair.vertex_places),
              ("vertex, bound", bound_places))

    print(f"{'structure':<16}{'axes':>7}{'blank':>7}{'top-1':>9}{'mean rank':>12}")
    print("-" * 62)
    best = {}
    for name, build in builds:
        _, axes = build(list(ontology.STATEMENTS))
        for blank in (False, True):
            t, r, n = fair.score(build, weighted=False, floored=False,
                                 blank_relation=blank, candidates=candidates)
            print(f"{name:<16}{len(axes):>7}{str(blank):>7}{t:>9.2f}{r:>12.1f}")
            if name not in best or t > best[name][0]:
                best[name] = (t, r)
    print("-" * 62)
    print()
    for name, (t, r) in best.items():
        print(f"  {name:<16} best {t:.2f} top-1, mean rank {r:.1f}")

    typed, plain, bound = best["typed"][0], best["vertex"][0], best["vertex, bound"][0]
    print()
    print(f"  binding the coordinate moves the vertex model {bound - plain:+.2f}, "
          f"leaving {typed - bound:+.2f} to typed")
    print()

    # A difference of means over the questions asked cannot say whether either difference is there, and
    # deciding it against a chosen margin would only be choosing the answer. The questions are the
    # same statements for all three, so the comparison is paired and settles itself.
    import test_is_the_gap_real as paired
    got = {name: paired.outcomes(build, weighted=False, candidates=candidates)
           for name, build in builds}
    vs_plain = paired.mcnemar(got["vertex, bound"], got["vertex"])
    vs_typed = paired.mcnemar(got["vertex, bound"], got["typed"])
    print(f"  paired: bound vs vertex {vs_plain[0]}-{vs_plain[1]}, p {vs_plain[3]:.3f}")
    print(f"          bound vs typed  {vs_typed[0]}-{vs_typed[1]}, p {vs_typed[3]:.3f}")
    print()

    record.figures(
        best={k: {'top1': v[0], 'mean_rank': v[1]} for k, v in best.items()},
        axes={n: len(b(list(ontology.STATEMENTS))[1])
              for n, b in builds},
        paired={'bound vs vertex': dict(zip(('wins', 'losses', 'shared', 'p'), vs_plain)),
                'bound vs typed': dict(zip(('wins', 'losses', 'shared', 'p'), vs_typed))},
    )

    recovers = vs_plain[3] < 0.05 and vs_plain[0] > vs_plain[1]
    matches = vs_typed[3] >= 0.05
    if recovers and matches:
        print("  **The structure was never what cost the accuracy -- the coordinate was.** Pairing")
        print("  a concept's position with the rest of its statement beats the unbound coordinate")
        print("  on the questions they disagree about, and is not separable from the typed axis on")
        print("  the questions it disagrees with that. The pairs are read off the store rather than")
        print("  declared, so the vertex model can be had at no cost.")
    elif recovers:
        print("  Binding beats the unbound coordinate, but typed still separates from it, so")
        print("  expressiveness was part of the cause and not all of it.")
    else:
        print("  **Refused.** Binding does not separate from the unbound coordinate on the paired")
        print("  test, so expressiveness is not the remaining cause either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
