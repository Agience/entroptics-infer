"""Does the bound coordinate score by inference, or by having seen the answer?

The worry
-----------
Built once, from every statement, holding out only the operator, the bound coordinate scores
about as well as typed edges on fewer axes. For the typed basis that is survivable: the axis ``(is_a, out, mammal)`` is shared by nine
statements, so removing one leaves it standing.

The bound basis is different. ``dog`` at position 0 of "dog is_a mammal" gets the axis
``(0, ((1,'is_a'), (2,'mammal')))``, and almost nothing else in this store occupies it. So ``dog``'s
place may be carrying *the very statement being held out*, and the read would be recalling rather
than inferring.

If that is what is happening, the built-once score is leakage and any conclusion drawn from it --
that the structure is not the cost -- is wrong.

The test
----------
Rebuild the places without the held-out statement, for every statement in turn, in all three
structures. Nothing about the removed statement is anywhere: not in the coordinate, not in the
operator, not in the basis.

Slow, because it is one rebuild per statement per structure rather than one, and that is the price
of the question being worth asking. Every structure is placed by the shipped walk on its own basis.

What each outcome means
-------------------------
If the bound coordinate holds its score, it infers, and the built-once comparison means what it
says. If it falls to the unbound model's level, the built-once score was reading its own answer back
and the gap between the two structures is open. If every structure falls, the leak is in all of them
and no built-once comparison means what it says.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

from ontology import Engine                                    # the shipped engine

import record

import ontology
import test_binding_the_coordinate as binding
import test_both_structures_fairly as fair
from measure import along as _along  # noqa: E402,F401


def score_rebuilt(build, *, weighted, candidates, rebuild):
    """Top-1 and mean rank, rebuilding the places per held-out statement when asked to."""
    from entroptics_infer.bind import association_matrix

    all_statements = list(ontology.STATEMENTS)
    whole, whole_axes = build(all_statements, weighted=weighted, floored=False)

    by_relation: dict = {}
    for i, (a, r, b) in enumerate(fair.triples(all_statements)):
        by_relation.setdefault(r, []).append((i, a, b))

    top1, ranks, skipped = [], [], 0
    for relation, items in by_relation.items():
        if len(items) < 3:
            continue
        for si, src, dst in items:
            held = [p for j, p in enumerate(all_statements) if j != si]
            if rebuild:
                vectors, _ = build(held, weighted=weighted, floored=False)
            else:
                vectors = whole
            if src not in vectors or dst not in vectors:
                skipped += 1
                continue
            present = [c for c in candidates if c in vectors]
            index = {c: i for i, c in enumerate(present)}
            if dst not in index:
                skipped += 1
                continue
            allp = np.stack([vectors[c] for c in present])

            pairs = [(a, b) for a, r, b in fair.triples(held) if r == relation]
            if len(pairs) < 2:
                skipped += 1
                continue
            A = np.stack([vectors[a] for a, _ in pairs])
            B = np.stack([vectors[b] for _, b in pairs])
            v = vectors[src]
            scores = _along(allp, Engine._land(v, A, B))
            if src in index:
                scores[index[src]] = -np.inf
            rank = int(np.where(np.argsort(-scores) == index[dst])[0][0]) + 1
            top1.append(rank == 1)
            ranks.append(rank)
    return float(np.mean(top1)), float(np.mean(ranks)), len(ranks), skipped


def main() -> int:
    candidates = fair.answerable(ontology.STATEMENTS)
    # every structure on the shipped walk, which carries no weight
    builds = (("typed", fair.typed_places, False),
              ("vertex", fair.vertex_places, False),
              ("vertex, bound", binding.bound_places, False))

    print(f"{len(ontology.STATEMENTS)} statements, {len(candidates)} candidates, "
          f"chance 1 in {len(candidates)}\n")
    print(f"{'structure':<16}{'places':>10}{'top-1':>9}{'mean rank':>12}{'asked':>7}{'skipped':>9}")
    print("-" * 65)

    results: dict = {}
    asked: dict = {}
    for name, build, weighted in builds:
        for rebuild in (False, True):
            t, r, n, sk = score_rebuilt(build, weighted=weighted,
                                        candidates=candidates, rebuild=rebuild)
            label = "rebuilt" if rebuild else "built once"
            print(f"{name:<16}{label:>10}{t:>9.2f}{r:>12.1f}{n:>7}{sk:>9}")
            results[(name, rebuild)] = t
            asked[f"{name}, {label}"] = n
    print("-" * 65)
    print()

    for name, _, _ in builds:
        once, rebuilt = results[(name, False)], results[(name, True)]
        print(f"  {name:<16} built once {once:.2f} -> rebuilt {rebuilt:.2f}  ({rebuilt - once:+.2f})")
    print()

    # the two arms ask different numbers of questions -- a rebuilt store can no longer pose some --
    # so a built-once score and a rebuilt one are not a paired comparison, and the counts say so
    record.figures(top1={f"{n}, {'rebuilt' if r else 'built once'}": v
                        for (n, r), v in results.items()},
                   asked=asked)

    bound = results[("vertex, bound", True)]
    typed = results[("typed", True)]
    plain = results[("vertex", True)]
    drop = results[("vertex, bound", False)] - bound
    if drop > 0.10:
        print(f"  **The bound coordinate was reading its own answer.** It loses {drop:.2f} once the")
        print("  held-out statement is kept out of the places too, so the built-once score is")
        print("  leakage and no conclusion can be drawn from it.")
    else:
        print(f"  The bound coordinate holds up with the statement kept out of the places entirely")
        print(f"  ({drop:+.2f}), so it was inferring. ")
    print(f"  Rebuilt, like for like: typed {typed:.2f}, vertex {plain:.2f}, bound {bound:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
