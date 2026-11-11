"""The inference test: a store that states only part of a structure, and a read for the rest.

The gap this is built to cross
--------------------------------
Two results between them close every other route tried. On a store that says things twice the
operator works and a chain is measurable, but the endpoint set is *defined* as what the stored
statements reach, so a hit is a tautology and nothing is inferred. On a store that says everything
once the question is genuinely inferential, and there the operator cannot be told apart from a
destruction of itself, so nothing is measurable. Measurable or inferential, never both.

The reason is that in both, truth is whatever the store says. So this changes what truth is.

The structure, and what the store is allowed to know
------------------------------------------------------
Concepts are drawn from ``G`` hidden groups. A relation is a map from group to group, drawn once per
relation. The true edges are then fixed and total: ``a`` stands in relation ``r`` to every
concept in the group ``r`` sends ``a``'s group to. Nothing about any of this is written into the
store.

The store is given a random fraction of those true edges and nothing else. It is a partial
observation of a structure that exists independently of it, and the fraction is swept.

The question, which the store cannot answer by lookup
-------------------------------------------------------
For a concept ``a`` and a relation ``r``, the read is scored only over concepts the store never
gives as a destination of ``(a, r)``. Every candidate it may return is therefore a pair the store
does not state. The answer is right when the concept it returns is in the group the structure says
``r`` sends ``a`` to.

That is inference and not retrieval, and the difference from everything before it is that the
grader is the generator rather than the store. A reader that can only return what it was told scores
at the floor here by construction, however well it retrieves.

What would make a hit mean nothing
------------------------------------
``re-paired``   each relation's stored sources and destinations re-paired before the operator is
                formed. Same coordinate, same statements, same counts. A read that survives this is
                the geometry.
``the floor``   naming one of the same candidates at random. It is exactly the share of unstated
                concepts that happen to be in the right group, so it is what a reader that knows
                nothing scores, and it is reported in every row.

No statement is held out per question, because none needs to be: the pairs being asked about were
never in the store. What is held out is most of the structure.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections

import numpy as np

import record
from draw import observed

import ask
import operators
import place

from measure import mcnemar
from operators import operators_over as _ops  # noqa: E402

GROUPS = 6
PER_GROUP = 6
RELATIONS = 3
DRAWS = 5
#: what share of the true edges the store is given. Swept; every share is reported.
SHARES = (0.2, 0.4, 0.6, 0.8)


def structure(rng):
    """``(concepts, relations, group of each concept, where each relation sends each group)``."""
    concepts = [f"c{i}" for i in range(GROUPS * PER_GROUP)]
    group = {c: i // PER_GROUP for i, c in enumerate(concepts)}
    relations = [f"r{j}" for j in range(RELATIONS)]
    sends = {r: {g: int(rng.integers(GROUPS)) for g in range(GROUPS)} for r in relations}
    return concepts, relations, group, sends


def main() -> int:
    print(f"{GROUPS} hidden groups of {PER_GROUP}, {RELATIONS} relations, each a map from group to "
          f"group")
    print(f"the store is given a share of the true edges and nothing else; the read is scored only "
          f"over\npairs the store does not state, and graded by the structure\n")
    print(f"  {'share'.rjust(7)}{'statements':>12}{'asked':>8}{'the read':>11}{'re-paired':>12}"
          f"{'the floor':>11}")
    print("  " + "-" * 61)

    rows: dict = {}
    for share in SHARES:
        real_got, fake_got, floors, sizes = {}, {}, [], []
        for seed in range(DRAWS):
            rng = np.random.default_rng(7000 + seed)
            concepts, relations, group, sends = structure(rng)
            statements = observed(concepts, relations, group, sends, share, rng)
            sizes.append(len(statements))
            stated: dict = collections.defaultdict(set)
            for a, r, b in statements:
                stated[(a, r)].add(b)

            vectors, _ = place.places(statements=statements)
            real = _ops(vectors, statements)
            fake = _ops(vectors, statements, rng=np.random.default_rng(seed))
            present = [c for c in concepts if c in vectors]
            at = {c: i for i, c in enumerate(present)}
            allp = np.stack([vectors[c] for c in present])

            for a in concepts:
                if a not in vectors:
                    continue
                for r in relations:
                    if r not in real or r not in fake:
                        continue
                    # only pairs the store does not state, so nothing on offer can be looked up
                    unstated = [c for c in present if c not in stated[(a, r)] and c != a]
                    if not unstated:
                        continue
                    want = {c for c in unstated if group[c] == sends[r][group[a]]}
                    if not want or len(want) == len(unstated):
                        continue          # no question: every candidate right, or none is
                    idx = [at[c] for c in unstated]
                    key = (seed, a, r)

                    for ops, into in ((real, real_got), (fake, fake_got)):
                        carried = operators.carry(vectors, r, a, ops)
                        if carried is None:
                            continue
                        scores = ask._along(allp, carried)[idx]
                        into[key] = unstated[int(np.argmax(scores))] in want
                    floors.append(len(want) / len(unstated))

        # the share is recorded as a value and not only as the key it is filed under: a claim
        # quotes "20% of the true edges observed", and a gate can only check a figure it can see
        rows[share] = {"share": float(share),
                       "statements": float(np.mean(sizes)), "asked": len(real_got),
                       "read": float(np.mean(list(real_got.values()))),
                       "repaired": float(np.mean(list(fake_got.values()))),
                       "floor": float(np.mean(floors)), "_real": real_got, "_fake": fake_got}
        r = rows[share]
        print(f"  {share:>7.0%}{r['statements']:>12.0f}{r['asked']:>8}{r['read']:>11.0%}"
              f"{r['repaired']:>12.0%}{r['floor']:>11.0%}")
    print()

    print(f"  {'share'.rjust(7)}{'the read wins':>16}{'the re-pairing wins':>22}{'p':>9}")
    print("  " + "-" * 55)
    paired = {}
    for share in SHARES:
        w, l, n, p = mcnemar(rows[share]["_real"], rows[share]["_fake"])
        paired[f"{share:.1f}"] = {"wins": w, "losses": l, "questions": n, "p": p}
        print(f"  {share:>7.0%}{w:>16}{l:>22}{p:>9.3f}")
    print()

    record.figures(
        shares={f"{s:.1f}": {k: v for k, v in r.items() if not k.startswith("_")}
                for s, r in rows.items()},
        shares_paired=paired)

    over_floor = [s for s in SHARES if rows[s]["read"] > rows[s]["floor"]]
    over_null = [s for s in SHARES if paired[f"{s:.1f}"]["wins"] > paired[f"{s:.1f}"]["losses"]]

    if len(over_floor) == len(SHARES) and len(over_null) == len(SHARES):
        print("  **The read reaches pairs the store does not state, at every share.** Every")
        print("  candidate offered is a pair the store is silent about, the grader is the structure")
        print("  rather than the store, and the read beats both the floor and a re-pairing of its")
        print("  own operators on the same questions. **Nothing was trained and nothing was fitted:")
        print("  the operator is a one-pass association over the statements the store happens to")
        print("  hold, formed at the moment of the question.**")
    elif over_floor and over_null:
        print(f"  **It reaches them at some shares and not others**: above both floor and null at")
        print(f"  {[f'{s:.0%}' for s in over_null]} and not at the rest. How much of the structure")
        print("  the store has to hold before the rest can be read is then the quantity, and it is")
        print("  the first thing here that a real ontology could be checked against.")
    elif over_floor and not over_null:
        print("  **The read clears the floor but not its own re-pairing**, so what carries it is the")
        print("  coordinate and not which source went with which destination. That is not")
        print("  inference; it is the geometry of a store whose groups sit apart.")
    else:
        print("  **The read does not reach them.** On pairs the store does not state, it scores at")
        print("  the floor, so the operator generalizes to nothing it was not given and the method")
        print("  retrieves rather than infers on this structure.")
    print()
    print("  The structure is synthesized and the generator is stated above. What it settles is")
    print("  whether the method can reach an unstated pair at all, which no store in this line has")
    print("  been able to ask.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
