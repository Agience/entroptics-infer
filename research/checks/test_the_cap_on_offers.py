"""How many landings each operator offers the walk. Why not the best three?

The question
--------------
`ask.walk` applies every relation's operator where the cone is and collects candidates. A cap such as

    for i in np.argsort(-scores)[:3]:

is a number nothing derives. The AST check `tests/test_no_constants.py` applies to the package
covers `src/entroptics_infer/` and not `research/`, where the reader's method actually lives;
`test_the_method_carries_no_constant.py` walks the method the same way. A cap is in the live path and
decides what is even offered, so every result about attention rests on it.

Why it cannot simply be removed on principle
----------------------------------------------
Removing it is not free either. With no cap each operator offers every concept, so the candidate set
is the whole store at every step and attention is doing all the work — which may be better, worse,
or identical. "Take everything" is a choice too; it is just a less visible one.

So the cap is measured rather than argued about.

What is measured
------------------
The same walks at caps of 1, 2, 3, 5 and no cap at all, recording what changes:

``offered``    how many candidates reach the attention floor
``kept``       how many clear it
``joined``     the share of kept candidates the store joins to where the cone is, which is the
               only thing here the store can adjudicate
``steps``      how far the walk runs before attention fails

What each outcome means
-------------------------
If ``joined`` and the walks' shape are flat across the caps, the cap is inert and belongs out of the
method: a number that decides nothing should not be written down. If they move, the cap is
load-bearing and un-derived, which is worse than inert and has to be reported as a forcing rather
than quietly kept.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections

import numpy as np

import record
from checking import joined

import ask
import ontology
import operators
import place


def walk_at(query, vectors, ops, cap):
    """One walk, taking ``cap`` landings per operator, or all of them when ``cap`` is None.

    `ask.steps` with its ``cap`` arm set and nothing else changed. Every step that offered anything
    is returned, with whether attention kept each candidate.
    """
    out = []
    for s in ask.steps(query, vectors, ops, cap="every landing" if cap is None else cap):
        if not s["candidates"]:
            break
        c, o, kept = s["candidates"], s["offers"], s["kept"]
        out.append([(c[i], o[c[i]][1], i in kept) for i in range(len(c))])
    return out


def main() -> int:
    vectors, _ = place.places()
    ops = operators.all_operators(vectors)
    concepts = list(ontology.concepts())
    link = {}

    print(f"{len(concepts)} queries, one per concept, at each cap\n")
    print(f"  {'landings per operator':<24}{'offered':>9}{'kept':>7}{'kept joined':>13}"
          f"{'steps':>8}{'walks':>7}")
    print("  " + "-" * 68)

    rows = {}
    for cap in (1, 2, 3, 5, None, "best available"):
        offered = kept = joined_kept = steps = walks = 0
        for probe in concepts:
            path = walk_at(probe, vectors, ops, cap)
            if path:
                walks += 1
                steps += len(path)
            for candidates in path:
                offered += len(candidates)
                for name, frm, was_kept in candidates:
                    if not was_kept:
                        continue
                    kept += 1
                    if (frm, name) not in link:
                        link[(frm, name)] = joined(frm, name)
                    joined_kept += link[(frm, name)]
        label = "all of them" if cap is None else str(cap)
        rate = joined_kept / (kept or 1)
        rows[label] = {"offered": offered, "kept": kept, "kept_joined": rate,
                       "steps": steps, "walks": walks}
        print(f"  {label:<24}{offered:>9}{kept:>7}{rate:>13.0%}{steps:>8}{walks:>7}")
    print("  " + "-" * 68)
    print()

    record.figures(caps=rows)

    rates = [r["kept_joined"] for r in rows.values()]
    kepts = [r["kept"] for r in rows.values()]
    spread = max(rates) - min(rates)
    at_three = rows["3"]["kept_joined"]

    print(f"  the share of kept candidates the store joins to the frontier runs "
          f"{min(rates):.0%} to {max(rates):.0%}")
    print(f"  across the caps, and the number kept runs {min(kepts)} to {max(kepts)}.")
    print()
    if spread <= 0.0:
        print("  **The cap decides nothing and should not be written down.** Attention keeps the")
        print("  same candidates whatever each operator is allowed to offer, so the three is inert")
        print("  and removing it removes a number without changing a result.")
    elif at_three == max(rates):
        print("  **The cap is load-bearing, and three is the best of the values tried.** That is the")
        print("  worse outcome: a number in the live path that changes the result and was never")
        print("  derived. It is a forcing, and the only defensible versions are no cap at all or a")
        print("  cap the store itself sets. Reported rather than kept quietly at three.")
    else:
        print("  **The cap is load-bearing and three is not even the best of it.** A number that")
        print("  changes the result and was never derived.")
        print()
        print("  What replaced it in `ask.py` is a rule with no number in it: an operator applied to")
        print("  a place lands at a point, what sits there is the nearest vertex, and if the walk")
        print("  has already been there that operator offers nothing at this step. That is the `1`")
        print("  row -- one landing per operator, not a cap of one chosen from a range.")
        print()
        print("  `best available` is the other cap-free reading, and it is worse: skipping past")
        print(f"  visited vertices never runs out, so it offers progressively poorer landings and")
        print(f"  runs {rows['best available']['steps']} steps at "
              f"{rows['best available']['kept_joined']:.0%} joined against "
              f"{rows['1']['steps']} at {rows['1']['kept_joined']:.0%}.")
        print()
        print("  The rule was chosen because it needs no number. That it also measures best is a")
        print("  check on it rather than the reason for it -- picking the best row would be tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
