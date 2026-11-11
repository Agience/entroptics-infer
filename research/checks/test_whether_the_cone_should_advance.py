"""The loop says advance the cone. The reader computes the advance and throws it away.

What is in the code
---------------------
`ask.walk` holds two directions. ``Q0`` is where the query landed, fixed for the whole walk.
``V`` starts equal to it and is updated at the end of every step::

    V = _unit(V + np.mean([vectors[r["reached"]] for r in reached], axis=0))

**``V`` is never read after that.** Every candidate is scored against ``Q0``. Deleting the line
entirely changes no answer -- checked by replacing it with ``del V`` and running the reader, which
raises nothing.

So the loop this work is built on --

    Get a vector V from the query Q. Place the light cone in that direction. Collect artifacts K and
    put them on the screen with the query. Advance the light cone and update V from the new Q.
    Repeat until there is no more attention.

-- does not advance the cone. The direction is set once and never moves, and a vestigial line makes
it look otherwise.

Why it is not simply a bug
----------------------------
The case against scoring against the frontier -- where the cone is now -- is that it measures
itself: scoring a candidate against where you just arrived rewards whatever is near where you just
arrived, and the walk never ends.

An accumulated ``V`` is not that. It is the query plus everything the walk has reached, which is
what the loop above describes: the query is still in it, so the walk cannot drift free of what was
asked, but what has been found is in it too. That is a different reference, and this is where it is
tested.

The three references
----------------------
``the query``     score against ``Q0``. The reader as it stands.
``the frontier``  score against where the cone is now. The version already refused, included so the
                  refusal is visible beside the others rather than only asserted.
``accumulated``   score against the query plus what has been reached, normalized. The loop's own
                  version.

What is measured
------------------
The same two populations as everywhere in this line: every concept crossed with every relation,
split by whether the store has ever seen that concept as a source of that relation. How often the
walk runs, how far it goes, and how much of what it keeps the store joins to where the cone was.

What each outcome means
-------------------------
If the accumulated reference walks further while keeping connected steps, the reader is leaving
something on the table and the loop was right. If it drifts like the frontier does, the fixed query
is doing necessary work and the loop's step needs restating rather than implementing.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections

import numpy as np

import record

import ask
import ontology
import operators
import place


def walk_with(reference, query, vectors, ops, names, allp):
    """`ask.steps`, with what candidates are scored against swapped and nothing else changed."""
    path = []
    for s in ask.steps(query, vectors, ops, names=names, allp=allp, ceiling=len(names),
                       reference=reference):
        if not s["kept"]:
            break
        path.append([(s["candidates"][i], s["offers"][s["candidates"][i]][1]) for i in s["kept"]])
    return path


def main() -> int:
    concepts = list(ontology.concepts())
    relations = list(operators.relations())
    sources = collections.defaultdict(set)
    for p in ontology.STATEMENTS:
        if len(p) == 3:
            sources[p[1]].add(p[0])

    vectors, _ = place.places()
    ops = operators.all_operators(vectors)
    names = list(vectors)
    allp = np.stack([vectors[c] for c in names])
    link: dict = {}

    def joined(a, b):
        if (a, b) not in link:
            link[(a, b)] = any(a in q and b in q for q in ontology.STATEMENTS)
        return link[(a, b)]

    references = ("the query", "the frontier", "accumulated")
    rows: dict = {}
    print(f"{len(concepts) - len(relations)} concepts x {len(relations)} relations, "
          f"scored against each reference\n")
    print(f"  {'scored against':<16}{'group':<11}{'the walk ran':>14}{'steps':>8}"
          f"{'reached':>9}{'joined':>9}")
    print("  " + "-" * 68)

    for reference in references:
        for group in ("has it", "has none"):
            rows[(reference, group)] = {"ran": [], "steps": [], "reached": 0, "joined": 0}
        for concept in concepts:
            if concept in relations:
                continue
            for relation in relations:
                group = "has it" if concept in sources[relation] else "has none"
                path = walk_with(reference, f"{concept} {relation}", vectors, ops, names, allp)
                r = rows[(reference, group)]
                r["ran"].append(len(path) > 0)
                r["steps"].append(len(path))
                for step in path:
                    for c, src in step:
                        r["reached"] += 1
                        r["joined"] += joined(src, c)
        for group in ("has it", "has none"):
            r = rows[(reference, group)]
            share = r["joined"] / r["reached"] if r["reached"] else float("nan")
            print(f"  {reference:<16}{group:<11}{np.mean(r['ran']):>14.0%}"
                  f"{np.mean(r['steps']):>8.2f}{r['reached']:>9}{share:>9.0%}")
        print()

    record.figures(references={
        f"{ref}, {g}": {"ran": float(np.mean(r["ran"])), "steps": float(np.mean(r["steps"])),
                        "reached": r["reached"],
                        "joined": r["joined"] / r["reached"] if r["reached"] else float("nan")}
        for (ref, g), r in rows.items()})

    def of(ref, group, key):
        r = rows[(ref, group)]
        if key == "joined":
            return r["joined"] / r["reached"] if r["reached"] else float("nan")
        return float(np.mean(r[key]))

    print("  the two things that matter: does it separate a question the store can speak to from")
    print("  one it cannot, and is what it keeps connected")
    print(f"    {'reference':<16}{'gap in whether it ran':>24}{'joined, has it':>17}"
          f"{'steps, has it':>16}")
    for ref in references:
        print(f"    {ref:<16}{of(ref, 'has it', 'ran') - of(ref, 'has none', 'ran'):>+24.0%}"
              f"{of(ref, 'has it', 'joined'):>17.0%}{of(ref, 'has it', 'steps'):>16.2f}")
    print()

    q_gap = of("the query", "has it", "ran") - of("the query", "has none", "ran")
    a_gap = of("accumulated", "has it", "ran") - of("accumulated", "has none", "ran")
    f_gap = of("the frontier", "has it", "ran") - of("the frontier", "has none", "ran")

    print(f"  the frontier reads {f_gap:+.0%}, which is the refusal `ask.py` already states, now")
    print("  measured rather than asserted.")
    print()
    if a_gap > q_gap and of("accumulated", "has it", "joined") >= of("the query", "has it",
                                                                    "joined"):
        print("  **The loop was right and the reader is not doing it.** Advancing the cone")
        print(f"  separates the populations better ({a_gap:+.0%} against {q_gap:+.0%}) and keeps")
        print("  steps at least as connected, so the vestigial line should be read rather than")
        print("  deleted.")
    elif abs(a_gap - q_gap) < 0.02:
        print("  **Advancing the cone changes almost nothing here.** The accumulated reference and")
        print("  the fixed query separate the populations alike, so on this store the loop's step")
        print("  is not load-bearing -- and the line that computes it should either be read or")
        print("  removed, because a line that is neither is a claim the code does not keep.")
    else:
        print(f"  **The fixed query is doing necessary work.** Accumulating drifts: {a_gap:+.0%}")
        print(f"  against {q_gap:+.0%}, in the direction the frontier drifts, which is what")
        print("  `ask.py` refused it for. The loop's step wants restating rather than implementing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
