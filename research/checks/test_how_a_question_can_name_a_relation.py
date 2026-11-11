"""The reader must be told which relation acts. This asks how narrow the telling has to be.

Why it matters more than it looks
-----------------------------------
§47.72 measures what happens when a question does not name its relation: the reader's choosing is
worse than picking one at random, and §47.74 shows a correct answer was on the table the whole time.
So on this method, as it stands, a question has to name its relation.

That makes how a question is allowed to name one a load-bearing property rather than a detail.
`ask.land` matches the query's words against the store's own identifiers -- there is no vocabulary,
so there is nothing else it could match against -- and `lives_in` is one token because the alphabet
is read off the store. A question that says `lives_in` names the relation. A question that says
*lives*, or *live*, or *lives in*, may not, and nothing has measured which.

What is measured, and where the variants come from
----------------------------------------------------
Nothing is invented. Each relation's identifier is taken apart mechanically and each piece put
back to the reader:

``as written``     `lives_in` -- the identifier itself.
``the first part`` `lives` -- everything before the first underscore.
``the last part``  `in` -- everything after it.
``spaced``         `lives in` -- the underscore replaced by a space.
``joined``         `livesin` -- the underscore removed.

Every one of those is derived from the store's own identifier by a rule with no judgment in it,
which is the point: what a person would *actually* type is not something this file can know, and
guessing it would be putting an answer into the data. What it can measure is which mechanical
neighbours of the identifier the reader accepts.

For each, whether the relation is among what the query lands on, and whether `ask.acting_in`
selects it as the operator to apply -- which is the thing that decides whether the read is §47.72's
"told" or its "choosing".
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

import ask
import ontology
import operators
import place


def variants(relation):
    """The identifier taken apart by rule, with no judgment about what a person would type."""
    out = {"as written": relation}
    if "_" in relation:
        head, _, tail = relation.partition("_")
        out["the first part"] = head
        out["the last part"] = tail
        out["spaced"] = relation.replace("_", " ")
        out["joined"] = relation.replace("_", "")
    return out


def main() -> int:
    vectors, _ = place.places()
    ops = operators.all_operators(vectors)
    relations = [r for r in operators.relations() if r in ops]
    sources = {}
    for parts in ontology.STATEMENTS:
        if len(parts) == 3 and parts[1] in ops:
            sources.setdefault(parts[1], parts[0])

    forms = ["as written", "the first part", "the last part", "spaced", "joined"]
    print(f"{len(relations)} relations with an operator, each named to the reader five ways\n")
    print(f"  {'relation':<14}{'source':<10}" + "".join(f"{f:>16}" for f in forms))
    print("  " + "-" * 104)

    rows = {}
    for relation in relations:
        src = sources.get(relation)
        if src is None:
            continue
        got = {}
        for name, word in variants(relation).items():
            hits, V = ask.land(f"{src} {word}", vectors)
            named = [c for c, _, _ in hits]
            got[name] = {
                "query": f"{src} {word}",
                "landed_at_all": V is not None,
                "landed_on_the_relation": relation in named,
                "selected_it": relation in ask.acting_in(hits)}
        rows[relation] = got
        print(f"  {relation:<14}{src:<10}"
              + "".join(f"{('yes' if got[f]['selected_it'] else 'no'):>16}"
                        if f in got else f"{'-':>16}" for f in forms))
    print()

    record.figures(naming={f"{r}, {f}": v for r, got in rows.items() for f, v in got.items()})

    by_form = {}
    for f in forms:
        seen = [got[f] for got in rows.values() if f in got]
        if not seen:
            continue
        by_form[f] = {
            "asked": len(seen),
            "selected_it": sum(v["selected_it"] for v in seen) / len(seen),
            "landed_on_the_relation": sum(v["landed_on_the_relation"] for v in seen) / len(seen),
            "landed_at_all": sum(v["landed_at_all"] for v in seen) / len(seen)}
    record.figures(by_form=by_form)

    print(f"  {'how it was named':<18}{'selects the relation':>22}{'lands on it':>14}"
          f"{'lands at all':>14}")
    print("  " + "-" * 70)
    for f in forms:
        if f not in by_form:
            continue
        v = by_form[f]
        print(f"  {f:<18}{v['selected_it']:>22.0%}{v['landed_on_the_relation']:>14.0%}"
              f"{v['landed_at_all']:>14.0%}")
    print()

    exact = by_form["as written"]["selected_it"]
    others = {f: v["selected_it"] for f, v in by_form.items() if f != "as written"}

    if exact == 1.0 and not any(others.values()):
        print("  **Only the identifier itself names the relation.** Every mechanical neighbour of")
        print("  it — the word before the underscore, the word after, the two spaced, the two")
        print("  joined — selects nothing, so the reader is told which relation acts only by a")
        print("  question that spells it exactly as the store does.")
        print()
        print("  That is not a defect to fix here: **there is no vocabulary**, so there is nothing")
        print("  for a near-miss to be matched against, and inventing one would be putting names")
        print("  into a store whose whole claim is that it holds none. What it means is that")
        print("  §47.72's 'told' is a narrow door, and the reader needs the question to come")
        print("  through it — which is a fact about what this method needs from whatever asks it.")
    elif exact == 1.0:
        print("  **The identifier names the relation and some of its neighbours do too**, so the")
        print("  door is wider than the identifier alone and the table above says how much.")
    else:
        print("  **The identifier itself does not always name the relation**, which is a fault")
        print("  rather than a limit: the reader cannot be told what it needs to be told, and the")
        print("  rows above say for which relations.")
    print()
    print("  The variants are the store's own identifiers taken apart by rule. What a person would")
    print("  actually type is not something this file can know, and guessing it would be putting")
    print("  an answer into the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
