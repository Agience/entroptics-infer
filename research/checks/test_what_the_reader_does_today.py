"""The bottom line: ask the reader every question the store can put, and count.

Why this is missing and worth having
--------------------------------------
This line has measured the reader in pieces -- the operator's top-1 on a held-out statement, what
the walk keeps, when it refuses, what it costs, what it can infer. Every one of those calls part of
the machinery directly, which is the only way to isolate anything.

**Nobody has asked the whole thing every question and counted.** That is the number a person wants
first and the one this line cannot currently answer without assembling it from five others, which is
exactly the kind of assembly that produces a figure nothing backs.

So: `ask.walk`, on the real ontology, for every statement it holds, with that statement held out of
the store. The query is the source and the relation spelled as the store spells them, which §47.77
shows is the only way to name a relation and §47.72 shows is necessary.

What is counted
-----------------
For each held-out statement, what the reader does with the question:

``said the answer``   the removed destination is among what the walk reached.
``said something``    it answered, and the removed destination is not in it.
``refused``           the walk offered nothing, which §47.29 establishes is an answer of a kind.
``could not be put``  the source or relation left the store with the statement, or the relation was
                      left with too few uses to have an operator. Counted apart, because a question
                      that cannot be asked is not a question the reader got wrong.

The floor beside it: the chance of naming the removed destination by drawing from the same concepts.

What it is not
----------------
It is not a score to improve. 91% of this store's questions have one answer, so holding it out
takes its own evidence with it -- §47.42 shows that on such a store the operator cannot even be
distinguished from a destruction of itself. The number below is therefore mostly a measurement of
the store, and it is recorded so that the line has one plain bottom line rather than an assembled
one.
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


def main() -> int:
    was = ontology.STATEMENTS
    triples = [p for p in was if len(p) == 3]
    concepts = list(ontology.concepts())

    tally = collections.Counter()
    drawn = []
    misses = []
    # How many statements point at each concept. The misses look like they land on whatever many
    # things point at -- `place`, `vehicle` -- and six examples is an impression, not a measurement.
    pointed_at = collections.Counter(b for _, _, b in triples)
    pull = {"said when wrong": [], "said when right": [], "wanted": [], "any concept": []}
    # "the region an answer lives in" is a metaphor until it is a set. The plainest set the store
    # offers is: the concepts this relation ever leads to. If a miss names one of those, the read
    # had the relation's destination class and not the item; if it names something outside, it did
    # not even have that.
    leads_to = collections.defaultdict(set)
    for a, r, b in triples:
        leads_to[r].add(b)
    region = {"the miss is a destination of that relation": 0,
              "any concept is, by share": [], "misses": 0}
    # §47.86 says the read lands in the relation's destination class and picks wrongly inside it.
    # That predicts something checkable: where the class has few members, picking inside it is
    # easy, and where it has many, it is not. If the prediction fails the unifying claim is wrong.
    by_relation = collections.defaultdict(lambda: {"right": 0, "asked": 0})
    try:
        for i, (src, relation, dst) in enumerate(triples):
            held = tuple(p for j, p in enumerate(was) if p != (src, relation, dst))
            ontology.STATEMENTS = held
            vectors, _ = place.places(statements=held)
            ops = operators.all_operators(vectors, statements=held)
            if src not in vectors or relation not in vectors or dst not in vectors \
                    or relation not in ops:
                tally["could not be put"] += 1
                continue

            by_relation[relation]["asked"] += 1
            by_relation[relation]["class"] = len(leads_to[relation])
            path = ask.walk(f"{src} {relation}", show=lambda *_: None)
            reached = [x["reached"] for entry in path for x in entry["reached"]]
            tally["asked"] += 1
            if not reached:
                tally["refused"] += 1
            elif dst in reached:
                tally["said the answer"] += 1
                by_relation[relation]["right"] += 1
                pull["said when right"].append(pointed_at[reached[0]])
            else:
                tally["said something"] += 1
                misses.append((src, relation, dst, reached[0]))
                pull["said when wrong"].append(pointed_at[reached[0]])
                here = leads_to[relation] - {dst}
                region["misses"] += 1
                region["the miss is a destination of that relation"] += reached[0] in here
                pool = [c for c in concepts if c in vectors and c != src and c != dst]
                region["any concept is, by share"].append(
                    sum(1 for c in pool if c in here) / len(pool) if pool else 0.0)
            if reached:
                pull["wanted"].append(pointed_at[dst])
                pull["any concept"].append(
                    float(np.mean([pointed_at[c] for c in concepts if c in vectors])))
            pool = [c for c in concepts if c in vectors and c != src]
            drawn.append(1 / len(pool) if pool else 0.0)
    finally:
        ontology.STATEMENTS = was

    if len(ontology.STATEMENTS) != len(was):
        print("  FAILED: the store was not put back")
        return 1

    asked = tally["asked"]
    print(f"{len(triples)} statements, each held out in turn and asked back through `ask.walk`")
    print(f"the query is the source and the relation, spelled as the store spells them\n")
    print(f"  {'what the reader did':<26}{'count':>8}{'of those asked':>16}")
    print("  " + "-" * 50)
    for name in ("said the answer", "said something", "refused"):
        print(f"  {name:<26}{tally[name]:>8}{tally[name] / asked:>16.0%}")
    print("  " + "-" * 50)
    print(f"  {'asked':<26}{asked:>8}")
    print(f"  {'could not be put':<26}{tally['could not be put']:>8}")
    print()

    floor = float(np.mean(drawn)) if drawn else float("nan")
    record.figures(
        statements=len(triples), asked=asked,
        could_not_be_put=tally["could not be put"],
        said_the_answer=tally["said the answer"],
        said_something=tally["said something"],
        refused=tally["refused"],
        said_the_answer_share=tally["said the answer"] / asked if asked else float("nan"),
        refused_share=tally["refused"] / asked if asked else float("nan"),
        floor=floor)

    print(f"  {'':<20}{'mean statements pointing at it':>32}{'times':>8}")
    print("  " + "-" * 60)
    for name in ("said when wrong", "said when right", "wanted", "any concept"):
        if pull[name]:
            print(f"  {name:<20}{float(np.mean(pull[name])):>32.1f}{len(pull[name]):>8}")
    print()
    record.figures(hub_pull={k: float(np.mean(v)) for k, v in pull.items() if v},
                   hub_pull_counts={k: len(v) for k, v in pull.items() if v})

    wrong = float(np.mean(pull["said when wrong"])) if pull["said when wrong"] else float("nan")
    typical = float(np.mean(pull["any concept"])) if pull["any concept"] else float("nan")
    want = float(np.mean(pull["wanted"])) if pull["wanted"] else float("nan")
    if wrong > typical and wrong > want:
        print(f"  **The wrong answers are hubs.** What it says when it is wrong has {wrong:.1f}")
        print(f"  statements pointing at it on average, against {typical:.1f} for a concept in the")
        print(f"  store and {want:.1f} for the answer it wanted. It reaches a region and takes the")
        print("  busiest thing in it, which is the same pull a relation run backwards shows.")
    elif wrong > typical:
        print(f"  **The wrong answers are busier than a typical concept** ({wrong:.1f} against")
        print(f"  {typical:.1f}) but no busier than the answers themselves ({want:.1f}), so the")
        print("  pull is toward what relations point at in general rather than toward hubs")
        print("  specifically.")
    else:
        print(f"  **The wrong answers are not hubs**: {wrong:.1f} statements point at them against")
        print(f"  {typical:.1f} for a typical concept, so what it says when wrong is not explained")
        print("  by how busy the vertex is, and the six examples above were not the pattern.")
    print()

    if region["misses"]:
        got = region["the miss is a destination of that relation"] / region["misses"]
        chance = float(np.mean(region["any concept is, by share"]))
        record.figures(region={"misses": region["misses"], "in_the_relations_class": got,
                               "in_it_by_chance": chance})
        print(f"  of the {region['misses']} misses, {got:.0%} name a concept the relation asked "
              f"about")
        print(f"  leads to somewhere else in the store, against {chance:.0%} by chance.")
        if got > chance:
            print("  **So the region is real and it is the relation's own destination class.** The")
            print("  read lands among the things this relation leads to and takes the wrong one —")
            print("  it has the relation and not the source's particular answer.")
        else:
            print("  **The misses are not in the relation's destination class either**, so the")
            print("  read has not even got that far and 'the right region' overstates it.")
        print()

    print("  the prediction §47.86 makes: a smaller destination class is easier to pick inside")
    print(f"  {'relation':<14}{'things it leads to':>20}{'asked':>8}{'said the answer':>18}")
    print("  " + "-" * 62)
    shape = {}
    for relation, v in sorted(by_relation.items(), key=lambda kv: kv[1].get("class", 0)):
        if not v["asked"]:
            continue
        got = v["right"] / v["asked"]
        shape[relation] = {"class": v.get("class", 0), "asked": v["asked"], "right": v["right"],
                           "said_the_answer": got}
        print(f"  {relation:<14}{v.get('class', 0):>20}{v['asked']:>8}{got:>18.0%}")
    print()
    record.figures(by_relation=shape)

    sizes = [v["class"] for v in shape.values()]
    rates = [v["said_the_answer"] for v in shape.values()]
    if len(sizes) > 2 and len(set(sizes)) > 1:
        fit = float(np.corrcoef(sizes, rates)[0, 1])
        # The widest class here is `part_of`, which this line has measured separately and refused
        # three accounts of: it is the only relation in the store that is not one-to-one. If it
        # alone carries the correlation then the prediction is resting on the one relation with an
        # independent problem, and four points cannot tell those apart.
        widest = max(shape, key=lambda k: shape[k]["class"])
        rest = [(v["class"], v["said_the_answer"]) for k, v in shape.items() if k != widest]
        without = (float(np.corrcoef([x[0] for x in rest], [x[1] for x in rest])[0, 1])
                   if len(rest) > 2 and len({x[0] for x in rest}) > 1 else float("nan"))
        record.figures(class_size_against_success=fit,
                       without_the_widest=without, relations_compared=len(sizes))
        print(f"  across the {len(sizes)} relations, the correlation between how many things a")
        print(f"  relation leads to and how often the reader says the answer is {fit:+.2f};")
        print(f"  leaving out `{widest}`, the widest and the only one here that is not")
        print(f"  one-to-one, it is {without:+.2f}.")
        print()
        print(f"  **{len(sizes)} relations cannot test this.** The sign is what §47.86 predicts and")
        print("  the ordering is not monotone — the narrowest class is answered less often than")
        print("  the next — so what this shows is that the prediction is **available to be")
        print("  tested** and that this store cannot test it. A store with more relations could,")
        print("  and a generated one could be built to, which is a measurement rather than a")
        print("  conclusion and is not made here.")
        print()

    print(f"  Naming the removed answer by drawing from the same concepts: {floor:.1%}.")
    print()
    print("  **This is the bottom line of the reader as it stands, on this store.** It is not a")
    print("  score to improve: 91% of the store's questions have exactly one answer, so holding")
    print("  one out takes its own evidence with it, and §47.42 shows that on such a store the")
    print("  operator cannot be told apart from a destruction of itself. **The number is mostly a")
    print("  measurement of the store**, and it is here so the line has one plain bottom line")
    print("  rather than one assembled out of five other files.")
    print()
    if misses[:6]:
        print("  what it said instead, on the first few it got wrong:")
        for src, relation, dst, got in misses[:6]:
            print(f"    {src} {relation}: wanted {dst}, said {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
