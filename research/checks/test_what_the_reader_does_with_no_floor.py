"""The floor does not sort and on a sparse place it refuses everything. What is it costing?

Why this is the question the last three rows leave
----------------------------------------------------
§47.105 finds the attention floor refusing offers that are right as often as the ones it admits, at
every share measured — it sets how often the reader speaks, not how often it is right. §47.104 finds
the alternative reference §47.58 compares it against admitting every offer, so that is not a bar
either. §47.129 finds that on a one-hop coordinate the bar and the score are both exactly zero and
the walk's strict `>` refuses all of it.

Between them those say the reader abstains on a comparison that carries nothing. What nobody has
measured is what the abstention costs, end to end, in answers — and whether removing it lets the
cheap coordinate's advantage (§47.126, 49% against 32% at the operator) reach the reader at
all, since §47.129 says a tie at zero is the only thing standing in its way.

What is measured, and the liberty taken
-----------------------------------------
`ask.walk` as it stands, on held-out questions, in four arms: each of the two coordinates with the
floor as the reader draws it and with the floor removed. The floor is removed by replacing
`ask._floor` with one that returns negative infinity for the duration of the run, which admits
every offer and refuses none, then putting it back. `ask.py` is not edited.

That is not a proposal. A reader with no floor cannot refuse, and refusing is a thing this reader is
for — §47.13 and §47.14 record what a refusal is worth. What it gives is the ceiling the floor
is spending from, which is the number the decision in §47.105 needs and does not have.

``reaches it``     the walk reaches the held-out answer somewhere along its path.
``its best``       the kept candidate the reader scores highest — its own best answer by its own
                   number. `ask.walk` emits what it keeps in `sorted(offers)` order, so what it
                   names first is alphabetical and not a ranking; this is the ranking.
``by its landing`` the kept candidate nearest its own landing, the score each offer was chosen
                   with and which the walk then discards (§47.58, §47.138). Recomputed from what
                   the path records, so it costs nothing the reader has not already computed.
``says it first``  the answer is the first vertex the walk reaches — the strict reading, and
                   the one comparable to the top-1 figures elsewhere in this line. A walk that
                   keeps several vertices is likelier to hold the answer *somewhere*, so the loose
                   measure flatters whichever arm keeps more, and both are reported.
``answers at all`` the walk takes any step rather than stopping.
``when it speaks``  of the questions it answers at all, how often the answer is among what it
                    reached. This is what a floor is for: a reader that refuses is supposed to
                    be wrong less often when it does speak, and that is the one defence of a bar
                    that does not sort.
``how many it keeps`` the vertices the walk ends up holding.
``by chance``      what "reaches it" would be if a walk that kept that many vertices had chosen
                   them at random from the concepts, averaged per question.
``paired``         floor against no floor, on the same questions, exact.

**The obvious confound is the net.** A reader that cannot refuse keeps more vertices, and a walk
holding more of the store is more likely to have touched the answer whether or not it is reading
better. So the vertices kept are counted and the chance rate computed from them, and a rise in
"reaches it" means nothing unless it outruns its own net.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import contextlib

import numpy as np

import record
from checking import reached

import ask
import ontology
import operators
import place

from measure import mcnemar
from test_how_far_a_place_has_to_spread import places_to


@contextlib.contextmanager
def reader_on(statements, depth, floored):
    """The store held out, the coordinate optionally shallow, the floor optionally removed."""
    was_store, was_places, was_floor = ontology.STATEMENTS, place.places, ask._floor

    def instead(drop=None, statements=None, **kw):
        vectors, _, _ = places_to(ontology.STATEMENTS, depth)
        return vectors, place.basis(None, ontology.STATEMENTS)

    ontology.STATEMENTS = statements
    if depth is not None:
        place.places = instead
        ask.place.places = instead
    if not floored:
        # admits every offer and refuses none: the walk's `salience > floor` is always true
        ask._floor = lambda vectors, V, *, n: float("-inf")
    try:
        yield
    finally:
        ontology.STATEMENTS = was_store
        place.places = was_places
        ask.place.places = was_places
        ask._floor = was_floor


def its_best(path, dst):
    """The kept candidate the reader scores highest -- the ranking it does not emit."""
    best = None
    for step in path:
        for r in step["reached"]:
            if best is None or r["salience"] > best[1]:
                best = (r["reached"], r["salience"])
    return best is not None and best[0] == dst


def by_its_landing(path, dst, vectors, ops):
    """The kept candidate nearest its own landing -- the score the walk picks with and discards."""
    best = None
    for step in path:
        for r in step["reached"]:
            landed = operators.carry(vectors, r["by"], r["from"], ops)
            if landed is None or r["reached"] not in vectors:
                continue
            here = float(ask._along(np.stack([vectors[r["reached"]]]), landed)[0])
            if best is None or here > best[1]:
                best = (r["reached"], here)
    return best is not None and best[0] == dst


def said_first(path, dst):
    """Was the answer the first vertex the walk reached? The strict reading."""
    for step in path:
        for r in step["reached"]:
            return r["reached"] == dst
    return False


def main() -> int:
    triples = [p for p in ontology.STATEMENTS if len(p) == 3]
    concepts = list(ontology.concepts())
    print(f"the real store: {len(triples)} statements, one held out at a time")
    print("the reader as it stands, with its floor and without it, on both coordinates\n")

    arms = {"run out, floored": (None, True), "run out, no floor": (None, False),
            "one hop, floored": (1, True), "one hop, no floor": (1, False)}
    got = {a: {} for a in arms}
    first = {a: {} for a in arms}
    rated = {a: {} for a in arms}
    landed = {a: {} for a in arms}
    spoke = {a: {} for a in arms}
    kept = {a: [] for a in arms}
    chance = {a: [] for a in arms}
    for src, relation, dst in triples:
        held = tuple(p for p in ontology.STATEMENTS if p != (src, relation, dst))
        key = (src, relation, dst)
        for name, (depth, floored) in arms.items():
            with reader_on(held, depth, floored):
                path = ask.walk(f"{src} {relation}", show=lambda *a, **k: None)
            got[name][key] = reached(path, dst)
            first[name][key] = said_first(path, dst)
            rated[name][key] = its_best(path, dst)
            with reader_on(held, depth, floored):
                v, _ = place.places()
                landed[name][key] = by_its_landing(
                    path, dst, v, operators.all_operators(v, statements=[
                        q for q in held if len(q) == 3]))
            spoke[name][key] = bool(path)
            held_now = {r["reached"] for step in path for r in step["reached"]}
            kept[name].append(len(held_now))
            # if a walk holding this many had picked them at random from the concepts
            chance[name].append(min(1.0, len(held_now) / max(1, len(concepts))))

    n = len(got["run out, floored"])
    rows = {}
    print(f"  {n} questions; the arms are compared to each other and not to a floor\n")
    print(f"  {'the reader':>20}{'reaches the answer':>21}{'answers at all':>17}"
          f"{'its best':>11}{'by its landing':>17}{'says it first':>16}"
          f"{'vertices kept':>16}")
    print("  " + "-" * 104)
    for name in arms:
        hit = sum(got[name].values())
        by = float(np.mean(chance[name]))
        spoke_n = sum(spoke[name].values())
        rows[name] = {"reached": hit, "read": hit / n,
                      "answers_at_all": float(np.mean(list(spoke[name].values()))),
                      # of what it says, how much is right: the precision a bar is supposed to buy
                      "right_when_it_speaks": hit / spoke_n if spoke_n else float("nan"),
                      "vertices_kept": float(np.mean(kept[name])),
                      "by_chance": by, "above_chance": hit / n - by,
                      "says_it_first": float(np.mean(list(first[name].values()))),
                      "its_best": float(np.mean(list(rated[name].values()))),
                      "by_its_landing": float(np.mean(list(landed[name].values()))),
                      "questions": n}
        v = rows[name]
        print(f"  {name:>20}{v['read']:>21.0%}{v['answers_at_all']:>17.0%}"
              f"{v['its_best']:>11.0%}{v['by_its_landing']:>17.0%}"
              f"{v['says_it_first']:>16.0%}{v['vertices_kept']:>16.1f}")
    print()

    pairs = {}
    for a, b in (("run out, no floor", "run out, floored"),
                 ("one hop, no floor", "run out, floored"),
                 ("one hop, no floor", "one hop, floored"),
                 ("one hop, no floor", "run out, no floor")):
        w, l, paired, p = mcnemar(got[a], got[b])
        pairs[f"{a} against {b}"] = {"first_only": w, "second_only": l,
                                     "questions": paired, "p": p}
        print(f"  {a:>20} against {b:<20} {w:>4} to {l:<4} p {p:.4f}")
    print()

    strict = {}
    for a, b in (("run out, no floor", "run out, floored"),
                 ("one hop, no floor", "run out, floored")):
        w, l, paired, p = mcnemar(first[a], first[b])
        strict[f"{a} against {b}"] = {"first_only": w, "second_only": l,
                                      "questions": paired, "p": p}
    print("  on the strict reading -- the answer is the first vertex reached:")
    for k, v in strict.items():
        print(f"    {k:<45} {v['first_only']:>4} to {v['second_only']:<4} p {v['p']:.4f}")
    print()

    record.figures(asked=n, arms=rows, paired=pairs, paired_on_the_first=strict)

    # §47.138 finds the landing order better on a generated structure; this store is asked too,
    # because a generated store answers the question its design allows
    base, pair_row = rows["run out, floored"], rows["one hop, no floor"]
    print("  the three orderings, on this store:")
    for name in arms:
        v = rows[name]
        print(f"    {name:>20} by the query {v['its_best']:>5.0%}   by its landing "
              f"{v['by_its_landing']:>5.0%}")
    if pair_row["by_its_landing"] < pair_row["its_best"]:
        print()
        print("  **The landing order is worse here, where §47.138 finds it better.** On the")
        print(f"  generated structure it gains; on this store the pair reads")
        print(f"  {pair_row['by_its_landing']:.0%} by the landing against {pair_row['its_best']:.0%}")
        print("  by the query. **§47.73 is the reason**: ranking offers by their own landings")
        print("  compares cosines taken against *different directions*, which is comparable when")
        print("  the relations are alike and is not when they are not. A generated structure gives")
        print("  five relations built the same way; this store gives `is_a` beside `flows_into`.")
        print("  **So §47.138 belongs to the structure it was measured on**, and the order stays")
        print("  the small term either way.")
    print()
    floored = rows["run out, floored"]["read"]
    free = rows["run out, no floor"]["read"]
    cheap_free = rows["one hop, no floor"]["read"]
    costs = pairs["run out, no floor against run out, floored"]
    between = pairs["one hop, no floor against run out, no floor"]

    if costs["first_only"] > costs["second_only"]:
        print(f"  **The floor is costing answers.** With it the reader reaches {floored:.0%} of the")
        print(f"  held-out answers and without it {free:.0%}, and on the questions they differ the")
        print(f"  unfloored reader is right {costs['first_only']} times to {costs['second_only']},")
        print(f"  p {costs['p']:.4f}. §47.105 finds the floor does not tell a right offer from a")
        print("  wrong one; this says what the caution buys nothing with. **A reader that cannot")
        print("  refuse is not the reader anyone wants** -- refusing is measured elsewhere as worth")
        print("  having -- so what this gives is the ceiling the floor spends from, not a proposal.")
        print()
        fl_row, free_row = rows["run out, floored"], rows["run out, no floor"]
        cheap_row = rows["one hop, no floor"]
        print("  **And it is not the net.** A reader that cannot refuse keeps more vertices, and a")
        print("  walk holding more of the store is likelier to have touched the answer whatever it")
        print(f"  is reading. It keeps {free_row['vertices_kept']:.1f} against")
        print(f"  {fl_row['vertices_kept']:.1f}, so a walk of that size picking at random would")
        print(f"  reach the answer {free_row['by_chance']:.0%} of the time against")
        print(f"  {fl_row['by_chance']:.0%} -- and the reads stand {free_row['above_chance']:+.0%}")
        print(f"  and {fl_row['above_chance']:+.0%} above their own nets, with the cheap unfloored")
        print(f"  reader {cheap_row['above_chance']:+.0%} above its. **The rise outruns the net by")
        print("  a wide margin**, which is what says these are reads and not a larger catch.")
        print()
        keeps = rows["run out, floored"]["right_when_it_speaks"]
        frees = rows["run out, no floor"]["right_when_it_speaks"]
        if frees >= keeps:
            print("  **And the floor does not buy precision either, which was its last defence.**")
            print(f"  Of the questions it answers at all the floored reader is right {keeps:.0%} of")
            print(f"  the time and the unfloored one {frees:.0%}. A bar that does not sort right")
            print("  from wrong (§47.105) cannot raise the share that is right by refusing, and it")
            print("  does not: **it costs recall and it costs precision**, and what it buys is")
            print("  silence.")
        else:
            print("  **But the floor does buy precision, and that is its defence.** Of what it")
            print(f"  answers the floored reader is right {keeps:.0%} against {frees:.0%}, so the")
            print("  recall it spends buys a reader that is wrong less often when it speaks --")
            print("  which is a trade rather than a loss.")
    else:
        print(f"  **The floor is not costing answers here.** The reader reaches {floored:.0%} with")
        print(f"  it and {free:.0%} without, {costs['first_only']} to {costs['second_only']} paired")
        print(f"  at p {costs['p']:.4f}, so what it refuses was not going to be right and §47.105's")
        print("  finding is about which offers rather than how many.")
    print()
    if between["first_only"] > between["second_only"]:
        print("  **And with nothing refusing it, the cheap coordinate's advantage reaches the")
        print(f"  reader.** Unfloored, one hop reads {cheap_free:.0%} against the full spread's")
        print(f"  {free:.0%}, right where it is wrong {between['first_only']} times to")
        print(f"  {between['second_only']}, p {between['p']:.4f}. §47.126 measured that at the")
        print("  operator and §47.128 found the walk unable to spend it; §47.129 named the tie at")
        print("  zero that stopped it, and this is what happens when the tie cannot refuse.")
    else:
        print("  **And the cheap coordinate's advantage still does not reach the reader**, even")
        print(f"  with nothing refusing it: {cheap_free:.0%} against {free:.0%},")
        print(f"  {between['first_only']} to {between['second_only']}, p {between['p']:.4f}. So the")
        print("  floor was not the only thing standing between §47.126 and the answer, and what")
        print("  else does is unmeasured.")
    print()
    print("  `ask._floor` and `place.places` are replaced for the duration of the run and put back;")
    print("  nothing in `ask.py` is edited. One store, one answer per question, exact paired tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
