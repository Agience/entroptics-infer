"""A generated structure whose relations are used at very different rates, from a process with no
parameter for it.

Why this generator exists
---------------------------
§47.148 finds two halves of one specification pulling against each other, and §47.149 sweeps
`partial.py` over sixteen settings to find that they fall together everywhere: `reaches` and `share`
are not independent knobs, so no partial-reach store is in both regimes at once.

§47.149 also measures what the ontology has that none of them do, and it is not size. The
ontology has *fewer* statements per concept than the generated stores. What it has is a skew across
its relations -- two that carry most of the store and several that carry almost nothing -- where
every generated relation carries about the same. Where the frequencies are uneven the two properties
sit in different relations instead of in tension: the dense relations keep the truth inside the
support, the sparse ones leave offers unsupported, and a flat generator has to trade one for the
other.

So this generator makes the skew, and the question is where the skew comes from.

**The skew has no parameter**
-------------------------------
A distribution could be chosen -- a power law, a fixed set of shares -- and then the store would
answer its own design, which is the failure this line keeps meeting. It is not chosen here.

Relations are drawn by a Polya urn: each relation begins having been used once, which is the
least that can be assumed about any of them, and every statement after picks a relation in proportion
to how often it has already been used. Nothing sets how uneven the result is. A relation that
happens to be used early is likelier to be used again, and the spread that produces is whatever that
process gives -- which for ``RELATIONS`` relations is a uniform draw from the simplex of shares.

**This is the design rule applied rather than quoted**: would this happen if it evolved from nothing.
A store that grows by use, starting from no preference at all, is skewed without anything deciding
to skew it. The alternative -- every relation used equally often -- is the thing that needs a
mechanism, not this.

The generator, stated in full
-------------------------------
* ``GROUPS`` groups of ``PER_GROUP`` concepts, as `partial.py` has them; a concept's group is the
  truth about it.
* ``RELATIONS`` relations, each given a map from source group to destination group, drawn once.
* ``count`` statements are emitted one at a time. Each draws a relation from the urn -- proportional
  to its uses so far, all starting at one -- then a source concept uniformly, and states that
  relation from it into a concept drawn from the destination group. The relation's count goes up by
  one, which is what makes the next draw uneven.

Nothing else. What is designed is that relation frequencies are uneven, because §47.149 measured
that as the difference. What is not designed is the answer: the maps are drawn from the seed,
the grader knows the structure, and the reader never sees the map.
"""
from __future__ import annotations

import numpy as np

#: groups of concepts, and concepts in each -- as `partial.py` has them, so the two are comparable
GROUPS = 6
PER_GROUP = 6
#: how many relations the structure has
RELATIONS = 5


def structure(rng):
    """``(concepts, relations, group of each, where each relation sends each group)``."""
    concepts = [f"c{i}" for i in range(GROUPS * PER_GROUP)]
    group = {c: i // PER_GROUP for i, c in enumerate(concepts)}
    relations = [f"r{j}" for j in range(RELATIONS)]
    sends = {r: {g: int(rng.integers(GROUPS)) for g in range(GROUPS)} for r in relations}
    return concepts, relations, group, sends


def observed(concepts, relations, group, sends, count, rng):
    """``count`` statements, the relation of each drawn from an urn that starts flat.

    Every relation begins having been used once. Each statement picks a relation in proportion to its
    uses so far and then increments it, so the frequencies spread on their own: nothing here says how
    far. Duplicates are dropped, so the store holds fewer than ``count`` statements and the shortfall
    is itself a measure of how concentrated the draw became.
    """
    by_group: dict = {}
    for c in concepts:
        by_group.setdefault(group[c], []).append(c)

    uses = np.ones(len(relations))
    out = []
    seen = set()
    for _ in range(count):
        r = relations[int(rng.choice(len(relations), p=uses / uses.sum()))]
        uses[relations.index(r)] += 1
        a = str(rng.choice(concepts))
        here = [c for c in by_group[sends[r][group[a]]] if c != a]
        if not here:
            continue
        s = (a, r, str(rng.choice(here)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def observed_without_the_urn(concepts, relations, group, sends, count, rng):
    """The same store with the urn replaced by a uniform draw over relations — the control.

    Everything else is identical: the same structure, the same number of statements drawn, the same
    duplicate handling. Only the skew is removed. Without this, a store built by this generator
    and found to have some property cannot be told apart from a store that merely holds more
    statements than an older generator could reach, which is the confound that matters here: drawing
    ``(source, relation)`` with replacement lets this generator hold more statements than
    `partial.py` can produce at all, so any comparison against that family confounds skew with size.

    A control that changes one thing and is the same size is the comparison that separates them.
    """
    by_group: dict = {}
    for c in concepts:
        by_group.setdefault(group[c], []).append(c)
    out = []
    seen = set()
    for _ in range(count):
        r = relations[int(rng.integers(len(relations)))]
        a = str(rng.choice(concepts))
        here = [c for c in by_group[sends[r][group[a]]] if c != a]
        if not here:
            continue
        s = (a, r, str(rng.choice(here)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)
