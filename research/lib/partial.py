"""A generated structure whose relations are partial, so that support can vary.

Why this generator exists
---------------------------
§47.142 finds the store's own statements separating right offers from wrong ones — the first
constraint measured in this line that does. §47.143 tried to put that somewhere else and could not:
the structure `test_inference_on_a_partly_stated_structure.py` builds gives every relation a
destination group for every source group, so a relation leads into almost every concept
somewhere and the constraint excludes nothing. A constraint that excludes nothing cannot be tested.

That sets a specification: the store needed has to ask enough questions, and
has to be sparse in support — a real share of offers the store does not support, about a quarter
as the ontology is, rather than the few per cent a total relation gives.

**What is designed here, and what is not**
--------------------------------------------
What is designed is that the question can be *asked*: relations are partial, so some
``(relation, candidate)`` pairs have no support anywhere and the constraint has something to exclude.
What is not designed is the answer. Whether a supported offer is more often right is measured
against a grader that knows the structure, and nothing here arranges for support and correctness to
agree — the generator fixes which groups a relation maps and the reader never sees that map.

The distinction matters and is the one this line keeps meeting: a generated store answers the
question its design allows, so the design must allow the question and not the answer.

The generator, stated in full
-------------------------------
* ``GROUPS`` groups of ``PER_GROUP`` concepts; a concept's group is the truth about it.
* ``RELATIONS`` relations. Each relation is given a partial map: for each source group it either
  sends to one destination group or sends nowhere, with the share that send decided by
  ``REACHES``. A relation that sends nowhere from a group states nothing from those concepts, so
  nothing supports it there.
* A statement ``(a, r, b)`` exists when ``r`` sends ``a``'s group somewhere and ``b`` is drawn from
  that destination group. ``observed`` then keeps a share of them, as the older generator does.

Nothing else. The maps are drawn once per structure from the seed and are not tuned: what varies
with ``REACHES`` is how partial the relations are, and it is swept rather than chosen.
"""
from __future__ import annotations

import numpy as np

#: groups of concepts, and concepts in each -- sized so a sweep asks a few hundred questions
GROUPS = 6
PER_GROUP = 6
#: how many relations the structure has
RELATIONS = 5


def structure(rng, reaches):
    """``(concepts, relations, group of each, where each relation sends each group or None)``.

    ``reaches`` is the share of source groups a relation sends anywhere from. At 1.0 this is the
    total map the older generator builds; below it, relations are partial and support can vary.
    """
    concepts = [f"c{i}" for i in range(GROUPS * PER_GROUP)]
    group = {c: i // PER_GROUP for i, c in enumerate(concepts)}
    relations = [f"r{j}" for j in range(RELATIONS)]
    sends = {}
    for r in relations:
        sends[r] = {g: (int(rng.integers(GROUPS)) if rng.random() < reaches else None)
                    for g in range(GROUPS)}
    return concepts, relations, group, sends


def everything(concepts, relations, group, sends, rng):
    """Every statement the structure implies: one per concept per relation that reaches its group."""
    by_group = {}
    for c in concepts:
        by_group.setdefault(group[c], []).append(c)
    out = []
    for a in concepts:
        for r in relations:
            to = sends[r][group[a]]
            if to is None:
                continue
            here = [c for c in by_group[to] if c != a]
            if here:
                out.append((a, r, str(rng.choice(here))))
    return out


def observed(concepts, relations, group, sends, share, rng):
    """A share of what the structure implies -- the store, as a partial view of the truth."""
    every = everything(concepts, relations, group, sends, rng)
    keep = rng.random(len(every)) < share
    return tuple(s for s, k in zip(every, keep) if k)
