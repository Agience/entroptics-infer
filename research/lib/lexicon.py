"""WordNet as a store the reader can be put to — a corpus nobody wrote for this program.

Why this corpus
-----------------
§47.153 measures the bare operator on `ontology.py` and finds it naming held-out facts **23.8** times
a draw, and **3.6** times a draw that already knows which candidates the relation reaches. That is
the result the line rests on, and it rests on **84 statements**. §47.154 shows why a generated store
cannot stand in: its destinations are drawn uniformly inside a group, so it has a ceiling the
decomposition exposes, and a synthesized corpus answers the question its design allows.

WordNet is neither. It is a hand-built lexical database of synsets, recorded by
lexicographers over decades, with **asymmetric named relations** — and nobody who wrote it had this
reader in mind. It is the corpus criterion met without a physical channel to carry the relation:
**an edge is there because someone recorded it**.

**One direction per relation, and why that is not a preference**
-----------------------------------------------------------------
WordNet stores both directions of every relation it has: ``hypernyms`` and ``hyponyms``,
``part_meronyms`` and ``part_holonyms``, and so on. Each pair is the same fact written twice.

**Keeping both would break the held-out protocol.** Holding out ``(a, hypernym, b)`` would leave
``(b, hyponym, a)`` in the store, so the answer would be stated in the very material the coordinate
and the operator are built from, and a read that looked up the inverse would score as inference. That
is the failure `hold out of the representation too` names, in its worst form — it would not look like
a bug, it would look like a result.

So exactly one direction of each inverse pair is kept. **Which one is not a choice about the data**:
the pairs are symmetric in content and the store would be identical under relabelling, so the
selection carries no information about whether a read succeeds.

**And that is not sufficient, because a relation can be its own inverse.** ``attribute`` and
``also_see`` are symmetric in WordNet — both directions are in the database under the same name, so
there is no second name to leave out and the rule above never reaches them. :func:`one_direction_only`
counts the reciprocity and drops any relation that has it, which is what removes ``attribute`` from
a core: the rule above leaves it in.

What is designed here, and what is not
----------------------------------------
What is designed is that the store is **connected**: statements are collected by walking outward from
a seed rather than sampling edges at random. A random sample of a corpus this size would be almost
entirely disjoint, every question would ask about a concept sharing no structure with any other, and
a failure would say nothing about the method. **What is not designed is the answer** — which relations
reach which synsets is WordNet's, the walk does not consult it, and the reader never sees it.

The seed is swept rather than chosen, because a single neighbourhood is one topic and a result that
held only there would be a fact about that topic.

**And what is ingested is the core, not the ball.** A ball in a taxonomy is mostly leaves, and a
concept named once cannot be held out at all — so most of a ball is unaskable by construction and
going bigger does not help, because the frontier is always leaves. `core` drops concepts named once
until none is left, which is the held-out protocol written down rather than a filter chosen for its
effect. See its docstring for what that does to the store's shape.
"""
from __future__ import annotations

import collections

from heldout import core  # noqa: F401  (the WordNet checks call it as ``lexicon.core``)
from nltk.corpus import wordnet as wn

#: The asymmetric relations, **one direction of each inverse pair**. Keeping a pair's other half
#: would state every held-out fact twice and turn a lookup into an apparent inference.
RELATIONS = ("hypernym", "part_meronym", "member_meronym", "substance_meronym",
             "entailment", "cause", "attribute", "also_see", "verb_group")

#: How each is read off a synset. Named here so the store's content is stated in full.
_READ = {
    "hypernym": lambda s: s.hypernyms() + s.instance_hypernyms(),
    "part_meronym": lambda s: s.part_meronyms(),
    "member_meronym": lambda s: s.member_meronyms(),
    "substance_meronym": lambda s: s.substance_meronyms(),
    "entailment": lambda s: s.entailments(),
    "cause": lambda s: s.causes(),
    "attribute": lambda s: s.attributes(),
    "also_see": lambda s: s.also_sees(),
    "verb_group": lambda s: s.verb_groups(),
}


def _out(synset):
    """Every statement this synset makes, as ``(relation, other)``."""
    out = []
    for r in RELATIONS:
        try:
            for other in _READ[r](synset):
                out.append((r, other))
        except Exception:
            # a relation a part of speech does not carry is not an error, it is an absence
            continue
    return out


def neighbourhood(size, seed="entity.n.01"):
    """``(statements, concepts)`` for a connected region of WordNet around ``seed``.

    Collected by walking outward until ``size`` synsets have been reached, then keeping **every**
    statement whose two ends are both inside. The walk follows the same relations the store is made
    of, in either direction, so reachability is not itself a claim about which way a relation points.
    """
    start = wn.synset(seed)
    seen = {start.name(): start}
    queue = collections.deque([start])
    while queue and len(seen) < size:
        here = queue.popleft()
        nbrs = [o for _, o in _out(here)]
        # inverses are walked for reach only; they never become statements
        nbrs += here.hyponyms() + here.part_holonyms() + here.member_holonyms()
        # **Sorted, because a walk that stops at a size must stop at the same place twice.**
        # The frontier is cut off the moment `size` is reached, so whichever neighbours are
        # considered first are the ones that get in -- and taken in the order the corpus reader
        # happens to return them, that order moves with the interpreter's hash seed, and the
        # same ball at the same seed gives cores of different sizes under different hash seeds:
        # unsorted, the store being measured is not the same store twice. Ordering by name
        # is arbitrary in the sense that any fixed order would do, and that is the point -- it
        # carries no information about which synsets are worth reaching, only that the answer does
        # not depend on how a set was laid out in memory.
        for other in sorted(nbrs, key=lambda o: o.name()):
            if other.name() not in seen:
                seen[other.name()] = other
                queue.append(other)
                if len(seen) >= size:
                    break

    inside = set(seen)
    statements = []
    for s in (seen[k] for k in sorted(seen)):
        for r, other in _out(s):
            if other.name() in inside and other.name() != s.name():
                statements.append((s.name(), r, other.name()))
    statements = sorted(set(statements))
    statements = one_direction_only(statements)
    concepts = sorted({c for a, _, b in statements for c in (a, b)})
    return tuple(statements), concepts


def one_direction_only(statements):
    """Drop any relation that holds **both** directions, measured rather than assumed.

    ``RELATIONS`` keeps one half of each inverse *pair* -- ``hypernym`` and not ``hyponym``, and so
    on -- and that is not sufficient. **A relation can be its own inverse.** WordNet's ``attribute``
    and ``also_see`` are symmetric: both ``(a, attribute, b)`` and ``(b, attribute, a)`` are in the
    database under the same name, so there is no second name to leave out and the rule above never
    applies to them.

    Measured, before this existed: ``attribute`` accounted for **16 of 77** statements in the core
    of ``entity.n.01`` at size 600 -- **20.8%** -- and with ``also_see`` for **18 of 190** at size
    1200. On every one of those, holding the statement out left the store saying it backwards, and a
    read that found the reverse scored as inference.

    So reciprocity is **counted** here rather than decided from a list of which relations sound
    symmetric. A relation with any reverse at all goes, because one reversed statement is one
    question whose answer is stated.
    """
    present = set(statements)
    keep = []
    for label in {r for _, r, _ in statements}:
        mine = [(a, b) for a, r, b in statements if r == label]
        if not any((b, label, a) in present for a, b in mine):
            keep.append(label)
    return tuple(s for s in statements if s[1] in keep)


def shape(statements):
    """How the store's statements are spread across its relations."""
    per = collections.Counter(r for _, r, _ in statements)
    return sorted(per.items(), key=lambda kv: -kv[1])
