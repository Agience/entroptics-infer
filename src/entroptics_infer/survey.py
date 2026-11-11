"""What to measure about a store before taking a single reading off it.

Why this is a module and not advice
-------------------------------------
A held-out read means nothing unless the store, after the hold-out, has stopped saying the answer.
There are three ways it can still say it, and each one has produced a wrong result in this
program's own history:

``in the coordinate``   the operator is rebuilt for the question but the basis is not, so the
                        vertex being asked about still carries the edge. Costs a factor of 2.4.
                        Not a property of the store -- the engine rebuilds the coordinate per
                        question -- so it is not surveyed here.
``backwards``           the store holds ``(a, L, b)`` and ``(b, L, a)``. Hold one out and the other
                        states it in reverse.
``under another name``  the store holds ``(a, L1, b)`` and ``(a, L2, b)``. Hold one out and the
                        other states it exactly. This is the one that hides: neither label is
                        the other's inverse, nothing in their spelling suggests they overlap, and a
                        reciprocity check passes it clean.

The last two are properties of the store, they are cheap to measure, and measuring them is not
optional -- on one lattice read by this package, 180 of 502 pairs in a region carry two labels,
53% of the statements, and no figure read from that store stands.

So this module measures them, and returns numbers rather than a verdict. A function that answered
"yes, your store is fine" would be one more thing to trust.

What it does not do
---------------------
It does not fix anything and it does not refuse. What to do about what it finds is the caller's, and
the two answers this package uses are worth naming:

* **drop the reciprocal labels**, because one direction of an inverse pair carries the same content
  and which one is kept says nothing about whether a read succeeds;
* **hold out every statement joining the two vertices**, not only the one asked about, because the
  question is about the pair.

And one that is not obvious: a null built by permuting each label independently will dissolve the
doubling, leaving a surrogate with more distinct edges than the real store. That is a different
graph, and a claim compared against it is compared against something differing by more than the
thing under test. Permute whole label-bundles where :func:`survey` reports any doubled pair.
"""
from __future__ import annotations

import collections
from typing import Iterable, Sequence

__all__ = ["survey", "core"]


def core(statements: Iterable[Sequence[str]]) -> tuple:
    """The part of a store a held-out protocol can actually ask about.

    A vertex named once cannot be held out -- removing its only statement removes the vertex, so it
    has no place in the coordinate and the question cannot be put. Those are dropped, repeatedly,
    until every vertex left is named at least twice.

    **This is the protocol written down, not a filter chosen for its effect.** What it removes is
    exactly what the protocol cannot ask about. In a taxonomy collected by walking outward it is
    most of the store, and going further out does not help because the frontier is always leaves.
    """
    st = [tuple(s) for s in statements]
    while True:
        seen = collections.Counter()
        for parts in st:
            for v in (parts[0], parts[-1]):
                seen[v] += 1
        keep = [s for s in st if seen[s[0]] > 1 and seen[s[-1]] > 1]
        if len(keep) == len(st):
            return tuple(st)
        st = keep
        if not st:
            return ()


def survey(statements: Iterable[Sequence[str]]) -> dict:
    """Everything worth knowing about a store before reading it. Counts, not a verdict.

    ``labels``                  how many statements each label carries.
    ``reciprocity``             per label, the share of its statements whose reverse is also in the
                                store. Anything above zero means holding one out leaves the other.
    ``pairs``                   distinct ``(first, last)`` pairs.
    ``pairs_under_two_labels``  how many carry more than one, and ``combinations`` names which
                                labels travel together. Read this before any figure.
    ``share_of_statements``     what fraction of the store those pairs account for. This is the size
                                of the leak a naive hold-out would allow.
    ``askable``                 the 2-core's size, and how many statements fall outside it -- the
                                part of the store no held-out question can be put to.
    ``biggest_label_share``     where one label is nearly the whole store, knowing which label was
                                asked says almost nothing, so a factor has to come from elsewhere.
    """
    st = [tuple(s) for s in statements]
    if not st:
        return {"statements": 0}

    labels = collections.Counter(s[1] for s in st if len(s) > 2)
    present = set(st)

    reciprocity = {}
    for label in labels:
        mine = [(s[0], s[-1]) for s in st if len(s) > 2 and s[1] == label]
        back = sum(1 for a, b in mine if (b, label, a) in present)
        reciprocity[label] = back / len(mine) if mine else float(0)

    under = collections.defaultdict(set)
    for s in st:
        under[(s[0], s[-1])].add(s[1] if len(s) > 2 else None)
    doubled = {k: v for k, v in under.items() if len(v) > 1}
    involved = sum(len(v) for v in doubled.values())

    inner = core(st)
    biggest = labels.most_common(1)[0] if labels else (None, 0)

    return {
        "statements": len(st),
        "vertices": len({v for s in st for v in (s[0], s[-1])}),
        "labels": dict(labels),
        "reciprocity": reciprocity,
        "labels_holding_both_directions": sorted(l for l, r in reciprocity.items() if r),
        "pairs": len(under),
        "pairs_under_two_labels": len(doubled),
        "statements_on_those_pairs": involved,
        "share_of_statements": involved / len(st),
        "combinations": {"+".join(sorted(str(x) for x in v)): n for v, n in
                         collections.Counter(tuple(sorted(str(x) for x in v))
                                             for v in doubled.values()).items()},
        "askable": len(inner),
        "not_askable": len(st) - len(inner),
        "biggest_label": biggest[0],
        "biggest_label_share": biggest[1] / len(st) if st else float(0),
    }
