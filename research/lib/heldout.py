"""What a held-out protocol can ask about, read off a store of statements.

This module imports nothing beyond the standard library, so a check that needs it does not take on
WordNet or any other corpus by importing it.
"""
from __future__ import annotations

import collections


def core(statements):
    """The part of a store a held-out protocol can actually ask about.

    **A concept named once cannot be held out**: removing its only statement removes the concept, so
    it has no place in the coordinate and the question cannot be put. In a taxonomy collected by
    walking outward that is most of the store — the frontier of a ball is always leaves, and it stays
    that way however far the walk goes.

    So concepts named once are dropped, repeatedly, until every concept left is named at least twice.
    This is not a filter chosen for its effect: it is the held-out protocol written down. What it
    removes is exactly what that protocol cannot ask about, and what it keeps is every statement whose
    two ends both survive their own removal.

    What it leaves is not a smaller taxonomy. Pruning leaves strips the tree and leaves the
    cross-linked structure, so the relations that are not the taxonomy survive preferentially.
    """
    st = list(statements)
    while True:
        seen = collections.Counter()
        for a, _, b in st:
            seen[a] += 1
            seen[b] += 1
        keep = [s for s in st if seen[s[0]] > 1 and seen[s[2]] > 1]
        if len(keep) == len(st):
            return tuple(st)
        st = keep
        if not st:
            return ()
