"""Does matching a query against glosses carry information, or does it fire on anything?

Why this is the one to check
------------------------------
Everything else in this system is structure. A place is read off the edges, an operator is read off
a relation's own uses, attention is scored against a floor drawn from the ontology. A gloss route
would be the single point where text decides anything: when a query names no concept, fall back
to matching its words against concept glosses, weighting each shared word by how few glosses hold it.

It fires on nothing. "what is quantum chromodynamics" names nothing this ontology holds and should
say so; by gloss it lands on ``used_for``, ``handle`` and ``part_of``, corroborated by the words
what and is. ``used_for``'s gloss is "what a made thing is for", so "what" occurs in exactly
one gloss out of sixty-six and scores as the rarest, most decisive word available.

The route lives here, not in `ask.py`
---------------------------------------
`ask.land` matches the query against concept names and nothing else. The gloss fallback this
measures is kept here, in the test that measures it, so the claim stays runnable without the live
path carrying a mechanism that was shown not to work.

Two arms
----------
``signal``  A concept's own gloss, with the concept's name struck out, used as the query. The words
            are as informative as this store gets -- they are the store's own description of that
            concept -- so if the gloss route is evidence at all it lands here.

``noise``   Words drawn from the glosses in proportion to how many glosses hold them, the same
            number of words as the signal query it is matched to. These are the store's *function*
            words by construction: nothing is on a stopword list, the store's own distribution
            decides what is common.

Nothing is fitted and no threshold is introduced. The two arms are scored the same way, and the
question is whether they differ.

What each outcome means
-------------------------
If noise lands about as often as signal, the gloss route is not evidence and the refusal it is
supposed to produce is not being produced -- in which case the fallback should go, and a query that
names nothing should say so. If signal lands far more often, the route carries information and the
job is to find what separates the two.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import re

import numpy as np

import ask
import ontology
import place

WORD = re.compile(r"[a-z]+")


def gloss_words():
    return {c: set(WORD.findall(ontology.content(c).lower())) for c in ontology.concepts()}


def commonness():
    """``{word: how many glosses hold it}`` -- the store's own account of which words are common."""
    held = collections.Counter()
    for words in gloss_words().values():
        held.update(words)
    return held


def land_by_gloss(query, vectors):
    """The fallback as it stood: with nothing named, match the query's words against glosses.

    Each shared word is weighted by how few glosses hold it, and the landing is required to be
    corroborated -- two different query words pointing into the same region -- because one
    incidental word was already known not to be evidence.
    """
    words = set(WORD.findall(query.lower()))
    glosses = gloss_words()
    holding = {w: sum(1 for g in glosses.values() if w in g) or 1 for w in words}

    named = [(c, "names it", 1.0) for c in ontology.concepts() if c in words]
    if named:
        return named

    weighted = []
    for c in ontology.concepts():
        shared = words & glosses[c]
        if shared:
            weighted.append((c, sum(1.0 / holding[s] for s in shared), "gloss"))
    pointing = {w for c, _, _ in weighted for w in (words & glosses[c])}
    if len(pointing) < 2 or not weighted:
        return []
    top = max(w for _, w, _ in weighted)
    return [(c, why, w) for c, w, why in weighted if w >= top / 10.0]


def signal_query(concept):
    """The concept's own gloss with its name struck out -- the store describing it, not naming it."""
    parts = [w for w in WORD.findall(ontology.content(concept).lower())
             if w not in concept.split("_")]
    return " ".join(parts)


def noise_query(n, held, rng):
    """``n`` words drawn in proportion to how many glosses hold them."""
    words = sorted(held)
    weights = np.array([held[w] for w in words], dtype=float)
    picked = rng.choice(len(words), size=n, replace=True, p=weights / weights.sum())
    return " ".join(words[int(i)] for i in picked)


def main() -> int:
    vectors, _ = place.places()
    held = commonness()
    rng = np.random.default_rng(len(ontology.STATEMENTS))
    concepts = [c for c in ontology.concepts() if ontology.content(c)]

    rows = []
    for c in concepts:
        q = signal_query(c)
        if not q:
            continue
        hits = land_by_gloss(q, vectors)
        names = [h[0] for h in hits]
        sig = {"landed": bool(hits), "first": bool(names) and names[0] == c,
               "among": c in names, "breadth": len(names)}

        nq = noise_query(len(q.split()), held, rng)
        nhits = land_by_gloss(nq, vectors)
        noi = {"landed": bool(nhits), "breadth": len(nhits)}
        rows.append((c, sig, noi))

    print(f"{len(rows)} concepts, each asked with its own gloss and with drawn words of equal "
          f"length\n")
    print(f"{'arm':<10}{'landed':>9}{'first':>9}{'among':>9}{'mean breadth':>15}")
    print("-" * 52)
    s_land = np.mean([r[1]["landed"] for r in rows])
    s_first = np.mean([r[1]["first"] for r in rows])
    s_among = np.mean([r[1]["among"] for r in rows])
    s_br = np.mean([r[1]["breadth"] for r in rows if r[1]["landed"]] or [0])
    n_land = np.mean([r[2]["landed"] for r in rows])
    n_br = np.mean([r[2]["breadth"] for r in rows if r[2]["landed"]] or [0])
    print(f"{'signal':<10}{s_land:>9.2f}{s_first:>9.2f}{s_among:>9.2f}{s_br:>15.1f}")
    print(f"{'noise':<10}{n_land:>9.2f}{'-':>9}{'-':>9}{n_br:>15.1f}")
    print("-" * 52)
    print()

    # the refusal that started this: a query about something the store has never heard of, put to
    # the route and to what `ask.py` actually does now
    print(f"  {'query':<34}{'by gloss':<26}{'live (names only)':<20}")
    for q in ("what is quantum chromodynamics", "how do i renew my passport",
              "where does a salmon live"):
        by_gloss = land_by_gloss(q, vectors)
        live, _ = ask.land(q, vectors)
        show = lambda h: ", ".join(x[0] for x in h) if h else "refused"   # noqa: E731
        print(f'  "{q}"'.ljust(36) + f"{show(by_gloss):<26}{show(live):<20}")
    print()

    # Paired, because each concept's gloss is matched to a drawn query of its own length: only the
    # concepts where the two arms disagree carry any information, and under "the route is not
    # evidence" a disagreement falls either way with equal chance.
    import test_is_the_gap_real as paired
    a = {r[0]: r[1]["landed"] for r in rows}
    b = {r[0]: r[2]["landed"] for r in rows}
    w, l, n, p = paired.mcnemar(a, b)
    print(f"  paired: signal lands where noise does not {w}x, the other way {l}x, p {p:.3f}")
    print()

    if p >= 0.05:
        print("  **The gloss route is not evidence.** Words drawn by how common they are land as")
        print("  readily as a concept's own description of itself, so a landing says nothing about")
        print("  whether the query is about anything here. A query that names nothing this ontology")
        print("  holds should say so, and the fallback is what stops it saying so.")
    else:
        print(f"  The gloss route separates: its own description lands {s_land:.2f} against")
        print(f"  {n_land:.2f} for drawn words. It carries information, and what remains is why it")
        print("  still fires on function words.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
