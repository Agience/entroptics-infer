"""A query, carried through the ontology by its own operators, answered from what it reaches.

The loop
----------
    Get a vector V from the query Q. Place the light cone in that direction. Collect artifacts K
    and put them on the screen with the query. Advance the light cone and update V from the new Q.
    Repeat until there is no more attention.

The triple, and where each part is
------------------------------------
**Context** is the location — how a thing is found. `place.py` computes it from the edges; nothing
is stored on a vertex and nothing is trained.

**Content** is what the thing holds — its gloss. The answer is made of content; the context is only
how the answer was reached.

**Operator** is what a relation *does* to a location. `operators.py` derives one per relation from
that relation's own edges in a single pass — the edge list is the shared ordered axis and the
operator is the cross-covariance, so nothing is fitted.

A step applies an operator, it does not follow an edge
--------------------------------------------------------
This is the difference between inference and traversal, and it is why a stated edge is not required
for the walk to move. At each step the cone applies an operator to where it is, landing somewhere in
the coordinate, and asks what actually sits there. Where the ontology happens to state that edge, the
step is marked stated; where it does not, the step is marked inferred — the relation took the
cone somewhere the ontology never wrote down.

Which operator, and why the query decides
-------------------------------------------
A relation here is a concept, so a query can name one: ``salmon lives_in`` names two. Naming it
selects the operator rather than moving the cone toward it — a relation is what acts, not
somewhere to go, and its place is the hub every statement using it passes through.

That is measured rather than assumed. On the same held-out statements, reaching the true destination
first:

    the query names the source only          9%   every operator applied, the walk picks
    source and relation summed into Q        3%   the relation treated as a place
    the relation selects the operator       25%   the relation treated as what acts

Selecting beats picking 11-0 and beats summing 17-2, both p 0.001. `test_naming_the_relation.py`
runs all three. So choosing the relation is most of the difficulty, and a query that says which
is not being given a hint — it is supplying the operator, which is one leg of the triple.

Name none and every relation is applied, which is the 9% case, and the reader says which it
is doing.

Measured, with the statement held out of the operator and of the places, so nothing about it
survives anywhere: the destination of a statement never seen is placed first out of 31 candidates
28% of the time, against 1 in 31 by chance and 8% for the same operator with its pairs re-paired at
random. `test_is_the_gap_real.py` runs all three. The re-paired arm is what says the relation is
carrying it: under a third of the read is the coordinate's geometry alone, and the rest is which
source went with which destination.

Attention, and stopping
-------------------------
A candidate is scored against the query. Two other references are available -- the frontier the
cone has reached, and the query accumulated with everything found so far, which is what this
module's own loop describes -- and on this store the three cannot be told apart: each separates
a question the ontology can speak to from one it cannot by the same +37%, and keeps steps the store
joins to the frontier at the same 97%. They differ only in how far the walk runs, 1.2 steps against
1.4. `test_whether_the_cone_should_advance.py` measures all three.

So the choice here does not carry a result. In particular, none of the three walks from `salmon` to
`paw`: all three answer `salmon` with `fish, river` in one step.

It holds attention when it lies further along the query than the best of as many concepts taken from
the ontology at random. That floor is computed from the ontology, not set — in closed form, with
no sample size and no seed — and it rises with the number of candidates on offer, which is a
multiple-comparison bar rather than a level anyone chose. The walk stops when nothing clears it, so
depth is a property of the query and there is no argument to pass one.

That the floor rises with the count is why what each operator is allowed to offer decides a great
deal, and the reader offers one landing per operator: where it lands, and nothing if the walk has
been there. `test_the_cap_on_offers.py` measures the alternatives -- taking the best three per
operator, the share of kept candidates the store joins to the frontier is 32% against 83% at one.

What is deliberately not here
-------------------------------
No corpus, no message stream, no file fetched from anywhere. If the answer names something, the
ontology holds it, and the path shows how it was reached.
"""
from __future__ import annotations

import argparse
import re

import numpy as np

import ontology
import operators
import place
from measure import along as _along, unit as _unit  # noqa: E402,F401


def _tokens(query: str) -> set:
    """The query split on everything the store's own names do not use.

    A concept is named by a query when its name is in here, so the alphabet has to be the one the
    ontology writes names in rather than the one English words use. ``[a-z]+`` would split
    ``lives_in`` into "lives" and "in" -- and the concepts whose names carry an underscore are
    exactly the six used as relations, so the only concepts that could tell the reader *which*
    relation to apply would be the only ones a query could never name.
    """
    alphabet = "".join(sorted({c for name in ontology.concepts() for c in name if not c.isalnum()}))
    return set(re.findall(f"[a-z0-9{re.escape(alphabet)}]+", query.lower()))


def _floor(vectors, V, *, n):
    """What the best of ``n`` vertices drawn at random scores against this direction.

    The null is every other place in the coordinate — concepts, relations and statements alike,
    because all three are vertices here and ``vectors`` holds them all. On this store that is 150
    places of which 66 are concepts, so more than half the null is things the walk can never offer.
    §47.130 finds the same closed form over the concepts alone differs a little and decides the
    same on every question; the two populations would diverge on a larger store, since a statement
    is a vertex and statements are what a store gains.

    Shuffling a candidate's own coordinates instead asks "does this concept have structure", and
    every concept in an ontology does — so nothing was ever refused and the walk only stopped when
    the graph ran out.

    Computed rather than sampled. The scores are a finite set, so the best of ``n`` drawn from it
    has ``P(best <= t) = F(t)**n`` for the empirical ``F``, and the median of that is the value at
    quantile ``0.5 ** (1/n)`` — read off the step function, because the distribution is discrete and
    interpolating between its points answers a different question.

    This was drawn, with a sample size of 200, and the sample size was doing work: at forty
    candidates the floor read 0.70 against a converged 0.75, so the bar was low and attention kept
    what it should have refused. The closed form reproduces a 200,000-draw brute force exactly and
    takes no sample size, no seed and no draw count with it.

    The ``0.5`` is the definition of a median, not a level chosen for this store.
    """
    scores = _along(np.stack(list(vectors.values())), V)
    return float(np.quantile(scores, 0.5 ** (1.0 / n), method="inverted_cdf"))


def land(query: str, vectors):
    """Where the query falls in the ontology: the concepts it names, and their place.

    **What the query names is what it is about**, and naming is the whole of it. A concept's name
    is part of the ontology; its gloss is content — what the concept *holds* — and content is not
    how a thing is found. Matching a query against glosses uses one to do the other's job.

    That fallback existed and was measured out. `test_the_gloss_route_is_evidence.py` asks each
    concept using its own gloss with the name struck out, and asks again with words drawn from the
    glosses by how common they are. Both land every time, 66 of 66, disagreeing on nothing --
    so a landing said nothing about whether the query was about anything here. On the arm where it
    should be strongest, its own description of itself, it put the right concept first 15% of the
    time and anywhere in its hits 29%.

    So a query that names nothing this ontology holds gets no direction, and that is an answer.
    """
    words = _tokens(query)
    named = [(c, "names it", 1.0) for c in ontology.concepts() if c in words]
    if not named:
        return [], None

    # A named relation says what should act, not where to go, so it is reported but kept out of
    # the direction. Its place is the hub every statement using it passes through, and adding it
    # makes the query less specific: measured on the same held-out statements, a step from source
    # and relation summed together reaches the destination 3% of the time against 9% from the
    # source alone and 25% when the relation selects the operator instead.
    # `test_naming_the_relation.py` runs all three.
    acting = set(operators.relations())
    where = [c for c, _, _ in named if c not in acting]
    if where:
        named = [(c, "names it" if c in where else "names the relation", 1.0)
                 for c, _, _ in named]
    else:
        # nothing but relations named, so they are the location -- "what is is_a" is a fair
        # question and the store holds an answer to it
        where = [c for c, _, _ in named]
    return named, _unit(np.sum([vectors[c] for c in where], axis=0))


def acting_in(hits):
    """The relations a query named, which are the operators it is asking to have applied.

    Empty when the query names none, and then every relation is applied -- which is the reader
    choosing for itself, and is most of where the accuracy goes.
    """
    acting = set(operators.relations())
    return [c for c, _, _ in hits if c in acting]


def _stated(src, relation, dst):
    """Does a statement say this outright? Reported, never required.

    A step applies a relation's operator and lands somewhere. Whether the ontology also states that
    fact is worth reporting -- it is the difference between the walk recalling and the walk
    inferring -- but it is never a condition for the step.
    """
    return (src, relation, dst) in ontology.STATEMENTS


def walk(query: str, *, show=print):
    """The loop: apply every operator where the cone is, keep what holds attention, advance."""
    vectors, _ = place.places()
    ops = operators.all_operators(vectors)
    hits, V = land(query, vectors)
    if V is None:
        show("  the query names nothing this ontology holds, so the cone has no direction to")
        show("  point in and the walk does not start. That is an answer.")
        return []

    show(f"  Q lands on {len(hits)}:")
    for c, why, w in sorted(hits, key=lambda h: -h[2]):
        show(f"      {c:<12} {w:>5.2f}  {why}")

    # Whatever relations the query named are the operators it asked for. Naming none means every
    # relation is applied and the walk picks, which is the expensive case: on the same held-out
    # statements, choosing reaches the destination 9% of the time and being told 25%.
    asked_for = [r for r in acting_in(hits) if r in ops]
    applying = asked_for or list(ops)
    if asked_for:
        show(f"  applying {', '.join(asked_for)}: the query said which")
    else:
        show(f"  applying all {len(applying)} relations: the query did not say which")
    show("")

    path = []
    for s in steps(query, vectors, ops, landing=(hits, V)):
        step, candidates, offers, salience, kept = (s["step"], s["candidates"], s["offers"],
                                                    s["salience"], s["kept"])
        if not candidates:
            show(f"  step {step}: every operator lands where the walk has already been. It stops.")
            break
        if not kept:
            best = int(np.argmax(salience))
            show(f"  step {step}: {len(candidates):>2} offered, none still resolve against the "
                 f"query (best {candidates[best]} at {salience[best]:.3f}, floor {s['floor']:.3f}). "
                 f"No more attention.")
            break

        reached = []
        for i in kept:
            c = candidates[i]
            _, src, label = offers[c]
            reached.append({"reached": c, "from": src, "by": label,
                            "stated": _stated(src, label, c), "salience": float(salience[i])})
        stated = sum(1 for r in reached if r["stated"])
        show(f"  step {step}: {len(candidates):>2} offered -> attention keeps {len(reached)}: "
             f"{', '.join(r['reached'] for r in reached)}"
             f"   [{stated} stated, {len(reached) - stated} inferred]")
        path.append({"step": step, "reached": reached})
    return path


def steps(query, vectors, ops, *, landing=None, names=None, allp=None, ceiling=None, cap=None,
          memory="perfect", reference="the query", keep=None):
    """The walk's loop, one step at a time: what :func:`walk` runs, and what its studies vary.

    Each step yields ``{"step", "here", "candidates", "offers", "salience", "floor", "kept"}``:
    the frontier it started from, the candidates offered (sorted), ``{candidate: (score, source,
    relation)}``, each candidate's salience, the floor it had to clear (``None`` under a ``keep``
    rule) and the indices of the candidates kept. A step that offers nothing yields no candidates
    and a step that keeps nothing yields an empty ``kept``; the walk ends after either. A walk that
    yields a kept step last ran to ``ceiling``.

    The defaults are the reader's walk. Each parameter is one arm a study varies, and nothing else
    changes with it:

    ``ceiling``    how many steps at most; ``len(names) - 1`` by default, which a walk that refuses
                   what it has taken cannot reach.
    ``cap``        what each operator offers. ``None`` is the reader's rule: one landing per
                   operator, and no number written down -- an operator applied to a place lands at
                   a point, and what sits there is the nearest vertex; if the walk has been there,
                   that operator has nothing new to offer this step. An integer offers that many
                   nearest landings, ``"every landing"`` all of them, and ``"best available"`` the
                   nearest one the walk has not taken. Taking each operator's best few would make
                   that count decide a great deal: the share of kept candidates the store joins to
                   the frontier runs 85% at one landing, 32% at three, 20% at five, and with every
                   landing offered the floor is never cleared. `test_the_cap_on_offers.py` has the
                   table.
    ``memory``     what the walk refuses to offer again: ``"perfect"`` everything it has taken,
                   ``"one step"`` the frontier and where the query landed, ``"none"`` the frontier.
    ``reference``  what a candidate's salience is measured against: ``"the query"`` as it landed,
                   ``"the frontier"`` the mean of where the walk stands, or ``"the cone"`` (also
                   ``"accumulated"``) the query advanced by every step kept.
    ``keep``       ``keep(salience, against, here) -> [index]`` in place of the floor.

    Under the reader's reference the advanced cone ``V`` is not read: every candidate is scored
    against the query as it landed. It is advanced anyway, because the three references are
    indistinguishable on this store (+37% separation and 97% connected steps for each), so which
    one the reader should use is undecided rather than settled, and the other two read it.
    """
    hits, V = land(query, vectors) if landing is None else landing
    if V is None:
        return
    asked_for = [r for r in acting_in(hits) if r in ops]
    applying = asked_for or list(ops)
    names = list(vectors) if names is None else names
    allp = np.stack([vectors[c] for c in names]) if allp is None else allp
    Q0 = V.copy()
    here = [c for c, _, _ in hits if c not in asked_for] or [c for c, _, _ in hits]
    landed_on = {c for c, _, _ in hits}
    taken = set(landed_on)
    ceiling = len(names) - 1 if ceiling is None else ceiling

    for step in range(1, ceiling + 1):
        if memory == "perfect":
            refuse = taken
        elif memory == "one step":
            refuse = set(here) | landed_on
        elif memory == "none":
            refuse = set(here)              # a step to where you already are is not a step
        else:
            raise ValueError(memory)

        # Apply the relations the query asked for where the cone is. Each lands somewhere in the
        # coordinate; what sits nearest is the candidate, whether or not an edge says so.
        offers: dict = {}
        for src in here:
            for label in applying:
                landed = operators.carry(vectors, label, src, ops)
                if landed is None:
                    continue
                scores = _along(allp, landed)
                if cap is None:
                    i = int(np.argmax(scores))
                    c = names[i]
                    if c in refuse or c == src:
                        continue
                    prior = offers.get(c)
                    if prior is None or scores[i] > prior[0]:
                        offers[c] = (float(scores[i]), src, label)
                    continue
                order = np.argsort(-scores)
                for i in (order if cap in ("every landing", "best available") else order[:cap]):
                    c = names[int(i)]
                    if c in refuse or c == src:
                        continue
                    prior = offers.get(c)
                    if prior is None or scores[i] > prior[0]:
                        offers[c] = (float(scores[i]), src, label)
                    if cap == "best available":
                        break
        if not offers:
            yield {"step": step, "here": here, "candidates": [], "offers": offers,
                   "salience": None, "floor": None, "kept": []}
            return

        if reference == "the query":
            against = Q0
        elif reference == "the frontier":
            against = _unit(np.mean([vectors[c] for c in here], axis=0))
        elif reference in ("the cone", "accumulated"):
            against = V
        else:
            raise ValueError(reference)

        candidates = sorted(offers)
        salience = _along(np.stack([vectors[c] for c in candidates]), against)
        if keep is None:
            floor = _floor(vectors, against, n=len(candidates))
            kept = [i for i in range(len(candidates)) if salience[i] > floor]
        else:
            floor = None
            kept = keep(salience, against, here)
        yield {"step": step, "here": here, "candidates": candidates, "offers": offers,
               "salience": salience, "floor": floor, "kept": kept}
        if not kept:
            return
        taken.update(candidates[i] for i in kept)
        here = [candidates[i] for i in kept]
        V = _unit(V + np.mean([vectors[candidates[i]] for i in kept], axis=0))


def answer(path, hits) -> str:
    """The content of what the chain reached, in the order it was reached.

    **Context is how it was found; content is what it holds.** Reciting the edges -- "a salmon
    lives in a river; a river flows into an ocean" -- would read the search aloud, and a reader
    asking where a salmon lives would be told the shape of the walk.
    """
    if not hits:
        return "Nothing in the ontology held attention for that."
    statements = set(ontology.statements())
    lines = []
    for c, _, _ in sorted(hits, key=lambda h: -h[2]):
        lines.append(f"{c} - {ontology.content(c) or '(no content)'}")
    for entry in path:
        for r in entry["reached"]:
            how = "stated" if r["stated"] else "inferred"
            reached = r["reached"]
            # A statement is the larger concept and holds itself; a concept holds its gloss. Which
            # level the path used is settled by the walk, not by one of them being the real one.
            whole = " (the whole statement)" if reached in statements else ""
            lines.append(f"  from {r['from']} by {r['by'].replace('_', ' ')} ({how}): "
                         f"{reached}{whole} - {ontology.content(reached) or '(no content)'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the ontology something.")
    ap.add_argument("query", nargs="+", help="the question, in words")
    args = ap.parse_args(argv)
    q = " ".join(args.query)

    print(f'"{q}"\n')
    print(ontology.summary())
    print()
    vectors, _ = place.places()
    hits, _ = land(q, vectors)
    path = walk(q)
    print()
    print(answer(path, hits))
    total = sum(len(e["reached"]) for e in path)
    inferred = sum(1 for e in path for r in e["reached"] if not r["stated"])
    if total:
        print(f"\n{total} reached over {len(path)} steps; {inferred} by a relation the ontology "
              f"does not state outright.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
