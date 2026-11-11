"""A curriculum that starts where a curriculum should start.

Why it starts at the floor
----------------------------
*Analogical completion between two competing relations over thirty-two-channel fields* is not a
first lesson. Analogy is among the last things a person learns, long after telling two things apart,
and starting there leaves every earlier ability untested: whether the reads can see that anything is there
at all, whether they can tell two things apart, whether they can find a partner when only one
relationship exists.

So the lessons start at the floor and each adds exactly one thing:

    nursery        is anything there?               one record, structure or noise
    pre-school     same, or different?              a thing, a near-copy, and a stranger
    kindergarten   which one goes with this?        one relationship only, nothing to confuse it
    grade one      two relationships, kept apart    each on its own things
    grade two      two relationships, mixed         both leave every source
    grade three    a name with nothing behind it    a relationship that is not one
    grade four     a relationship made of two       gamma is alpha then beta
    grade five     relationships that hold partly   the map is blurred
    grade six      taught a piece at a time         is it right while still learning
    grade seven    a relationship taught last       arrives after the reads have started

Each lesson states its own answer, so a wrong result is a defect in the read rather than a property
of the data, and can be fixed. The first lesson to fail is the thing to work on; nothing past it is
worth running until it passes.

What a lesson asks
--------------------
Lessons do not all ask the same kind of question. Posing an analogy at every level would include
levels where an analogy makes no sense.

    ``presence``      given one record, is there anything to read?
    ``discriminate``  given two, are they the same thing or not?
    ``match``         given a pair and a third, which candidate is its partner?
    ``relate``        given a pair standing in one of several relationships, which candidate stands
                      in *that* one -- the question only a matrix read can answer

Two operating requirements, both found by starting small enough to fail
------------------------------------------------------------------------
**A record needs structure, not just numbers.** Content drawn as i.i.d. noise makes every read
return nothing, correctly: white noise has no modes above its own floor. Measured, `rank` reads 0 on
the record and 0 on its association matrix, and `shared_modes` goes to -3. Easy is not the same as
empty -- which is what `nursery` tests on purpose.

**The feature axis has to be wide enough for the association matrix to resolve.** At 8 channels an
`8 x 8` association matrix reads `rank` 0 while the records themselves read 3, so the breadth gate
in `infer.complete` has nothing to confirm with and refuses every question. At 32 it answers. This
is where the floor is calibrated and not a defect: the same read on the forward-pass corpus of PAPER §5.3 returns `bound`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from draw import structured as _structured

KIND_PREFIX = "kind:"


@dataclass(frozen=True)
class Lesson:
    """One lesson, with the answer it expects.

    ``asks``      what this lesson poses: ``presence``, ``discriminate``, ``match`` or ``relate``.
    ``records``   vertex id -> its content.
    ``edges``     ``(src, dst, kind_vertex)``; empty where the lesson is not about relationships.
    ``carries``   how many named kinds genuinely stand behind a map. A name with nothing behind it
                  is still a name, and a read that counts it has counted a label.
    ``answers``   the lesson's own answer key, shaped by what it asks.
    """

    name: str
    asks: str
    records: dict
    edges: list = field(default_factory=list)
    carries: int = 0
    answers: dict = field(default_factory=dict)
    note: str = ""

    def named_edges(self):
        out = {}
        for s, d, k in self.edges:
            out.setdefault(k.removeprefix(KIND_PREFIX), []).append((s, d))
        return out


def _orthogonal(rng, d):
    return np.linalg.qr(rng.normal(size=(d, d)))[0]


def _noise(rng, T, D):
    """Numbers with nothing behind them. There is no structure here and the answer is 'no'."""
    return rng.normal(size=(T, D))


def nursery(*, rng, n=8, T=96, D=32):
    """Is anything there?

    Half the records have structure and half are noise. Nothing is related to anything; the only
    question is whether a thing can be read at all. A read that says yes to noise is imagining
    structure and a read that says no to structure is blind, and both are worth knowing before any
    relationship is asked about.
    """
    records, answers = {}, {}
    for i in range(n):
        records[f"real{i}"] = _structured(rng, T, D)
        answers[f"real{i}"] = True
        records[f"empty{i}"] = _noise(rng, T, D)
        answers[f"empty{i}"] = False
    return Lesson("nursery", "presence", records, answers=answers,
                  note="half structure, half noise; nothing is related to anything")


def pre_school(*, rng, n=8, T=96, D=32, blur=0.15):
    """Same, or different?

    Each thing has a near-copy of itself and a stranger. No maps and no relationships -- only
    whether two records are the same thing seen twice. Every later lesson rests on this.
    """
    records, answers = {}, {}
    for i in range(n):
        base = _structured(rng, T, D)
        records[f"thing{i}"] = base
        records[f"copy{i}"] = (1 - blur) * base + blur * _structured(rng, T, D)
        records[f"stranger{i}"] = _structured(rng, T, D)
        answers[(f"thing{i}", f"copy{i}")] = True
        answers[(f"thing{i}", f"stranger{i}")] = False
    return Lesson("pre-school", "discriminate", records, answers=answers,
                  note=f"a thing, a copy of it blurred by {blur:.0%}, and a stranger")


def kindergarten(*, rng, n=12, T=96, D=32):
    """Which one goes with this?

    **One** relationship, so the only question is which candidate is the partner. There is nothing
    to confuse it with: this asks whether a partner can be found at all, not which relationship it
    stands in, which is a later lesson.
    """
    records, edges, answers = {}, [], {}
    M = _orthogonal(rng, D)
    for i in range(n):
        src, dst = f"s{i}", f"partner{i}"
        records[src] = _structured(rng, T, D)
        records[dst] = records[src] @ M
        edges.append((src, dst, KIND_PREFIX + "goes-with"))
        answers[src] = dst
    return Lesson("kindergarten", "match", records, edges, 1, answers,
                  "one relationship; find the partner")


def grade_one(*, rng, n=12, T=96, D=32):
    """Still matching, but now with two relationships in the room rather than one.

    Each relationship lives on its own things, so no source has a partner under both and the two
    never compete for the same question. That makes this a matching lesson, not a relating one:
    the distractors belong to other sources, so "which of these is related to me at all" settles it
    and an exemplar is not needed. It was graded as a relating lesson at first, where it answered
    1.000 against a no-exemplar control at 0.975 -- the read was not wrong, the question was, and a
    question its own control can answer teaches nothing about supplying a relation by example.

    `grade two` is the first that genuinely needs the exemplar, because there both relationships
    leave every source and the distractor is that same source's other partner, which relatedness
    alone cannot separate.
    """
    records, edges, answers = {}, [], {}
    for kind, tag in (("alpha", "a"), ("beta", "b")):
        M = _orthogonal(rng, D)
        for i in range(n):
            src, dst = f"{tag}s{i}", f"{tag}d{i}"
            records[src] = _structured(rng, T, D)
            records[dst] = records[src] @ M
            edges.append((src, dst, KIND_PREFIX + kind))
            answers[src] = dst
    return Lesson("grade one", "match", records, edges, 2, answers,
                  "two relationships, each on its own things; find the partner")


def grade_two(*, rng, n=12, T=96, D=32):
    """The same two, now both leaving every source, so only the relationship distinguishes."""
    records, edges = {}, []
    Ma, Mb = _orthogonal(rng, D), _orthogonal(rng, D)
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    return Lesson("grade two", "relate", records, edges, 2, {},
                  "both relationships leave every source")


def grade_three(*, rng, n=12, T=96, D=32):
    """A name with nothing behind it: three carry a map and a fourth joins unrelated things."""
    records, edges = {}, []
    maps = {k: _orthogonal(rng, D) for k in ("alpha", "beta", "gamma")}
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in maps.items():
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
        records[f"loose{i}"] = _structured(rng, T, D)
    for i in range(n):
        edges.append((f"s{i}", f"loose{(i + 5) % n}", KIND_PREFIX + "hollow"))
    return Lesson("grade three", "relate", records, edges, 3, {},
                  "a fourth name with no relationship behind it")


def grade_four(*, rng, n=12, T=96, D=32):
    """A relationship made of two others: gamma is alpha, then beta."""
    records, edges = {}, []
    Ma, Mb = _orthogonal(rng, D), _orthogonal(rng, D)
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        records[f"alpha{i}"] = records[src] @ Ma
        records[f"beta{i}"] = records[src] @ Mb
        records[f"gamma{i}"] = records[src] @ Ma @ Mb
        edges += [(src, f"alpha{i}", KIND_PREFIX + "alpha"),
                  (src, f"beta{i}", KIND_PREFIX + "beta"),
                  (src, f"gamma{i}", KIND_PREFIX + "gamma")]
    return Lesson("grade four", "relate", records, edges, 3, {},
                  "gamma is alpha then beta")


def grade_five(*, rng, n=12, T=96, D=32, blur=0.6):
    """Relationships that hold only partly: the target is the map's answer plus content of its own."""
    records, edges = {}, []
    Ma, Mb = _orthogonal(rng, D), _orthogonal(rng, D)
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            records[f"{kind}{i}"] = ((1 - blur) * (records[src] @ M)
                                     + blur * _structured(rng, T, D))
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    return Lesson("grade five", "relate", records, edges, 2, {},
                  f"the relationships hold only {1 - blur:.0%}")


def grade_six(*, rng, n=12, T=96, D=32):
    """Taught a piece at a time. :func:`in_pieces` delivers it; the whole is returned here."""
    g = grade_two(rng=rng, n=n, T=T, D=D)
    return Lesson("grade six", "relate", g.records, g.edges, g.carries, {},
                  "the same lattice, taught a piece at a time")


def grade_seven(*, rng, n=12, T=96, D=32):
    """A relationship taught last: it exists on only the final third of the sources."""
    records, edges = {}, []
    Ma, Mb, Mc = (_orthogonal(rng, D) for _ in range(3))
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    for i in range(2 * n // 3, n):
        records[f"late{i}"] = records[f"s{i}"] @ Mc
        edges.append((f"s{i}", f"late{i}", KIND_PREFIX + "late"))
    return Lesson("grade seven", "relate", records, edges, 3, {},
                  "a third relationship taught last")


def grade_eight(*, rng, n=12, T=96, D=32, seen=0.6):
    """Partial observation: a source does not always have a target under every relationship.

    Every lesson to here gives each source a partner under each relationship it participates in, so
    a question always has a full set of candidates and the lattice is a complete grid. A lattice
    that arrives from the world is not like that -- a thing is observed in one relationship and not
    yet in another, and the gap means *not seen*, which is a different thing from *not related*.

    That distinction is the whole lesson, and it is the one a hollow name does not test. `hollow`
    asks whether a relationship that is not there is refused. This asks whether a relationship
    that is there is still read when only some of its instances have been observed, and whether
    the screen counts it without counting the absences as evidence of anything.

    Each source keeps a share `seen` of the relationships it could stand in, drawn independently, so
    the lattice is ragged rather than thinned uniformly -- some sources carry both, some one, some
    neither.
    """
    records, edges = {}, []
    Ma, Mb = _orthogonal(rng, D), _orthogonal(rng, D)
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            if rng.random() > seen:
                continue                      # observed in this relationship: not yet
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    return Lesson("grade eight", "relate", records, edges, 2, {},
                  f"each source observed in {seen:.0%} of the relationships it could stand in")


def on_real_records(sources, *, rng, hollow=True):
    """The lessons' own structure, built on content that was measured rather than generated.

    Every lesson above draws its sources from :func:`_structured` -- a few latent channels through a
    fixed basis, rows independent. Real records are not like that. They are autocorrelated along
    their axis, their channels carry wildly different scales, and their distributions are not
    Gaussian. Any of those could break a read that works on generated content, and until a lesson is
    built on real material there is no way to tell that apart from a real corpus simply not holding
    several distinct relations.

    So the sources here are supplied by the caller -- real segments, whatever they are -- and the
    relations are imposed and therefore known: `alpha` and `beta` are orthogonal maps of the source,
    and `hollow` joins sources that stand in no relation at all. The answer is known exactly, as in
    every other lesson, while the content is not synthetic.

    What it isolates: if the screen counts two relations here and the read answers, real content is
    not what stands between these reads and a real corpus. If it does not, this says which property
    of real material is responsible, on a lattice where the truth is not in doubt.
    """
    names = list(sources)
    D = np.asarray(sources[names[0]]).shape[1]
    Ma, Mb = _orthogonal(rng, D), _orthogonal(rng, D)
    records, edges = {}, []
    for i, key in enumerate(names):
        src = f"s{i}"
        records[src] = np.asarray(sources[key], dtype=float)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    if hollow:
        n = len(names)
        for i in range(n):
            edges.append((f"s{i}", f"s{(i + n // 2) % n}", KIND_PREFIX + "hollow"))
    return Lesson("on real records", "relate", records, edges, 2 if hollow else 2, {},
                  f"the lesson's structure on {len(names)} measured records, "
                  f"{D} channels each")


def near_duplicates(*, rng, n=12, T=96, D=32, near=0.9):
    """Two relationships that are nearly the same map.

    Not a graded lesson. How close two may be before they stop being two is a quantity to be
    measured rather than a lesson to be passed, by sweeping `near`. Declaring
    them the same above 0.99 would be a number chosen rather than found.
    """
    records, edges = {}, []
    Ma, Mother = _orthogonal(rng, D), _orthogonal(rng, D)
    Mb = near * Ma + (1 - near) * Mother
    for i in range(n):
        src = f"s{i}"
        records[src] = _structured(rng, T, D)
        for kind, M in (("alpha", Ma), ("beta", Mb)):
            records[f"{kind}{i}"] = records[src] @ M
            edges.append((src, f"{kind}{i}", KIND_PREFIX + kind))
    return Lesson("near duplicates", "relate", records, edges, 2, {},
                  f"beta is {near:.0%} alpha")


def in_pieces(lesson, pieces):
    """The same lattice delivered a piece at a time, each piece a lattice in its own right."""
    step = max(1, len(lesson.edges) // pieces)
    for cut in range(step, len(lesson.edges) + step, step):
        es = lesson.edges[:cut]
        keep = {v for s, d, _ in es for v in (s, d)}
        yield Lesson(f"{lesson.name} [{min(cut, len(lesson.edges))}/{len(lesson.edges)} edges]",
                     lesson.asks, {k: v for k, v in lesson.records.items() if k in keep},
                     es, lesson.carries, {}, lesson.note)


def kinds(edges):
    """Read off the edges. Nothing here declares how many kinds a lattice has."""
    return tuple(sorted({k.removeprefix(KIND_PREFIX) for _, _, k in edges}))


CURRICULUM = (nursery, pre_school, kindergarten,
              grade_one, grade_two, grade_three, grade_four,
              grade_five, grade_six, grade_seven, grade_eight)
