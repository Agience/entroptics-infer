"""A lattice you can add to and ask, with nothing trained and nothing rebuilt.

What this is for
----------------
The reads in this package are functions: hand :func:`complete` an exemplar and some candidates and
it answers. That is the right shape for measuring them and the wrong shape for using them, because
a running system does not have the exemplar to hand — it has a store that things arrive into, and a
question about one of them.

This is that store. Records arrive, edges name what stands in what relation, and a question is
answered from whatever has arrived by the time it is asked.

Where the weights are
---------------------
There are none, and that is the whole design rather than a slogan. A relation holds no parameters:
:meth:`Lattice.operator` derives it from the edges citing it, at the moment it is asked for, by
:func:`entroptics_infer.relation_over`. An edge arriving changes that relation by entering the
derivation; an edge withdrawn changes it by leaving. There is no build step, no fitting, nothing to
retrain and nothing to unlearn.

How much of a relation's own history goes into it is read off the relation rather than supplied —
the longest run of its edges whose aggregate still resolves at the reader's level. That choice looks
at the relation and never at the question, which is what keeps the reader's level intact.

What arriving cannot do
-----------------------
It cannot move an answer already given. `Infer/Arrival.lean` proves it: a completion reads only the
records its question names, so a store that grows elsewhere is the same store to that question. The
experiments measure the same thing at 0 of 80 answers moved across four arrivals.

It *can* move what :meth:`Lattice.carries` reports, and that is not a contradiction. A separation is
scored against a baseline built from pairs standing in no named relation, so it reads the whole
store and arriving records join that baseline. Measured: an already-present relation moving from
7.36 to 4.78 as a lattice went from 48 to 80 records, without its verdict changing. An answer is a
property of the records it was read from; a separation is a property of a relation in the lattice it
currently sits in, and only the first may be cached.

What it refuses
---------------
Everything the reads refuse, unchanged and for the same reasons. A question whose candidates do not
resolve gets nothing back rather than a guess, and a relation with no edges behind it cannot be
asked for at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import Unreadable
from .infer import Completion, complete_with, relation_over
from .relations import carries_kinds


@dataclass(frozen=True)
class Verdict:
    """What the lattice holds, and what it can be asked to exemplify — two different things.

    ``holding``     the relations this lattice carries: their edges' mean stands above the mean
                    of as many pairs standing in no named relation. This is what
                    :meth:`Lattice.operator` needs, because it derives a relation from every edge
                    citing it.
    ``carrying``    the narrower set: relations a single exemplar pair could name, which also
                    tell apart from the others. This is what :func:`entroptics_infer.complete`
                    needs, and it is a stronger requirement.
    ``named``       every relation any edge mentions, held or not.
    ``separation``  per relation, how far a single edge reads past a single unrelated pair.
    ``strength``    per relation, how far its mean reads past the mean of as many unrelated pairs.

    The two sets are not the same and the gap is where real corpora live. A road network's
    ``downstream`` reads 0.45 to 1.42 on ``separation`` — outside ``carrying`` in all 30 cells
    swept — and 7.9 to 32.6 on ``strength``, against a control at −1.2 to 1.3. A store that reported
    only ``carrying`` would say it holds nothing while holding that.

    ``in`` tests ``holding``, because the question a running lattice is usually asking is whether it
    has the relation at all.
    """

    holding: tuple
    carrying: tuple
    named: tuple
    separation: dict
    strength: dict

    def __contains__(self, relation) -> bool:
        return relation in self.holding


@dataclass
class Lattice:
    """Records, edges, and the reads over them. Everything is derived when asked for."""

    far: float
    records: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)

    # --- arrival -------------------------------------------------------------------

    def add(self, name, record) -> "Lattice":
        """A record arrives. Nothing is rebuilt, because nothing was built."""
        self.records[name] = np.asarray(record, dtype=float)
        return self

    def relate(self, src, dst, relation) -> "Lattice":
        """An edge arrives: `src` stands in `relation` to `dst`.

        The relation is whatever string the edge carries. Nothing here holds a vocabulary, nothing
        validates the name against a list, and a name never seen before needs no registration —
        which is the arrangement the screen needs and an enumeration would defeat.
        """
        for end in (src, dst):
            if end not in self.records:
                raise Unreadable(
                    f"{end!r} has no record, so an edge to it stands between nothing. Add it first."
                )
        self.edges.setdefault(relation, []).append((src, dst))
        return self

    def forget(self, src, dst, relation) -> "Lattice":
        """An edge is withdrawn. The relation changes by that edge leaving its derivation."""
        kept = [e for e in self.edges.get(relation, []) if e != (src, dst)]
        if kept:
            self.edges[relation] = kept
        else:
            self.edges.pop(relation, None)
        return self

    # --- what it holds -------------------------------------------------------------

    def operator(self, relation) -> np.ndarray:
        """The relation, derived now from the edges citing it.

        This is the weight, and it exists only for as long as this call. How many of the relation's
        edges go into it is read off the relation itself.
        """
        edges = self.edges.get(relation)
        if not edges:
            raise Unreadable(
                f"nothing stands in {relation!r} here, so there is no relation to derive. Name it "
                f"on a pair first."
            )
        return relation_over(self.records, edges, far=self.far)

    def carries(self, *, seed: int, draws: int) -> Verdict:
        """Which relations this lattice actually holds, as against merely names.

        Asked of the whole store, so it moves as the store grows — see this module's docstring.
        Both readings are returned because they answer different questions: see :class:`Verdict`.
        """
        if len(self.edges) < 1:
            return Verdict(holding=(), carrying=(), named=(), separation={}, strength={})
        screen = carries_kinds(self.records, self.edges, far=self.far, seed=seed, draws=draws)
        holding = tuple(sorted(k for k in screen.carries if screen.carries[k]))
        carrying = tuple(sorted(k for k in screen.present
                                if screen.present[k] and screen.distinct[k]))
        return Verdict(holding=holding, carrying=carrying, named=tuple(screen.names),
                       separation={k: float(v) for k, v in screen.separation.items()},
                       strength={k: float(v) for k, v in screen.carried.items()})

    # --- asking --------------------------------------------------------------------

    def ask(self, source, relation, *, candidates=None) -> Completion:
        """**What stands in `relation` to `source`?**

        The relation is derived from its edges, not supplied by an exemplar — which is the
        difference between this and :func:`complete`, and the reason a running system can be asked
        a question it has no example to hand for.

        With no `candidates`, everything else in the lattice is offered. That is the harder
        question and usually the right one: most records stand in no relation to the source at all,
        and a read that cannot refuse them is not worth asking.
        """
        if source not in self.records:
            raise Unreadable(f"{source!r} has no record here, so there is nothing to ask about it.")
        if candidates is None:
            candidates = [k for k in sorted(self.records) if k != source]
        if not candidates:
            raise Unreadable("no candidates were offered and the lattice holds nothing else.")
        return complete_with(self.records, operator=self.operator(relation),
                             source=source, candidates=list(candidates), far=self.far)

    def answer(self, source, relation, *, candidates=None):
        """:meth:`ask`, reduced to the name it returned, or ``None`` where it refused."""
        cands = (list(candidates) if candidates is not None
                 else [k for k in sorted(self.records) if k != source])
        out = self.ask(source, relation, candidates=cands)
        return None if out.key is None else cands[out.key]

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        edges = sum(len(v) for v in self.edges.values())
        return (f"Lattice({len(self.records)} records, {edges} edges, "
                f"{len(self.edges)} relations named, far={self.far})")
