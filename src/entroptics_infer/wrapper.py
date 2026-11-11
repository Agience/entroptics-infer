"""The swap-in surface: the layer a model trains, derived from a store at read time instead.

What this is for
------------------
Someone building a model holds an embedding table, projections ``W_Q``/``W_K``/``W_V``, and a
softmax over ``q·k/√d``. Every one of those is a fitted object: it exists because gradient
descent put it there, it is fixed once training stops, and a fact learned after training cannot
enter it without retraining. This module is the same layer with each of those replaced by something
read off a store at the moment the question is asked.

    a trained model                     this engine
    ------------------------------------------------------------------------------
    embedding table                     :meth:`Engine.places` -- a vertex's coordinate is what it
                                        is connected to, recomputed from the edges, stored nowhere
    ``W_Q``/``W_K``/``W_V``             nothing. There is no projection to learn; the two sides of
                                        a relation are already in one basis
    the relation's weight               :meth:`Engine.operator` -- the cross-covariance of the
                                        statements that use that relation, computed in one pass
    ``softmax(q·k/√d)``                 :meth:`Engine.attend` -- alignment standardized by the
                                        exact permutation null. No temperature, and a row of
                                        candidates that resolve nothing sums to zero rather than
                                        to one
    a fact learned by retraining        :meth:`Engine.relate` -- one write. The next read derives a
                                        different operator because the statement is in the sum

**Nothing here is trained and nothing here is stored between reads.** The only number the caller
supplies is ``far``, the false-alarm level, which by Neyman-Pearson belongs to the reader; this
module defaults it nowhere, exactly as the rest of the package does not.

Where the seams are, and why this module is not one
-----------------------------------------------------
:mod:`entroptics_infer.engine` is the one seam onto the instrument and :mod:`entroptics_infer.store`
is the one seam onto mantle. This module is a facade over both and is a seam onto neither -- it
composes what they expose and adds no reading of its own. That distinction is the reason the
load-bearing test can still neuter the instrument in one place and see every read here change.

The hold-out protocol is part of the API, deliberately
--------------------------------------------------------
Every read takes ``without``. Passing the statement being asked about removes it from the operator
and from the coordinate, which is the protocol every published number in ``research/`` is taken
under. It is exposed rather than kept internal because it is the thing a sceptical caller most needs
to run for themselves: with ``without`` the answer cannot have been stated, and the difference
between the two calls is what the claim is about. `hold out of the representation too` records what
happens when only half of it is done -- every score inflated by a factor of two and a half.

How a store is laid out in mantle
-----------------------------------
A statement is a vertex, and its parts hang off it as edges carrying the position they arrive at.
``dog is_a mammal`` is one statement vertex with three edges, to ``dog`` at position 0, ``is_a`` at
position 1 and ``mammal`` at position 2. Nothing records that position 1 means "the relation": the
roles come from order, and no read here branches on the number.

That layout is what lets mantle back this without a relation vocabulary. ``mantle.entities.relation``
holds no enumeration and :mod:`entroptics_infer.store` writes one uninformative label for every edge;
what kind a relation is, is measured (see :mod:`entroptics_infer.relations`), and a *name* like
``is_a`` is just another vertex that happens to be used in the middle position a lot.
"""
from __future__ import annotations

import heapq
import math

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from entroptics.entropy import shannon_bits

from . import bind
from .engine import Unreadable

#: The multiplicative identity, carried into the coordinate walk. Not a weight: it is what "the
#: vertex is entirely itself before any step is taken" is spelled as, and nothing tunes it.
_WHOLE = float(1)

#: The edge property a statement's parts carry. A position is an order, not a type, and the store
#: seam refuses a type. Named once here so a reader of the lattice knows what to look for.
POSITION = "position"



def _aperture(count, total, weighted_log, smallest, vertices):
    """How many cells this vantage occupies, or ``None`` while the unsettled tail could still move it.

    The participation number ``2^H`` of the power spectrum -- the entroptics application's "effective
    count of occupied cells" [0019] -- read while the spectrum is still being built.

    **It is read in constant time, from two running sums.** Entropy over weights ``w`` normalized by
    their total ``S`` is

        H = -sum (w/S) log2(w/S) = log2(S) - L/S,      L = sum w log2 w

    so a walk that settles one more vertex adds one term to ``S`` and one to ``L`` and knows ``H``
    again, instead of summing the whole prefix afresh. Summing it afresh is what the profile measures
    as the cost: 33.5 s of 77.8 s inside this function, 31.4 million calls to ``math.log2`` for
    one prose coordinate. The rule is the same either way; only how often it is arithmetic differs.

    **The walk settles largest first, so every value still to come is at most ``smallest``.** That
    bounds the unsettled tail: at most ``vertices - count`` of them, each at most ``smallest``. So
    ``H`` is taken twice, once with the tail at nothing and once with it at that maximum spread as
    widely as it can be, and the count is returned only when both round the same way. Then no amount
    of further walking can change it and the walk stops -- which is what makes the aperture
    affordable at all.
    """
    if count < 2 or total <= 0:
        return None
    low = math.log2(total) - weighted_log / total
    left = max(int(vertices) - count, 0)
    tail = left * smallest * smallest
    if tail <= 0:
        return max(int(round(float(2) ** low)), 1)
    grown = total + tail
    # the same two sums renormalized, plus the most entropy that much tail mass could carry
    high = (total / grown) * math.log2(grown) - weighted_log / grown
    share = tail / grown
    if share > 0 and left > 0:
        high += share * math.log2(left / share)
    if int(round(float(2) ** low)) == int(round(float(2) ** high)):
        return max(int(round(float(2) ** low)), 1)
    return None



@dataclass(frozen=True)
class Row:
    """One vertex's coordinate, holding only the axes it reaches.

    A dense row is the width of the basis whatever it carries, and under the aperture a row carries
    0.14% of it. That is the difference between a corpus this program can read and one it
    cannot: the 5,000-sentence point was killed for memory twice, once on an idle box with 21 GB
    free, holding rows that were almost entirely zero.

    ``where`` are the axis indices, ascending, and ``what`` their values. ``dense()`` gives back the
    array for a caller that wants one; ``dot`` reads against a dense vector without building it.
    """

    where: np.ndarray
    what: np.ndarray
    width: int

    def dense(self) -> np.ndarray:
        out = np.zeros(self.width)
        out[self.where] = self.what
        return out

    def dot(self, other) -> float:
        """This row against a dense vector, or against another row, without densifying either."""
        if isinstance(other, Row):
            shared = np.intersect1d(self.where, other.where, assume_unique=True)
            if not shared.size:
                return float(0)
            mine = self.what[np.searchsorted(self.where, shared)]
            theirs = other.what[np.searchsorted(other.where, shared)]
            return float(np.dot(mine, theirs))
        return float(np.dot(self.what, np.asarray(other)[self.where]))

    def norm(self) -> float:
        return float(np.linalg.norm(self.what))


@dataclass(frozen=True)
class Answer:
    """What the engine read, and enough of the working to disbelieve it.

    ``named``        the candidate the operator landed nearest, or ``None`` where it landed on
                     nothing the caller offered.
    ``scores``       the attention row over ``candidates``, in the order they were given.
    ``stated``       whether the store, as read, already contains this statement. A true here
                     means the answer was looked up, not inferred, and the field exists so that
                     can never be discovered later.
    ``reached``      how many candidates this relation points into elsewhere in the store. The
                     draw an inference has to beat is one over this, not one over the pool.
    ``pool``         how many candidates were offered.
    """

    named: str | None
    scores: np.ndarray
    stated: bool
    reached: int
    pool: int


def _rest_of(parts: Sequence[str], position: int) -> tuple:
    """Everything in the statement except what stands at ``position``, with its own places kept."""
    return tuple((q, parts[q]) for q in range(len(parts)) if q != position)


class Engine:
    """A store, read as the layer a model would otherwise have trained.

    Built from any iterable of statements -- tuples of vertex names, of any width -- or from a
    mantle lattice. The statements are held as given and nothing is precomputed: a coordinate is
    built when a read asks for one, because a coordinate that survived a write would be a trained
    object by another name.
    """

    def __init__(self, statements: Iterable[Sequence[str]]) -> None:
        self._statements = tuple(tuple(str(p) for p in s) for s in statements)
        if not self._statements:
            raise Unreadable(
                "an engine over no statements has nothing to derive an operator from. This raises "
                "rather than returning an engine that answers everything with None, because a "
                "reader that cannot tell an empty store from a store with nothing to say has lost "
                "the distinction this package exists to keep."
            )

    # ---- where a store comes from -------------------------------------------------------------

    @classmethod
    def on_statements(cls, statements: Iterable[Sequence[str]]) -> "Engine":
        """An engine over statements held in memory. The form to use in a test or a notebook."""
        return cls(statements)

    @classmethod
    def on_lattice(cls, lattice: Any, *, seed: str, size: int) -> "Engine":
        """An engine over a connected region of a mantle lattice, walked outward from ``seed``.

        A region rather than the whole store, and walked rather than sampled, for the reason
        ``research/lib/lexicon.py`` states in full: a random sample of a large store is almost entirely
        disjoint, every question would ask about a vertex sharing no structure with any other, and
        a failure would say nothing about the method. ``size`` is the caller's declared reach, not
        a tuning parameter -- it says how much of their store they want read, and the cost of a
        read grows with it.

        The lattice is not enumerated. :class:`entroptics_infer.store.Lattice` refuses that on
        purpose, and every read in this program starts from a vertex.
        """
        return cls(_walk(lattice, seed=str(seed), size=int(size)))

    def write_to(self, lattice: Any) -> int:
        """Put this engine's statements into a mantle lattice, and return how many edges that took.

        Each statement becomes a vertex and each of its parts an edge carrying the position it
        arrives at. No relation name is written as a type -- a relation is a vertex like any
        other and the store seam writes one uninformative label for every edge, so nothing about
        which relations exist is decided at write time.
        """
        edges = []
        for i, parts in enumerate(self._statements):
            sid = self.statement_id(i)
            for position, part in enumerate(parts):
                edges.append((sid, part, {POSITION: position}))
        return int(lattice.relate_many(edges))

    # ---- what the store says ------------------------------------------------------------------

    @property
    def statements(self) -> tuple:
        return self._statements

    @staticmethod
    def statement_id(i: int) -> str:
        """A statement's name as a vertex. Positional, so a held-out build names the same things."""
        return f"statement:{i}"

    def vertices(self, *, without: Iterable[Sequence[str]] = ()) -> tuple:
        """Every vertex some surviving statement mentions, and the statements themselves."""
        live = self._kept(without)
        concepts = sorted({c for _, parts in live for c in parts})
        return tuple(concepts) + tuple(sid for sid, _ in live)

    def concepts(self, *, without: Iterable[Sequence[str]] = ()) -> tuple:
        """The vertices that are not statements, in a stable order."""
        return tuple(sorted({c for _, parts in self._kept(without) for c in parts}))

    def _kept(self, without: Iterable[Sequence[str]]) -> tuple:
        """``((statement id, parts), ...)`` -- the store minus the statements held out.

        Ids stay positional in the whole store, so a held-out build and a whole one name the same
        statements and a caller can compare the two directly.
        """
        drop = {tuple(str(p) for p in s) for s in without}
        return tuple((self.statement_id(i), parts)
                     for i, parts in enumerate(self._statements) if parts not in drop)

    # ---- the embedding table, derived ---------------------------------------------------------

    def basis(self, *, without: Iterable[Sequence[str]] = ()) -> tuple:
        """The axes a coordinate lives on: every ``(position, rest-of-statement)``, plus each
        statement's own axis.

        The basis is counted off the store, so it grows when a statement arrives and no size is
        chosen. Binding the position to the rest of the statement is what keeps which relation went
        with which target; an axis per ``(position, vertex)`` sums those together and loses the
        pairing, which ``research/checks/test_binding_the_coordinate.py`` measures at seven questions to
        nothing.
        """
        live = self._kept(without)
        found = {(p, _rest_of(parts, p)) for _, parts in live for p in range(len(parts))}
        return tuple(sorted(found, key=repr)) + tuple(("in", sid) for sid, _ in live)

    def _adjacency(self, without: Iterable[Sequence[str]]) -> dict:
        """The edges, both ways, each carrying where it arrives."""
        out: dict = {v: [] for v in self.vertices(without=without)}
        for sid, parts in self._kept(without):
            for position, concept in enumerate(parts):
                out[concept].append((sid, (position, _rest_of(parts, position))))
                out[sid].append((concept, ("in", sid)))
        return out

    def places(self, *, without: Iterable[Sequence[str]] = (),
               only: Iterable[str] | None = None, sparse: bool = False) -> dict:
        """``{vertex: coordinate}`` -- this is the embedding table, and it is not stored.

        A vertex's place is its own row plus its neighbours' rows, each decayed by the branching it
        passed through: a statement with three parts divides by three, a vertex used in twenty
        statements divides by twenty, so a hub does not flood everything hanging off it. The decay
        is the store's own structure, not a number written here -- there is no rate to tune and
        nothing to fit.

        Recomputed on every call. That is the expensive property and it is the load-bearing one: a
        coordinate that persisted across a write would be a trained object, and a fact added to the
        store would not reach it without a rebuild.

        **There is one walk and it stops at its own aperture.** :meth:`_settle` settles every
        vertex at the largest value that reaches it, largest first; :func:`_aperture` reads the
        participation number ``2^H`` of the settled values as they arrive and stops the walk once
        the unsettled tail can no longer change it; :meth:`_row` adds each kept vertex's value into
        every axis it points at. Nothing about where the walk stops is supplied by a caller: the
        place keeps the vertices its own spectrum resolves and no others.
        ``research/checks/test_whether_the_walk_can_stop_at_its_own_aperture.py`` reads it against
        the same walk read to its end.

        **``only`` computes the rows named and no others, and the rows are the same rows.** Each
        start's walk reads the basis and the adjacency and nothing another start produced, so a row
        is the same whether it was built beside every other row or alone.
        ``research/checks/test_what_a_read_actually_touches.py`` checks that cell for cell.

        ``sparse`` changes how a row is held and not what it holds: a :class:`Row` of the axes it
        reaches instead of an array the width of the basis.
        """
        axes = self.basis(without=without)
        index = {a: i for i, a in enumerate(axes)}
        adj = self._adjacency(without)

        wanted = self.vertices(without=without)
        if only is not None:
            keep = set(only)
            wanted = tuple(v for v in wanted if v in keep)

        vectors = {}
        for start in wanted:
            kept = self._resolved(adj, start)
            vectors[start] = self._row(adj, index, kept, sparse=sparse)
        return vectors

    @staticmethod
    def _settle(adj: dict, start: str):
        """``(vertex, value)`` for every vertex the walk reaches, largest value first.

        **A vertex contributes once, at the largest value that reaches it.** Settled by priority
        queue, which is a max-product shortest path, and a shortest path cannot depend on the order
        its edges were offered in. A store is a set; the order it was written down in carries
        nothing, and nothing read here moves when it changes
        (``research/checks/test_whether_the_order_a_store_was_given_in_reaches_the_read.py``).

        **A queue entry is a vertex whose neighbours are all available at one value**, not a
        neighbour: every out-edge of a vertex carries the same step -- the value divided by the
        fanout, and there are exactly that many neighbours -- so a vertex is pushed once and its
        neighbours are offered when it is popped. Popping in decreasing order of step is what keeps
        that correct: the first entry that offers a vertex offers it at the largest value anything
        will. Neighbours are taken in name order, a property of the vertices and not of the store's
        order.

        This is a generator and it never stops on its own: read to its end it is the whole connected
        region, every vertex at its settled value. :meth:`_resolved` is what stops reading it.
        """
        yield start, _WHOLE
        done = {start}
        queue = [(-(_WHOLE / (len(adj[start]) or 1)), 0, start)]
        pushed = 0
        while queue:
            negative, _, parent = heapq.heappop(queue)
            step = -negative
            for other, _axis in sorted(adj[parent]):
                if other in done:
                    continue
                done.add(other)
                yield other, step
                pushed += 1
                heapq.heappush(queue, (-(step / (len(adj[other]) or 1)), pushed, other))

    @classmethod
    def _resolved(cls, adj: dict, start: str) -> list:
        """``[(vertex, value), ...]`` -- the settled vertices this vantage resolves, and no others.

        The settled values are read as a power spectrum while :meth:`_settle` produces them, and the
        walk stops once :func:`_aperture` says the tail cannot change the count of occupied cells.

        **The cut is at a value, never at a count.** Several vertices are reached by exactly the
        same value -- every neighbour of one vertex is -- so keeping a count of them keeps whichever
        the queue handed over first. Kept is every vertex at or above the value of the last one the
        count admits.
        """
        settled = []
        total, weighted_log, keep = float(0), float(0), None
        for here, carried in cls._settle(adj, start):
            settled.append((here, carried))
            power = carried * carried
            total += power
            weighted_log += power * math.log2(power)
            keep = _aperture(len(settled), total, weighted_log, carried, len(adj))
            if keep is not None and len(settled) > keep and carried < settled[keep - 1][1]:
                break
        if keep is not None:
            edge = settled[min(max(keep, 1), len(settled)) - 1][1]
            settled = [(here, value) for here, value in settled if value >= edge]
        return settled

    @staticmethod
    def _row(adj: dict, index: dict, settled: Iterable[tuple], *, sparse: bool = False):
        """One place: each settled vertex's value added into every axis it points at."""
        if sparse:
            got: dict = {}
            for here, carried in settled:
                for _other, axis in adj[here]:
                    at = index[axis]
                    got[at] = got.get(at, float(0)) + carried
            where = np.fromiter(sorted(got), dtype=np.intp, count=len(got))
            return Row(where, np.array([got[i] for i in where]), len(index))
        v = np.zeros(len(index))
        for here, carried in settled:
            for _other, axis in adj[here]:
                v[index[axis]] += carried
        return v

    # ---- the relation's weight, derived -------------------------------------------------------

    def uses(self, relation: str, *, without: Iterable[Sequence[str]] = ()) -> tuple:
        """Every statement that contains ``relation``, with where in it the relation stands.

        A relation is a vertex, not a label, so anything used as one has an operator by the same
        rule and a vertex never used as one has none. No list of relation names exists anywhere.
        """
        found = []
        for _, parts in self._kept(without):
            for position, part in enumerate(parts):
                if part == relation and position > 0 and position < len(parts) - 1:
                    found.append((parts, position))
        return tuple(found)

    def relations(self, *, without: Iterable[Sequence[str]] = ()) -> tuple:
        """Whatever the store has used as a relation, counted rather than declared."""
        found = set()
        for _, parts in self._kept(without):
            for position, part in enumerate(parts):
                if position > 0 and position < len(parts) - 1:
                    found.add(part)
        return tuple(sorted(found))

    def operator(self, relation: str, vectors: dict | None = None, *,
                 without: Iterable[Sequence[str]] = ()) -> np.ndarray | None:
        """**The relation's weight matrix, observed in one pass rather than fitted.**

        Stack the places of what stands *before* the relation as the rows of ``A`` and what stands
        *after* it as the rows of ``B``. The statement list is the shared ordered axis -- row ``i``
        of both is the same statement -- and the operator is ``ÃᵀB̃``, the cross-covariance.

        Observed, not fitted: no least squares, no target, no convergence, no learning rate, and
        nothing iterated. It is a statistic of the statements, and it changes when a statement
        arrives because the statement enters the sum. That is the whole of learning here.

        ``None`` where the relation is used by no surviving statement -- which is a relation the
        held-out store no longer contains -- and also where it is used exactly once: a
        cross-covariance over one row is that row, and it predicts that row and nothing else. Both
        are legitimate answers rather than errors, and returning a matrix in the second case would
        be reporting a lookup as an operator.
        """
        vectors = self.places(without=without) if vectors is None else vectors
        before, after = self._sides(relation, vectors, without)
        if len(before) < 2:
            return None
        return bind.association_matrix(np.stack(before), np.stack(after))

    def _sides(self, relation: str, vectors: dict,
               without: Iterable[Sequence[str]]) -> tuple[list, list]:
        """The placed rows either side of ``relation``, in statement order -- the shared axis."""
        before, after = [], []
        for parts, position in self.uses(relation, without=without):
            src, dst = parts[position - 1], parts[position + 1]
            if src in vectors and dst in vectors:
                before.append(vectors[src])
                after.append(vectors[dst])
        return before, after

    def carry(self, source: str, relation: str, vectors: dict | None = None, *,
              without: Iterable[Sequence[str]] = ()) -> np.ndarray | None:
        """Where ``relation`` takes ``source`` -- the forward step, with the matrix never formed.

        ``u @ (ÃᵀB̃)`` equals ``(Ãu) @ B̃`` by associativity, so the landing is computed without
        materializing the operator. That identity is what keeps the cost linear in the number of
        statements using the relation rather than quadratic in the width of the basis, and it is
        the reason a wide store is readable at all.
        """
        vectors = self.places(without=without) if vectors is None else vectors
        if source not in vectors:
            return None
        before, after = self._sides(relation, vectors, without)
        if len(before) < 2:
            return None
        return self._land(vectors[source], before, after)

    @staticmethod
    def _land(u, before, after) -> np.ndarray:
        """``ũ @ Ãᵀ @ B̃`` -- a place put through the operator its two sides define, never formed.

        The one landing computation: :meth:`carry` uses it, and so does every read in
        ``research/``, so a landing taken there and one taken here are the same arithmetic.
        """
        A, B = np.stack(before), np.stack(after)
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        # the source is connected the same way the operator's own sides are, so the read is of the
        # structure and not of the offset every coordinate in this basis shares
        u = u - u.mean()
        # (u @ Ãᵀ) @ B̃ -- the operator is never built
        return np.asarray(u @ A.T @ B)

    # ---- the attention row, derived -----------------------------------------------------------

    def attend(self, source: str, relation: str, candidates: Sequence[str], *,
               without: Iterable[Sequence[str]] = (),
               vectors: dict | None = None) -> np.ndarray:
        """**The attention row**: how much each candidate is where this relation leads.

        What a transformer computes here is ``softmax(q·k/√d)``, and both halves of that carry a
        constant -- ``√d`` is a temperature and the softmax forces the row to sum to one whether or
        not anything in it resolved. Here the row is the alignment of each candidate's place with
        where the operator landed. Nothing forces it to sum to anything, so a row of candidates
        that are all wrong is a row of small numbers rather than a confident distribution over
        rubbish.

        Returned in the order the candidates were given, with ``0`` for any the held-out store does
        not place.
        """
        vectors = self.places(without=without) if vectors is None else vectors
        landed = self.carry(source, relation, vectors, without=without)
        row = np.zeros(len(candidates))
        if landed is None:
            return row
        norm = float(np.linalg.norm(landed))
        if norm <= float(0):
            return row
        for i, c in enumerate(candidates):
            v = vectors.get(c)
            if v is None:
                continue
            d = float(np.linalg.norm(v))
            if d > float(0):
                row[i] = float(np.dot(v, landed)) / (d * norm)
        return row

    def between(self, source: str, target: str, *,
                among: Sequence[str] | None = None,
                without: Iterable[Sequence[str]] = (),
                vectors: dict | None = None) -> tuple:
        """**Which operator maps ``source`` to ``target``** -- the third reading of the triple.

        ``content + context -> operator``. The paper states three readings and this engine had two:
        ``carry`` and ``infer`` both answer *given the arrow and one end, find the other*. This
        answers *given both ends, which arrow* -- and it is the one that removes the last forcing,
        because a store whose operators can be found does not need anyone to name them. ``is_a`` is
        a label somebody chose; the arrow it labels is in the statements whether or not it was
        named.

        **The operator is a vertex and this is what makes it one.** Without this, an operator is
        only an index: ``carry`` uses it to select which statements to build a cross-covariance
        from, and its own place in the store plays no part in any read. A thing that can only be looked up
        is not a thing that can be activated, composed, or asked for.

        Returns ``((operator, alignment), ...)`` strongest first, over the operators offered or over
        every operator the store has. The alignment is the same one :meth:`attend` reports -- how
        much where this operator lands is where the target actually is -- and nothing forces the
        scores to sum to anything, so a pair no operator maps comes back as a list of small
        numbers rather than a confident ranking of rubbish.
        """
        vectors = self.places(without=without) if vectors is None else vectors
        want = vectors.get(target)
        if source not in vectors or want is None:
            return ()
        far = float(np.linalg.norm(want))
        if far <= float(0):
            return ()
        out = []
        for relation in (self.relations(without=without) if among is None else among):
            landed = self.carry(source, relation, vectors, without=without)
            if landed is None:
                continue
            norm = float(np.linalg.norm(landed))
            if norm <= float(0):
                continue
            out.append((relation, float(np.dot(want, landed)) / (far * norm)))
        # strongest first, ties on the operator's own name so the order is not the store's
        return tuple(sorted(out, key=lambda row: (-row[1], row[0])))

    def ahead(self, source: str, candidates: Sequence[str], *,
              among: Sequence[str] | None = None,
              without: Iterable[Sequence[str]] = (),
              vectors: dict | None = None) -> tuple:
        """**What comes next, with no arrow supplied** -- the read the finish line actually needs.

        ``carry``, ``attend`` and ``infer`` all require the caller to hand in a relation, and
        ``between`` requires *both* ends. Nothing here answered "given a context, what follows"
        without being told which arrow to travel -- and that is tokens in, tokens out.

        Each candidate is scored by the strongest alignment any relation in the store achieves
        between ``source`` and it: ``max_r align(carry(source, r), c)``. The name is not supplied,
        it is searched for, so a store whose relation names are tokens gets no help from the fact.
        This is :meth:`between` read the other way round -- that one fixes both ends and ranks
        arrows, this one fixes the source and ranks ends.

        **The arithmetic is identical to calling ``between`` once per candidate and taking its
        best row**, and the cost is not: ``between`` recomputes every landing for every target, so
        ranking ``C`` candidates over ``R`` relations costs ``R x C`` carries. A landing depends on
        the source and the relation and not on the candidate, so it is computed once and scored
        against every candidate -- ``R`` carries, then ``R x C`` dot products.

        Returns ``(scores, via)``: the alignment per candidate in the order given, and the relation
        each candidate's best came through, which says *which arrow the read found* and is the
        thing to inspect when it is wrong.

        **No floor is invented here.** A candidate nothing aligns with scores negative or zero and
        is returned as it stands; the refusal is the caller's ``> 0``, which is the rule
        :meth:`infer` already applies, rather than a level chosen in this method.
        """
        vectors = self.places(without=without) if vectors is None else vectors
        scores = np.full(len(candidates), -np.inf)
        via: list = [None] * len(candidates)
        if source not in vectors:
            return scores, tuple(via)
        # the candidates' own places, gathered once -- a candidate the held-out store does not
        # place cannot be scored by anything and stays at -inf
        places, norms = [], []
        for c in candidates:
            v = vectors.get(c)
            d = float(np.linalg.norm(v)) if v is not None else float(0)
            places.append(v if d > float(0) else None)
            norms.append(d)
        # relations in name order, so which arrow wins a tie is a property of the arrows and not
        # of the order the store was written down in
        for relation in (self.relations(without=without) if among is None else sorted(among)):
            landed = self.carry(source, relation, vectors, without=without)
            if landed is None:
                continue
            norm = float(np.linalg.norm(landed))
            if norm <= float(0):
                continue
            for i, v in enumerate(places):
                if v is None:
                    continue
                score = float(np.dot(v, landed)) / (norms[i] * norm)
                if score > scores[i]:
                    scores[i], via[i] = score, relation
        return scores, tuple(via)

    def infer(self, source: str, relation: str, *,
              among: Sequence[str] | None = None,
              without: Iterable[Sequence[str]] = (),
              hold_out: bool = True) -> Answer:
        """**The forward pass**: name what ``relation`` reaches from ``source``.

        With ``hold_out`` true -- the default -- any statement of exactly this relation from this
        source is removed from the operator *and* from the coordinate before anything is read, so
        the answer cannot be the store repeating itself. :attr:`Answer.stated` reports whether it
        would have been, and it is on the answer rather than in a log because a caller who has to
        go looking for that will not.
        """
        held = list(without)
        if hold_out:
            held += [parts for parts in self._statements
                     for position, part in enumerate(parts)
                     if part == relation and position > 0 and position < len(parts) - 1
                     and parts[position - 1] == source]
        vectors = self.places(without=held)
        candidates = list(among) if among is not None else [
            c for c in self.concepts(without=held) if c != source]
        row = self.attend(source, relation, candidates, without=held, vectors=vectors)

        into = {parts[position + 1] for _, parts in self._kept(held)
                for position, part in enumerate(parts)
                if part == relation and position > 0 and position < len(parts) - 1}
        stated = any(parts[position - 1] == source and parts[position + 1] in candidates
                     for _, parts in self._kept(held)
                     for position, part in enumerate(parts)
                     if part == relation and position > 0 and position < len(parts) - 1)

        named = None
        if len(row) and float(np.max(row)) > float(0):
            named = candidates[int(np.argmax(row))]
        return Answer(named=named, scores=row, stated=bool(stated),
                      reached=len({c for c in candidates if c in into}),
                      pool=len(candidates))

    # ---- learning, with nothing to retrain ----------------------------------------------------

    def relate(self, *parts: str) -> "Engine":
        """Add a statement. This is the whole of learning here.

        Returns a new engine rather than mutating, so a caller comparing before and after is
        comparing two stores rather than one store at two times. Nothing is retrained and no
        coordinate is updated, because no coordinate was stored: the next read derives a different
        operator because the new statement is in the sum it is derived from.
        """
        return Engine(self._statements + (tuple(str(p) for p in parts),))


def _walk(lattice: Any, *, seed: str, size: int) -> tuple:
    """Statements reconstructed from a mantle lattice, walking outward from ``seed``.

    A statement vertex is recognized by what it *is* rather than by a type: it is a vertex whose
    edges carry positions, and its parts are those edges read back in position order. Nothing in
    the store says "this is a statement", which is the same rule the rest of the package keeps.
    """
    seen, queue, found = {str(seed)}, [str(seed)], {}
    while queue and len(seen) < size:
        here = queue.pop(0)
        for direction in ("out", "in"):
            for r in lattice.relations(here, direction=direction):
                for other in (r.src, r.dst):
                    if other in seen:
                        continue
                    seen.add(other)
                    queue.append(other)
                    if len(seen) >= size:
                        break
    for vertex in seen:
        rows = lattice.relations(vertex, direction="out")
        placed = [(r.props.get(POSITION), r.dst) for r in rows if POSITION in (r.props or {})]
        if len(placed) > 2 and all(p is not None for p, _ in placed):
            parts = [d for _, d in sorted(placed, key=lambda pd: pd[0])]
            if all(p in seen for p in parts):
                found[vertex] = tuple(parts)
    return tuple(found[k] for k in sorted(found))
