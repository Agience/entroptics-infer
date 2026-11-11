"""Corpora, generated fresh from a stated process. Nothing here is loaded from disk.

Why synthetic, and what it is allowed to prove
---------------------------------------------
A synthetic corpus cannot show that an effect exists in the world. What it can do, and what real
data cannot, is let a *calculation be checked against a known answer*: the generator knows which key
is correct and how many facets two records share, so a method's output is compared with the truth
rather than with another method's output.

**That is the whole of what these corpora establish, and it is worth being blunt about the limit.**
The multi-facet advantage measured on them has been run on five real corpora in `research/` and
reproduces on none of them — `exp25` supplies the adversarial configuration the construction is
built around, measured in 70% of a real corpus's questions, and the advantage is still absent. So
nothing generated here should be read as evidence that the effect occurs outside a generator. It is
evidence that the arithmetic is what it is claimed to be.

The risk, named
----------------------
A generator designed around the method it is about to validate proves nothing. Two things guard
against that here. The generative process below is stated in full and is not derived from the
instrument — independent latent channels loaded onto fixed directions, which is the standard
factor model and predates all of this. And the discriminating parameter (how much amplitude a
narrow distractor carries against a broad true key) is swept, not chosen, so every experiment
reports the curve and the crossover rather than one point picked from it.

The data policy, which has two tracks
-------------------------------------
**Nothing in this module reads a file.** Every corpus here is built from a stated seed at the moment
it is consumed, and the generator below is the record of it.

That is one track. On the other, thirteen experiments fetch real corpora — books, weather
stations, air quality, inertial sensors — into `research/corpora/`, which is in `.gitignore` because it is a cache and not
a source. Each of those records the SHA-256 of what it received, so the corpus a figure was
measured on is identified even though it is not committed.

So the rule is not "nothing is loaded from disk". It is: a corpus is either generated from a stated
seed or fetched fresh with its digest recorded, and never carried along as an artefact nobody can
trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ChainCorpus", "FacetCorpus", "Record", "RelationCorpus", "TripleCorpus",
           "chain_corpus", "facet_corpus", "relation_corpus", "triple_corpus"]


@dataclass(frozen=True)
class Record:
    """One record, and the truth about how it was made.

    ``frame``   the ``(T, D)`` field a reader is given.
    ``facets``  which latent channels it actually carries. The method never sees this.
    ``amps``    the amplitude it carries each of them at. Also never seen.
    """

    frame: np.ndarray
    facets: tuple[int, ...]
    amps: tuple[float, ...]


@dataclass(frozen=True)
class FacetCorpus:
    """A query, a set of candidate keys, and which key is the correct one.

    ``truth`` is the index into ``keys`` of the record that shares *every* facet the query carries.
    It is the correct answer by the generator's construction, not by any method's opinion.
    """

    query: Record
    keys: tuple[Record, ...]
    truth: int
    loadings: np.ndarray = field(repr=False)


def _channels(rng, T: int, n: int) -> np.ndarray:
    """``(n, T)`` independent latent time-courses — the facets, as the factor model has them."""
    return rng.normal(size=(n, T))


def _loadings(rng, n: int, D: int) -> np.ndarray:
    """``(min(n, D), D)`` fixed directions the facets occupy in the shared basis.

    Orthonormalized so that facets are genuinely independent *directions* and not accidentally
    correlated ones — the generator's claim that a record "carries R facets" has to mean R
    independent things, or the ground truth is not what it says it is.

    **It returns `min(n, D)` rows and not always `n`, and a caller must not assume otherwise.**
    There are no more than `D` orthonormal directions in `D` dimensions, so asking for more is
    asking for something that does not exist; this returns what does. The rows it returns are always
    orthonormal, so a corpus that only ever indexes below `len(...)` is exactly what it claims to
    be — and every corpus in this module is, which is why the published figures are unaffected by
    this note.

    The failure mode is loud rather than silent: a caller indexing past the end gets an
    ``IndexError``, not a quietly wrong direction. `facet_corpus` checks for the one configuration
    that can reach past it and says so in words instead.
    """
    A = rng.normal(size=(n, D))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:n]


def _emit(rng, channels, loadings, facets, amps, noise) -> np.ndarray:
    """A record: its facets' channels loaded onto their directions, under noise.

    ``X = sum_f a_f * s_f u_f^T + e``  — the standard factor model, nothing instrument-specific.
    """
    T, D = channels.shape[1], loadings.shape[1]
    X = np.zeros((T, D))
    for f, a in zip(facets, amps):
        X += a * np.outer(channels[f], loadings[f])
    return X + rng.normal(size=(T, D)) * noise


def facet_corpus(
    *,
    seed: int,
    facets: int,
    n_keys: int,
    T: int,
    D: int,
    narrow_gain: float,
    noise: float,
    subset_size: int | None = None,
    facet_coupling: float = 0.0,
) -> FacetCorpus:
    """One trial of the discriminating case: a broad true key against narrow strong distractors.

    The construction, stated so it can be argued with:

    * The query carries facets ``0..R-1``, at amplitude ``1/sqrt(R)`` each. R is ``facets``.
    * The true key carries the same ``0..R-1``, at the same amplitude. It shares everything the
      query has, moderately.
    * Each distractor carries a strict, randomly chosen subset of size ``m``, at amplitude
      ``narrow_gain / sqrt(m)``. It shares *less of the query* but *more loudly*.

    **Every record carries the same total signal energy**, which is what ``1/sqrt(k)`` is for. Giving
    each facet unit amplitude instead would grow a record's signal energy with R while its noise
    energy stayed at ``noise^2 * D``, so the task would get harder as R rose for a reason that had
    nothing to do with facets — and a falling resolved rank would then look like a saturating counter
    when it was the instrument correctly refusing a frame whose SNR had collapsed. Holding the energy
    fixed keeps the facet count and the signal-to-noise ratio separable.

    ``narrow_gain`` is the discriminating parameter and it is swept, never chosen. A distractor
    carrying ``m`` of ``R`` facets shares ``g * sqrt(m/R)`` of the query's energy against the true
    key's ``1``, so the crossover sits at ``g = sqrt(R/m)``. Above it the distractor carries more
    total shared energy than the true key does, and any method ranking by shared magnitude *must*
    prefer it. That crossover is arithmetic, known before the run, and is what the experiment plots
    against.

    **``subset_size`` fixes ``m``, and an experiment should fix it.** The signal separating the true
    key from a distractor is *breadth* — the true key is the only record carrying every facet the
    query carries. How strong that signal is depends entirely on ``m``: at ``m = 1`` of ``R = 6``
    breadth is six-to-one and unmistakable, while at ``m = 5`` of ``R = 6`` the distractor differs
    by one facet and carries each of the five it has at ``g/sqrt(5)`` against the true key's
    ``1/sqrt(6)`` — more than double the amplitude per facet. At ``m`` close to ``R`` the distractor
    is, by any reasonable reading of the records, the better match, and calling the broad key "true"
    there labels the corpus against its own geometry.

    So ``m`` is a dimension of the design, not a nuisance to average over. Left as ``None`` it is
    drawn uniformly from ``1..R-1``, which mixes unmistakable and ambiguous trials into one number;
    experiments fix it and report per ``m``.

    ``facet_coupling`` makes the facets non-independent by giving each of them a share of one common
    time-course. At the default ``0`` they are independent, which is what every experiment before
    `exp15` used. Real channels are not: eleven Beijing air-quality channels resolve 3.5 modes,
    because pollutants and meteorology each move together. The parameter exists so that difference
    can be varied rather than argued about.

    At ``facets = 1`` there is no strict non-empty subset, so a distractor carries a *disjoint*
    facet instead. This is the control: with one facet there is one thing to match on, a scalar
    carries it perfectly, and every method must tie. A method that wins here is reading something
    other than facet structure.
    """
    rng = np.random.default_rng(seed)
    R = int(facets)
    # Enough channels for the query's facets plus a disjoint one per key for the R = 1 control.
    n_channels = R + n_keys
    if R == 1 and D < 1 + n_keys:
        # The R = 1 control gives every distractor its own *disjoint* facet, so it needs one
        # direction per key on top of the query's — and there are only D of them in D dimensions.
        # At R > 1 the distractors draw from the query's own facets and any D > R will do, which is
        # why this is checked here and not in `_loadings`.
        raise ValueError(
            f"the R = 1 control needs a disjoint direction for each of {n_keys} keys plus the "
            f"query's, so it needs D >= {1 + n_keys}; got D = {D}. At R > 1 the distractors share "
            f"the query's facets and this width is fine — it is the control that cannot be built."
        )
    channels = _channels(rng, T, n_channels)
    if facet_coupling:
        # Facets that are not independent of each other. `facet_coupling` is how much of a single
        # common time-course every facet carries, so at 0 they are independent — the default, and
        # what every earlier experiment used — and as it rises they increasingly move together.
        #
        # This is what real channels look like. Beijing air quality has eleven of them and a station
        # record resolves 3.5: PM2.5 moves with PM10, temperature with dew point. A generated corpus
        # with orthonormal loadings and independent latents has no such structure, and whether that
        # difference is why the advantage does not transfer is a question worth being able to ask.
        c = float(facet_coupling)
        common = rng.normal(size=T)
        channels = np.sqrt(max(0.0, 1.0 - c * c)) * channels + c * common
    loadings = _loadings(rng, n_channels, D)

    q_facets = tuple(range(R))
    broad = (1.0 / np.sqrt(R),) * R                          # unit total signal energy
    query = Record(
        frame=_emit(rng, channels, loadings, q_facets, broad, noise),
        facets=q_facets,
        amps=broad,
    )

    true_key = Record(
        frame=_emit(rng, channels, loadings, q_facets, broad, noise),
        facets=q_facets,
        amps=broad,
    )

    keys = [true_key]
    for j in range(n_keys - 1):
        if R > 1:
            m = int(rng.integers(1, R)) if subset_size is None else int(subset_size)
            if not 1 <= m < R:
                raise ValueError(
                    f"a distractor must carry a strict, non-empty subset: 1 <= m < R, got m={m} "
                    f"with R={R}. At m = R the distractor IS the true key and the corpus has no "
                    f"answer."
                )
            sub = tuple(int(f) for f in rng.choice(R, size=m, replace=False))
        else:
            sub = (R + j,)                                   # the control: a disjoint facet
            m = 1
        amps = (narrow_gain / np.sqrt(m),) * m               # same energy budget, spent narrowly
        keys.append(
            Record(frame=_emit(rng, channels, loadings, sub, amps, noise), facets=sub, amps=amps)
        )

    order = rng.permutation(len(keys))
    shuffled = tuple(keys[i] for i in order)
    truth = int(np.where(order == 0)[0][0])
    return FacetCorpus(query=query, keys=shuffled, truth=truth, loadings=loadings)


# ── a corpus whose relations are matrices, and whose count is not declared ───────────────────────


@dataclass(frozen=True)
class RelationCorpus:
    """Vertices, edges between them, and how many kinds of relation the generator used.

    ``n_relations`` and ``true_type`` are the answer, and no method is given either. They exist
    so a measurement can be scored, and nothing reads them before it is scored.
    """

    records: dict                      # vertex id -> (T, D) frame
    edges: tuple                       # (src, dst) pairs, in generation order
    true_type: tuple                   # the generating relation index per edge — never shown
    n_relations: int                   # ditto
    roles: np.ndarray = field(repr=False)   # (R, 2, D): each relation's source and target direction


def relation_corpus(
    *,
    seed: int,
    n_relations: int,
    edges_per_relation: int,
    n_vertices: int,
    T: int,
    D: int,
    noise: float,
    share_vertices: bool = True,
) -> RelationCorpus:
    """A corpus in which a relation is a matrix, and the number of them is not written down.

    The construction, which is RESCAL's ``X_k ≈ A R_k Aᵀ`` run forwards rather than fitted:

    * Relation ``k`` owns two directions in the shared basis, a source ``u_k`` and a target
      ``w_k``, orthonormalized across all relations so distinct relations are genuinely distinct.
    * An edge of relation ``k`` between vertices ``a`` and ``b`` is created by drawing one latent
      time-course ``s`` for that edge, adding ``s uₖᵀ`` to ``a``'s record and ``s wₖᵀ`` to ``b``'s.

    The consequence is the point of the construction. The cross-covariance of the two endpoint
    records carries ``|s|² uₖ wₖᵀ`` — a rank-one matrix that depends only on which relation it
    is, not on which vertices or which edge. Every edge of relation ``k`` therefore contributes
    the same direction in the space of ``D × D`` matrices, and edges of different relations
    contribute different ones. A relation is a matrix here in the literal sense, and the number of
    distinct relations is the number of independent directions the observed matrices occupy.

    ``share_vertices`` sets which of two regimes the corpus is in, and they measure different
    things.

    * ``False`` — every edge gets its own fresh pair of vertices, so a record participates in
      exactly one relation. Each observed matrix is then a clean multiple of its relation's
      ``u_k w_kᵀ`` and the stack occupies exactly ``R`` directions. This is the calibration
      regime: it is where a counted answer can be checked against a known one.
    * ``True`` — edges are drawn over a pool of ``n_vertices``, so a record is the sum of every
      relation its vertex takes part in, and the cross-covariance of two endpoints carries terms
      from their *other* edges as well as the one between them. The observed structure then occupies
      more directions than ``R``, and legitimately so: what is being counted is the effective
      dimensionality of the relational structure as it appears in the records, which is not the same
      quantity as the number of relations the generator used. This is the realistic regime and the
      one a corpus is actually in.

    Nothing is labelled in either. The edges carry no type, the records carry no annotation, and
    ``n_relations`` is returned only so a measurement can be scored against it afterwards.
    """
    rng = np.random.default_rng(seed)
    R = int(n_relations)
    if n_vertices < 2:
        raise ValueError("a relation needs two endpoints; n_vertices must be at least 2")
    if 2 * R > D:
        raise ValueError(
            f"{R} relations need {2 * R} orthogonal role directions but the shared basis has only "
            f"{D}. Beyond that the roles are no longer independent and the corpus cannot claim to "
            f"contain R distinct relations — so this refuses rather than silently generating fewer."
        )

    # Source and target directions for every relation, orthonormal across the whole set.
    roles = _loadings(rng, 2 * R, D).reshape(R, 2, D)

    records: dict = {}
    if share_vertices:
        records = {f"v{i}": np.zeros((T, D)) for i in range(n_vertices)}

    edges, types = [], []
    for k in range(R):
        for e in range(edges_per_relation):
            if share_vertices:
                a, b = (f"v{i}" for i in rng.choice(n_vertices, size=2, replace=False))
            else:
                a, b = f"s{k}_{e}", f"t{k}_{e}"
                records.setdefault(a, np.zeros((T, D)))
                records.setdefault(b, np.zeros((T, D)))
            s = rng.normal(size=T)
            records[a] = records[a] + np.outer(s, roles[k, 0])
            records[b] = records[b] + np.outer(s, roles[k, 1])
            edges.append((a, b))
            types.append(k)

    for key in records:
        records[key] = records[key] + rng.normal(size=(T, D)) * noise

    return RelationCorpus(
        records=records,
        edges=tuple(edges),
        true_type=tuple(types),
        n_relations=R,
        roles=roles,
    )


# ── a corpus a signal has to travel through ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChainCorpus:
    """Vertices strung in a line, with content transmitted from each to the next.

    ``transmission`` is the fraction of a record that survives one step. It is the generator's
    parameter, swept by the experiment, and no method is given it — the point is to recover how
    far a signal travels without being told how fast it decays.
    """

    records: dict
    order: tuple                       # vertex ids, in chain order
    transmission: float
    channels: int


def _structured(rng, T: int, D: int, channels: int) -> np.ndarray:
    """A record carrying ``channels`` independent channels at unit total signal energy."""
    dirs = _loadings(rng, channels, D)
    X = np.zeros((T, D))
    for k in range(channels):
        X += np.outer(rng.normal(size=T), dirs[k]) / np.sqrt(channels)
    return X


def chain_corpus(
    *,
    seed: int,
    length: int,
    transmission: float,
    T: int,
    D: int,
    noise: float,
    channels: int = 4,
) -> ChainCorpus:
    """A line of records, each carrying a fraction of the one before it.

        X_0       = structured content
        X_{t+1}   = a * X_t + sqrt(1 - a^2) * fresh content

    with ``a = transmission``. Each record is then observed under its own noise. The construction
    keeps every record's total signal energy the same, so what changes along the chain is *what* the
    content is and not how loud it is — otherwise a decaying signal and a quietening one would be the
    same measurement.

    The fraction of the origin surviving at step ``t`` is ``a^t``, so the origin's association with
    the chain decays geometrically and crosses any positive floor after finitely many steps. Where it
    crosses is what an experiment reads off, and it is never supplied.

    ``transmission = 0`` is the control: every record is independent of the one before it, so the
    origin reaches nothing and a horizon of 0 is the correct answer.
    """
    if not 0.0 <= transmission <= 1.0:
        raise ValueError(
            f"transmission is the fraction of a record surviving one step and must lie in [0, 1]; "
            f"got {transmission}"
        )
    rng = np.random.default_rng(seed)
    carried = _structured(rng, T, D, channels)
    keep = float(transmission)
    fresh_share = float(np.sqrt(max(0.0, 1.0 - keep * keep)))

    order, records = [], {}
    for t in range(length):
        if t:
            carried = keep * carried + fresh_share * _structured(rng, T, D, channels)
        name = f"c{t}"
        order.append(name)
        records[name] = carried + rng.normal(size=(T, D)) * noise

    return ChainCorpus(records=records, order=tuple(order),
                       transmission=keep, channels=int(channels))


# ── a corpus of triples, where every source stands in every relation ─────────────────────────────


@dataclass(frozen=True)
class TripleCorpus:
    """Sources, each related to one target per relation kind.

    The construction is what makes the operator's role checkable. Every source is related to every
    one of its targets — just by a different kind — so a method that ignores which relation is
    asked for cannot do better than chance among them. Any accuracy above chance is the operator
    doing work.

    ``targets[(source, kind)]`` is the correct answer. ``n_relations`` is the generator's count.
    Neither is shown to a method: the operator is supplied as an exemplar pair, never as an index.
    """

    records: dict
    sources: tuple
    targets: dict                      # (source, kind) -> target vertex
    n_relations: int
    roles: np.ndarray = field(repr=False)


def triple_corpus(
    *,
    seed: int,
    n_relations: int,
    n_sources: int,
    T: int,
    D: int,
    noise: float,
    n_targets: int | None = None,
) -> TripleCorpus:
    """Every source related to one target per kind, so the kind is the only thing that separates them.

    Relation ``k`` owns a source direction ``u_k`` and a target direction ``w_k``. For each source
    ``a`` and each kind ``k`` a fresh latent ``s`` is placed along ``u_k`` in ``a``'s record and along
    ``w_k`` in the target's, so the cross-covariance of the pair carries ``|s|² u_k w_kᵀ`` — the same
    matrix for every pair of that kind, and a different one for every other kind.

    A source therefore carries all ``R`` relations at once and each of its targets carries one. The
    question *"which of a's targets is the one reached by this relation?"* has no answer from
    association strength alone, which is exactly the control the experiment needs.
    """
    rng = np.random.default_rng(seed)
    R = int(n_relations)
    if 2 * R > D:
        raise ValueError(
            f"{R} relations need {2 * R} orthogonal role directions but the basis has {D}"
        )
    if n_targets is not None and n_targets < R:
        raise ValueError(
            f"a source needs {R} distinct targets but the pool holds {n_targets}; with fewer, two "
            f"of its relations would point at the same record and the question would have no answer"
        )
    roles = _loadings(rng, 2 * R, D).reshape(R, 2, D)

    records: dict = {}
    targets: dict = {}
    sources = tuple(f"a{i}" for i in range(n_sources))
    pool = tuple(f"t{j}" for j in range(n_targets)) if n_targets else None
    stride = max(1, n_targets // R) if n_targets else 0

    for i, a in enumerate(sources):
        records.setdefault(a, np.zeros((T, D)))
        for k in range(R):
            b = pool[(i + k * stride) % n_targets] if pool else f"{a}_k{k}"
            records.setdefault(b, np.zeros((T, D)))
            s = rng.normal(size=T)
            records[a] = records[a] + np.outer(s, roles[k, 0])
            records[b] = records[b] + np.outer(s, roles[k, 1])
            targets[(a, k)] = b

    for v in records:
        records[v] = records[v] + rng.normal(size=(T, D)) * noise

    return TripleCorpus(records=records, sources=sources, targets=targets,
                        n_relations=R, roles=roles)


def line_neighbours(order):
    """Adjacency of the chain, as the callable the walk takes."""
    index = {v: i for i, v in enumerate(order)}

    def nbrs(v):
        i = index[v]
        out = []
        if i:
            out.append(order[i - 1])
        if i + 1 < len(order):
            out.append(order[i + 1])
        return out

    return nbrs
