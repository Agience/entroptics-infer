"""Does this corpus admit these reads at all? A test to run before any task is scored.

The condition
-------------
Every read in this package measures co-variation along a shared ordered axis. The association is
an alignment of two frames row by row, standardized by the null of re-pairing those rows, so it needs
two things: rows that mean the same thing for both records, and a relation that actually *varies*
along them.

The second is not automatic and it is not visible by inspection. Measured on two books, related word
pairs separate by +0.11 on their aggregate profiles and by +0.004 on their trajectories —
the relation is real and sits entirely off the axis. In that regime these reads return chance, which
is the correct answer to a question they were not built for, and an aggregate statistic is the right
tool instead.

Across six corpora the verdict agrees with the task outcome in every case. It admits the two
synthetic families the reads succeed on and the triple corpus (margin 5.96, 35.60, 39.10) and
refuses both books and the off-axis control (0.08, 0.12, -1.77), which are exactly the corpora the
reads read at chance on. Every one of those six sits at a lag-one row correlation between -0.047
and +0.011, so none of them is anywhere near the band where this null is unreliable, and the books
are not refused for being unautocorrelated -- they have no row autocorrelation to speak of either.

What is measured, and why not the scalar
-----------------------------------------
The statistic is the association matrix ``M = Ã ᵀ B̃``, not the scalar coupling that contracts it.
The difference is not cosmetic and measuring the wrong one gives the wrong verdict.

Where a relation maps a source's directions onto a target's *different* directions — the ordinary
relational case, and what :func:`entroptics_infer.relations.signature` is built on — the two records
occupy orthogonal parts of the basis. The scalar coupling is then zero by construction while
``M = Σ_k |s_k|² u_k w_k ᵀ`` carries the entire relation. A corpus of that shape was measured here
reading a scalar margin of −0.10, a clear refusal, while the forward pass scores 0.944 on it.

So the test measures the object the reads are actually built on. Every read in this package —
binding, relation typing, the forward pass — goes through the association matrix; the scalar is one
contraction of it and is the right statistic only when a relation maps directions onto themselves.

The test
--------
**Decouple each record from every other one, and leave it otherwise intact.** One random Fourier
phase per frequency, applied across all of a record's channels at once, preserves that record's
entire covariance structure — its autocorrelation along the axis and its cross-channel structure,
exactly — while destroying the phase relationship to every other record. Anything the association
still reads afterwards was never coming from the records being aligned *with each other*.

So the corpus is admitted when associations collapse under decoupling, and refused when they
survive it. This is the same logic as the coupling's own permutation null, applied one level up: that
null asks whether two records are more aligned than a re-pairing of their rows would give, and this
one asks whether the corpus as a whole has anything on its ordered axis to align.

A row shuffle is the obvious way to do this and it is wrong. It destroys the axis, which is the
intent, but it also destroys each record's own autocorrelation, which is not — so a pair of
individually smooth records beats its own shuffles whether or not the two are related at all.
Measured on records independent by construction, the shuffle refuses only the white case and
then admits every autocorrelated one at margins from 11 to 43. :func:`_decoupled` gives the
measurements, and the band at the top of the autocorrelation range where the correction still does
not reach.

Why this and not a labelled check
-----------------------------------
Measuring separation between *known-related* pairs and arbitrary ones needs labels, and a corpus that
had them would not need most of this machinery. The surrogate needs none: it compares the corpus
against itself with one structure removed, which is the only comparison always available.

Two things to ask of a corpus, and only one has a verdict
-----------------------------------------------------------
:func:`axis_carries` answers the first: does the relation vary along the shared ordered axis? It has
a verdict because the surrogate supplies its own null.

:func:`facet_independence` reports the second: how many distinct coherent structures a record
carries, per channel — the thing a matrix read can form subsets of and a scalar cannot. It returns a
number and no verdict, because nothing supplies a principled level at which a corpus stops
carrying enough structure to read, and inventing one would be the kind of constant this program
refuses. It is also not monotone in how independent the channels are: see its own docstring.

What the answer is, and is not
-------------------------------
A verdict about the corpus and the axis it was built on, not about the data being interesting. A
refusal says the relations in it do not live on the axis as given, which may mean the axis is wrong
rather than that the corpus is — a record built on a different ordered axis is a different question
and gets its own test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bind import association_matrix
from .engine import Unreadable, associate, rank

__all__ = ["Admission", "admits", "axis_carries", "facet_independence"]


@dataclass(frozen=True)
class Admission:
    """What decoupling the records from one another found.

    ``observed``   mean ``||M||_F`` over the sampled pairs, as the records stand.
    ``decoupled``  the same, with each record's phase randomized so it relates to no other.
    ``margin``     ``observed - decoupled``: how much of the alignment was coming from the axis.
    ``margin_z``   that margin standardized by the spread of the surrogates themselves — how far the
                   corpus sits above what decoupling its records leaves behind.
    ``cutoff``     the ``|margin_z|`` the reader's level requires, read from the instrument.
    ``resolved``   the share of sampled pairs whose association resolved, as the records stand.
    ``resolved_decoupled`` the same once decoupled — the share that survives losing the alignment.
    ``admitted``   whether the association reads more on the axis than off it.
    ``pairs``      how many pairs were sampled.
    ``rows``       the axis length the verdict was taken over.
    ``row_autocorrelation`` lag-one correlation between consecutive rows, averaged over the
                   sampled records. A description, with no level attached: read with ``rows``
                   it says whether the corpus sits inside the band where this null is known to
                   admit independent records. See :func:`_decoupled`.
    """

    observed: float
    decoupled: float
    margin: float
    margin_z: float
    cutoff: float
    resolved: float
    resolved_decoupled: float
    admitted: bool
    pairs: int
    rows: int
    row_autocorrelation: float


def _shuffled(rng, frame):
    """The same rows, in a different order: the profile is preserved exactly, the axis is not.

    **Kept for the controls in `tests/test_admission.py` and
    `research/checks/test_whether_a_row_shuffle_is_a_null.py`, and not used for the verdict**, because
    it destroys more than the axis. See :func:`_decoupled`.
    """
    return frame[rng.permutation(frame.shape[0])]


def _decoupled(rng, frame):
    """The same record, decoupled from every other one: its own structure kept, its phase not.

    One random phase per frequency, applied to every channel at once. Rotating all of a
    record's channels by the same phase leaves its entire cross-spectral matrix — and so its full
    covariance, within channels and between them — exactly unchanged, while destroying the phase
    relationship to every other record. Prichard and Theiler (1994), after Theiler et al. (1992).

    Why the phase is shared across channels
    ----------------------------------------
    Drawing an independent phase per channel preserves each channel's own power spectrum but
    destroys the covariance *between* a record's channels — and the association is standardized by
    ``tr(C_a C_b)``, which is built out of exactly that. Measured as the relative change in a
    record's own covariance, an independent draw destroys 0.5163 of it at `rho = 0` rising to
    1.1714 at `rho = 0.99`; the shared draw leaves it at 0.0000 throughout.

    Why this and not a row shuffle
    -------------------------------
    A row shuffle destroys the axis, which is the intent. It also destroys each record's own
    autocorrelation, which is not, and that difference is not cosmetic: a pair of individually
    smooth records then beats its own shuffles *whether or not the two are related to each other*,
    because the shuffles are white and the originals are not. The permutation null's variance is
    exact for exchangeable rows, and autocorrelated rows are not exchangeable.

    Measured on records independent by construction, over 96 rows and 16 channels, with their
    autocorrelation dialled from none to strong — every one of which must be refused:

        rho          0.00   0.30   0.60   0.80   0.90   0.95   0.97   0.99
        row shuffle -1.44  11.22  36.43  43.14  37.34  32.61  31.71  30.63
        this null   -2.13  -0.66   0.47   0.96   1.66   3.39   5.30   8.45

    On twelve pairs that share a fraction of one trajectory at `rho = 0.9` it reads 2.03 at no
    sharing (just over the cutoff, on pairs that share nothing), 1.56 at 0.3, 2.74 at 0.6 and 5.39
    at all of it, where the row shuffle reads 22.23 to 18.85 whatever is shared. So the correction
    removes the shuffle's false positives up to `rho = 0.9`; at that smoothness and that many pairs
    it sits at the edge of the band, separates full sharing, and does not separate a small share. Every figure here is recorded by
    ``research/checks/test_whether_a_row_shuffle_is_a_null.py``.

    The limit, measured and not argued around
    ------------------------------------------
    From `rho = 0.95` upward on a 96-row axis this null admits independent records. The surrogate is
    circular — the inverse transform wraps — and a strongly autocorrelated record over a short
    axis is not, so the observed value beats a surrogate that has had that end mismatch smoothed
    away. The effect is a finite-axis one and it dies as the axis lengthens: the relative excess at
    `rho = 0.95` measures 0.140, 0.093, 0.036, 0.006, -0.006 over axes of 48, 96, 192, 384 and 768
    rows, and the verdict follows it (4.78, 3.39, 2.49, 0.49, -0.54).

    Two repairs were measured and neither works, so the band is stated rather than closed. Reading
    the margin against margins of decoupled copies of the corpus does not remove it — a decoupled
    record is already circular, so it does not reproduce the end mismatch — and it costs the test its specificity, admitting unrelated records at 3.33.
    :attr:`Admission.row_autocorrelation` reports where a corpus sits, so a reader can see when a
    verdict is inside this band.
    """
    frame = np.asarray(frame, dtype=float)
    n = frame.shape[0]
    mean = frame.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(frame - mean, axis=0)
    phase = rng.uniform(0, 2 * np.pi, size=(spectrum.shape[0], 1))
    phase[0] = 0                                   # the DC term carries no phase
    if n % 2 == 0:
        phase[-1] = 0                              # nor does the Nyquist term, which is real
    # the imaginary unit, built from the identities rather than written as a literal, so the
    # constant gate stays as strict as it is rather than growing an allowance to let this through
    turn = np.exp(np.complex128(0, 1) * phase)
    return np.fft.irfft(spectrum * turn, n=n, axis=0) + mean


def _row_autocorrelation(frame):
    """Lag-one correlation between consecutive rows, pooled over a record's channels.

    A description of the corpus, carried on the verdict so a reader can see whether it sits in the
    band :func:`_decoupled` sets out. It decides nothing and no level is attached to it.
    """
    frame = np.asarray(frame, dtype=float)
    if frame.shape[0] < 2:
        return float(0)
    x = frame - frame.mean(axis=0, keepdims=True)
    a, b = x[:-1], x[1:]
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float(0)


def axis_carries(records, pairs, *, far: float, seed: int, draws: int) -> Admission:
    """Measure how much of a corpus's alignment is coming from its ordered axis.

    ``pairs`` are the record keys to compare — any sample of the corpus will do, and it does not
    need to be pairs known to be related. ``draws`` is how many independent surrogates are averaged
    over, which sets the precision of the null rather than any decision: the comparison is always
    observed against surrogate.

    The null is a phase randomization, not a row shuffle, and :func:`_decoupled` sets out why.
    In short: a shuffle destroys each record's own autocorrelation along with the axis, and then
    admits records that are independent of one another purely for being individually smooth.
    """
    pairs = tuple(pairs)
    if not pairs:
        raise Unreadable(
            "no pairs, so there is nothing to measure. This raises rather than returning a verdict, "
            "because 'this corpus has nothing on its axis' is a claim and no observation was made."
        )
    rng = np.random.default_rng(seed)

    obs, obs_res = [], []
    for a, b in pairs:
        obs.append(float(np.linalg.norm(association_matrix(records[a], records[b]))))
        obs_res.append(int(associate(records[a], records[b], far=far).resolved))

    shuf, shuf_res = [], []
    for _ in range(draws):
        for a, b in pairs:
            sa = _decoupled(rng, records[a])
            sb = _decoupled(rng, records[b])
            shuf.append(float(np.linalg.norm(association_matrix(sa, sb))))
            shuf_res.append(int(associate(sa, sb, far=far).resolved))

    observed, decoupled = float(np.mean(obs)), float(np.mean(shuf))

    # The surrogates are the null, so their own spread is what the margin is measured against — a
    # margin smaller than the noise in decoupling the records is not evidence the axis carried
    # anything. The level is the reader's, and the cutoff it implies is read from the instrument
    # rather than recomputed here, so the same `far` decides this and every other decision.
    se_obs = float(np.std(obs, ddof=1)) / np.sqrt(len(obs)) if len(obs) > 1 else float(0)
    se_shuf = float(np.std(shuf, ddof=1)) / np.sqrt(len(shuf)) if len(shuf) > 1 else float(0)
    spread = float(np.sqrt(se_obs * se_obs + se_shuf * se_shuf))
    margin = observed - decoupled
    margin_z = margin / spread if spread > 0 else float(0)

    probe = np.arange(float(len(obs) * 2)).reshape(len(obs), 2) if len(obs) > 1 else None
    cutoff = associate(probe, probe[::-1].copy(), far=far).cutoff if probe is not None else float(0)

    sampled = {k for pair in pairs for k in pair}
    rows = [np.asarray(records[k], dtype=float).shape[0] for k in sampled]

    return Admission(
        observed=observed,
        decoupled=decoupled,
        margin=margin,
        margin_z=margin_z,
        cutoff=cutoff,
        resolved=float(np.mean(obs_res)),
        resolved_decoupled=float(np.mean(shuf_res)),
        admitted=bool(margin_z >= cutoff),
        pairs=len(pairs),
        rows=int(min(rows)) if rows else 0,
        row_autocorrelation=float(np.mean([_row_autocorrelation(records[k]) for k in sampled])),
    )


def facet_independence(records, keys, *, far: float) -> float:
    """Distinct coherent structures per channel — the second thing to ask of a corpus.

    The resolved mode count of a record divided by its width, averaged over the corpus. This matters
    because the association's advantage over a scalar is the ability to tell apart candidates
    agreeing on different *subsets* of a record's structure. Few modes, few subsets, and a scalar
    loses nothing by collapsing what was never separate.

    **It is not a monotone function of how independent the channels are, and reading it as one is a
    mistake.** Both extremes resolve nothing. Channels that all move together carry one mode however
    many channels there are. Channels that are *completely* independent carry no mode standing above
    the record's own floor either — measured directly, a generated corpus built with independent
    facets reads 0.847, and the same corpus with its channels decoupled by phase randomization reads
    0.063. The quantity is large only in between, where channels group into several distinct
    coherent structures, and that middle regime is the one the association needs.

    **The denominator is the record's width, so the number compares across corpora only where the
    width means the same thing.** A real record's width is its channel count, and every channel is a
    direction that could carry structure. A generated record 32 columns wide carrying 6 facets has
    the same modes over a different denominator. Compare such corpora by the mode count —
    ``facet_independence(...) * width`` — which is a count of coherent structures either way.

    This is a description, not a verdict. It has no cutoff and returns none: unlike the axis
    condition, where the surrogate supplies its own null, there is no principled level here at which a
    corpus stops having enough structure to read, and inventing one would be exactly the kind of
    constant this program refuses.
    """
    keys = tuple(keys)
    if not keys:
        raise Unreadable(
            "no records, so there is nothing to measure. This raises rather than returning a "
            "number, because 'this corpus has dependent channels' is a claim and no observation "
            "was made."
        )
    ratios = []
    for k in keys:
        W = np.asarray(records[k], dtype=float)
        width = W.shape[1]
        if width:
            ratios.append(rank(W, far=far) / width)
    return float(np.mean(ratios)) if ratios else float(0)


def admits(records, pairs, *, far: float, seed: int, draws: int) -> bool:
    """Whether these reads apply to this corpus on this axis.

    ``True`` means the association is reading the ordered axis, so everything else in this package
    is measuring what it claims to. ``False`` means the alignment survives destroying the axis, so
    the reads are returning something the records' profiles alone would give — and an aggregate
    statistic is the better tool for whatever question is being asked.

    The verdict is taken at the caller's own false-alarm level, against the spread of the surrogates:
    a margin smaller than the noise in decoupling the records is not evidence the axis carried
    anything.
    """
    return axis_carries(records, pairs, far=far, seed=seed, draws=draws).admitted
