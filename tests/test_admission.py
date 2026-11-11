"""The admission test: does a corpus have anything on its ordered axis?

The property under test is that the verdict tracks where the relation lives, and nothing else.
Two corpora with identical per-record structure must be told apart when one carries its relation on
the shared axis and the other does not — which is exactly what the shuffle is for.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import admission, engine

FAR = 0.05
T, D = 96, 24
SEED, DRAWS = 7, 3


def _dial(rng, lam, n_pairs):
    """Pairs whose relation lies on the ordered axis to degree ``lam``.

    A record is an activation trajectory times a context profile. The partner always shares the
    profile; it shares the trajectory to degree ``lam``. So at ``lam = 0`` the two are as related as
    ever — same profile — and *none* of that relation is on the axis.
    """
    records, pairs = {}, []
    for i in range(n_pairs):
        A = rng.normal(size=(2, D))
        Q, _ = np.linalg.qr(A.T)
        profile = Q.T[0]
        a = rng.normal(size=T)
        b = lam * a + np.sqrt(max(0.0, 1.0 - lam * lam)) * rng.normal(size=T)
        records[f"q{i}"] = np.outer(a, profile) + rng.normal(size=(T, D)) * 0.1
        records[f"c{i}"] = np.outer(b, profile) + rng.normal(size=(T, D)) * 0.1
        pairs.append((f"q{i}", f"c{i}"))
    return records, pairs


def test_a_relation_across_orthogonal_roles_is_admitted():
    """A relation across orthogonal roles, which a scalar statistic would refuse.

    Where a relation maps a source's directions onto a target's *different* directions — the
    ordinary relational case — the two records occupy orthogonal parts of the basis and the scalar
    coupling is zero by construction, while the association matrix carries the whole relation. A
    test measuring the scalar refuses such a corpus, and this one must not: the forward pass scores
    0.944 on exactly this shape.
    """
    rng = np.random.default_rng(11)
    records, pairs = {}, []
    for i in range(20):
        A = rng.normal(size=(2, D))
        Q, _ = np.linalg.qr(A.T)
        source_role, target_role = Q.T[0], Q.T[1]      # orthogonal by construction
        s = rng.normal(size=T)
        records[f"s{i}"] = np.outer(s, source_role) + rng.normal(size=(T, D)) * 0.1
        records[f"t{i}"] = np.outer(s, target_role) + rng.normal(size=(T, D)) * 0.1
        pairs.append((f"s{i}", f"t{i}"))
    verdict = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
    assert verdict.admitted, (
        f"a relation across orthogonal roles was refused at margin {verdict.margin_z:.2f}; the "
        f"statistic is contracting the association matrix away"
    )


def test_a_relation_on_the_axis_is_admitted():
    rng = np.random.default_rng(0)
    records, pairs = _dial(rng, lam=1.0, n_pairs=24)
    verdict = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
    assert verdict.admitted
    assert verdict.margin_z > verdict.cutoff


def test_a_relation_off_the_axis_is_refused():
    """The case that matters.

    These records are just as related as the ones above — identical profiles — but none of the
    relation varies along the shared axis. The reads would return chance here, and the test has to
    say so *before* a task is scored rather than after it is lost.
    """
    rng = np.random.default_rng(1)
    records, pairs = _dial(rng, lam=0.0, n_pairs=24)
    verdict = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
    assert not verdict.admitted
    assert verdict.margin_z < verdict.cutoff


def test_the_margin_grows_with_how_much_lies_on_the_axis():
    """The verdict is a threshold on a graded quantity, and the grade is the useful part."""
    margins = []
    for lam in (0.0, 0.3, 0.6, 1.0):
        rng = np.random.default_rng(2)
        records, pairs = _dial(rng, lam=lam, n_pairs=20)
        margins.append(admission.axis_carries(records, pairs, far=FAR,
                                              seed=SEED, draws=DRAWS).margin_z)
    assert margins == sorted(margins), f"the margin did not grow with the dial: {margins}"


def _ar1(rng, rho, t=96, d=16):
    """One record: `d` independent AR(1) columns, shared with nothing."""
    x = np.zeros((t, d))
    e = rng.normal(size=(t, d))
    x[0] = e[0]
    for i in range(1, t):
        x[i] = rho * x[i - 1] + np.sqrt(max(0.0, 1 - rho * rho)) * e[i]
    return x


def test_what_each_surrogate_preserves_and_what_it_destroys():
    """The test's validity rests on the null removing the alignment and nothing else.

    A row shuffle preserves a record's *marginal* profile exactly -- the same rows, reordered -- which
    makes it look like the right null. It is not enough. What the association is
    standardized by is the record's covariance, and a shuffle destroys that along with the axis,
    which is why it admits records that share nothing (see the two tests at the end of this file).

    This pins both halves: the shuffle keeps the marginal and loses the covariance; decoupling keeps
    the covariance. A change that broke either would change verdicts on every real corpus.
    """
    rng = np.random.default_rng(3)
    frame = admission._decoupled(np.random.default_rng(5), _ar1(rng, 0.8))

    shuffled = admission._shuffled(np.random.default_rng(4), frame)
    assert shuffled.shape == frame.shape
    assert np.allclose(np.sort(frame, axis=0), np.sort(shuffled, axis=0)), (
        "the shuffle changed a record's marginal profile; it is meant to be the same rows reordered"
    )
    assert np.allclose(frame.sum(axis=0), shuffled.sum(axis=0))

    # A shuffle permutes rows identically across channels, so the CROSS-CHANNEL covariance is
    # untouched -- which is exactly why it looked like a clean null. What it destroys is the
    # record's structure along the axis, and that is what the association was reading.
    before = np.cov(frame, rowvar=False)
    across = float(np.linalg.norm(np.cov(shuffled, rowvar=False) - before) / np.linalg.norm(before))
    assert across < 1e-9, (
        f"the row shuffle changed the cross-channel covariance by {across:.4f}; permuting rows "
        f"reorders samples and cannot change a covariance taken over them"
    )
    along = admission._row_autocorrelation(frame)
    assert along > 0.5, f"this fixture should be strongly autocorrelated, and reads {along:.3f}"
    assert abs(admission._row_autocorrelation(shuffled)) < along / 2, (
        f"the row shuffle left a lag-one row correlation of "
        f"{admission._row_autocorrelation(shuffled):+.3f} against the original's {along:+.3f}. It "
        f"is meant to destroy exactly this, and the whole correction rests on it doing so."
    )

    decoupled = admission._decoupled(np.random.default_rng(4), frame)
    kept = float(np.linalg.norm(np.cov(decoupled, rowvar=False) - before) / np.linalg.norm(before))
    assert kept < 1e-9, f"decoupling changed the record's own covariance by {kept:.4f}"
    assert abs(admission._row_autocorrelation(decoupled) - along) < 0.15, (
        f"decoupling moved the lag-one row correlation from {along:+.3f} to "
        f"{admission._row_autocorrelation(decoupled):+.3f}; it is meant to preserve it"
    )


def test_the_margin_separates_a_weak_pass_from_a_strong_one():
    """The verdict is a floor; the margin is the grade, and both are reported.

    Across the corpora measured the verdict agrees with the task outcome every time, but a boolean
    carries less than the number behind it — a corpus barely over the line and one far past it are
    different facts, and a caller deciding whether to trust a read wants the second.
    """
    rng = np.random.default_rng(5)
    weak_records, weak_pairs = _dial(rng, lam=0.15, n_pairs=20)
    rng = np.random.default_rng(5)
    strong_records, strong_pairs = _dial(rng, lam=0.9, n_pairs=20)

    weak = admission.axis_carries(weak_records, weak_pairs, far=FAR, seed=SEED, draws=DRAWS)
    strong = admission.axis_carries(strong_records, strong_pairs, far=FAR, seed=SEED, draws=DRAWS)
    assert strong.margin_z > weak.margin_z, "the margin does not separate a weak pass from a strong"


def test_no_pairs_is_refused_rather_than_answered():
    """"This corpus has nothing on its axis" is a claim, and it needs an observation."""
    with pytest.raises(engine.Unreadable):
        admission.axis_carries({}, (), far=FAR, seed=SEED, draws=DRAWS)


def test_the_verdict_is_taken_at_the_callers_level():
    """No default level, and a looser level cannot demand more evidence than a stricter one."""
    rng = np.random.default_rng(6)
    records, pairs = _dial(rng, lam=0.5, n_pairs=20)
    strict = admission.axis_carries(records, pairs, far=0.001, seed=SEED, draws=DRAWS)
    loose = admission.axis_carries(records, pairs, far=0.20, seed=SEED, draws=DRAWS)
    assert strict.cutoff > loose.cutoff
    assert strict.margin_z == pytest.approx(loose.margin_z), (
        "the evidence moved with the level; only the cutoff should"
    )


def _phase_randomised(rng, W):
    """Each channel's power spectrum kept exactly, its phase alignment with the others destroyed."""
    n = W.shape[0]
    F = np.fft.rfft(W - W.mean(axis=0, keepdims=True), axis=0)
    phase = rng.uniform(0, 2 * np.pi, size=F.shape)
    phase[0] = 0
    if n % 2 == 0:
        phase[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * phase), n=n, axis=0)


def test_facet_independence_falls_at_both_extremes():
    """The measure counts coherent structures, and is not monotone in channel independence.

    Reading it as "higher means more independent channels" is the mistake this pins. A record whose
    channels all move together carries one mode; a record whose channels are *completely*
    independent carries none standing above its own floor. The quantity is large only in between.
    """
    rng = np.random.default_rng(SEED)
    facets = 4

    grouped, several, apart = {}, {}, {}
    for i in range(12):
        loads = np.linalg.qr(rng.normal(size=(D, facets)))[0]
        latents = rng.normal(size=(T, facets))
        # one structure shared by every channel
        grouped[i] = np.outer(latents[:, 0], loads[:, 0]) * 2 + rng.normal(size=(T, D)) * 0.1
        # several distinct structures — the regime the association needs
        several[i] = latents @ loads.T * 2 + rng.normal(size=(T, D)) * 0.1
        # the same record with its channels decoupled: as independent as channels get
        apart[i] = _phase_randomised(rng, several[i])

    one = admission.facet_independence(grouped, sorted(grouped), far=FAR)
    mid = admission.facet_independence(several, sorted(several), far=FAR)
    none = admission.facet_independence(apart, sorted(apart), far=FAR)

    assert mid > one, f"several structures {mid} should read above one shared structure {one}"
    assert mid > none, f"several structures {mid} should read above fully decoupled {none}"
    assert none < mid / 2, (
        f"fully independent channels read {none}, not far below the middle regime's {mid} — "
        "if this holds, the measure really is monotone in independence and its docstring is wrong"
    )


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.8, 0.9])
def test_independent_records_are_refused_however_smooth_they_are(rho):
    """The false-positive mode that a row-shuffle null has and this one must not.

    These records share nothing: every column of every record is its own AR(1) draw. Only their
    individual smoothness varies. A null that destroys each record's autocorrelation along with the
    axis admits every autocorrelated one -- the row shuffle reads margins of 11.22 to 43.14 on these
    records from a lag-one row correlation of 0.3 upward, recorded by
    ``research/checks/test_whether_a_row_shuffle_is_a_null.py`` -- because the surrogates are white
    and the originals are not, and that has nothing to do with whether the two are related.
    """
    rng = np.random.default_rng(11)
    records = {f"r{i}": _ar1(rng, rho) for i in range(12)}
    pairs = [(f"r{i}", f"r{j}") for i in range(12) for j in range(i + 1, 12)]
    v = admission.axis_carries(records, pairs, far=FAR, seed=3, draws=DRAWS)
    assert not v.admitted, (
        f"independent records admitted at rho={rho}, margin {v.margin_z:.2f}. The null is "
        f"reading their own smoothness as a relation between them."
    )


def test_the_surrogate_leaves_a_records_own_covariance_exactly_alone():
    """The property the null rests on, asserted directly rather than inferred from a verdict.

    The association is standardized by ``tr(C_a C_b)``, so a surrogate that changes a record's own
    covariance is not testing what it claims to. Drawing an independent phase per channel does
    change it -- badly, and worse the smoother the record -- which is why the phase is shared.
    """
    rng = np.random.default_rng(SEED)
    for rho in (0.0, 0.6, 0.9, 0.99):
        frame = _ar1(rng, rho)
        before = np.cov(frame, rowvar=False)
        after = np.cov(admission._decoupled(rng, frame), rowvar=False)
        relative = float(np.linalg.norm(after - before) / np.linalg.norm(before))
        assert relative < 1e-9, (
            f"decoupling changed the record's own covariance by {relative:.4f} at rho={rho}. "
            f"A shared phase per frequency leaves the cross-spectrum -- and so the whole "
            f"covariance -- untouched; an independent phase per channel does not."
        )


@pytest.mark.parametrize("rows,rho", [(96, 0.95), (384, 0.95), (96, 0.99), (768, 0.99)])
def test_the_band_this_null_does_not_reach_is_where_it_is_said_to_be(rows, rho):
    """The limit, pinned so it cannot quietly move.

    Above `rho = 0.9` on a short axis this null admits records that are independent by
    construction. The surrogate is circular and a strongly autocorrelated record over a short axis
    is not, so the observed value beats a surrogate whose end mismatch has been smoothed away.
    It is a finite-axis effect and it dies as the axis lengthens.

    This asserts the shape of that decay rather than a verdict: the margin at the longer axis must
    be a fraction of the margin at the short one. If a later change makes the short axis pass, this
    test fails and says so -- which is the point, because the docstring states the band.
    """
    rng = np.random.default_rng(11)
    records = {f"r{i}": _ar1(rng, rho, t=rows) for i in range(12)}
    pairs = [(f"r{i}", f"r{j}") for i in range(12) for j in range(i + 1, 12)]
    v = admission.axis_carries(records, pairs, far=FAR, seed=3, draws=DRAWS)
    assert v.rows == rows
    assert v.row_autocorrelation > 0.5, (
        f"the verdict reports a row autocorrelation of {v.row_autocorrelation:.3f} on records "
        f"built at rho={rho}; a reader cannot see this band without that number being right"
    )
    if rows <= 96:
        assert v.admitted, (
            f"independent records at rho={rho} over {rows} rows were refused (margin "
            f"{v.margin_z:.2f}). That is better than documented -- the band in _decoupled's "
            f"docstring is now wrong and should be re-measured and narrowed."
        )
    else:
        assert v.margin_z < 4.0, (
            f"at rho={rho} the margin is {v.margin_z:.2f} over {rows} rows. The band is supposed "
            f"to shrink with axis length; if it does not, it is not a finite-axis effect and the "
            f"explanation in _decoupled's docstring is wrong."
        )


def test_the_row_shuffle_null_is_the_thing_that_fails_here():
    """Prove the defect the check above guards against is real.

    A test that only shows the current code passing does not establish that it is doing anything.
    This runs the row-shuffle null on the same independent records and asserts that it fails, so
    the check above is pinned to a defect rather than to a preference.
    """
    rng = np.random.default_rng(11)
    records = {f"r{i}": _ar1(rng, 0.9) for i in range(12)}
    pairs = [(f"r{i}", f"r{j}") for i in range(12) for j in range(i + 1, 12)]

    def margin(surrogate):
        r = np.random.default_rng(3)
        obs = [abs(engine.associate(records[a], records[b], far=FAR).z) for a, b in pairs]
        null = []
        for _ in range(DRAWS):
            s = {k: surrogate(r, np.asarray(v, dtype=float)) for k, v in records.items()}
            null += [abs(engine.associate(s[a], s[b], far=FAR).z) for a, b in pairs]
        null = np.array(null)
        return (np.mean(obs) - null.mean()) / (null.std() / np.sqrt(null.size))

    assert margin(admission._shuffled) > 10, "the row shuffle should show the false positive"
    assert abs(margin(admission._decoupled)) < 5, "the decoupling null should not"
