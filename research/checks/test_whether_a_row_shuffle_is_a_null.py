"""Whether a row shuffle can stand in for the admission test's null, measured beside the shipped one.

The question
--------------
`entroptics_infer.admission.axis_carries` asks whether a corpus has anything on its ordered axis by
comparing each pair's association against the same pair with the records decoupled from each
other. The decoupling is a phase randomization with one phase per frequency shared across a
record's channels (`admission._decoupled`). The obvious alternative is a row shuffle
(`admission._shuffled`): the same rows of each record in a random order.

A shuffle destroys the axis, which is the intent, and it also destroys each record's own
autocorrelation along the axis, which is not. Two records that are individually smooth then beat
their own shuffles whether or not they are related to each other, because the shuffles are white
and the originals are not.

What is measured
------------------
``independent``  twelve records, every column of every record its own AR(1) draw over 96 rows and 16
                 channels, sharing nothing; all 66 pairs. Only the records' own smoothness, the
                 lag-one row correlation, varies. Every one of these must be refused.
``sharing``      twelve pairs whose two records share a fraction of one AR(1) trajectory and draw
                 the rest independently, at a fixed lag-one row correlation. These should be
                 admitted as the shared fraction grows.
``band``         the independent records at a lag-one row correlation of 0.95, over axes of growing
                 length, read by the shipped null alone: where it admits records that share nothing,
                 and whether that dies as the axis lengthens. The relative excess is the margin over
                 the decoupled mean.
``phase``        what a surrogate does to a record's own covariance, as the relative change in it:
                 the shipped draw, one phase per frequency shared by every channel, against an
                 independent phase per channel.

Each is read by the shipped `axis_carries`, once as it ships and once with the row shuffle in the
decoupling's place. Nothing here computes a margin of its own: the swap replaces the one
surrogate function `axis_carries` calls, for the length of one call, and restores it.

The level is the caller's ``far`` and the cutoff is the one the verdict reads, so "admitted" here is
the shipped verdict, not a threshold written for this check.
"""
from __future__ import annotations

import contextlib
import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

_sys.path[:0] = [str(_pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import admission                          # noqa: E402

#: CHOSEN: the caller's false-alarm level, the same one `tests/test_admission.py` reads at.
FAR = 0.05
#: CHOSEN: the corpus shape -- records, rows and channels -- the same as `tests/test_admission.py`.
RECORDS, ROWS, CHANNELS = 12, 96, 16
#: CHOSEN: the seeds and the number of surrogate draws, the same as `tests/test_admission.py`.
SEED, NULL_SEED, DRAWS = 11, 3, 3
#: CHOSEN: the lag-one row correlations swept, from white to very smooth
RHOS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.97, 0.99)
#: CHOSEN: the shared fractions swept for power, and the smoothness they are read at
SHARED = (0.0, 0.3, 0.6, 1.0)
POWER_RHO = 0.9
#: CHOSEN: the axis lengths and the smoothness the band is read over
BAND_ROWS, BAND_RHO = (48, 96, 192, 384, 768), 0.95
#: CHOSEN: the smoothness the two phase draws are compared at
PHASE_RHOS = (0.0, 0.6, 0.9, 0.99)


def _ar1(rng, rho, t=ROWS, d=CHANNELS):
    """``d`` independent AR(1) columns over ``t`` rows, unit variance."""
    x = np.zeros((t, d))
    e = rng.normal(size=(t, d))
    x[0] = e[0]
    for i in range(1, t):
        x[i] = rho * x[i - 1] + np.sqrt(max(0.0, 1 - rho * rho)) * e[i]
    return x


def _phase_per_channel(rng, frame):
    """The control: each channel's power spectrum kept, with an independent phase per channel."""
    n = frame.shape[0]
    mean = frame.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(frame - mean, axis=0)
    phase = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    phase[0] = 0
    if n % 2 == 0:
        phase[-1] = 0
    return np.fft.irfft(spectrum * np.exp(np.complex128(0, 1) * phase), n=n, axis=0) + mean


@contextlib.contextmanager
def _null(surrogate):
    """`axis_carries` with ``surrogate`` in place of the decoupling, for one block only."""
    shipped = admission._decoupled
    admission._decoupled = surrogate
    try:
        yield
    finally:
        admission._decoupled = shipped


def _margins(records, pairs):
    """``{null: (margin_z, admitted)}`` for the shipped null and the row shuffle."""
    out = {}
    for name, surrogate in (("decoupling", admission._decoupled),
                            ("row_shuffle", admission._shuffled)):
        with _null(surrogate):
            v = admission.axis_carries(records, pairs, far=FAR, seed=NULL_SEED, draws=DRAWS)
        out[name] = (float(v.margin_z), bool(v.admitted), float(v.cutoff))
    return out


def main() -> int:
    print(f"whether a row shuffle is a null -- {RECORDS} records, {ROWS} rows, {CHANNELS} channels, "
          f"far {FAR}\n")

    independent = {}
    cutoff = None
    print("  records independent by construction: every row must be refused")
    print(f"  {'lag-1 row correlation':>24}{'row shuffle':>14}{'decoupling':>13}")
    print("  " + "-" * 51)
    pairs = [(f"r{i}", f"r{j}") for i in range(RECORDS) for j in range(i + 1, RECORDS)]
    for rho in RHOS:
        rng = np.random.default_rng(SEED)
        records = {f"r{i}": _ar1(rng, rho) for i in range(RECORDS)}
        got = _margins(records, pairs)
        cutoff = got["decoupling"][2]
        independent[str(rho)] = {k: {"margin_z": round(m, 2), "admitted": a}
                                 for k, (m, a, _c) in got.items()}
        print(f"  {rho:>24.2f}{got['row_shuffle'][0]:>14.2f}{got['decoupling'][0]:>13.2f}")
    print()

    sharing = {}
    print(f"  pairs sharing a fraction of one trajectory, at lag-one row correlation {POWER_RHO}")
    print(f"  {'shared fraction':>24}{'row shuffle':>14}{'decoupling':>13}")
    print("  " + "-" * 51)
    for f in SHARED:
        rng = np.random.default_rng(SEED)
        records, shared_pairs = {}, []
        for i in range(RECORDS):
            common = _ar1(rng, POWER_RHO)
            records[f"a{i}"] = np.sqrt(f) * common + np.sqrt(1 - f) * _ar1(rng, POWER_RHO)
            records[f"b{i}"] = np.sqrt(f) * common + np.sqrt(1 - f) * _ar1(rng, POWER_RHO)
            shared_pairs.append((f"a{i}", f"b{i}"))
        got = _margins(records, shared_pairs)
        sharing[str(f)] = {k: {"margin_z": round(m, 2), "admitted": a}
                           for k, (m, a, _c) in got.items()}
        print(f"  {f:>24.2f}{got['row_shuffle'][0]:>14.2f}{got['decoupling'][0]:>13.2f}")
    print()

    band = {}
    print(f"  the band: independent records at lag-one row correlation {BAND_RHO}, by axis length")
    print(f"  {'rows':>24}{'relative excess':>17}{'decoupling':>13}")
    print("  " + "-" * 54)
    for rows in BAND_ROWS:
        rng = np.random.default_rng(SEED)
        records = {f"r{i}": _ar1(rng, BAND_RHO, t=rows) for i in range(RECORDS)}
        v = admission.axis_carries(records, pairs, far=FAR, seed=NULL_SEED, draws=DRAWS)
        band[str(rows)] = {"relative_excess": round(float(v.margin / v.decoupled), 3),
                           "margin_z": round(float(v.margin_z), 2), "admitted": bool(v.admitted)}
        print(f"  {rows:>24}{band[str(rows)]['relative_excess']:>17.3f}"
              f"{band[str(rows)]['margin_z']:>13.2f}")
    print()

    phase = {}
    print("  a record's own covariance, relative change under each surrogate")
    print(f"  {'lag-1 row correlation':>24}{'shared phase':>14}{'per channel':>13}")
    print("  " + "-" * 51)
    rng = np.random.default_rng(SEED)
    for rho in PHASE_RHOS:
        frame = _ar1(rng, rho)
        before = np.cov(frame, rowvar=False)
        changed = {}
        for name, surrogate in (("shared", admission._decoupled),
                                ("per_channel", _phase_per_channel)):
            after = np.cov(surrogate(rng, frame), rowvar=False)
            changed[name] = round(float(np.linalg.norm(after - before) / np.linalg.norm(before)), 4)
        phase[str(rho)] = changed
        print(f"  {rho:>24.2f}{changed['shared']:>14.4f}{changed['per_channel']:>13.4f}")
    print()

    false_shuffle = sum(v["row_shuffle"]["admitted"] for v in independent.values())
    false_decoupled = sum(v["decoupling"]["admitted"] for v in independent.values())
    record.figures(records=RECORDS, pairs=len(pairs), rows=ROWS, channels=CHANNELS, far=FAR,
                   cutoff=round(cutoff, 2), independent=independent, sharing=sharing,
                   power_rho=POWER_RHO, band_rho=BAND_RHO, band=band, phase=phase,
                   independent_admitted_by_row_shuffle=false_shuffle,
                   independent_admitted_by_decoupling=false_decoupled,
                   correlations_swept=len(RHOS), correlations=list(RHOS),
                   phase_correlations=list(PHASE_RHOS), shared_fractions=list(SHARED),
                   band_rows=list(BAND_ROWS))
    print(f"  at a cutoff of {cutoff:.2f}, the row shuffle admits **{false_shuffle}** of "
          f"{len(RHOS)} independent corpora and")
    print(f"  the decoupling **{false_decoupled}**.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
