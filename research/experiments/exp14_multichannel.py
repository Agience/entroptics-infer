"""Stage 9 — a real corpus that is both admitted and multi-facet.

The gap this closes
-------------------
Two real corpora have been run. Text is multi-facet and refused by the admission criterion.
Weather is admitted and single-facet — a station record resolves 1.00 of its 4 measurements — so
the tie it produced was the facet control doing what it was built to do, not a test of the claim.

The claim itself is about the multi-facet regime: that where evidence spreads across several
independent channels, a matrix read carries what a scalar cannot. The other stages measure it only
on generated corpora. This is a real corpus with both properties.

The corpus
----------
Beijing air quality, fetched fresh: 12 monitoring stations, 11 numeric channels each — six pollutants
and five meteorological variables — on a shared daily axis. A station record resolves 1.75 modes
with the city mean removed, against weather's 1.00, and the admission criterion admits it outright.

The task, and why its ground truth is not a statistic
-------------------------------------------------------
Every real-text task here was compromised by a ground truth that some baseline estimated directly.
So the truth here is held-out measurement: one channel of one station is hidden, a method chooses
which *other* station to read it from, and the answer is scored against what that channel actually
did.

A method that identifies the more genuinely-related station predicts the hidden channel better. The
choice is made on the remaining ten channels only, so nothing about the hidden channel informs it.

The baselines, and the forcing they expose
--------------------------------------------
A correlation-based method must decide how to combine ten channels into one judgment of relatedness,
and the data does not say how. Three obvious choices are run separately — the mean of the
per-channel correlations, the maximum, and the correlation of the leading component — because picking
one and calling it *the* baseline would conceal the very thing under test. The association makes no
such choice.

Two bounds are reported with them: a random partner, and an oracle that picks the partner
which actually predicts the hidden channel best. The oracle is not achievable — it sees the answer —
and it is there to say how much of the gap between random and perfect is available at all.

What would falsify this
-----------------------
Stated before the run. The stage fails if the corpus is not admitted, if the record is not
multi-facet, or if the corpus contains no narrow-loud pairs — a candidate agreeing strongly on
few channels. Any of those would mean this is not the test it claims to be, and a null result would
say the corpus lacks the case rather than that the claim fails on it. It is reported as no advance if
the association does not beat the best of the three correlation aggregations.

Run
---
    python research/experiments/exp14_multichannel.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from entroptics_infer import admission, engine                     # noqa: E402
from measure import correlation as _corr  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
INNER = "PRSA2017_Data_20130301-20170228.zip"
CHANNELS = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM")
MIN_HOURS = 20          # a day counts only if this many of its hours are complete
FAR = 0.05
SEED = 20260915
DRAWS = 3


def fetch(into: pathlib.Path):
    into.mkdir(parents=True, exist_ok=True)
    dest = into / "beijing-air.zip"
    if not dest.exists():
        subprocess.run(["curl", "-sS", "-L", "--max-time", "180", URL, "-o", str(dest)], check=True)
    raw = dest.read_bytes()
    return dest, hashlib.sha256(raw).hexdigest()


def build(path, *, remove_city_mean: bool):
    """One `(days, channels)` record per station, on one shared daily axis.

    Hourly readings are averaged to days, and a day counts for a station only when enough of its
    hours are complete. Only days complete at every station are kept, so the axis is genuinely
    shared rather than shared where convenient.

    Channels are put on a common footing by the corpus's own spread — parts per billion and
    hectopascals are not otherwise commensurable — using the corpus's own statistics.
    """
    outer = zipfile.ZipFile(path)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER)))
    names = sorted(n for n in inner.namelist() if n.endswith(".csv"))

    daily = {}
    for n in names:
        station = n.split("/")[-1].split("_")[2]
        acc = {}
        with inner.open(n) as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")):
                vals, ok = [], True
                for c in CHANNELS:
                    if r[c] == "NA":
                        ok = False
                        break
                    vals.append(float(r[c]))
                if ok:
                    acc.setdefault((r["year"], r["month"], r["day"]), []).append(vals)
        daily[station] = {k: np.mean(v, axis=0) for k, v in acc.items() if len(v) >= MIN_HOURS}

    days = sorted(set.intersection(*[set(d) for d in daily.values()]))
    stations = sorted(daily)
    X = np.stack([np.stack([daily[s][d] for d in days]) for s in stations])

    scale = X.reshape(-1, X.shape[2]).std(axis=0)
    scale[scale == 0] = 1.0
    X = X / scale
    if remove_city_mean:
        # what the whole city did that day: shared by every station, and no part of what
        # distinguishes one from another — the same component the seasonal mean is in weather
        X = X - X.mean(axis=0, keepdims=True)

    return {s: X[i] for i, s in enumerate(stations)}, stations, days


def evaluate(records, stations, label):
    """Hide one channel of one station; choose whom to read it from; score against the truth."""
    rng = np.random.default_rng(SEED)
    methods = ("association", "mean channel correlation", "max channel correlation",
               "leading-component correlation", "random", "oracle")
    total = {m: 0.0 for m in methods}
    asked = 0

    per_question = {m: [] for m in methods}
    for qi, q in enumerate(stations):
        others = [s for s in stations if s != q]
        for ci in range(len(CHANNELS)):
            keep = [k for k in range(len(CHANNELS)) if k != ci]
            qr = records[q][:, keep]
            truth = records[q][:, ci]

            picks = {}
            zs = [engine.associate(qr, records[s][:, keep], far=FAR).z for s in others]
            picks["association"] = others[int(np.argmax(zs))]

            per_channel = [[_corr(qr[:, j], records[s][:, keep][:, j]) for j in range(len(keep))]
                           for s in others]
            picks["mean channel correlation"] = others[int(np.argmax([np.mean(r)
                                                                     for r in per_channel]))]
            picks["max channel correlation"] = others[int(np.argmax([np.max(r)
                                                                     for r in per_channel]))]
            # the leading component of the query's remaining channels, correlated against each
            # candidate's own leading component: one number per candidate, as a classical read
            def lead(M):
                Mc = M - M.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
                return Mc @ Vt[0]
            ql = lead(qr)
            picks["leading-component correlation"] = others[
                int(np.argmax([abs(_corr(ql, lead(records[s][:, keep]))) for s in others]))]
            picks["random"] = others[int(rng.integers(len(others)))]
            picks["oracle"] = others[int(np.argmax([abs(_corr(truth, records[s][:, ci]))
                                                    for s in others]))]

            for m, t in picks.items():
                v = abs(_corr(truth, records[t][:, ci]))
                total[m] += v
                # kept per question, because every method answers the same question and a
                # difference between two of them is a paired quantity. A point difference with no
                # error attached cannot say whether one read beat another or the two are the same.
                per_question[m].append(v)
            asked += 1

    out = {"arm": label, "questions": asked, **{m: total[m] / asked for m in methods}}
    scalars = [m for m in methods if m not in ("association", "random", "oracle")]
    best_scalar = max(scalars, key=lambda m: out[m])
    diff = np.asarray(per_question["association"]) - np.asarray(per_question[best_scalar])
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    out["best_scalar_read"] = best_scalar
    out["edge_over_the_best_scalar"] = float(diff.mean())
    out["paired_standard_error"] = se
    out["edge_in_paired_standard_errors"] = float(diff.mean() / se) if se > 0 else 0.0
    out["separated_at_two_standard_errors"] = bool(abs(out["edge_in_paired_standard_errors"]) >= 2)
    print(f"\n  {label}: {asked} questions "
          f"({len(stations)} stations x {len(CHANNELS)} held-out channels)")
    for m in methods:
        print(f"     {m:<32}{out[m]:.3f}")
    print(f"     {'edge over ' + best_scalar:<32}{out['edge_over_the_best_scalar']:+.3f}"
          f"  ({out['edge_in_paired_standard_errors']:+.2f} paired standard errors, "
          f"{'separated' if out['separated_at_two_standard_errors'] else 'NOT separated'})")
    return out


def run():
    started = time.time()
    path, digest = fetch(HERE.parents[1] / "corpora")
    print(f"\n{URL}\n  sha256 {digest[:16]}…")

    verdicts, results, facets, configuration = [], [], [], []
    for label, strip in (("raw", False), ("city mean removed", True)):
        records, stations, days = build(path, remove_city_mean=strip)
        print(f"\n=== {label} ===")
        print(f"  {len(stations)} stations, {len(days)} days complete at every station, "
              f"{len(CHANNELS)} channels")

        pairs = [(stations[i], stations[j])
                 for i in range(len(stations)) for j in range(i + 1, len(stations))]
        v = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
        verdicts.append({"arm": label, "margin_z": v.margin_z, "cutoff": v.cutoff,
                         "admitted": v.admitted, "rows": v.rows,
                         "row_autocorrelation": v.row_autocorrelation})
        print(f"  admission: margin_z {v.margin_z:.2f} against cutoff {v.cutoff:.2f} "
              f"-> {'ADMITTED' if v.admitted else 'REFUSED'}"
              f"   ({v.rows} rows, lag-1 row correlation {v.row_autocorrelation:+.3f})")

        ks = [engine.rank(records[s], far=FAR) for s in stations]
        facets.append({"arm": label, "mean_rank": float(np.mean(ks)), "max_rank": int(max(ks)),
                       "channels": len(CHANNELS)})
        print(f"  a station record resolves {np.mean(ks):.2f} modes on average, max {max(ks)}, "
              f"of {len(CHANNELS)} channels")

        # Does this corpus contain the configuration the claim is about? A narrow-loud distractor
        # is a candidate agreeing strongly on few channels. Without any, a null result would mean
        # the corpus lacks the case rather than that the claim fails on it, and the two are very
        # different findings.
        br, mg = [], []
        for q in stations:
            for c in stations:
                if c == q:
                    continue
                r = np.array([abs(_corr(records[q][:, j], records[c][:, j]))
                              for j in range(len(CHANNELS))])
                br.append(float(r.sum() / r.max()))
                mg.append(float(r.max()))
        br, mg = np.array(br), np.array(mg)
        narrow = br < np.percentile(br, 25)
        loud = mg > np.percentile(mg, 75)
        shape = {"arm": label, "breadth_mean": float(br.mean()), "breadth_min": float(br.min()),
                 "breadth_max": float(br.max()),
                 "breadth_vs_magnitude": float(np.corrcoef(br, mg)[0, 1]),
                 "narrow_and_loud": int((narrow & loud).sum()), "pairs": int(len(br))}
        configuration.append(shape)
        print(f"  effective breadth of agreement: {br.min():.2f} to {br.max():.2f} of "
              f"{len(CHANNELS)} channels, mean {br.mean():.2f}")
        print(f"  breadth against peak magnitude: r = {shape['breadth_vs_magnitude']:.3f}; "
              f"{shape['narrow_and_loud']} of {shape['pairs']} pairs are both narrow and loud")

        results.append(evaluate(records, stations, label))

    out = {
        "source": {"url": URL, "sha256": digest},
        "design": {"channels": CHANNELS, "min_hours": MIN_HOURS, "far": FAR,
                   "seed": SEED, "draws": DRAWS},
        "admission": verdicts, "facets": facets, "configuration": configuration,
        "results": results,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp14.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp14.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
