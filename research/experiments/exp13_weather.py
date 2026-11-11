"""Stage 8 — a real corpus that the admission test says should work.

Why this one
------------
Everything the reads succeed on in the earlier stages is synthetic, and the one real corpus they
try — text — is refused by the admission test and loses every task. That is consistent, but it is not evidence that
the admitted regime occurs outside a generator. This is the test of that.

US weather stations, fetched fresh: 318 stations, 53 weekly dates through 2016, four measurements
each. The shape is exactly what the reads need and what text lacked — a shared ordered axis that
the relation varies along. Two nearby stations are not merely similar on average; they are warm in
the same week and wet in the same week. That is co-variation along a shared axis, which is the thing
the association measures.

And the ground truth is independent of the signal. Which state a station is in is administrative
geography, recorded by someone who never looked at the temperature. Every real-text task here failed
partly because its ground truth was a statistic some baseline estimated directly; nothing about a
state boundary is a statistic of the weather.

The identity component, again
-----------------------------
Every station in the country is warm in July and cold in January. That seasonal signal is shared by
all of them and says nothing about *which* station a record is — the same shape as the corpus unigram
distribution in a text record, which was measured there to be hiding almost everything a record had.
So the per-date national mean is removed, and the experiment reports both arms so the size of the
effect on real data is visible rather than assumed.

What is measured
----------------
**A — admission, run first.** The program's own precondition. The prediction is that this corpus is
admitted, where both books were refused.

**B — the task.** Given a station, pick the one of eight candidates that shares its state. Baselines
are the classical ones on the same data: correlation of the aggregate temperature series, and
correlation after the seasonal mean is removed.

**C — both arms.** With and without the national mean removed, so the identity component's effect on
a real corpus is a measurement.

**D — how many facets the corpus actually has.** The association's measured advantage lives in the
*multi-facet* regime; `exp01`'s control requires every method to tie at one facet. So the record's
own resolved rank decides which of those two predictions this corpus falls under, and it is measured
rather than assumed.

What would falsify this
-----------------------
Stated before the run. The stage fails if the corpus is refused by admission — the criterion
would then have no positive case and would only ever have said no — or if the association does not
beat chance. It is reported as no advance if the association does not beat the classical
correlation baselines.

Run
---
    python research/experiments/exp13_weather.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from entroptics_infer import admission, engine                     # noqa: E402
from measure import correlation as _corr  # noqa: E402

URL = "https://corgis-edu.github.io/corgis/datasets/csv/weather/weather.csv"
MEASUREMENTS = ("Data.Precipitation", "Data.Temperature.Avg Temp",
                "Data.Temperature.Max Temp", "Data.Temperature.Min Temp")
N_CANDIDATES = 8
FAR = 0.05
SEED = 20260915
DRAWS = 3


def fetch(into: pathlib.Path) -> tuple:
    """Fetch the dataset, or reuse the copy this function already fetched."""
    into.mkdir(parents=True, exist_ok=True)
    dest = into / "weather.csv"
    if not dest.exists():
        subprocess.run(["curl", "-sS", "-L", "--max-time", "90", URL, "-o", str(dest)], check=True)
    raw = dest.read_bytes()
    rows = list(csv.DictReader(dest.open(encoding="utf-8")))
    return rows, hashlib.sha256(raw).hexdigest()


def build(rows, *, remove_seasonal: bool):
    """One `(dates, measurements)` record per station, on one shared date axis.

    Only stations observed on every date are kept, so the axis is genuinely shared: a record with
    gaps would be compared against one without them, and the missing rows would read as agreement.

    Each measurement is put on a common footing by the corpus's own spread — precipitation in inches
    and temperature in Fahrenheit are not otherwise commensurable, and without it one column would
    decide every read. The statistics are the corpus's own, not brought from anywhere.
    """
    dates = sorted({r["Date.Full"] for r in rows})
    index = {d: i for i, d in enumerate(dates)}
    by_station = defaultdict(dict)
    state = {}
    for r in rows:
        code = r["Station.Code"]
        state[code] = r["Station.State"]
        try:
            by_station[code][r["Date.Full"]] = [float(r[m]) for m in MEASUREMENTS]
        except ValueError:
            continue

    complete = [s for s, obs in by_station.items() if len(obs) == len(dates)]
    frames = {}
    for s in complete:
        X = np.zeros((len(dates), len(MEASUREMENTS)))
        for d, vals in by_station[s].items():
            X[index[d]] = vals
        frames[s] = X

    stack = np.stack([frames[s] for s in complete])          # (stations, dates, measurements)
    scale = stack.reshape(-1, stack.shape[2]).std(axis=0)
    scale[scale == 0] = 1.0
    stack = stack / scale

    if remove_seasonal:
        # the country's own weather on each date: shared by every station, and no part of what
        # distinguishes one from another
        stack = stack - stack.mean(axis=0, keepdims=True)

    return {s: stack[i] for i, s in enumerate(complete)}, {s: state[s] for s in complete}, dates


def evaluate(records, states, label):
    """Given a station, which of eight candidates shares its state?"""
    rng = np.random.default_rng(SEED)
    by_state = defaultdict(list)
    for s, st in states.items():
        by_state[st].append(s)
    codes = sorted(records)

    hits = {"association": 0, "aggregate correlation": 0, "random": 0.0}
    asked = 0
    for q in codes:
        same = [s for s in by_state[states[q]] if s != q]
        if not same:
            continue
        others = [s for s in codes if states[s] != states[q]]
        if len(others) < N_CANDIDATES - 1:
            continue
        truth = same[int(rng.integers(len(same)))]
        picked = [others[i]
                  for i in rng.choice(len(others), size=N_CANDIDATES - 1, replace=False)]
        cands = [truth] + picked
        order = rng.permutation(len(cands))
        cands = [cands[i] for i in order]
        gold = cands.index(truth)

        zs = [engine.associate(records[q], records[c], far=FAR).z for c in cands]
        # the classical read on the same data: correlate the average-temperature series
        ag = [_corr(records[q][:, 1], records[c][:, 1]) for c in cands]
        hits["association"] += int(int(np.argmax(zs)) == gold)
        hits["aggregate correlation"] += int(int(np.argmax(ag)) == gold)
        hits["random"] += 1.0 / len(cands)
        asked += 1

    out = {"arm": label, "questions": asked,
           **{k: v / asked for k, v in hits.items()}}
    print(f"\n  {label}: {asked} questions, {N_CANDIDATES} candidates")
    print(f"     random                 {out['random']:.3f}")
    print(f"     aggregate correlation  {out['aggregate correlation']:.3f}")
    print(f"     association            {out['association']:.3f}")
    return out


def run():
    started = time.time()
    rows, digest = fetch(HERE.parents[1] / "corpora")
    print(f"\n{URL}\n  sha256 {digest[:16]}…  {len(rows)} observations")

    results = []
    verdicts = []
    for label, strip in (("raw, seasonal signal present", False),
                         ("national mean removed", True)):
        records, states, dates = build(rows, remove_seasonal=strip)
        codes = sorted(records)
        print(f"\n=== {label} ===")
        print(f"  {len(codes)} stations observed on all {len(dates)} dates, "
              f"{len(set(states.values()))} states, {len(MEASUREMENTS)} measurements")

        rng = np.random.default_rng(3)
        pairs = [(codes[i], codes[j])
                 for i, j in rng.integers(0, len(codes), size=(80, 2)) if i != j]
        v = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
        verdicts.append({"arm": label, "observed": v.observed, "decoupled": v.decoupled,
                         "margin_z": v.margin_z, "cutoff": v.cutoff, "admitted": v.admitted,
                         "rows": v.rows, "row_autocorrelation": v.row_autocorrelation})
        print(f"  admission: margin_z {v.margin_z:.2f} against cutoff {v.cutoff:.2f} "
              f"-> {'ADMITTED' if v.admitted else 'REFUSED'}"
              f"   ({v.rows} rows, lag-1 row correlation {v.row_autocorrelation:+.3f})")

        # How much independent structure a record actually has. The association's measured
        # advantage lives in the multi-facet regime; at one facet `exp01`'s control requires every
        # method to tie, so this decides which of the two predictions applies here.
        ks = [engine.rank(records[s], far=FAR) for s in codes]
        stack = np.concatenate([records[s] for s in codes], axis=0)
        sv = np.linalg.svd(stack - stack.mean(axis=0), compute_uv=False)
        facets = {"mean_rank": float(np.mean(ks)), "max_rank": int(max(ks)),
                  "pooled_singular_values": [float(x) for x in (sv / sv[0])]}
        print(f"  a record resolves {facets['mean_rank']:.2f} modes on average, max "
              f"{facets['max_rank']}, of {len(MEASUREMENTS)} measurements")
        print(f"  pooled singular values {np.round(sv / sv[0], 3)} — the three temperature "
              f"columns are near-collinear")

        row = evaluate(records, states, label)
        row["facets"] = facets
        results.append(row)

    out = {
        "source": {"url": URL, "sha256": digest, "observations": len(rows)},
        "design": {"measurements": MEASUREMENTS, "n_candidates": N_CANDIDATES,
                   "far": FAR, "seed": SEED, "draws": DRAWS},
        "admission": verdicts, "results": results,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp13.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp13.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
