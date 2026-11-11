"""Stage 40 — the null's effective sample size, which is what autocorrelated rows actually cost.

The two failures this sits between
------------------------------------
`exp45`: `associate`'s permutation null does not deliver its stated level on records whose rows are
autocorrelated — 0.05 asked for, 0.748 delivered at a lag-one row correlation of 0.99.

`exp46`: the obvious repair, standardizing against decoupled surrogates, removes real skill on a
real corpus. The surrogate keeps each record's own covariance exactly, two records of the same
activity are similar in exactly that, and so its spread is 2.3x wider for the pairs the read has to
find. Standardizing by it divides the signal out.

Why that repair was the wrong shape
-------------------------------------
`z` already divides by `tr(C_a C_b)`. The decoupled surrogate's spread is *also* driven by
`tr(C_a C_b)`, so standardizing against it corrects for the covariance a second time — and the
covariance is exactly what relatedness consists of on that corpus. The defect was never that the
null's scale ignores covariance. It is that the null counts `T` independent observations when
autocorrelated rows do not supply `T` of them.

That is the effective sample size, and it has a derivation rather than a choice behind it. For two
series the variance of their cross-product runs over the product of their own autocorrelations
(Bartlett 1935), so

    factor  =  1 + 2 * sum_k (1 - k/T) * rho_a(k) * rho_b(k)
    z'      =  z / sqrt(factor)

with `rho_a`, `rho_b` each record's own lag-`k` row autocorrelation averaged over its channels, and
the taper `1 - k/T` read off the axis length. Nothing here is supplied: no cutoff, no bandwidth
chosen by eye, no level beyond the caller's own.

The property that makes it the right shape, and the surrogate the wrong one: this factor is built
from each record separately and never from the two together. It cannot know whether a pair is
related, so it cannot subtract a relation.

What is measured
------------------
1. **The level**, on records independent by construction, over the same autocorrelation sweep as
   `exp45`, so the two are directly comparable.
2. **Inertness**, which matters most for the published work: at zero autocorrelation the factor must
   be 1 and no figure in this paper may move.
3. **Power**, on records that genuinely share a trajectory. A correction that buys its level by
   refusing everything is the failure `exp46` found.
4. **The ranking on the wide corpus**, which is where the decoupled surrogate destroyed the read.
   This is the arm that separates a repair from a second defect.
5. **The bandwidth**, swept, because a `kmax` chosen here would be exactly the kind of constant this
   program removes.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import exp19_wide_corpus as W                                       # noqa: E402
from draw import ar1                                                # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from measure import autocorrelation  # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP47_SMOKE") == "1"
T, D = 125, 16
RHOS = (0.0, 0.6, 0.9) if SMOKE else (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99)
PAIRS = 80 if SMOKE else 400
SHARES = (0.3, 0.6, 1.0)
POWER_RHOS = (0.0, 0.6, 0.9)
POWER_PAIRS = 60 if SMOKE else 200
QUERIES = 20 if SMOKE else 120


def effective_factor(A, B, kmax=None):
    """How many fewer independent observations these two rows supply than their length claims.

    Built from each record separately. It never sees the two together, so it cannot know whether
    they are related and cannot subtract a relation — which is precisely what `exp46` found the
    decoupled surrogate doing.
    """
    n = A.shape[0]
    k = kmax if kmax is not None else n - 1
    k = max(1, min(k, n - 1))
    ra, rb = autocorrelation(A, k), autocorrelation(B, k)
    taper = 1.0 - np.arange(1, k + 1) / n            # read off the axis length, not chosen
    return max(float(1.0 + 2.0 * np.sum(taper * ra * rb)), 1e-9)


def corrected(a, b, kmax=None):
    r = engine.associate(a, b, far=FAR)
    return r.z / np.sqrt(effective_factor(a, b, kmax)), r.z, r.cutoff


def run():
    started = time.time()
    rng = np.random.default_rng(SEED)

    # --- 1 and 2: the level, and inertness where there is no autocorrelation ----------
    print("THE LEVEL — records independent by construction; every pair must be refused\n")
    print(f"{'rho':>6}{'as stated':>12}{'corrected':>12}{'mean factor':>14}")
    print("-" * 46)
    cells = []
    for rho in RHOS:
        stated = corr = 0
        fs = []
        for _ in range(PAIRS):
            a, b = ar1(rng, rho, T, D), ar1(rng, rho, T, D)
            z2, z, cut = corrected(a, b)
            fs.append(effective_factor(a, b))
            stated += int(abs(z) >= cut)
            corr += int(abs(z2) >= cut)
        cells.append({"rho": rho, "as_stated": stated / PAIRS,
                      "corrected": corr / PAIRS, "mean_factor": float(np.mean(fs))})
        print(f"{rho:>6.2f}{stated / PAIRS:>12.3f}{corr / PAIRS:>12.3f}{np.mean(fs):>14.2f}")
    print(f"\n  the level asked for is {FAR}. At no autocorrelation the factor is "
          f"{cells[0]['mean_factor']:.2f}, so the")
    print("  correction is inert exactly where the defect is absent.\n")

    # --- 3: power ---------------------------------------------------------------------
    print("=" * 46)
    print("POWER — records that genuinely share a trajectory\n")
    print(f"{'rho':>6}{'shared':>8}{'as stated':>12}{'corrected':>12}")
    power = []
    for rho in POWER_RHOS:
        for shared in SHARES:
            hs = hc = 0
            for _ in range(POWER_PAIRS):
                base, a, b = ar1(rng, rho, T, D), ar1(rng, rho, T, D), ar1(rng, rho, T, D)
                A = shared * base + (1 - shared) * a
                B = shared * base + (1 - shared) * b
                z2, z, cut = corrected(A, B)
                hs += int(abs(z) >= cut)
                hc += int(abs(z2) >= cut)
            power.append({"rho": rho, "shared": shared,
                          "as_stated": hs / POWER_PAIRS, "corrected": hc / POWER_PAIRS})
            print(f"{rho:>6.2f}{shared:>8.1f}{hs / POWER_PAIRS:>12.3f}{hc / POWER_PAIRS:>12.3f}")

    # --- 5: the bandwidth, swept rather than chosen ------------------------------------
    print("\n" + "=" * 46)
    print("BANDWIDTH — `kmax` swept, because choosing one would be a constant\n")
    widths = [max(1, T // f) for f in (12, 4, 2, 1)]
    print(f"{'rho':>6}" + "".join(f"{f'kmax={k}':>12}" for k in widths))
    bandwidth = []
    for rho in (0.0, 0.6, 0.9):
        row = {"rho": rho}
        for k in widths:
            hit = 0
            for _ in range(POWER_PAIRS):
                a, b = ar1(rng, rho, T, D), ar1(rng, rho, T, D)
                z2, _, cut = corrected(a, b, kmax=k)
                hit += int(abs(z2) >= cut)
            row[f"kmax_{k}"] = hit / POWER_PAIRS
        bandwidth.append(row)
        print(f"{rho:>6.2f}" + "".join(f"{row[f'kmax_{k}']:>12.3f}" for k in widths))

    # --- 4: the wide corpus, which is where the surrogate failed -----------------------
    print("\n" + "=" * 46)
    print("THE WIDE CORPUS — where the decoupled surrogate destroyed the read\n")
    path, _ = W.fetch(HERE.parents[1] / "corpora")
    records, keys, width = W.build(path, remove_corpus_mean=False)
    qrng = np.random.default_rng(W.SEED)
    pool = list(keys)
    hits = {"association": 0, "corrected": 0, "cos": 0, "random": 0}
    asked = 0
    for qi in qrng.permutation(len(pool))[:QUERIES]:
        q = pool[qi]
        tp = [k for k in pool if k[0] == q[0] and k[1] != q[1]]
        fp = [k for k in pool if k[0] != q[0]]
        if not tp or len(fp) < W.N_CANDIDATES - 1:
            continue
        truth = tp[int(qrng.integers(len(tp)))]
        cands = [truth] + [fp[i] for i in
                           qrng.choice(len(fp), W.N_CANDIDATES - 1, replace=False)]
        perm = qrng.permutation(W.N_CANDIDATES)
        cands = [cands[i] for i in perm]
        answer = int(np.where(perm == 0)[0][0])
        Q = records[q]
        frames = [records[c] for c in cands]
        zs = [engine.associate(Q, f, far=FAR).z for f in frames]
        hits["association"] += int(int(np.argmax(zs)) == answer)
        hits["corrected"] += int(int(np.argmax(
            [z / np.sqrt(effective_factor(Q, f)) for z, f in zip(zs, frames)])) == answer)
        hits["cos"] += int(int(np.argmax([W._flat_cos(Q, f) for f in frames])) == answer)
        hits["random"] += int(int(qrng.integers(W.N_CANDIDATES)) == answer)
        asked += 1
    corpus = {"asked": asked, "chance": 1.0 / W.N_CANDIDATES,
              **{m: hits[m] / asked for m in hits}}
    print(f"  same-activity task, {asked} questions, chance {corpus['chance']:.3f}")
    for m in hits:
        print(f"    {m:<22}{corpus[m]:>8.3f}")
    print(f"\n  the decoupled surrogate read 0.058 here, below chance; this reads "
          f"{corpus['corrected']:.3f}.")

    worst = max(c["corrected"] for c in cells)
    out = {"design": {"far": FAR, "seed": SEED, "rows": T, "channels": D, "rhos": list(RHOS),
                      "pairs": PAIRS, "queries": QUERIES, "smoke": SMOKE},
           "cells": cells, "power": power, "bandwidth": bandwidth, "wide_corpus": corpus,
           "factor_at_no_autocorrelation": cells[0]["mean_factor"],
           "level_at_no_autocorrelation": cells[0]["corrected"],
           "worst_corrected_level": worst,
           "worst_level_as_stated": max(c["as_stated"] for c in cells),
           "power_at_full_sharing": min(p["corrected"] for p in power if p["shared"] == 1.0),
           "wide_corpus_as_stated": corpus["association"],
           "wide_corpus_corrected": corpus["corrected"],
           "wide_corpus_cosine": corpus["cos"],
           "seconds": time.time() - started}
    print("\n" + "=" * 46)
    print(f"  the level, corrected, is at most {worst:.3f} across the sweep against "
          f"{out['worst_level_as_stated']:.3f} as stated;")
    print(f"  the factor is {out['factor_at_no_autocorrelation']:.2f} where there is no "
          f"autocorrelation, so nothing at")
    print(f"  zero moves; power at full sharing is {out['power_at_full_sharing']:.3f}; and the wide")
    print(f"  corpus reads {out['wide_corpus_corrected']:.3f} against "
          f"{out['wide_corpus_as_stated']:.3f} uncorrected.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp47.smoke.json" if SMOKE else "exp47.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
