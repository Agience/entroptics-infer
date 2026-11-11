"""Stage 28 — is the association load-bearing in the read that works best?

Why this has to be asked
--------------------------
The conditional read is the strongest context reader this program has produced. It weights every
cached context by how well it matches the query and sums the votes.

It weights them by cosine. There is no aperture in it, nothing is gated on `resolved`, and the
association matrix appears only in the column it is compared against. So the best-performing
inference read in this program currently uses none of the instrument the program is about, and
that is not a detail to leave unstated.

This stage puts the two weightings against each other with everything else held identical: the same
frames, the same cache, the same decisions, the same seed, the same candidates. The only thing that
differs is what decides how much a cached occasion is worth.

    cosine        the scalar competitor -- one number per pair, the angle between flattened frames
    association   the instrument's own read of the pair, standardized against its permutation null
    aperture      the association, but only occasions that resolve vote at all

The third is the one the program's own rule points at: `resolved` is the aperture deciding what
stands above a record's own floor, with no `k` and no threshold supplied. If it beats the other two
the instrument is doing work here; if it does not, it is not, and the read that works is a kernel
regression that happens to be running in this repository.

The cost, and what it forces
------------------------------
An association is far more expensive than a dot product -- it is a matrix read with a null behind it,
not an inner product -- so the cache and the decision count are cut until a million of them is
affordable. That makes this stage less powerful than the conditional read and it is not a replacement for it:
it answers which weighting is better on a smaller sample, not what the best achievable read is.

Held to the same two tests as every stage since `exp31`: clear the frequency floor, and beat the
context-blind rarity control on the rarest quartile, both at two standard errors.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import text as TX                                                   # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from measure import reciprocal_rank as _rr  # noqa: E402

FAR = 0.05
SEED = 11
SMOKE = os.environ.get("EXP35_SMOKE") == "1"

SOURCES = [
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]

# The cost of an association is set by the feature axis, not the window: it forms covariances, so a
# 1025-wide axis means a million-entry matrix per pair read and roughly sixteen times the work of a
# 257-wide one. This stage asks which weighting is better, not which vocabulary, so the axis is cut
# and the cache and decision count -- which set the power -- are kept.
VOCAB = 256
WINDOW = 4 if SMOKE else 8
CONE = 2
WEIGHTINGS = ("cosine", "association", "aperture")
NORMALISATIONS = ("none", "root")
# An association is a matrix read with a null behind it, not an inner product, and measures about
# six milliseconds a pair at this frame size. The cache and decision count are set by what a single
# pass over every pair can afford, which is what makes this stage less powerful than the conditional read.
CACHE = 120 if SMOKE else 500
TESTS = 30 if SMOKE else 500


def run():
    started = time.time()
    corpora, words = [], []
    for url in SOURCES:
        t = TX.fetch(url, into=HERE.parents[1] / "corpora")
        toks = list(t.tokens)
        corpora.append({"url": url, "sha256": t.sha256, "tokens": len(toks)})
        words.extend(toks)
    if SMOKE:
        words = words[:40000]
    counts = Counter(words)
    V, W = VOCAB, WINDOW
    keep = {w: i for i, (w, _) in enumerate(counts.most_common(V))}
    F = V + 1
    code = np.array([keep.get(w, V) for w in words], dtype=np.int64)

    usable = np.arange(W, len(code) - 1)
    split = int(usable[len(usable) * 2 // 3])
    rng = np.random.default_rng(SEED)
    cache_at = rng.choice(usable[usable < split], size=min(CACHE, int(split - W)), replace=False)
    held = usable[usable >= split]
    held = held[code[held] < V]
    test_at = rng.choice(held, size=min(TESTS, len(held)), replace=False)

    train_counts = np.zeros(F)
    for c in code[:split]:
        train_counts[c] += 1
    unigram_scores = train_counts[:V].astype(float)
    rarity_scores = -unigram_scores
    freqs = train_counts[[int(code[p]) for p in test_at]]
    cuts = np.quantile(freqs, [0.25, 0.5, 0.75])
    band_of = {int(p): int(np.searchsorted(cuts, train_counts[int(code[p])], side="right"))
               for p in test_at}

    tables = {}
    for n in range(1, W + 1):
        tab = {}
        for p in range(n, split):
            tab.setdefault(tuple(int(q) for q in code[p - n:p]), Counter())[int(code[p])] += 1
        tables[n] = tab

    def ngram_scores(p):
        s = unigram_scores / (unigram_scores.max() or 1.0) * 1e-6
        for n in range(1, W + 1):
            hit = tables[n].get(tuple(int(q) for q in code[p - n:p]))
            if hit:
                for k, v in hit.items():
                    if k < V:
                        s[k] += v * (10.0 ** n)
        return s

    ng_cached = {int(p): ngram_scores(int(p)) for p in test_at}
    back = np.exp(-np.arange(W - 1, -1, -1) / CONE)

    def frame2d(p):
        f = np.zeros((W, F))
        f[np.arange(W), code[p - W:p]] = back
        return f

    Kf = [frame2d(int(p)) for p in cache_at]
    K = np.asarray([f.ravel() for f in Kf])
    Kn = np.linalg.norm(K, axis=1)
    targets = np.asarray([int(code[p]) for p in cache_at])
    built = np.bincount(targets, minlength=F).astype(float)
    safe = np.where(built > 0, built, 1.0)
    onehot = np.zeros((len(cache_at), F))
    onehot[np.arange(len(cache_at)), targets] = 1.0
    onehot = onehot[:, :V]
    print(f"{len(words)} words, vocab {V}, window {W}, {len(test_at)} decisions, "
          f"cache {len(cache_at)}  ({len(test_at) * len(cache_at)} pair reads per weighting)")

    # One pass over the pairs, not three. The association and the aperture read differ only in
    # whether an unresolved pair is allowed to vote, so a single `associate` call fills both.
    weights = {w: np.zeros((len(test_at), len(cache_at))) for w in WEIGHTINGS}
    t0 = time.time()
    for i, p in enumerate(test_at):
        Q2 = frame2d(int(p))
        q = Q2.ravel()
        qn = float(np.linalg.norm(q))
        weights["cosine"][i] = np.clip((K @ q) / (Kn * qn + 1e-30), 0, None)
        for j, Kp in enumerate(Kf):
            try:
                a = engine.associate(Q2, Kp, far=FAR)
            except Exception:
                continue
            weights["association"][i, j] = abs(a.z)
            if a.resolved:
                weights["aperture"][i, j] = abs(a.z)
    print(f"  {len(test_at) * len(cache_at)} pair reads in {time.time() - t0:.0f}s")
    for wname in WEIGHTINGS:
        print(f"  {wname:>12}   mean live share "
              f"{float(np.mean(weights[wname] > 0)):.3f}")

    rows = []
    for wname in WEIGHTINGS:
        Wt = weights[wname]
        for norm in NORMALISATIONS:
            d = np.ones(len(cache_at)) if norm == "none" else np.sqrt(safe)[targets]
            acc = {k: [] for k in ("read", "n_gram", "unigram", "rarity")}
            by_band = {}
            for i, p in enumerate(test_at):
                p = int(p)
                truth = int(code[p])
                sc = (Wt[i] / d) @ onehot
                r = {"read": _rr(sc, truth),
                     "n_gram": _rr(ng_cached[p][:V], truth),
                     "unigram": _rr(unigram_scores, truth),
                     "rarity": _rr(rarity_scores, truth)}
                for k, v in r.items():
                    acc[k].append(v)
                by_band.setdefault(band_of[p], []).append(r)

            n_t = len(test_at)
            bands = [{"quartile": b, "n": len(by_band[b]),
                      **{k: float(np.mean([x[k] for x in by_band[b]])) for k in acc}}
                     for b in sorted(by_band)]
            row = {"weighting": wname, "normalisation": norm, "tests": n_t,
                   "cache": int(len(cache_at)),
                   **{f"mrr_{k}": float(np.mean(v)) for k, v in acc.items()},
                   "se_read": float(np.std(acc["read"], ddof=1) / np.sqrt(n_t)),
                   "live_share": float(np.mean(Wt > 0)),
                   "rarest_read": bands[0]["read"],
                   "rarest_n_gram": bands[0]["n_gram"],
                   "rarest_rarity_control": bands[0]["rarity"],
                   "se_rarest": float(np.std([x["read"] for x in by_band[0]], ddof=1)
                                      / math.sqrt(len(by_band[0]))),
                   "bands": bands}
            fm = row["mrr_read"] - row["mrr_unigram"]
            row["floor_margin"] = fm
            row["floor_margin_in_standard_errors"] = fm / row["se_read"] if row["se_read"] else 0.0
            over = row["rarest_read"] - row["rarest_rarity_control"]
            row["rarest_over_control"] = over
            resolvable = row["se_rarest"] > 1e-4
            row["rarest_band_is_resolvable"] = bool(resolvable)
            row["rarest_over_control_in_standard_errors"] = (
                over / row["se_rarest"] if resolvable else 0.0)
            row["clears_both"] = bool(row["floor_margin_in_standard_errors"] > 2.0 and resolvable
                                      and row["rarest_over_control_in_standard_errors"] > 2.0)
            rows.append(row)
            print(f"  {wname:>12} {norm:>4}   MRR {row['mrr_read']:.4f}+/-{row['se_read']:.4f}"
                  f"   floor {row['mrr_unigram']:.4f} "
                  f"({row['floor_margin_in_standard_errors']:+.1f} SE)   rarest "
                  f"{row['rarest_read']:.4f} vs {row['rarest_rarity_control']:.4f} "
                  f"({row['rarest_over_control_in_standard_errors']:+.1f} SE)"
                  f"{'   <-- CLEARS BOTH' if row['clears_both'] else ''}")

    best = {w: max(r["mrr_read"] for r in rows if r["weighting"] == w) for w in WEIGHTINGS}
    cos, asc = best["cosine"], max(best["association"], best["aperture"])
    se = math.sqrt(2) * rows[0]["se_read"]
    out = {"sources": corpora,
           "design": {"far": FAR, "seed": SEED, "vocabulary": V, "window": W, "cone": CONE,
                      "cache": CACHE, "tests": int(len(test_at)),
                      "weightings": list(WEIGHTINGS), "normalisations": list(NORMALISATIONS),
                      "smoke": SMOKE, "word_positions": len(words)},
           "cells": rows,
           "best_by_weighting": best,
           "association_over_cosine": asc - cos,
           "association_over_cosine_in_standard_errors": (asc - cos) / se if se else 0.0,
           "standard_error_of_a_difference": se,
           "cells_clearing_both": sum(1 for r in rows if r["clears_both"]),
           "cells_swept": len(rows),
           "seconds": time.time() - started}
    print("\n" + "=" * 78)
    print(f"  best by weighting: " + ", ".join(f"{k} {v:.4f}" for k, v in best.items()))
    print(f"  the instrument's reads over cosine: {asc - cos:+.4f} "
          f"({out['association_over_cosine_in_standard_errors']:+.1f} standard errors)")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    # A smoke run must never land on the evidence filename: it is a toy measurement and
    # would sit in the results directory looking exactly like a real one.
    name = "exp35.smoke.json" if SMOKE else "exp35.json"
    (dest / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
