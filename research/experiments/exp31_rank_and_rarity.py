"""Stage 24 — score the whole ranking, and stratify where a lookup is actually weak.

Two things a lookup's measurement has to get right.

**Top-1 accuracy is the wrong statistic.** Over a thousand candidates it throws away everything
except whether the single best guess was right, so it is dominated by the commonest words and needs
enormous samples to resolve a difference of 0.01. The reciprocal rank of the true word uses the
whole ordering, is bounded, and needs no temperature or softmax to define — which matters here,
because a softmax would introduce exactly the kind of constant this program refuses. Every read in
this stage already produces a score for every candidate, so the ranking is free.

**A short context is not a weak lookup.** Stratifying by how deep a context the n-gram found
treats depth 1 as the regime where a lookup is helpless. It is the opposite: depth 1 means the
bigram fired, and a bigram answering with a common word does *well*. Measured, the n-gram scored
higher in that band (0.095) than overall (0.093), so the stratifier selected for the lookup rather
than against it.

What a lookup is genuinely bad at is a rare continuation. Frequency-shaped answers carry it when
the next word is common and abandon it when the next word is not, whatever the context length. So
the questions are placed by how common the true next word is, in quartiles of the training half's
own counts — the corpus's cuts, not chosen ones. The rarest quartile is where a method that reads
context has to show itself.

What is compared
-----------------
* **operator** — the corpus folded into one `context -> next word` map, in the disjoint-role shared
  basis CLAIMS §8.2 requires. A sum over occasions; no gradient step, no learned projection.
* **cosine** — the scalar competitor, scoring candidates by cosine-weighted votes over the identical
  cache, so it ranks rather than answering once.
* **n-gram** — the incumbent, itself free of fixed weights, and given the strongest fit-free form
  available here: every context order that matched contributes to its ordering, weighted so a
  longer context dominates a shorter one and the shorter ones break its ties, with frequency
  underneath as the final fallback. Reading only the deepest order and discarding the rest is
  weaker, and measured weaker — it is the incumbent, so it gets the better version.
* **unigram** — ranking by frequency alone. This is the floor, and reporting each read's margin
  *over* it is the only comparison that means anything when the floor is close.
* **rarity** — ranking by frequency *inverted*, rarest first. This is the control the rarest
  quartile needs and it reads nothing at all: it knows no context and cannot, so any score it gets
  there is what the stratum hands out for free. A read that beats the lookup on rare targets has
  shown something only if it also beats this, because "does not favour common words" is a property
  of an operating point rather than of understanding.

Both bases are carried through: the one-hot frame and the counted profile, whose
scaling is swept rather than chosen. The representation is the only thing that differs between them.
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
from entroptics_infer.bind import association_matrix                # noqa: E402
from measure import reciprocal_rank as _reciprocal_rank  # noqa: E402

FAR = 0.05
SEED = 11
SMOKE = os.environ.get("EXP31_SMOKE") == "1"

SOURCES = [
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
]

VOCAB = 256 if SMOKE else 1024
CO_WINDOW = 4
WINDOW = 4 if SMOKE else 8
CONES = (4,) if SMOKE else (2, 4, 8)
SCALINGS = ("unit",) if SMOKE else ("unit", "shared")
CACHE = 400 if SMOKE else 8000
TESTS = 60 if SMOKE else 4000
QUARTILES = 4


def _profiles(code, V, split, scaling):
    """Each word's decayed company, counted over the training half only. Nothing is fitted."""
    F = V + 1
    P = np.zeros((F, F))
    decay = np.exp(-np.arange(1, CO_WINDOW + 1) / CO_WINDOW)
    for p in range(split):
        c = code[p]
        for j, q in enumerate(range(p - 1, max(-1, p - CO_WINDOW - 1), -1)):
            P[c, code[q]] += decay[j]
            P[code[q], c] += decay[j]
    if scaling == "shared":
        spread = P.sum(axis=0, keepdims=True)
        P = P / np.where(spread > 0, spread, 1.0)
    n = np.linalg.norm(P, axis=1, keepdims=True)
    return P / np.where(n > 0, n, 1.0)


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
    # frequency inverted: the control for the rarest quartile, which knows no context whatever
    rarity_scores = -unigram_scores

    # the corpus's own cuts: quartiles of how common the true next word is in the training half
    freqs = train_counts[[int(code[p]) for p in test_at]]
    cuts = np.quantile(freqs, [0.25, 0.5, 0.75])
    band_of = {int(p): int(np.searchsorted(cuts, train_counts[int(code[p])], side="right"))
               for p in test_at}
    print(f"{len(words)} words, vocab {V}, window {W}, {len(test_at)} decisions")
    print(f"target-frequency quartile cuts at counts {cuts.astype(int).tolist()}")

    tables = {}
    for n in range(1, W + 1):
        tab = {}
        for p in range(n, split):
            tab.setdefault(tuple(int(q) for q in code[p - n:p]), Counter())[int(code[p])] += 1
        tables[n] = tab

    def ngram_scores(p):
        """A full ordering over every candidate, built from every context order that matched.

        The incumbent is given the strongest fit-free form available here. Rather than reading the
        deepest matching order and discarding the rest, every order that matched contributes,
        weighted so that a longer context dominates a shorter one and the shorter ones break its
        ties. Frequency sits underneath as the final fallback, which is what a backoff model does
        when it has nothing else. Nothing is smoothed by a fitted parameter: these are counts.
        """
        s = unigram_scores / (unigram_scores.max() or 1.0) * 1e-6
        for n in range(1, W + 1):
            hit = tables[n].get(tuple(int(q) for q in code[p - n:p]))
            if hit:
                for k, v in hit.items():
                    if k < V:
                        s[k] += v * (10.0 ** n)
        return s

    ng_cached = {int(p): ngram_scores(int(p)) for p in test_at}
    profiles = {s: _profiles(code, V, split, s) for s in SCALINGS}

    rows, cells = [], []
    for basis in ("one-hot", "counted"):
        for scaling in (SCALINGS if basis == "counted" else ("n/a",)):
            for CONE in CONES:
                back = np.exp(-np.arange(W - 1, -1, -1) / CONE)
                P = profiles[scaling] if basis == "counted" else None

                def frame(p):
                    if basis == "one-hot":
                        f = np.zeros((W, F))
                        f[np.arange(W), code[p - W:p]] = back
                        return f.ravel()
                    return (P[code[p - W:p]] * back[:, None]).ravel()

                CW = W * F
                K = np.asarray([frame(int(p)) for p in cache_at])
                targets = np.asarray([int(code[p]) for p in cache_at])
                Y = np.zeros((len(cache_at), F))
                Y[np.arange(len(cache_at)), targets] = 1.0
                A = np.hstack([K, np.zeros((len(cache_at), F))])
                B = np.hstack([np.zeros((len(cache_at), CW)), Y])
                M = association_matrix(A, B)[:CW, CW:]
                Kmean, Kn = K.mean(axis=0), np.linalg.norm(K, axis=1)
                # full width first, then drop the collapsed bucket. Clipping it onto the last
                # real candidate instead would pile every rare word's vote onto one word, which
                # would corrupt exactly the rarest quartile this stage is built to read.
                onehot_cache = np.zeros((len(cache_at), F))
                onehot_cache[np.arange(len(cache_at)), targets] = 1.0
                onehot_cache = onehot_cache[:, :V]

                acc = {k: [] for k in ("operator", "cosine", "n_gram", "unigram",
                                      "rarity")}
                by_band = {}
                for p in test_at:
                    p = int(p)
                    truth = int(code[p])
                    q = frame(p)
                    op = ((q - Kmean) @ M)[:V]
                    qn = float(np.linalg.norm(q))
                    sim = (K @ q) / (Kn * qn + 1e-30)
                    cos = np.clip(sim, 0, None) @ onehot_cache
                    ng = ng_cached[p][:V]
                    r = {"operator": _reciprocal_rank(op, truth),
                         "cosine": _reciprocal_rank(cos, truth),
                         "n_gram": _reciprocal_rank(ng, truth),
                         "unigram": _reciprocal_rank(unigram_scores, truth),
                         "rarity": _reciprocal_rank(rarity_scores, truth)}
                    for k, v in r.items():
                        acc[k].append(v)
                    by_band.setdefault(band_of[p], []).append(r)

                n_t = len(test_at)
                row = {"basis": basis, "scaling": scaling, "window": W, "cone": CONE,
                       "tests": n_t,
                       **{f"mrr_{k}": float(np.mean(v)) for k, v in acc.items()},
                       **{f"se_{k}": float(np.std(v, ddof=1) / np.sqrt(n_t))
                          for k, v in acc.items()}}
                row["operator_over_unigram"] = row["mrr_operator"] - row["mrr_unigram"]
                row["n_gram_over_unigram"] = row["mrr_n_gram"] - row["mrr_unigram"]
                bands = []
                for b in sorted(by_band):
                    rs = by_band[b]
                    bands.append({"quartile": b, "n": len(rs),
                                  **{k: float(np.mean([x[k] for x in rs]))
                                     for k in ("operator", "cosine", "n_gram", "unigram",
                                               "rarity")}})
                row["rarest_operator"] = bands[0]["operator"] if bands else float("nan")
                row["rarest_n_gram"] = bands[0]["n_gram"] if bands else float("nan")
                row["rarest_rarity_control"] = bands[0]["rarity"] if bands else float("nan")
                rows.append(row)
                cells.append({"basis": basis, "scaling": scaling, "cone": CONE, "bands": bands})
                print(f"  {basis:>8} {scaling:>7} cone={CONE}   MRR operator "
                      f"{row['mrr_operator']:.4f}+/-{row['se_operator']:.4f}   "
                      f"n-gram {row['mrr_n_gram']:.4f}   unigram {row['mrr_unigram']:.4f}   "
                      f"rarest: op {row['rarest_operator']:.4f} / "
                      f"ng {row['rarest_n_gram']:.4f} / "
                      f"rarity-control {row['rarest_rarity_control']:.4f}")

    best = max(rows, key=lambda r: r["mrr_operator"])
    # Every comparison this stage publishes, recorded in standard errors rather than worked out in
    # prose where nothing can check the arithmetic. The error on a difference of two means is the
    # root of the sum of their squares; both are measured on the same decisions, so this is
    # conservative rather than optimistic.
    best_onehot = max((r for r in rows if r["basis"] == "one-hot"),
                      key=lambda r: r["mrr_operator"])
    sd = math.sqrt(best["se_operator"] ** 2 + best_onehot["se_operator"] ** 2)
    uni, ngr = rows[0]["mrr_unigram"], rows[0]["mrr_n_gram"]
    comparisons = {
        "best_operator": best["mrr_operator"],
        "best_one_hot_operator": best_onehot["mrr_operator"],
        "counted_over_one_hot": best["mrr_operator"] - best_onehot["mrr_operator"],
        "counted_over_one_hot_in_standard_errors":
            (best["mrr_operator"] - best_onehot["mrr_operator"]) / sd,
        "operator_over_unigram": best["mrr_operator"] - uni,
        "operator_over_unigram_in_standard_errors":
            (best["mrr_operator"] - uni) / sd,
        "lookup_over_operator": ngr - best["mrr_operator"],
        "lookup_over_operator_in_standard_errors": (ngr - best["mrr_operator"]) / sd,
        "standard_error_of_a_difference": sd,
    }
    beats = sum(1 for r in rows if r["mrr_operator"] > r["mrr_n_gram"])
    beats_rare = sum(1 for r in rows if r["rarest_operator"] > r["rarest_n_gram"])
    beats_control = sum(1 for r in rows
                        if r["rarest_operator"] > r["rarest_rarity_control"])
    out = {"sources": corpora,
           "design": {"far": FAR, "seed": SEED, "vocabulary": V, "window": W,
                      "co_window": CO_WINDOW, "cones": list(CONES), "scalings": list(SCALINGS),
                      "cache": CACHE, "tests": int(len(test_at)), "quartiles": QUARTILES,
                      "smoke": SMOKE, "word_positions": len(words)},
           "cells": rows, "bands": cells,
           "best_mrr_operator": best["mrr_operator"],
           "comparisons": comparisons,
           "best_cell": {k: best[k] for k in ("basis", "scaling", "cone")},
           "cells_where_the_operator_outranks_the_lookup": beats,
           "cells_where_it_outranks_it_on_the_rarest_quartile": beats_rare,
           "cells_where_it_also_outranks_the_rarity_control": beats_control,
           "cells_swept": len(rows),
           "seconds": time.time() - started}
    print("\n" + "=" * 78)
    print(f"  The operator outranks the lookup in {beats} of {len(rows)} cells overall, and in "
          f"{beats_rare} of {len(rows)}")
    print("  on the rarest quartile of targets — where frequency-shaped answers cannot help.")
    print(f"  It also outranks the context-blind rarity control there in {beats_control} of "
          f"{len(rows)} cells,")
    print("  which is what separates reading the context from merely not favouring common words.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    # A smoke run must never land on the evidence filename: it is a toy measurement and
    # would sit in the results directory looking exactly like a real one.
    name = "exp31.smoke.json" if SMOKE else "exp31.json"
    (dest / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
