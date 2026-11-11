"""Stage 4b — the same reads on a real text, with held-out ground truth.

The question
------------
Every corpus up to here was generated from a stated process, because the job was to check
calculations against known answers. That cannot show the method works on language. This does the
first real-text test: records built from the first half of a book, scored against what actually
happens in the second half.

The text is *Alice's Adventures in Wonderland*, fetched fresh from Project Gutenberg, ~27,000 tokens.
It is small on purpose — the program expands a corpus only after a gate passes on a smaller one.

What a word's record is
-----------------------
A `(segments, features)` field: the ordered axis is position in the first half, cut into segments,
and the feature axis is co-occurrence with the commonest types. Every word shares the same segment
axis, which is what allows two of them to be compared — the association re-pairs rows, so rows must
correspond to the same thing. Nothing is trained: no embedding table, no distillation, no fitted
matrix.

The task and its ground truth
------------------------------
For a query word `q`, which candidate is the word it genuinely goes with? Truth is taken from the
second half only — the candidate with the highest pointwise mutual information with `q` there —
and the method never sees it. Distractors are frequency-matched to the true collocate, so
frequency cannot be the giveaway.

The baseline that matters
--------------------------
Not random. First-half PMI — direct count memorization over exactly the data the method is given.
If the geometric read cannot beat counting on held-out data, it is doing nothing that counting does
not already do, and the result should say so.

What would falsify this stage
-----------------------------
Stated before the run. The stage fails if the association does not beat random; and it is reported as
no advance over counting if it does not beat first-half PMI on held-out truth.

Run
---
    python research/experiments/exp07_real_text.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import text as TX                                                  # noqa: E402
from entroptics_infer import bind, engine                          # noqa: E402

# ── the experiment's declared design ────────────────────────────────────────────────────────────
WINDOW = 6
N_CANDIDATES = 8
FAR = 0.05
SEED = 20260915
FREQ_TOLERANCE = 0.5          # a distractor's count must lie within this fraction of the truth's

#: Two texts, both public domain and fetched fresh. The segment count is set per text so that rows
#: are populated: a record whose rows are mostly empty reads to the aperture as noise does, which is
#: `paper-2` §65 (*Knowledge without weights*, in github.com/Agience/agience-pharos) and not a
#: fact about this method. Measured on Moby Dick's first half, every one of its 302 records
#: resolves something at each segment count from 32 to 128 (`exp11`, part B), so the setting is
#: chosen to keep rows populated and is reported with the result rather than tuned against the score.
TEXTS = (
    ("alice", "https://www.gutenberg.org/files/11/11-0.txt", 16, 128, 20),
    ("moby-dick", "https://www.gutenberg.org/files/2701/2701-0.txt", 48, 192, 40),
)


def _pmi(tokens, vocab, window):
    """Pointwise mutual information between types, from co-occurrence within a window."""
    unigram = Counter(tokens)
    total = len(tokens)
    joint: dict = {}
    n = len(tokens)
    keep = set(vocab)
    for p, w in enumerate(tokens):
        if w not in keep:
            continue
        lo, hi = max(0, p - window), min(n, p + window + 1)
        for q in range(lo, hi):
            if q == p:
                continue
            v = tokens[q]
            if v in keep:
                joint[(w, v)] = joint.get((w, v), 0) + 1
    pairs = sum(joint.values()) or 1
    out: dict = {}
    for (w, v), c in joint.items():
        pw, pv = unigram[w] / total, unigram[v] / total
        out[(w, v)] = math.log((c / pairs) / (pw * pv)) if pw and pv else 0.0
    return out, unigram


def one_text(name, url, segments, n_features, min_count):
    rng = np.random.default_rng(SEED)
    text = TX.fetch(url, into=HERE.parents[1] / "corpora")
    tokens = list(text.tokens)
    half = len(tokens) // 2
    first, second = tokens[:half], tokens[half:]
    print(f"\n{text.url}\n  sha256 {text.sha256[:16]}…  {len(tokens)} tokens, "
          f"split {len(first)} / {len(second)}\n")

    c_first, c_second = Counter(first), Counter(second)
    vocab = tuple(w for w in c_first
                  if c_first[w] >= min_count and c_second[w] >= min_count)
    print(f"vocabulary present in both halves at >= {min_count}: {len(vocab)} types")

    # Records from the first half only.
    first_text = TX.Text(url=text.url, sha256=text.sha256, characters=0, tokens=tuple(first))
    tr = TX.build_records(first_text, segments=segments, window=WINDOW,
                          n_features=n_features, min_count=min_count)
    usable = tuple(w for w in vocab if w in tr.records)
    print(f"with a record built from the first half: {len(usable)} types")

    pmi_second, _ = _pmi(second, usable, WINDOW)
    pmi_first, _ = _pmi(first, usable, WINDOW)

    ranks = {w: 0 for w in usable}
    for w in usable:
        ranks[w] = engine.rank(tr.records[w], far=FAR)
    resolved = [w for w in usable if ranks[w] > 0]
    print(f"whose record resolves anything at all: {len(resolved)}\n")

    hits = {"z": 0, "gated_correct": 0, "gated_answered": 0, "pmi_first": 0, "random": 0}
    asked = 0
    per_query = []

    for q in resolved:
        # truth: the strongest collocate in the second half
        collocates = [(v, pmi_second.get((q, v), -math.inf)) for v in resolved if v != q]
        collocates = [(v, s) for v, s in collocates if s > -math.inf]
        if not collocates:
            continue
        truth, _ = max(collocates, key=lambda x: x[1])

        # distractors, frequency-matched to the truth so frequency is not the giveaway
        target = c_first[truth]
        pool = [v for v, _ in collocates
                if v != truth and abs(c_first[v] - target) <= FREQ_TOLERANCE * target]
        if len(pool) < N_CANDIDATES - 1:
            continue
        picked = [pool[i] for i in rng.choice(len(pool), size=N_CANDIDATES - 1, replace=False)]
        cands = [truth] + picked
        order = rng.permutation(len(cands))
        cands = [cands[i] for i in order]
        gold = cands.index(truth)

        frames = [tr.records[v] for v in cands]
        zs = [engine.associate(tr.records[q], f, far=FAR).z for f in frames]
        hits["z"] += int(int(np.argmax(zs)) == gold)

        d = bind.decide(tr.records[q], frames, far=FAR)
        if d.key is not None:
            hits["gated_answered"] += 1
            hits["gated_correct"] += int(d.key == gold)

        counts = [pmi_first.get((q, v), -math.inf) for v in cands]
        hits["pmi_first"] += int(int(np.argmax(counts)) == gold)
        hits["random"] += 1.0 / len(cands)
        asked += 1
        per_query.append({"query": q, "truth": truth, "z_correct": int(int(np.argmax(zs)) == gold)})

    print(f"questions asked: {asked}  ({N_CANDIDATES} frequency-matched candidates each)\n")
    print(f"{'method':>26}{'top-1':>9}")
    print(f"{'random':>26}{hits['random'] / asked:>9.3f}")
    print(f"{'first-half PMI (counting)':>26}{hits['pmi_first'] / asked:>9.3f}")
    print(f"{'association z':>26}{hits['z'] / asked:>9.3f}")
    gated = (hits["gated_correct"] / hits["gated_answered"]) if hits["gated_answered"] else None
    cover = hits["gated_answered"] / asked
    print(f"{'gated decision':>26}{(gated if gated is not None else float('nan')):>9.3f}"
          f"   at coverage {cover:.3f}")

    out = {
        "name": name,
        "text": {"url": text.url, "sha256": text.sha256, "tokens": len(tokens),
                 "first_half": len(first), "second_half": len(second)},
        "design": {"segments": segments, "window": WINDOW, "n_features": n_features,
                   "min_count": min_count, "n_candidates": N_CANDIDATES, "far": FAR,
                   "seed": SEED, "freq_tolerance": FREQ_TOLERANCE},
        "vocabulary": {"both_halves": len(vocab), "with_record": len(usable),
                       "resolving": len(resolved)},
        "questions": asked,
        "top1": {
            "random": hits["random"] / asked,
            "pmi_first_half": hits["pmi_first"] / asked,
            "association_z": hits["z"] / asked,
            "gated_precision": gated,
            "gated_coverage": cover,
        },
        "per_query": per_query,
    }
    return out


def run():
    started = time.time()
    results = [one_text(*spec) for spec in TEXTS]
    out = {"texts": results, "seconds": time.time() - started}
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp07.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'text':>12}{'questions':>11}{'random':>9}{'PMI':>9}{'z':>9}{'gated':>9}{'cover':>8}")
    for r in results:
        t = r["top1"]
        g = "      n/a" if t["gated_precision"] is None else f"{t['gated_precision']:>9.3f}"
        print(f"{r['name']:>12}{r['questions']:>11}{t['random']:>9.3f}"
              f"{t['pmi_first_half']:>9.3f}{t['association_z']:>9.3f}{g}"
              f"{t['gated_coverage']:>8.3f}")
    print(f"\nwrote {dest / 'exp07.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
