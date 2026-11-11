"""Stage 4d — the text diagnostics, as a script rather than as prose.

Why this exists
---------------
CLAIMS Stage 4b/4c carries a dozen figures measured on the way to its conclusion: how much of a text
record the identity component hides, how collocations score against arbitrary pairs, where text
sits on the axis dial, what happens under a second-order task, and the four explanations that are
tested and rejected.

`CLAIMS.md` says a row may not cite a document, only a runnable script. This is the script those
rows cite. Nothing here is new work — it is the same measurements, kept runnable so `verify.py` can
check them and so a change to the record builder cannot move them silently.

What it measures
----------------
**A — the identity component.** Every word sits next to the same common words, so every record carries
the corpus unigram distribution as a large shared part that says nothing about which word it is.
Removing it is the largest effect measured on text here.

**B — the segment sweep.** A record whose rows are mostly empty reads to the aperture as noise does,
reproducing `paper-2` §65 on natural text rather than on an ontology. (`paper-2` is *Knowledge without weights*, ``research/paper-2-knowledge-without-weights.md`` in
https://github.com/Agience/agience-pharos.)

**C — collocations against arbitrary pairs**, and what `tightness` says about the difference between
a fixed bigram and a diffuse association.

**D — where text sits on the axis dial.** The measurement that closes PAPER §6.3's account: do related pairs
separate along the ordered axis, or only off it?

**E — the second-order task.** The fair test of the claim: the subspace read against the *same
information aggregated to a vector*, on held-out distributional similarity.

**F — two rejected explanations.** That PMI is unreliable at low counts, and that abstraction should
transfer better across corpora.

Run
---
    python research/experiments/exp11_text_diagnostics.py
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
sys.path.insert(0, str(HERE.parent))

import text as TX                                                  # noqa: E402
from entroptics_infer import engine                                # noqa: E402
from exp07_real_text import FREQ_TOLERANCE, N_CANDIDATES, SEED, TEXTS, WINDOW, _pmi  # noqa: E402
from measure import correlation as _corr  # noqa: E402

FAR = 0.05
ALICE, MOBY = TEXTS[0], TEXTS[1]
SEGMENT_SWEEP = (32, 48, 64, 96, 128)
RANDOM_PAIRS = 200
TOP_PAIRS = 200


def _halves(url):
    t = TX.fetch(url, into=HERE.parents[1] / "corpora")
    toks = list(t.tokens)
    h = len(toks) // 2
    return t, toks, toks[:h], toks[h:]


def _first_half_text(t, first):
    return TX.Text(url=t.url, sha256=t.sha256, characters=0, tokens=tuple(first))


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def part_a():
    """The identity component: what a text record has left once the corpus's own profile is gone."""
    print("\nPART A — the identity component (Moby-Dick, first half, 48 segments)\n")
    _, _, first, _ = _halves(MOBY[1])
    ft = TX.Text(url=MOBY[1], sha256="", characters=0, tokens=tuple(first))
    out = {}
    for label, strip in (("raw", False), ("baseline removed", True)):
        tr = TX.build_records(ft, segments=48, window=WINDOW, n_features=192,
                              min_count=40, remove_baseline=strip)
        ks = [engine.rank(tr.records[w], far=FAR) for w in tr.vocabulary]
        out[label] = {"resolving": int(sum(1 for k in ks if k > 0)), "total": len(ks),
                      "mean_rank": float(np.mean(ks)), "max_rank": int(max(ks))}
        print(f"  {label:<20} resolving {out[label]['resolving']:>3} of {len(ks)}   "
              f"mean rank {out[label]['mean_rank']:.2f}   max {out[label]['max_rank']}")
    return out


def part_b():
    """The segment sweep: a record whose rows empty out reads as noise does."""
    print("\nPART B — records resolving anything, against segment count\n")
    _, _, first, _ = _halves(MOBY[1])
    ft = TX.Text(url=MOBY[1], sha256="", characters=0, tokens=tuple(first))
    rows = []
    for seg in SEGMENT_SWEEP:
        tr = TX.build_records(ft, segments=seg, window=WINDOW, n_features=192,
                              min_count=40, remove_baseline=False)
        ks = [engine.rank(tr.records[w], far=FAR) for w in tr.vocabulary]
        rows.append({"segments": seg, "resolving": int(sum(1 for k in ks if k > 0)),
                     "total": len(ks)})
        print(f"  {seg:>4} segments   {rows[-1]['resolving']:>3} of {rows[-1]['total']}")
    return rows


def part_c():
    """Collocations against arbitrary pairs, and what tightness says about the difference."""
    print("\nPART C — collocations against arbitrary pairs (Alice)\n")
    # The whole book, not a half. This part is descriptive rather than predictive — it asks what a
    # collocation scores against an arbitrary pair, with nothing held out — so there is no split to
    # respect, and halving the text drops the rarer collocates below the count threshold entirely
    # (`march` occurs 34 times in the whole of Alice, and `hare` 31).
    t, _, _, _ = _halves(ALICE[1])
    tr = TX.build_records(t, segments=64, window=WINDOW, n_features=128,
                          min_count=25, remove_baseline=False)
    V = set(tr.vocabulary)
    named = []
    for a, b in (("mock", "turtle"), ("march", "hare"), ("white", "rabbit"),
                 ("gryphon", "turtle"), ("alice", "said")):
        if a in V and b in V:
            r = engine.associate(tr.records[a], tr.records[b], far=FAR)
            named.append({"pair": f"{a}/{b}", "z": r.z, "tightness": r.tightness})
            print(f"  {a:<9}{b:<9} z = {r.z:>6.2f}   tightness {r.tightness:.3f}")
    rng = np.random.default_rng(0)
    vocab = list(tr.vocabulary)
    zs = []
    for _ in range(RANDOM_PAIRS):
        i, j = rng.choice(len(vocab), size=2, replace=False)
        zs.append(engine.associate(tr.records[vocab[i]], tr.records[vocab[j]], far=FAR).z)
    zs = np.array(zs)
    ctrl = {"n": RANDOM_PAIRS, "mean": float(zs.mean()),
            "p95": float(np.percentile(zs, 95))}
    print(f"\n  {RANDOM_PAIRS} random pairs: mean z {ctrl['mean']:.2f}, "
          f"95th percentile {ctrl['p95']:.2f}")
    return {"named": named, "random": ctrl}


def part_d():
    """Where text sits on the axis dial: do related pairs separate along the axis, or off it?"""
    print("\nPART D — related pairs: separation on the axis against off it\n")
    print(f"{'text':>10}{'axis (related)':>17}{'axis (median)':>16}{'axis sep':>11}"
          f"{'profile sep':>14}")
    rows = []
    for name, url, seg, nf, mc in (("alice", *ALICE[1:]), ("moby", *MOBY[1:])):
        t, _, first, second = _halves(url)
        cf, cs = Counter(first), Counter(second)
        vocab = [w for w in cf if cf[w] >= mc and cs[w] >= mc]
        ft = _first_half_text(t, first)
        raw = TX.build_records(ft, segments=seg, window=WINDOW, n_features=nf,
                               min_count=mc, remove_baseline=False)
        V = [w for w in vocab if w in raw.records]
        traj = {w: raw.records[w].sum(axis=1) for w in V}
        prof = {w: raw.records[w].sum(axis=0) for w in V}
        v2 = TX.cooccurrence(second, window=WINDOW, features=raw.features)
        pairs = [(a, b, _cos(v2[a], v2[b]))
                 for i, a in enumerate(V) for b in V[i + 1:] if a in v2 and b in v2]
        pairs.sort(key=lambda x: -x[2])
        top = pairs[:TOP_PAIRS]
        mid = pairs[len(pairs) // 2: len(pairs) // 2 + TOP_PAIRS]
        at, am = np.mean([_corr(traj[a], traj[b]) for a, b, _ in top]), \
            np.mean([_corr(traj[a], traj[b]) for a, b, _ in mid])
        pt, pm = np.mean([_cos(prof[a], prof[b]) for a, b, _ in top]), \
            np.mean([_cos(prof[a], prof[b]) for a, b, _ in mid])
        row = {"text": name, "pairs": len(pairs), "axis_related": float(at),
               "axis_median": float(am), "axis_separation": float(at - am),
               "profile_related": float(pt), "profile_separation": float(pt - pm)}
        rows.append(row)
        print(f"{name:>10}{at:>17.3f}{am:>16.3f}{at - am:>+11.3f}{pt - pm:>+14.3f}")
    print("\n(related pairs are the 200 most similar in HELD-OUT text; median pairs for contrast)")
    return rows


def part_e():
    """The second-order task: the subspace against the same information aggregated to a vector."""
    print("\nPART E — a second-order task with a second-order baseline\n")
    print(f"{'text':>10}{'n':>6}{'random':>9}{'subspace':>11}{'vector cosine':>16}{'PMI':>9}")
    rows = []
    for name, url, seg, nf, mc in (("alice", *ALICE[1:]), ("moby", *MOBY[1:])):
        t, _, first, second = _halves(url)
        cf, cs = Counter(first), Counter(second)
        vocab = tuple(w for w in cf if cf[w] >= mc and cs[w] >= mc)
        ft = _first_half_text(t, first)
        tr = TX.build_records(ft, segments=seg, window=WINDOW, n_features=nf, min_count=mc)
        v1 = TX.cooccurrence(first, window=WINDOW, features=tr.features)
        v2 = TX.cooccurrence(second, window=WINDOW, features=tr.features)
        p1, _ = _pmi(first, vocab, WINDOW)
        pool = [w for w in vocab if w in tr.records and engine.rank(tr.records[w], far=FAR) > 0
                and w in v1 and w in v2]
        rng = np.random.default_rng(SEED)
        hit = cos = pmi = n = 0
        for q in pool:
            sims = [(v, _cos(v2[q], v2[v])) for v in pool if v != q]
            truth = max(sims, key=lambda x: x[1])[0]
            tgt = cf[truth]
            cand_pool = [v for v, _ in sims
                         if v != truth and abs(cf[v] - tgt) <= FREQ_TOLERANCE * tgt]
            if len(cand_pool) < N_CANDIDATES - 1:
                continue
            picked = [cand_pool[i]
                      for i in rng.choice(len(cand_pool), size=N_CANDIDATES - 1, replace=False)]
            cands = [truth] + picked
            order = rng.permutation(len(cands))
            cands = [cands[i] for i in order]
            gold = cands.index(truth)
            zs = [engine.associate(tr.records[q], tr.records[v], far=FAR).z for v in cands]
            cs_ = [_cos(v1[q], v1[v]) for v in cands]
            pm = [p1.get((q, v), -math.inf) for v in cands]
            hit += int(int(np.argmax(zs)) == gold)
            cos += int(int(np.argmax(cs_)) == gold)
            pmi += int(int(np.argmax(pm)) == gold)
            n += 1
        row = {"text": name, "questions": n, "random": 1.0 / N_CANDIDATES,
               "subspace": hit / n, "vector_cosine": cos / n, "first_order_pmi": pmi / n}
        rows.append(row)
        print(f"{name:>10}{n:>6}{row['random']:>9.3f}{row['subspace']:>11.3f}"
              f"{row['vector_cosine']:>16.3f}{row['first_order_pmi']:>9.3f}")
    print("\n(truth is held-out distributional similarity; the only difference between the two "
          "methods is vector against subspace)")
    return rows


def run():
    started = time.time()
    a, b, c, d, e = part_a(), part_b(), part_c(), part_d(), part_e()
    out = {
        "design": {"far": FAR, "window": WINDOW, "segment_sweep": SEGMENT_SWEEP,
                   "random_pairs": RANDOM_PAIRS, "top_pairs": TOP_PAIRS, "seed": SEED},
        "identity_component": a, "segment_sweep": b, "collocations": c,
        "axis_dial": d, "second_order": e,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp11.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp11.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
