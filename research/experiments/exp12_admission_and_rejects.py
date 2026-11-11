"""Stage 4e — the admission table, and the explanations that were tested and rejected.

Why this exists
---------------
The same reason as `exp11`. Four figures in the paper need a runnable script behind them: the
admission verdict across every corpus the program has used, and three explanations for the
real-text deficit that are tried and do not hold. `CLAIMS.md` says a row must cite a runnable
script, and this is the one those rows cite.

The admission table is the most load-bearing of the four — it is the evidence that the criterion
predicts where the reads work — so it belongs in a script rather than in prose.

What it measures
----------------
**A — the admission verdict on every corpus.** Six of them: the axis dial at three settings, the
triple corpus the forward pass succeeds on, and both books. The claim is that the verdict agrees with
the task outcome in every case.

**B — the statistic must be the association matrix.** Measuring the scalar coupling instead refuses a
corpus the forward pass scores 0.944 on, because a relation across orthogonal roles contracts to zero.
This is kept as a live measurement rather than a remark, since it is the difference between a working
criterion and a broken one.

**C — a low-count explanation, rejected.** PMI is unreliable at low counts, so the real-text deficit
might have been an artefact of plentiful data. Stratified by query frequency, it is not.

**D — cross-corpus transfer, partial and inconsistent.** Abstraction might transfer better across
corpora than counting does. It does in one direction and not the other.

**E — absence handling, real but small.** A segment where a word does not occur carries no
observation, and treating it as one is an artefact worth measuring — and not large enough to explain
anything.

Run
---
    python research/experiments/exp12_admission_and_rejects.py
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
from corpus import triple_corpus                                   # noqa: E402
from entroptics_infer import admission, bind, engine                # noqa: E402
from exp07_real_text import FREQ_TOLERANCE, N_CANDIDATES, SEED, TEXTS, WINDOW, _pmi  # noqa: E402
from exp09_axis_condition import trial as axis_trial               # noqa: E402

FAR = 0.05
BASE_SEED = 20260915
DRAWS = 3
ALICE, MOBY = TEXTS[0], TEXTS[1]


def _matrix_norm(records, a, b):
    return float(np.linalg.norm(bind.association_matrix(records[a], records[b])))


def _scalar_admission(records, pairs, *, far, seed, draws):
    """The admission test as it would be with the scalar coupling instead of the matrix.

    Kept so the difference between a working criterion and a broken one is a measurement rather than
    a remark.
    """
    rng = np.random.default_rng(seed)
    obs = [abs(engine.associate(records[a], records[b], far=far).z) for a, b in pairs]
    shuf = []
    for _ in range(draws):
        for a, b in pairs:
            ra = np.asarray(records[a], dtype=float)
            rb = np.asarray(records[b], dtype=float)
            sa = ra[rng.permutation(ra.shape[0])]
            sb = rb[rng.permutation(rb.shape[0])]
            shuf.append(abs(engine.associate(sa, sb, far=far).z))
    se_o = float(np.std(obs, ddof=1)) / np.sqrt(len(obs))
    se_s = float(np.std(shuf, ddof=1)) / np.sqrt(len(shuf))
    spread = float(np.sqrt(se_o ** 2 + se_s ** 2))
    return (float(np.mean(obs)) - float(np.mean(shuf))) / spread if spread > 0 else 0.0


def part_a_and_b():
    print("\nPARTS A & B — the admission verdict on every corpus\n")
    print(f"{'corpus':>28}{'observed':>10}{'decoupled':>10}{'margin_z':>10}{'admitted':>10}"
          f"{'rows':>7}{'row corr':>10}")
    rows = []

    for lam in (0.0, 0.2, 1.0):
        rng = np.random.default_rng(7)
        recs, pairs = {}, []
        for i in range(30):
            q, cands, gold = axis_trial(rng, lam)
            recs[f"q{i}"], recs[f"c{i}"] = q, cands[gold]
            pairs.append((f"q{i}", f"c{i}"))
        v = admission.axis_carries(recs, pairs, far=FAR, seed=1, draws=DRAWS)
        rows.append({"corpus": f"synthetic lambda={lam}", "observed": v.observed,
                     "decoupled": v.decoupled, "margin_z": v.margin_z, "admitted": v.admitted,
                     "rows": v.rows, "row_autocorrelation": v.row_autocorrelation})
        print(f"{rows[-1]['corpus']:>28}{v.observed:>10.1f}{v.decoupled:>10.1f}"
              f"{v.margin_z:>10.2f}{str(v.admitted):>10}{v.rows:>7}"
              f"{v.row_autocorrelation:>+10.3f}")

    c = triple_corpus(seed=BASE_SEED, n_relations=4, n_sources=30, T=96, D=32, noise=0.10)
    pr = [(a, c.targets[(a, k)]) for a in c.sources[:20] for k in range(4)]
    v = admission.axis_carries(c.records, pr, far=FAR, seed=1, draws=2)
    scalar = _scalar_admission(c.records, pr, far=FAR, seed=1, draws=2)
    rows.append({"corpus": "triple corpus", "observed": v.observed, "decoupled": v.decoupled,
                 "margin_z": v.margin_z, "admitted": v.admitted, "rows": v.rows,
                 "row_autocorrelation": v.row_autocorrelation,
                 "margin_z_with_scalar_statistic": scalar})
    print(f"{'triple corpus':>28}{v.observed:>10.1f}{v.decoupled:>10.1f}"
          f"{v.margin_z:>10.2f}{str(v.admitted):>10}{v.rows:>7}"
          f"{v.row_autocorrelation:>+10.3f}")

    for name, url, seg, nf, mc in (("real text: alice", *ALICE[1:]), ("real text: moby", *MOBY[1:])):
        t = TX.fetch(url, into=HERE.parents[1] / "corpora")
        half = list(t.tokens)[:len(t.tokens) // 2]
        ft = TX.Text(url=url, sha256=t.sha256, characters=0, tokens=tuple(half))
        tr = TX.build_records(ft, segments=seg, window=WINDOW, n_features=nf, min_count=mc)
        V = list(tr.vocabulary)
        rng = np.random.default_rng(3)
        pairs = [(V[i], V[j]) for i, j in rng.integers(0, len(V), size=(60, 2)) if i != j]
        v = admission.axis_carries(tr.records, pairs, far=FAR, seed=1, draws=DRAWS)
        rows.append({"corpus": name, "observed": v.observed, "decoupled": v.decoupled,
                     "margin_z": v.margin_z, "admitted": v.admitted, "rows": v.rows,
                     "row_autocorrelation": v.row_autocorrelation})
        print(f"{name:>28}{v.observed:>10.1f}{v.decoupled:>10.1f}"
              f"{v.margin_z:>10.2f}{str(v.admitted):>10}{v.rows:>7}"
              f"{v.row_autocorrelation:>+10.3f}")

    print(f"\n  the same triple corpus, scored with the SCALAR coupling instead of the matrix: "
          f"margin_z {scalar:+.2f}")
    print(f"  (the forward pass scores 0.944 on it — a scalar statistic refuses a corpus that works)")
    return rows


def _collocate_task(records, first, second, vocab, pool, rank_ok=True):
    """The held-out collocate task, shared by the two rejected explanations."""
    p2, _ = _pmi(second, vocab, WINDOW)
    p1, _ = _pmi(first, vocab, WINDOW)
    cf = Counter(first)
    rng = np.random.default_rng(SEED)
    out = []
    for q in pool:
        col = [(v, p2.get((q, v), -math.inf)) for v in pool if v != q]
        col = [(v, s) for v, s in col if s > -math.inf]
        if not col:
            continue
        truth = max(col, key=lambda x: x[1])[0]
        tgt = cf[truth]
        cand_pool = [v for v, _ in col if v != truth and abs(cf[v] - tgt) <= FREQ_TOLERANCE * tgt]
        if len(cand_pool) < N_CANDIDATES - 1:
            continue
        picked = [cand_pool[i]
                  for i in rng.choice(len(cand_pool), size=N_CANDIDATES - 1, replace=False)]
        cands = [truth] + picked
        order = rng.permutation(len(cands))
        cands = [cands[i] for i in order]
        gold = cands.index(truth)
        zs = [engine.associate(records[q], records[v], far=FAR).z for v in cands]
        pm = [p1.get((q, v), -math.inf) for v in cands]
        out.append((cf[q], int(int(np.argmax(zs)) == gold), int(int(np.argmax(pm)) == gold)))
    return out


def part_c():
    print("\nPART C — a low-count explanation, stratified by query frequency\n")
    print(f"{'text':>8}{'third':>16}{'n':>6}{'z':>9}{'PMI':>9}")
    rows = []
    for name, url, seg, nf, mc in (("alice", *ALICE[1:]), ("moby", *MOBY[1:])):
        t = TX.fetch(url, into=HERE.parents[1] / "corpora")
        toks = list(t.tokens)
        h = len(toks) // 2
        first, second = toks[:h], toks[h:]
        cf, cs = Counter(first), Counter(second)
        vocab = tuple(w for w in cf if cf[w] >= mc and cs[w] >= mc)
        ft = TX.Text(url=url, sha256=t.sha256, characters=0, tokens=tuple(first))
        tr = TX.build_records(ft, segments=seg, window=WINDOW, n_features=nf, min_count=mc)
        pool = [w for w in vocab if w in tr.records and engine.rank(tr.records[w], far=FAR) > 0]
        res = sorted(_collocate_task(tr.records, first, second, vocab, pool))
        third = max(1, len(res) // 3)
        for label, sl in (("low", res[:third]), ("middle", res[third:2 * third]),
                          ("high", res[2 * third:])):
            if not sl:
                continue
            row = {"text": name, "third": label, "n": len(sl),
                   "z": float(np.mean([r[1] for r in sl])),
                   "pmi": float(np.mean([r[2] for r in sl]))}
            row["z_leads"] = row["z"] > row["pmi"]
            rows.append(row)
            print(f"{name:>8}{label:>16}{len(sl):>6}{row['z']:>9.3f}{row['pmi']:>9.3f}")
    leads = sum(1 for r in rows if r["z_leads"])
    print(f"\n  the association leads in {leads} of {len(rows)} cells")
    return {"cells": rows, "z_leads_in": leads, "of": len(rows)}


def part_d():
    print("\nPART D — cross-corpus transfer\n")
    print(f"{'built on':>10}{'tested on':>12}{'n':>6}{'z':>9}{'PMI':>9}{'edge':>9}")
    rows = []
    texts = {"alice": ALICE, "moby": MOBY}
    for build, test, seg, nf, mc in (("moby", "moby", 48, 192, 40), ("moby", "alice", 64, 192, 40),
                                     ("alice", "alice", 16, 128, 20), ("alice", "moby", 24, 128, 20)):
        tb = TX.fetch(texts[build][1], into=HERE.parents[1] / "corpora")
        tt = TX.fetch(texts[test][1], into=HERE.parents[1] / "corpora")
        btok, ttok = list(tb.tokens), list(tt.tokens)
        if build == test:
            h = len(btok) // 2
            train, held = btok[:h], btok[h:]
        else:
            train, held = btok, ttok
        ctr, cte = Counter(train), Counter(held)
        vocab = tuple(w for w in ctr if ctr[w] >= mc and cte[w] >= 20)
        ft = TX.Text(url=texts[build][1], sha256=tb.sha256, characters=0, tokens=tuple(train))
        tr = TX.build_records(ft, segments=seg, window=WINDOW, n_features=nf, min_count=mc)
        pool = [w for w in vocab if w in tr.records and engine.rank(tr.records[w], far=FAR) > 0]
        res = _collocate_task(tr.records, train, held, vocab, pool)
        if not res:
            continue
        z = float(np.mean([r[1] for r in res]))
        pm = float(np.mean([r[2] for r in res]))
        rows.append({"built_on": build, "tested_on": test, "n": len(res),
                     "z": z, "pmi": pm, "edge": z - pm})
        print(f"{build:>10}{test:>12}{len(res):>6}{z:>9.3f}{pm:>9.3f}{z - pm:>+9.3f}")
    print("\n  (both degrade under corpus shift; whether the gap narrows depends on direction)")
    return rows


def part_e():
    print("\nPART E — absence: how much of a record sits in segments where the word never occurs\n")
    t = TX.fetch(MOBY[1], into=HERE.parents[1] / "corpora")
    half = list(t.tokens)[:len(t.tokens) // 2]
    ft = TX.Text(url=MOBY[1], sha256=t.sha256, characters=0, tokens=tuple(half))
    raw = TX.build_records(ft, segments=48, window=WINDOW, n_features=192,
                           min_count=40, remove_baseline=False)
    centred = TX.build_records(ft, segments=48, window=WINDOW, n_features=192, min_count=40)
    V = list(raw.vocabulary)
    present = {w: (raw.records[w].sum(axis=1) > 0) for w in V}
    shares = []
    for w in V:
        absent = ~present[w]
        if absent.sum() == 0:
            continue
        R = centred.records[w]
        shares.append(float(np.sum(R[absent] ** 2) / max(np.sum(R ** 2), 1e-30)))
    out = {"records": len(shares), "mean_share": float(np.mean(shares)),
           "max_share": float(np.max(shares)),
           "mean_segments_present": float(np.mean([p.sum() for p in present.values()])),
           "segments": 48}
    print(f"  a word is present in {out['mean_segments_present']:.1f} of 48 segments on average")
    print(f"  share of record energy in ABSENT rows: mean {out['mean_share']:.3f}, "
          f"max {out['max_share']:.3f}  ({len(shares)} records)")
    return out


def run():
    started = time.time()
    a, c, d, e = part_a_and_b(), part_c(), part_d(), part_e()
    out = {
        "design": {"far": FAR, "base_seed": BASE_SEED, "draws": DRAWS, "window": WINDOW},
        "admission": a, "frequency_strata": c, "transfer": d, "absence": e,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp12.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp12.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
