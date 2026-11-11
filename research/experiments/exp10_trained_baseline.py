"""Stage 6 — against a trained baseline, with the protocol fixed before it runs.

The baseline, and why this one
-------------------------------
RESCAL (Nickel, Tresp & Kriegel, ICML 2011) is the method this work positions against: it scores a
triple as `x_sᵀ R_k x_o` with entity vectors and a matrix per relation, both obtained by gradient
descent. That is the same object this program measures instead of fitting, so it is the comparison
that means something. It is written from scratch here — plain gradient descent, no framework — on the
identical corpus.

The asymmetries, all favouring the baseline
--------------------------------------------
Stated up front because they are the point, not a caveat.

1. **The baseline is given the relation index.** Every training triple arrives labelled with which
   relation it is. The measured path is given no labels at all and must take its operator from an
   exemplar pair.
2. **The baseline is fitted to this corpus.** It sees the training triples repeatedly and adjusts
   until it explains them.
3. **The baseline's hyperparameters are tuned on a validation split** — latent dimension, learning
   rate and training budget — as the literature does. Under-resourcing a baseline proves
   nothing, and the training budget is the easiest way to do it by accident: at 400 epochs this
   model fits its own triples at 0.576 and scores at chance held-out, while at 2000 it fits at 1.000
   and reaches 0.921. Epochs are therefore selected, not fixed.

The measured path gets none of these. If it wins anywhere, it wins against a steelman.

The two regimes, and only the second is the claim
---------------------------------------------------
**Seen entities.** Held-out *triples* over sources the baseline trained on — the standard link
prediction setting. This is the baseline's home ground and it is expected to win. Reported either
way.

**Unseen entities.** Sources that appear in no training triple. A trained model has no vector for
an entity it never saw, which is not a deficiency of RESCAL but a property of every method that
learns per-entity parameters. The measured path has nothing to train, so an unseen record is read
exactly like a seen one.

**This second regime is the claim.** "Inference without a trained model" is worth something
precisely where there was no opportunity to train.

What would make this uninformative
-----------------------------------
Stated before the run. Targets are drawn from a shared pool so each appears in many triples: with
a private target per source, a held-out triple names an entity the baseline has never seen, its
vector is random, and the task is unlearnable by construction — a trained model would then score at
chance for a reason that has nothing to do with the method. The comparison is also vacuous if the
baseline does not fit: a model that
cannot score its own training triples is not a baseline, so training accuracy is reported and the
result is void below chance-plus-a-margin. And the cold-start result means nothing unless the
baseline is given its best available fallback rather than being scored at zero by construction —
here it falls back to the mean entity vector, which is what an implementer would do.

Run
---
    python research/experiments/exp10_trained_baseline.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import triple_corpus                                   # noqa: E402
from entroptics_infer import admission, bind, engine, infer        # noqa: E402

T, D = 96, 32
N_RELATIONS = 4
N_TRAIN_SOURCES = 60
N_TEST_SOURCES = 16
N_TARGETS = 24
NOISE = 0.10
FAR = 0.05
BASE_SEED = 20260915

# the baseline's own hyperparameters, selected on a validation split (see `_select`)
#: **Epochs are part of the grid, not a fixed setting.** Held at 400 the baseline fits its own
#: training triples at 0.576 and scores 0.238 held-out — at chance — while the same model at 2000
#: epochs fits at 1.000 and reaches 0.921. A comparison run at the first would have reported a
#: number about the training budget and called it a result about the method.
RANKS = (4, 8, 16)
RATES = (0.2, 0.5)
EPOCH_GRID = (2000, 8000)
NEGATIVES = 4


def _rescal_fit(triples, n_entities, n_relations, *, rank, rate, epochs, negatives, seed):
    """RESCAL by plain gradient descent: entity vectors and one matrix per relation.

    `score(a, k, b) = e_a · R_k · e_b`, trained to separate observed triples from corrupted ones.
    Written out rather than imported so the comparison has no framework's defaults hiding in it.
    """
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_entities, rank)) / np.sqrt(rank)
    R = rng.normal(size=(n_relations, rank, rank)) / rank
    triples = np.asarray(triples)

    for _ in range(epochs):
        a, k, b = triples[:, 0], triples[:, 1], triples[:, 2]
        # corrupt the object, which is the standard negative for this task
        neg_b = rng.integers(0, n_entities, size=(len(triples), negatives))

        ea, eb = E[a], E[b]
        Rk = R[k]
        pos = np.einsum("nr,nrs,ns->n", ea, Rk, eb)
        negs = np.einsum("nr,nrs,nms->nm", ea, Rk, E[neg_b])

        # logistic loss on the margin, summed over negatives
        gp = -1.0 / (1.0 + np.exp(pos))                      # d/d pos of -log sigmoid(pos)
        gn = 1.0 / (1.0 + np.exp(-negs))                     # d/d neg of -log sigmoid(-neg)

        dE = np.zeros_like(E)
        dR = np.zeros_like(R)
        np.add.at(dE, a, gp[:, None] * np.einsum("nrs,ns->nr", Rk, eb))
        np.add.at(dE, b, gp[:, None] * np.einsum("nr,nrs->ns", ea, Rk))
        np.add.at(dR, k, gp[:, None, None] * np.einsum("nr,ns->nrs", ea, eb))
        for m in range(negatives):
            bm = neg_b[:, m]
            gm = gn[:, m]
            np.add.at(dE, a, gm[:, None] * np.einsum("nrs,ns->nr", Rk, E[bm]))
            np.add.at(dE, bm, gm[:, None] * np.einsum("nr,nrs->ns", ea, Rk))
            np.add.at(dR, k, gm[:, None, None] * np.einsum("nr,ns->nrs", ea, E[bm]))

        scale = rate / len(triples)
        E -= scale * dE
        R -= scale * dR
    return E, R


def _rescal_pick(E, R, a, k, candidates, *, cold=None):
    """Which candidate the trained model chooses. `cold` supplies a vector for an unseen entity."""
    ea = E[a] if a is not None else cold
    scores = [float(ea @ R[k] @ E[c]) for c in candidates]
    return int(np.argmax(scores))


def _select(train, valid, n_entities, seed):
    """Choose the baseline's hyperparameters on a validation split, as the literature does."""
    best = None
    for rank in RANKS:
        for rate in RATES:
            for epochs in EPOCH_GRID:
                E, R = _rescal_fit(train, n_entities, N_RELATIONS, rank=rank, rate=rate,
                                   epochs=epochs, negatives=NEGATIVES, seed=seed)
                hit = sum(int(_rescal_pick(E, R, a, k, cands) == gold)
                          for a, k, cands, gold in valid)
                acc = hit / max(len(valid), 1)
                if best is None or acc > best[0]:
                    best = (acc, rank, rate, epochs, E, R)
    return best


def run():
    started = time.time()
    rng = np.random.default_rng(BASE_SEED)

    # One corpus, split by SOURCE: the baseline trains on the first block and never sees the second.
    total_sources = N_TRAIN_SOURCES + N_TEST_SOURCES
    # Targets are drawn from a shared pool, so each appears in many triples. With private targets
    # a held-out triple names an entity the baseline has never seen, its vector is random, and the
    # task is unlearnable by construction — a trained model then scores at chance for a reason that
    # has nothing to do with the method, which would be a statement about the corpus and not a
    # comparison.
    c = triple_corpus(seed=BASE_SEED, n_relations=N_RELATIONS, n_sources=total_sources,
                      T=T, D=D, noise=NOISE, n_targets=N_TARGETS)
    train_src = c.sources[:N_TRAIN_SOURCES]
    test_src = c.sources[N_TRAIN_SOURCES:]

    # Admission first — the program's own precondition, run before anything is scored.
    probe = [(a, c.targets[(a, k)]) for a in train_src[:20] for k in range(N_RELATIONS)]
    verdict = admission.axis_carries(c.records, probe, far=FAR, seed=BASE_SEED, draws=2)
    print(f"\nadmission: margin_z {verdict.margin_z:.2f} against cutoff {verdict.cutoff:.2f} "
          f"-> {'admitted' if verdict.admitted else 'REFUSED'}")
    if not verdict.admitted:
        print("the corpus does not admit these reads; the comparison would measure that instead")
        return None

    entities = sorted(c.records)
    idx = {e: i for i, e in enumerate(entities)}

    # Training triples: every (source, relation, target) over the training sources, except a
    # held-out slice used to test the seen-entity regime.
    held = {(a, k) for a in train_src for k in range(N_RELATIONS)
            if rng.random() < 0.2}
    train = [(idx[a], k, idx[c.targets[(a, k)]])
             for a in train_src for k in range(N_RELATIONS) if (a, k) not in held]

    def questions(sources, only_held):
        out = []
        for a in sources:
            cands = [c.targets[(a, k)] for k in range(N_RELATIONS)]
            for k in range(N_RELATIONS):
                if only_held and (a, k) not in held:
                    continue
                out.append((a, k, cands, k))
        return out

    seen_q = questions(train_src, only_held=True)
    unseen_q = questions(test_src, only_held=False)
    valid = seen_q[: max(1, len(seen_q) // 3)]
    seen_q = seen_q[max(1, len(seen_q) // 3):]

    print(f"\n{len(train)} training triples over {len(train_src)} sources; "
          f"{len(seen_q)} held-out questions on seen entities, "
          f"{len(unseen_q)} on {len(test_src)} entities never trained on")

    acc, rank, rate, epochs, E, R = _select(
        train, [(idx[a], k, [idx[x] for x in cands], gold) for a, k, cands, gold in valid],
        len(entities), BASE_SEED)
    print(f"baseline selected on validation: rank {rank}, rate {rate}, epochs {epochs}, "
          f"validation top-1 {acc:.3f}")

    # is the baseline fitting at all? the comparison is void if not
    fit = sum(int(_rescal_pick(E, R, idx[a], k, [idx[x] for x in cands]) == gold)
              for a, k, cands, gold in questions(train_src, only_held=False)
              if (a, k) not in held)
    n_fit = len([1 for a in train_src for k in range(N_RELATIONS) if (a, k) not in held])
    print(f"baseline fit on its own training triples: {fit / n_fit:.3f}  "
          f"(the comparison is void if this is near chance {1 / N_RELATIONS:.3f})")

    cold = E[[idx[a] for a in train_src]].mean(axis=0)

    rows = []
    for label, qs, use_cold in (("seen entities", seen_q, False),
                                ("UNSEEN entities", unseen_q, True)):
        trained = measured = 0
        for a, k, cands, gold in qs:
            ci = [idx[x] for x in cands]
            if use_cold:
                trained += int(_rescal_pick(E, R, None, k, ci, cold=cold) == gold)
            else:
                trained += int(_rescal_pick(E, R, idx[a], k, ci) == gold)
            # the measured path: operator from an exemplar pair, no labels, no training
            other = train_src[0] if a != train_src[0] else train_src[1]
            ex = (other, c.targets[(other, k)])
            op = infer.operator_of(c.records, *ex)
            zs = [engine.associate(op, bind.association_matrix(c.records[a], c.records[x]),
                                   far=FAR).z for x in cands]
            measured += int(int(np.argmax(zs)) == gold)
        rows.append({"regime": label, "questions": len(qs),
                     "trained": trained / len(qs), "measured": measured / len(qs),
                     "chance": 1.0 / N_RELATIONS,
                     # the gap is what the regime is reported for, so it is stored rather
                     # than subtracted in prose where nothing can check the subtraction
                     "gap": (measured - trained) / len(qs)})
        print(f"\n{label}: {len(qs)} questions   chance {1 / N_RELATIONS:.3f}")
        print(f"   trained RESCAL (labels + fitting + tuning)   {trained / len(qs):.3f}")
        print(f"   measured path (no labels, no training)       {measured / len(qs):.3f}")

    out = {
        "design": {"T": T, "D": D, "n_relations": N_RELATIONS,
                   "train_sources": N_TRAIN_SOURCES, "test_sources": N_TEST_SOURCES,
                   "noise": NOISE, "far": FAR, "base_seed": BASE_SEED, "n_targets": N_TARGETS,
                   "ranks": RANKS, "rates": RATES, "epoch_grid": EPOCH_GRID,
                   "negatives": NEGATIVES},
        "admission": {"margin_z": verdict.margin_z, "cutoff": verdict.cutoff,
                      "admitted": verdict.admitted},
        "baseline": {"rank": rank, "rate": rate, "epochs": epochs, "validation_top1": acc,
                     "fit_on_training_triples": fit / n_fit},
        "results": rows,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp10.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp10.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
