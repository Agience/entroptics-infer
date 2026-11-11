"""Stage 4 — the forward pass: two of a triple in, the third out, with nothing trained or named.

The question
------------
`(context, content, operator)` is a three-terminal device. Inference is that device run forward:
supply two terminals and read the third. The hard terminal is the operator, because naming it would
need a vocabulary and there is none — the kinds a corpus contains are directions it turns out to
occupy, not members of a list.

So the operator is supplied by example. Given a pair `(c, d)` standing in some relation and a
source `a`, which candidate `b` stands in that same relation to `a`? That is an analogy, and it is
the only form the question can take where relations have no names.

Why the corpus is built this way
---------------------------------
Every source is related to every one of its candidates — each by a different kind. So a method
that ignores the operator cannot beat chance among them: they are all genuinely related to the
source. Any accuracy above chance is the operator doing work, and the experiment measures exactly
that gap rather than an accuracy in isolation.

What is measured
----------------
**A** — accuracy against the two controls that matter: chance (`1/R`), and the same method with the
operator removed, which ranks candidates by raw association strength.

**B** — the wrong operator. An exemplar of a *different* kind should return that different kind's
target, essentially never the asked-for one. This is what shows the operator determines the answer
rather than merely helping.

**C** — the envelope. Where the method breaks as the records get noisier.

**D** — the decision. The two-read gate from Stage 1b applied here: precision and coverage.

**E** — through the store. The candidate set recovered from a real `agience-mantle` lattice that
carries no relation type, rather than from a list held in the experiment.

What would falsify this stage
-----------------------------
Stated before the run. The stage fails if accuracy does not exceed chance; if removing the operator
does *not* drop the method to chance (which would mean the operator was never load-bearing); if a
wrong-kind exemplar returns the asked-for target at better than chance; or if the gated decision is
ever wrong where it answers.

Run
---
    python research/experiments/exp05_forward_pass.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import triple_corpus                                   # noqa: E402
from entroptics_infer import bind, engine, infer, store            # noqa: E402

T, D = 96, 32
N_SOURCES = 6
NOISE = 0.10
FAR = 0.05
BASE_SEED = 20260915
DRAWS = 12

RELATIONS = (2, 3, 4, 6, 8)
NOISE_SWEEP = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50)


def _corpus(seed, R, noise=NOISE):
    return triple_corpus(seed=seed, n_relations=R, n_sources=N_SOURCES, T=T, D=D, noise=noise)


def _trials(c):
    """Every (source, kind) question, with an exemplar drawn from a *different* source."""
    R = c.n_relations
    for i, a in enumerate(c.sources):
        other = c.sources[(i + 1) % len(c.sources)]
        cands = [c.targets[(a, k)] for k in range(R)]
        for k in range(R):
            yield a, k, cands, (other, c.targets[(other, k)])


def part_a():
    print("\nPART A — accuracy, against chance and against the operator removed\n")
    print(f"{'R':>3}{'chance':>9}{'analogy':>10}{'no operator':>13}{'gap':>9}")
    rows = []
    for R in RELATIONS:
        hit = ignored = n = 0
        for d in range(DRAWS):
            c = _corpus(BASE_SEED + d, R)
            for a, k, cands, ex in _trials(c):
                op = infer.operator_of(c.records, *ex)
                z = [engine.associate(op, bind.association_matrix(c.records[a], c.records[b]),
                                      far=FAR).z for b in cands]
                hit += int(int(np.argmax(z)) == k)
                strength = [engine.associate(c.records[a], c.records[b], far=FAR).z for b in cands]
                ignored += int(int(np.argmax(strength)) == k)
                n += 1
        row = {"R": R, "chance": 1.0 / R, "analogy": hit / n, "no_operator": ignored / n,
               "trials": n}
        row["gap"] = row["analogy"] - row["chance"]
        rows.append(row)
        print(f"{R:>3}{row['chance']:>9.3f}{row['analogy']:>10.3f}{row['no_operator']:>13.3f}"
              f"{row['gap']:>+9.3f}", flush=True)
    print("\n(every candidate is genuinely related to the source, so chance is the ceiling without "
          "the operator)")
    return rows


def part_b():
    print("\nPART B — the wrong operator returns the wrong target\n")
    print(f"{'R':>3}{'right exemplar':>16}{'wrong exemplar':>16}{'chance':>9}")
    rows = []
    for R in RELATIONS:
        right = wrong = n = 0
        for d in range(DRAWS):
            c = _corpus(BASE_SEED + d, R)
            for a, k, cands, ex in _trials(c):
                other = ex[0]
                good = infer.operator_of(c.records, other, c.targets[(other, k)])
                bad_k = (k + 1) % R
                bad = infer.operator_of(c.records, other, c.targets[(other, bad_k)])
                for op, tally in ((good, "right"), (bad, "wrong")):
                    z = [engine.associate(op, bind.association_matrix(c.records[a], c.records[b]),
                                          far=FAR).z for b in cands]
                    if tally == "right":
                        right += int(int(np.argmax(z)) == k)
                    else:
                        wrong += int(int(np.argmax(z)) == k)
                n += 1
        rows.append({"R": R, "right": right / n, "wrong": wrong / n, "chance": 1.0 / R})
        print(f"{R:>3}{right / n:>16.3f}{wrong / n:>16.3f}{1.0 / R:>9.3f}", flush=True)
    print("\n(a wrong-kind exemplar returning the asked-for target at chance or above would mean "
          "the operator is not deciding)")
    return rows


def part_c():
    print("\nPART C — the envelope\n")
    print(f"{'noise':>7}{'analogy':>10}{'wrong exemplar':>16}  (R = 4, chance 0.250)")
    rows = []
    for noise in NOISE_SWEEP:
        hit = bad = n = 0
        for d in range(DRAWS):
            c = _corpus(BASE_SEED + d, 4, noise)
            for a, k, cands, ex in _trials(c):
                other = ex[0]
                for kind, tally in (((k), "right"), ((k + 1) % 4, "wrong")):
                    op = infer.operator_of(c.records, other, c.targets[(other, kind)])
                    z = [engine.associate(op, bind.association_matrix(c.records[a], c.records[b]),
                                          far=FAR).z for b in cands]
                    if tally == "right":
                        hit += int(int(np.argmax(z)) == k)
                    else:
                        bad += int(int(np.argmax(z)) == k)
                n += 1
        rows.append({"noise": noise, "analogy": hit / n, "wrong_exemplar": bad / n})
        print(f"{noise:>7.2f}{hit / n:>10.3f}{bad / n:>16.3f}", flush=True)
    return rows


def part_d():
    print("\nPART D — the gated decision: precision and coverage\n")
    print(f"{'R':>3}{'raw':>8}{'coverage':>10}{'precision':>11}  refusal reasons")
    rows = []
    for R in RELATIONS:
        raw = answered = correct = n = 0
        reasons: dict = {}
        for d in range(DRAWS):
            c = _corpus(BASE_SEED + d, R)
            for a, k, cands, ex in _trials(c):
                res = infer.complete(c.records, exemplar=ex, source=a, candidates=cands, far=FAR)
                raw += int(int(np.argmax([s.z for s in res.scores])) == k)
                reasons[res.why] = reasons.get(res.why, 0) + 1
                if res.key is not None:
                    answered += 1
                    correct += int(res.key == k)
                n += 1
        row = {"R": R, "raw": raw / n, "coverage": answered / n,
               "precision": (correct / answered) if answered else None, "reasons": reasons}
        rows.append(row)
        prec = "     n/a" if row["precision"] is None else f"{row['precision']:>8.3f}"
        top = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:2])
        print(f"{R:>3}{row['raw']:>8.3f}{row['coverage']:>10.3f}{prec:>11}  {top}", flush=True)
    return rows


def part_e():
    print("\nPART E — candidates recovered from a lattice carrying no relation type\n")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for R in (3, 4, 6):
            c = _corpus(BASE_SEED, R)
            with store.open_lattice(pathlib.Path(tmp) / f"t{R}.db", origin="exp05") as lat:
                lat.relate_many((a, c.targets[(a, k)], {})
                                for a in c.sources for k in range(R))
                hit = n = 0
                for i, a in enumerate(c.sources):
                    cands = lat.neighbours(a)          # from the store, unordered and untyped
                    other = c.sources[(i + 1) % len(c.sources)]
                    for k in range(R):
                        ex = (other, c.targets[(other, k)])
                        res = infer.complete(c.records, exemplar=ex, source=a,
                                             candidates=cands, far=FAR)
                        truth = c.targets[(a, k)]
                        hit += int(res.key is not None and cands[res.key] == truth)
                        n += 1
            rows.append({"R": R, "accuracy_via_store": hit / n, "trials": n})
            print(f"R={R:<3} candidates from the store, gated accuracy {hit / n:.3f} "
                  f"over {n} questions", flush=True)
    print("\n(the store supplies who is adjacent; it says nothing about how, because it holds no "
          "type)")
    return rows


def run():
    started = time.time()
    a, b, c, d, e = part_a(), part_b(), part_c(), part_d(), part_e()
    out = {
        "design": {"T": T, "D": D, "n_sources": N_SOURCES, "noise": NOISE, "far": FAR,
                   "draws": DRAWS, "base_seed": BASE_SEED, "relations": RELATIONS,
                   "noise_sweep": NOISE_SWEEP},
        "accuracy": a, "wrong_operator": b, "envelope": c, "decision": d, "through_store": e,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp05.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp05.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
