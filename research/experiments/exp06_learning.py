"""Stage 5 — learning in real time, with nothing to retrain and nothing to rebuild.

The question
------------
A trained system learns by changing weights, which is why it needs a training step, why new data
perturbs answers it was never asked about, and why the order data arrives in matters. This program
has no weights. Every read is a pure function of the records as they stand, so *learning* here is
simply the corpus growing — and the question is whether that produces the behaviour learning is
supposed to produce.

Four properties are worth measuring, and one of them is the reason to prefer this arrangement at
all.

**A — evidence changes the answer.** With too few observations the system should refuse; as
observations arrive it should answer, and answer correctly. If refusal did not lift with evidence,
the refusal would be a property of the method rather than of the corpus.

**B — a new kind of relation is learned; more of an old one is not mistaken for one.** The count of
kinds should rise when genuinely new relational structure arrives and stay put when more of the same
arrives. A counter that rises either way is counting volume, not structure.

**C — order does not matter.** The same records in a different order must give the same answer,
exactly. A trained system cannot promise this; a pure function gets it for free, and the test is
here to show the promise is kept rather than assumed.

**D — adding a record does not disturb unrelated answers.** This is the property a trained system
cannot have. Fine-tuning on new data moves every weight, so answers to questions nobody asked about
change — catastrophic forgetting is the extreme of it. Here an answer depends only on the records it
was computed from, so an unrelated arrival must leave it bit-identical.

**E — is it fast enough to be called real time?** A read whose cost grows badly with corpus size
would make the whole arrangement academic.

What is deliberately not used
------------------------------
`entroptics.ResolvedScreen` offers a stateful, resumable screen with incremental Gram updates. It is
not used here, for two reasons: it carries amortization constants (`refresh_every`, `warmup`,
`forgetting`), and its read *converges* to the batch read for a stationary stream rather than
equalling it. Neither is acceptable where the claim is exactness. The cost of that choice is
measured in E.

What would falsify this stage
-----------------------------
Stated before the run. The stage fails if refusal does not lift as evidence accumulates; if the kind
count rises as fast for repeated structure as for new structure; if any reordering changes an
answer; or if an unrelated arrival changes an existing answer by anything at all.

Run
---
    python research/experiments/exp06_learning.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus, relation_corpus, triple_corpus    # noqa: E402
from entroptics_infer import bind, engine, infer, relations        # noqa: E402

T, D = 96, 32
FAR = 0.05
BASE_SEED = 20260915
DRAWS = 12
N_KEYS = 12

ROWS_SEEN = (4, 8, 12, 16, 24, 32, 48, 64, 96)
GROWTH_STEPS = (0, 1, 2, 3, 4)
CORPUS_SIZES = (10, 20, 40, 80, 160)


def part_a():
    """Refusal lifts as observations arrive — the corpus learning, not the method changing."""
    print("\nPART A — evidence accumulates: refusal lifts, accuracy arrives\n")
    print(f"{'rows seen':>10}{'refused':>10}{'answered':>10}{'precision':>11}")
    rows = []
    for seen in ROWS_SEEN:
        refused = answered = correct = 0
        for d in range(DRAWS):
            c = facet_corpus(seed=BASE_SEED + d, facets=3, n_keys=N_KEYS, T=T, D=D,
                             narrow_gain=2.0, noise=0.10, subset_size=1)
            q = c.query.frame[:seen]
            cands = [k.frame[:seen] for k in c.keys]
            res = bind.decide(q, cands, far=FAR)
            if res.key is None:
                refused += 1
            else:
                answered += 1
                correct += int(res.key == c.truth)
        row = {"rows_seen": seen, "refused": refused / DRAWS, "answered": answered / DRAWS,
               "precision": (correct / answered) if answered else None}
        rows.append(row)
        prec = "      n/a" if row["precision"] is None else f"{row['precision']:>9.3f}"
        print(f"{seen:>10}{row['refused']:>10.3f}{row['answered']:>10.3f}{prec:>11}", flush=True)
    print("\n(the same records throughout; only how many observations of them have arrived changes)")
    return rows


def part_b():
    """A new kind is learned; more of an old kind is not mistaken for one."""
    print("\nPART B — new structure against more of the same\n")
    print(f"{'edges added':>12}{'new kinds':>12}{'repeat kinds':>14}")
    rows = []
    base_R = 4
    for step in GROWTH_STEPS:
        new_counts, repeat_counts = [], []
        for d in range(DRAWS):
            # A corpus of base_R kinds, plus `step` further kinds — genuinely new structure.
            grown = relation_corpus(seed=BASE_SEED + d, n_relations=base_R + step,
                                    edges_per_relation=10, n_vertices=60, T=T, D=D,
                                    noise=0.05, share_vertices=False)
            new_counts.append(relations.relation_modes(grown.records, grown.edges, far=FAR))
            # The same base_R kinds, with `step` further batches of the same kinds — more volume,
            # no new structure. Edge count is matched to the arm above.
            repeated = relation_corpus(seed=BASE_SEED + d, n_relations=base_R,
                                       edges_per_relation=10 * (base_R + step) // base_R,
                                       n_vertices=60, T=T, D=D, noise=0.05, share_vertices=False)
            repeat_counts.append(relations.relation_modes(repeated.records, repeated.edges, far=FAR))
        row = {"step": step, "kinds_present": base_R + step,
               "new_structure": float(np.mean(new_counts)),
               "repeated_structure": float(np.mean(repeat_counts))}
        rows.append(row)
        print(f"{step:>12}{row['new_structure']:>12.2f}{row['repeated_structure']:>14.2f}",
              flush=True)
    print(f"\n(both arms grow by the same number of edges; only one carries new kinds. Base is "
          f"{base_R})")
    return rows


def part_c():
    """Order independence, asserted exactly rather than within a tolerance."""
    print("\nPART C — the same records in a different order\n")
    mismatches = 0
    checks = 0
    for d in range(DRAWS):
        c = triple_corpus(seed=BASE_SEED + d, n_relations=4, n_sources=5, T=T, D=D, noise=0.10)
        a = c.sources[0]
        other = c.sources[1]
        cands = [c.targets[(a, k)] for k in range(4)]
        for k in range(4):
            ex = (other, c.targets[(other, k)])
            straight = infer.complete(c.records, exemplar=ex, source=a,
                                      candidates=cands, far=FAR)
            order = np.random.default_rng(d).permutation(len(cands))
            shuffled = [cands[i] for i in order]
            reordered = infer.complete(c.records, exemplar=ex, source=a,
                                       candidates=shuffled, far=FAR)
            checks += 1
            same = (
                (straight.key is None and reordered.key is None)
                or (straight.key is not None and reordered.key is not None
                    and cands[straight.key] == shuffled[reordered.key])
            )
            mismatches += int(not same)
    print(f"  {checks} questions asked with candidates in two orders: "
          f"{mismatches} disagreements")
    return {"checks": checks, "mismatches": mismatches}


def part_d():
    """An unrelated arrival must leave an existing answer bit-identical."""
    print("\nPART D — adding an unrelated record to the corpus\n")
    rng = np.random.default_rng(BASE_SEED)
    drift = 0
    checks = 0
    for d in range(DRAWS):
        c = facet_corpus(seed=BASE_SEED + d, facets=3, n_keys=N_KEYS, T=T, D=D,
                         narrow_gain=2.0, noise=0.10, subset_size=1)
        before = bind.decide(c.query.frame, [k.frame for k in c.keys], far=FAR)
        # Three unrelated records arrive. They are real records, not zeros — the point is that
        # nothing about them touches the question already answered.
        newcomers = [rng.normal(size=(T, D)) for _ in range(3)]
        after = bind.decide(c.query.frame, [k.frame for k in c.keys] + newcomers, far=FAR)
        checks += 1
        drift += int(before.key != after.key or before.why != after.why)
        # and the scores of the original candidates must not have moved at all
        for b0, b1 in zip(before.bindings, after.bindings[: len(before.bindings)]):
            if b0.association.z != b1.association.z:
                drift += 1
                break
    print(f"  {checks} answers re-asked after unrelated records arrived: {drift} changed")
    print("\n(a trained system moves every weight on new data; here an answer depends only on the "
          "records it was computed from)")
    return {"checks": checks, "changed": drift}


def part_e():
    """Is a read fast enough to call this real time?"""
    print("\nPART E — read cost as the corpus grows\n")
    print(f"{'candidates':>12}{'per question (ms)':>20}{'per candidate (ms)':>21}")
    rows = []
    for n in CORPUS_SIZES:
        c = facet_corpus(seed=BASE_SEED, facets=3, n_keys=n, T=T, D=D,
                         narrow_gain=2.0, noise=0.10, subset_size=1)
        cands = [k.frame for k in c.keys]
        started = time.perf_counter()
        reps = 3
        for _ in range(reps):
            bind.decide(c.query.frame, cands, far=FAR)
        per_question = (time.perf_counter() - started) / reps * 1000.0
        rows.append({"candidates": n, "ms_per_question": per_question,
                     "ms_per_candidate": per_question / n})
        print(f"{n:>12}{per_question:>20.1f}{per_question / n:>21.2f}", flush=True)
    print("\n(no state is carried between questions; each is computed from the records as they "
          "stand)")
    return rows


def run():
    started = time.time()
    a, b, c, d, e = part_a(), part_b(), part_c(), part_d(), part_e()
    out = {
        "design": {"T": T, "D": D, "far": FAR, "draws": DRAWS, "base_seed": BASE_SEED,
                   "rows_seen": ROWS_SEEN, "growth_steps": GROWTH_STEPS,
                   "corpus_sizes": CORPUS_SIZES},
        "evidence": a, "new_vs_repeated": b, "order_independence": c,
        "locality": d, "cost": e,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp06.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp06.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
