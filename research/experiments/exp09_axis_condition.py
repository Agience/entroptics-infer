"""Stage 4d — testing the scope claim itself, rather than asserting it.

What is being tested
--------------------
Four real-text tasks were lost, and PAPER §6.3 explains them with a condition: the association measures
co-variation along a shared ordered axis, so the relation asked about must vary along that axis.
For a text record the axis is position in the book, and distributional similarity does not vary
along it — two words can be near-perfect substitutes and appear in different chapters.

That explanation was arrived at *after* the losses, which makes it the kind of claim this program
is supposed to distrust. So it is turned into a prediction and put at risk.

The prediction, stated before the run
--------------------------------------
Build a corpus where how much of a relation lives on the ordered axis is a dial, and nothing else
changes. Then:

1. Where the relation is carried entirely by co-variation along the axis, the association must
   beat the aggregated vector cosine.
2. Where the relation is carried entirely by matching aggregate profiles and the axis carries
   nothing, the cosine must beat the association.
3. There must be a crossover in between.

If the association wins everywhere, the condition is not what separates the synthetic wins from the
text losses and PAPER §6.3 is wrong. If it loses everywhere, the synthetic results have an explanation that
has nothing to do with the axis. Either outcome falsifies the account.

The construction
----------------
A record is an activation trajectory times a context profile: row ``t`` is ``a_t · p`` under noise.
Two things can be shared between two records and they are independent of each other — the profile
``p`` (what kind of contexts the word occurs in) and the trajectory ``a`` (when it is active).

* The query has profile ``p`` and trajectory ``a``.
* The true candidate always shares the profile ``p``. It shares the trajectory to degree
  ``lambda``: ``lambda * a + sqrt(1 - lambda^2) * independent``.
* **Trajectory distractors** share the query's trajectory ``a`` exactly but carry a *different*
  profile. These are the narrow-loud case in this setting: strongly related along the axis, unrelated
  in kind.
* **Unrelated distractors** share neither.

At ``lambda = 1`` the true candidate is the only record sharing both, and the axis carries the
relation. At ``lambda = 0`` it shares only the profile while the distractors own the axis, which is
the text case exactly.

Why the two methods must differ here
--------------------------------------
For rank-one records the arithmetic is explicit. The association's statistic is
``S = (sum_t a_t b_t)(p_q · p_c)`` — a product of trajectory alignment and profile overlap, so it
is zero if either is. The aggregated vector is ``(sum_t a_t) p``, so its cosine is
``cos(p_q, p_c)`` — profile overlap alone, blind to the axis. They are therefore different
statistics and not two estimators of one thing, which is what makes the dial meaningful.

Run
---
    python research/experiments/exp09_axis_condition.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from draw import unit_directions as _unit                          # noqa: E402
from entroptics_infer import engine                                # noqa: E402

T, D = 96, 32
N_CANDIDATES = 8
TRIALS = 200
FAR = 0.05
BASE_SEED = 20260915
NOISE = 0.10
LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)


def _record(rng, activation, profile, noise):
    return np.outer(activation, profile) + rng.normal(size=(T, D)) * noise


def cos_aggregate(A, B):
    """The classical read: collapse the ordered axis, then compare what is left."""
    a, b = A.sum(axis=0), B.sum(axis=0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def trial(rng, lam):
    profiles = _unit(rng, N_CANDIDATES + 1, D)
    p_query = profiles[0]
    a_query = rng.normal(size=T)

    query = _record(rng, a_query, p_query, NOISE)

    # the true candidate: same profile, trajectory shared to degree lam
    fresh = rng.normal(size=T)
    a_true = lam * a_query + np.sqrt(max(0.0, 1.0 - lam * lam)) * fresh
    cands = [_record(rng, a_true, p_query, NOISE)]

    # trajectory distractors: the query's own trajectory, a different profile
    half = (N_CANDIDATES - 1) // 2
    for i in range(half):
        cands.append(_record(rng, a_query, profiles[i + 1], NOISE))
    # unrelated distractors: neither shared
    for i in range(N_CANDIDATES - 1 - half):
        cands.append(_record(rng, rng.normal(size=T), profiles[half + i + 1], NOISE))

    order = rng.permutation(len(cands))
    cands = [cands[i] for i in order]
    gold = int(np.where(order == 0)[0][0])
    return query, cands, gold


def run():
    started = time.time()
    print(f"\n{'lambda':>8}{'association':>14}{'vector cosine':>16}{'edge':>9}"
          f"   (chance {1 / N_CANDIDATES:.3f})")
    rows = []
    for lam in LAMBDAS:
        rng = np.random.default_rng(BASE_SEED)
        hit_z = hit_c = 0
        for _ in range(TRIALS):
            q, cands, gold = trial(rng, lam)
            zs = [engine.associate(q, c, far=FAR).z for c in cands]
            cs = [cos_aggregate(q, c) for c in cands]
            hit_z += int(int(np.argmax(zs)) == gold)
            hit_c += int(int(np.argmax(cs)) == gold)
        row = {"lambda": lam, "association": hit_z / TRIALS, "cosine": hit_c / TRIALS}
        row["edge"] = row["association"] - row["cosine"]
        rows.append(row)
        print(f"{lam:>8.1f}{row['association']:>14.3f}{row['cosine']:>16.3f}{row['edge']:>+9.3f}",
              flush=True)

    wins_hi = [r for r in rows if r["lambda"] >= max(LAMBDAS) and r["edge"] > 0]
    wins_lo = [r for r in rows if r["lambda"] <= min(LAMBDAS) and r["edge"] < 0]
    crossing = [r for r in rows if r["edge"] > 0] and [r for r in rows if r["edge"] < 0]
    print("\nthe prediction, checked:")
    print(f"  association wins where the axis carries the relation   "
          f"{'YES' if wins_hi else 'NO'}")
    print(f"  cosine wins where the axis carries nothing             "
          f"{'YES' if wins_lo else 'NO'}")
    print(f"  a crossover exists between them                        "
          f"{'YES' if crossing else 'NO'}")

    out = {
        "design": {"T": T, "D": D, "n_candidates": N_CANDIDATES, "trials": TRIALS,
                   "far": FAR, "noise": NOISE, "base_seed": BASE_SEED, "lambdas": LAMBDAS,
                   # published alongside the results, so the gate can check it rather than
                   # having to trust that 1 / N_CANDIDATES was worked out correctly in prose
                   "chance": 1.0 / N_CANDIDATES},
        "rows": rows,
        "prediction_held": {"association_wins_on_axis": bool(wins_hi),
                            "cosine_wins_off_axis": bool(wins_lo),
                            "crossover": bool(crossing)},
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp09.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp09.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
