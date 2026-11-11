"""Stage 1 — is the association the attention weight, or is a scalar enough?

The question
------------
A transformer scores a query against a key with ``q·k``, a scalar. Between two records that meet on
one basis there is a whole cross-covariance ``M = Ã ᵀ B̃``, and the scalar is one contraction of it.
Does the contraction throw away something that decides the answer?

The four scalar baselines are the same alignment at four levels of normalization, which makes this
an ablation rather than a beauty contest:

===============  ==============================  ===========================================
``dot``          ``S = <Ã, B̃>_F``                 the raw alignment, nothing removed
``cos``          ``S / (||Ã|| ||B̃||)``            normalized by each side's total magnitude
``z``            ``S / sqrt(tr(C_a C_b)/(T-1))``  normalized by the exact permutation null
``pooled``       ``mean_t(A) · mean_t(B)``        the ordered axis collapsed — standard attention
===============  ==============================  ===========================================

``cos`` and ``z`` differ in exactly one thing: ``cos`` divides by the product of the Frobenius
norms, ``z`` by the cross-Gram ``tr(C_a C_b)``. Those agree when each record's covariance is
isotropic and diverge when it is not — precisely when the records have internal structure. So ``z``
is a scalar whose *normalizer* is a matrix functional, and that distinction is what Part B measures.

The three parts
---------------
**A — calibration.** Does the count recover a known channel count, where does it stop, and do the
two candidate shared-mode reads survive a negative control? Nothing is compared until the
measurement itself is checked against a known answer.

**B — selection.** A broad true key (shares every facet, moderately) against narrow
distractors (fewer facets, louder). ``narrow_gain`` is swept across the arithmetic crossover: a
distractor carrying ``m`` of ``R`` facets shares ``g·sqrt(m/R)`` of the query's energy against the
true key's ``1``, so above ``g = sqrt(R/m)`` it carries more shared energy and any method ranking
by shared magnitude must fail. Known before the run.

**C — the envelope.** Part B's first run returned 1.000 for ``z`` in every cell, which says the task
never reached the method's limit. A result that is perfect everywhere is reporting an easy problem,
so this part raises the noise until the method breaks and reports where.

How ``far`` is handled
----------------------
``z`` does not depend on the false-alarm level; it enters only the decision layer, through the
cutoff that ``resolved`` is taken against and the floor the modes are counted above. The cutoff is
read from the instrument rather than recomputed here.

Run
---
    python research/experiments/exp01_association_vs_scalar.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus                                    # noqa: E402
from entroptics_infer import bind, engine                          # noqa: E402
from measure import centred as _centred, centred_cos as cos, centred_dot as dot  # noqa: E402,F401

# ── the experiment's declared design, published with the result ─────────────────────────────────
T, D = 96, 32
N_KEYS = 12
BASE_SEED = 20260915
REPORT_FAR = 0.05

CAL_FACETS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
CAL_NOISE = (0.05, 0.10, 0.15, 0.20, 0.30)
CAL_DRAWS = 20
CONTROL_DRAWS = 40

SEL_FACETS = (1, 2, 3, 4, 5, 6)
SEL_GAINS = (1.0, 1.5, 2.0, 2.5, 3.0)
SEL_NOISE = 0.10
SEL_TRIALS = 80

ENV_FACETS = (2, 4, 6)
ENV_NOISE = (0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
ENV_GAIN = 2.0
ENV_TRIALS = 80


def pooled(A, B):
    return float(np.dot(A.mean(axis=0), B.mean(axis=0)))


def pooled_cos(A, B):
    a, b = A.mean(axis=0), B.mean(axis=0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


SCALARS = {"dot": dot, "cos": cos, "pooled": pooled, "pooled_cos": pooled_cos}


def matrix_rank_modes(A, B, *, far):
    """The read this module started with and rejected: modes counted on ``M`` itself.

    Kept so the negative control can be reported rather than described.
    """
    return int(engine.rank(bind.association_matrix(A, B), far=far))


def part_a():
    print("\nPART A — calibration: does the count recover the truth?\n")
    print(f"{'R':>3}" + "".join(f"{('n=' + str(n)):>9}" for n in CAL_NOISE))
    rows = []
    for R in CAL_FACETS:
        line, cells = f"{R:>3}", {}
        for noise in CAL_NOISE:
            ks = [
                engine.rank(
                    facet_corpus(seed=BASE_SEED + d, facets=R, n_keys=2, T=T, D=D,
                                 narrow_gain=1.0, noise=noise).query.frame,
                    far=REPORT_FAR,
                )
                for d in range(CAL_DRAWS)
            ]
            cells[str(noise)] = {
                "mean_K": float(np.mean(ks)),
                "exact_rate": float(np.mean([k == R for k in ks])),
            }
            line += f"{np.mean(ks):>9.2f}"
        rows.append({"facets": R, "by_noise": cells})
        print(line)
    print(f"\n(cell = mean resolved rank over {CAL_DRAWS} draws; exact recovery is mean == R)")

    # The negative control that decided which shared-mode read to keep.
    print("\nNEGATIVE CONTROL — two independent frames, sharing nothing. Truth is 0.\n")
    rng = np.random.default_rng(BASE_SEED)
    mr, ie = [], []
    for _ in range(CONTROL_DRAWS):
        A, B = rng.normal(size=(T, D)), rng.normal(size=(T, D))
        mr.append(matrix_rank_modes(A, B, far=REPORT_FAR))
        ie.append(bind.shared_modes(A, B, far=REPORT_FAR))
    control = {
        "draws": CONTROL_DRAWS,
        "matrix_rank_mean": float(np.mean(mr)),
        "inclusion_exclusion_mean": float(np.mean(ie)),
    }
    print(f"  rank of M              {control['matrix_rank_mean']:.2f}   <- outside the "
          f"instrument's family: M is a product of two fields, not a field")
    print(f"  inclusion-exclusion    {control['inclusion_exclusion_mean']:.2f}   <- every term is "
          f"a (T, F) field of observations")

    # And whether each recovers a known count.
    print("\nRECOVERY OF A KNOWN SHARED COUNT (true key, gain 2.0, noise 0.10)\n")
    print(f"{'R':>3}{'rank(M) t':>11}{'rank(M) d':>11}{'incl-exc t':>12}{'incl-exc d':>12}")
    recovery = []
    for R in SEL_FACETS:
        mt, md, it, idd = [], [], [], []
        for t in range(25):
            c = facet_corpus(seed=BASE_SEED + 500 + t, facets=R, n_keys=N_KEYS, T=T, D=D,
                             narrow_gain=2.0, noise=SEL_NOISE)
            Q = c.query.frame
            for j, k in enumerate(c.keys):
                (mt if j == c.truth else md).append(matrix_rank_modes(Q, k.frame, far=REPORT_FAR))
                (it if j == c.truth else idd).append(bind.shared_modes(Q, k.frame, far=REPORT_FAR))
        recovery.append({"facets": R, "matrix_rank_true": float(np.mean(mt)),
                         "matrix_rank_distractor": float(np.mean(md)),
                         "incl_excl_true": float(np.mean(it)),
                         "incl_excl_distractor": float(np.mean(idd))})
        print(f"{R:>3}{np.mean(mt):>11.2f}{np.mean(md):>11.2f}{np.mean(it):>12.2f}"
              f"{np.mean(idd):>12.2f}")
    return {"calibration": rows, "negative_control": control, "recovery": recovery}


def _trial_scores(c, far):
    Q = c.query.frame
    scal = {name: [fn(Q, k.frame) for k in c.keys] for name, fn in SCALARS.items()}
    assoc = [engine.associate(Q, k.frame, far=far) for k in c.keys]
    return Q, scal, assoc


def part_b():
    print("\nPART B — selection across the crossover\n")
    rows = []
    for R in SEL_FACETS:
        for gain in SEL_GAINS:
            hits = {name: 0 for name in list(SCALARS) + ["z", "gated"]}
            refused = 0
            tight_t, tight_d, modes_t, modes_d = [], [], [], []
            for t in range(SEL_TRIALS):
                c = facet_corpus(seed=BASE_SEED + t, facets=R, n_keys=N_KEYS, T=T, D=D,
                                 narrow_gain=gain, noise=SEL_NOISE)
                Q, scal, assoc = _trial_scores(c, REPORT_FAR)
                for name, s in scal.items():
                    hits[name] += int(int(np.argmax(s)) == c.truth)
                zs = [a.z for a in assoc]
                hits["z"] += int(int(np.argmax(zs)) == c.truth)

                live = [i for i, a in enumerate(assoc) if a.resolved]
                if not live:
                    refused += 1
                else:
                    ranked = [(bind.shared_modes(Q, c.keys[i].frame, far=REPORT_FAR), zs[i], i)
                              for i in live]
                    hits["gated"] += int(max(ranked)[2] == c.truth)
                    for m, _, i in ranked:
                        (modes_t if i == c.truth else modes_d).append(m)
                for i, a in enumerate(assoc):
                    (tight_t if i == c.truth else tight_d).append(a.tightness)

            row = {
                "facets": R, "narrow_gain": gain, "crossover_m1": float(np.sqrt(R)),
                "top1": {k: v / SEL_TRIALS for k, v in hits.items()},
                "refusal_rate": refused / SEL_TRIALS,
                "tightness_true": float(np.mean(tight_t)),
                "tightness_distractor": float(np.mean(tight_d)),
                "modes_true": float(np.mean(modes_t)) if modes_t else None,
                "modes_distractor": float(np.mean(modes_d)) if modes_d else None,
            }
            rows.append(row)
            print(f"R={R} g={gain:<4} (xover {np.sqrt(R):.2f})  z={row['top1']['z']:.3f} "
                  f"gated={row['top1']['gated']:.3f} cos={row['top1']['cos']:.3f} "
                  f"dot={row['top1']['dot']:.3f} pooled={row['top1']['pooled']:.3f}  "
                  f"tight t/d={row['tightness_true']:.3f}/{row['tightness_distractor']:.3f}",
                  flush=True)
    return rows


def part_c():
    print("\nPART C — the envelope: where does the association break?\n")
    rows = []
    for R in ENV_FACETS:
        for noise in ENV_NOISE:
            hits = {name: 0 for name in list(SCALARS) + ["z"]}
            refused = 0
            for t in range(ENV_TRIALS):
                c = facet_corpus(seed=BASE_SEED + t, facets=R, n_keys=N_KEYS, T=T, D=D,
                                 narrow_gain=ENV_GAIN, noise=noise)
                _, scal, assoc = _trial_scores(c, REPORT_FAR)
                for name, s in scal.items():
                    hits[name] += int(int(np.argmax(s)) == c.truth)
                hits["z"] += int(int(np.argmax([a.z for a in assoc])) == c.truth)
                if not any(a.resolved for a in assoc):
                    refused += 1
            row = {"facets": R, "noise": noise,
                   "top1": {k: v / ENV_TRIALS for k, v in hits.items()},
                   "refusal_rate": refused / ENV_TRIALS}
            rows.append(row)
            print(f"R={R} noise={noise:<5} z={row['top1']['z']:.3f} cos={row['top1']['cos']:.3f} "
                  f"dot={row['top1']['dot']:.3f} pooled={row['top1']['pooled']:.3f}  "
                  f"refused={row['refusal_rate']:.3f}", flush=True)
    return rows


def run():
    started = time.time()
    a = part_a()
    b = part_b()
    c = part_c()
    out = {
        "design": {
            "T": T, "D": D, "n_keys": N_KEYS, "base_seed": BASE_SEED, "report_far": REPORT_FAR,
            "calibration": {"facets": CAL_FACETS, "noise": CAL_NOISE, "draws": CAL_DRAWS,
                            "control_draws": CONTROL_DRAWS},
            "selection": {"facets": SEL_FACETS, "gains": SEL_GAINS, "noise": SEL_NOISE,
                          "trials": SEL_TRIALS},
            "envelope": {"facets": ENV_FACETS, "noise": ENV_NOISE, "gain": ENV_GAIN,
                         "trials": ENV_TRIALS},
        },
        "part_a": a, "part_b": b, "part_c": c,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp01.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp01.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
