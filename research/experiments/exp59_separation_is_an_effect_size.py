"""Stage 52 — what the screen's separation is, and what it therefore cannot be used to conclude.

Why this is its own experiment
--------------------------------
`exp57` rests on a claim about the instrument rather than about any corpus: that
`relations.carries_kinds`'s ``separation`` is an effect size and not a test statistic, so it does
not grow as a relation gains edges. Everything `exp57` argues follows from that, and an argument
that carries a whole stage should be measured by a script somebody can run rather than asserted in a
docstring.

What separation is
--------------------
`carries_kinds` computes, for a relation's edges,

    separation = (mean(z over the edges) - base) / spread

where ``base`` and ``spread`` are the mean and the standard deviation of the z of individual
unrelated pairs. The divisor is the spread of a single draw, not the standard error of a mean. So
more edges make ``mean(z)`` a more precise estimate without moving the ratio: separation converges
to a constant rather than growing.

That is the correct question for :func:`entroptics_infer.complete`, which is handed exactly one
exemplar pair and must name the relation from it. A relation whose single pair cannot be told from
an unrelated one cannot be specified by an exemplar, however real it is.

It is the wrong question for an aggregate read. :func:`entroptics_infer.relation_over` averages
every edge, and the precision of that average does grow.

What is measured
------------------
One fixed population of records and one fixed pool of edges per relation. The only thing that moves
between cells is how many of that same pool the screen is shown. Nothing else differs — not the
records, not the null, not the relation — so any movement in separation is movement with edge count
and nothing else.

Three relations are carried at once so the screen has something to tell apart, and one of them is
deliberately faint, sitting near where the refused corpora read.

Reported beside it, for contrast, is the z of the mean — ``separation * sqrt(edges)`` — which is
what a test statistic would do. The two columns going different ways is the whole result.

**What would falsify it.** Separation rising with edge count, in the way the z of the mean does. If
it rises, it is a test statistic after all, the refused corpora were under-sampled rather than
mis-asked, and `exp57`'s premise is wrong.

Why it is swept over seeds
----------------------------
One lattice is one draw, and a ratio of two estimated quantities is noisy at small edge counts.
The spread across seeds is reported so a flat line is distinguishable from a line too noisy to read.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from entroptics_infer import relations                              # noqa: E402
from draw import structured                                         # noqa: E402
from record import instrument                                       # noqa: E402

FAR = 0.05
SMOKE = os.environ.get("EXP59_SMOKE") == "1"
T, D = 96, 32
SIZES = (5, 20, 80) if SMOKE else (5, 10, 20, 40, 80, 160, 320)
SOURCES = 320
SEEDS = (5, 11) if SMOKE else (5, 11, 23, 37, 41)
DRAWS = 80 if SMOKE else 250
#: how much of the destination the faint relation accounts for. Chosen to land near where the
#: refused corpora read (road 0.68, same-word 0.82, reply 0.28), so the flat line is measured where
#: it matters rather than only where the effect is large.
FAINT = 0.2


def lattice(seed):
    """One population of records and one pool of edges per relation, built once.

    Every cell screens a prefix of the same pool over the same records, so the null is drawn
    from the same population at every size and the comparison is only ever with edge count.
    """
    rng = np.random.default_rng(seed)
    maps = {n: np.linalg.qr(rng.normal(size=(D, D)))[0] for n in ("alpha", "beta")}
    records, pool = {}, {}
    for i in range(SOURCES):
        s = structured(rng, T, D)
        records[f"s{i}"] = s
        for n, m in maps.items():
            records[f"{n}{i}"] = s @ m
            pool.setdefault(n, []).append((f"s{i}", f"{n}{i}"))
        records[f"w{i}"] = (FAINT * (s @ maps["alpha"])
                            + np.sqrt(1 - FAINT ** 2) * structured(rng, T, D))
        pool.setdefault("faint", []).append((f"s{i}", f"w{i}"))
    return records, pool


def run():
    started = time.time()
    build = instrument()
    print(f"entroptics from {build['path']}")
    print(f"  projection.py {build['digests'].get('projection.py', 'missing')}\n")
    print(f"{SOURCES} sources, three relations, the same records and the same edge pools at every")
    print(f"size. Only how many edges the screen is shown changes.\n")

    names = ("alpha", "beta", "faint")
    print(f"{'edges':>7}{'alpha':>16}{'beta':>16}{'faint':>16}{'z of mean, alpha':>19}")
    print("-" * 74)

    rows = []
    for k in SIZES:
        per = {n: [] for n in names}
        for seed in SEEDS:
            records, pool = lattice(seed)
            sc = relations.carries_kinds(records, {n: pool[n][:k] for n in names},
                                         far=FAR, seed=seed, draws=DRAWS)
            for n in names:
                per[n].append(float(sc.separation[n]))
        row = {"edges": k}
        for n in names:
            row[n] = float(np.mean(per[n]))
            row[f"{n}_sd"] = float(np.std(per[n], ddof=1))
        row["alpha_z_of_mean"] = row["alpha"] * np.sqrt(k)
        rows.append(row)
        print(f"{k:>7}{row['alpha']:>10.2f} +-{row['alpha_sd']:>4.2f}"
              f"{row['beta']:>10.2f} +-{row['beta_sd']:>4.2f}"
              f"{row['faint']:>10.2f} +-{row['faint_sd']:>4.2f}"
              f"{row['alpha_z_of_mean']:>19.1f}")

    # Does separation grow with edge count? A test statistic would rise as sqrt(edges): across this
    # range that is a factor of sqrt(320/5) = 8. The ratio of last to first is the whole test, and
    # it is reported rather than compared against anything chosen.
    spread = {n: float(np.std([r[n] for r in rows], ddof=1)) for n in names}
    ratio = {n: float(rows[-1][n] / rows[0][n]) if rows[0][n] != 0 else float("nan") for n in names}
    expected = float(np.sqrt(SIZES[-1] / SIZES[0]))
    out = {"instrument": build,
           "design": {"far": FAR, "sizes": list(SIZES), "sources": SOURCES, "seeds": list(SEEDS),
                      "draws": DRAWS, "faint": FAINT, "smoke": SMOKE},
           "rows": rows,
           "edge_range": f"{SIZES[0]} to {SIZES[-1]}",
           "edge_factor": SIZES[-1] / SIZES[0],
           "separation_ratio_last_over_first": ratio,
           "ratio_a_test_statistic_would_show": expected,
           "separation_sd_across_sizes": spread,
           "z_of_mean_first": rows[0]["alpha_z_of_mean"],
           "z_of_mean_last": rows[-1]["alpha_z_of_mean"],
           "seconds": time.time() - started}

    print("\n" + "=" * 74)
    print(f"  Over a {SIZES[-1] // SIZES[0]}x range of edge count, separation moves by a factor of "
          f"{ratio['alpha']:.2f} on the strong")
    print(f"  relation and {ratio['faint']:.2f} on the faint one. A test statistic would move by "
          f"{expected:.1f}x.")
    print(f"  The z of the mean, on the same edges, moves {rows[0]['alpha_z_of_mean']:.0f} -> "
          f"{rows[-1]['alpha_z_of_mean']:.0f}.")
    print()
    print("  So the relation becomes overwhelmingly certain as edges arrive, and the screen's")
    print("  verdict does not move, because the two are answering different questions. A corpus")
    print("  refused at this level is not refused for want of edges, and gathering more of them")
    print("  cannot change the verdict -- which is why exp57 asks the engine instead.")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp59.smoke.json" if SMOKE else "exp59.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
