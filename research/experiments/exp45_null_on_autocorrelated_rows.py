"""Stage 38 — the level `associate` states, and the level it delivers on autocorrelated records.

What prompted this
--------------------
`associate` standardizes an alignment by the exact permutation null of re-pairing the two
records' rows: closed-form moments, no sampling, no supplied scale. That null is exact for
*exchangeable* rows.

`admission.py` already says so, in the course of explaining why the admission test could not use a
row shuffle: *"The permutation null's variance is exact for exchangeable rows, and autocorrelated
rows are not exchangeable."* It measured the consequence for its own test and replaced the
shuffle with a phase-randomizing surrogate that survives autocorrelation.

The same sentence applies to `associate` itself, and that had not been measured. It matters because
`carries_kinds`, `complete` and every real-corpus read in this paper rest on `associate`'s `z` and
on the cutoff the caller's level implies.

What is measured here
-----------------------
**Arm 1 — the level delivered.** Records independent by construction, drawn as AR(1) with the
row autocorrelation dialled from none to strong. Every pair must be refused, so the share called
resolved *is* the realized false-alarm rate and should sit at the stated `far`.

**Arm 2 — the repair.** The same pairs standardized against `admission`'s decoupled surrogate
instead: each record's own covariance and spectrum kept exactly, its phase relationship to every
other record destroyed. This is not new machinery; it is the null this codebase already uses where
exchangeability fails.

**Arm 3 — whether the repair keeps its power.** A null that never fires is trivially calibrated. So
the same comparison runs on records that genuinely share a trajectory, over how much of it they
share. A repair that removes the false alarms by removing the reads is not a repair.

**Arm 4 — where this paper's corpora sit.** The autocorrelation of each corpus the paper reads,
against the curve from Arm 1, which says which results are inside the affected regime and which are
not.

What this does not do
-----------------------
It does not change `entroptics.reads.coupling`, which is the shared instrument and is not this
repository's to alter on a finding. The repair is measured here; whether it belongs in the
instrument is a decision with more than this paper in it.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import time
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import curriculum as C                                              # noqa: E402
from draw import ar1                                                # noqa: E402
from entroptics_infer import engine                                 # noqa: E402
from entroptics_infer.admission import _decoupled, _row_autocorrelation   # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP45_SMOKE") == "1"
T, D = 125, 16
RHOS = (0.0, 0.6, 0.9) if SMOKE else (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99)
PAIRS = 80 if SMOKE else 400
DRAWS = 80 if SMOKE else 300
SHARES = (0.0, 0.3, 0.6, 1.0)
POWER_RHOS = (0.0, 0.6, 0.9)
POWER_PAIRS = 60 if SMOKE else 200


def null_from_surrogates(rng, rho, draws):
    """The distribution of `z` between records that share nothing, with their own structure kept."""
    zs = [engine.associate(_decoupled(rng, ar1(rng, rho, T, D)), _decoupled(rng, ar1(rng, rho, T, D)),
                           far=FAR).z
          for _ in range(draws)]
    return float(np.mean(zs)), float(np.std(zs, ddof=1))


def run():
    started = time.time()
    rng = np.random.default_rng(SEED)
    cutoff = float(engine.associate(ar1(rng, 0.0, T, D), ar1(rng, 0.0, T, D), far=FAR).cutoff)

    # --- Arms 1 and 2 -------------------------------------------------------------
    print(f"ARM 1 & 2 — records independent by construction; every pair must be refused\n")
    print(f"{'rho':>6}{'permutation null':>19}{'decoupled null':>17}{'mean |z|':>11}")
    print("-" * 55)
    cells = []
    for rho in RHOS:
        mu, sd = null_from_surrogates(rng, rho, DRAWS)
        obs = np.array([engine.associate(ar1(rng, rho, T, D), ar1(rng, rho, T, D), far=FAR).z
                        for _ in range(PAIRS)])
        stated = float((np.abs(obs) >= cutoff).mean())
        repaired = float((np.abs((obs - mu) / sd) >= cutoff).mean()) if sd > 0 else float("nan")
        cells.append({"rho": rho, "as_stated": stated, "with_the_decoupled_null": repaired,
                      "mean_abs_z": float(np.abs(obs).mean()),
                      "surrogate_mean": mu, "surrogate_spread": sd})
        print(f"{rho:>6.2f}{stated:>19.3f}{repaired:>17.3f}{np.abs(obs).mean():>11.2f}")
    print(f"\n  the stated level is {FAR}. A cell above it is the read calling unrelated records")
    print("  related, at a rate the caller did not ask for.\n")

    # --- Arm 3: power -------------------------------------------------------------
    print("=" * 55)
    print("ARM 3 — records that genuinely share a trajectory\n")
    print(f"{'rho':>6}{'shared':>8}{'permutation fires':>19}{'decoupled fires':>17}")
    print("-" * 55)
    power = []
    for rho in POWER_RHOS:
        mu, sd = null_from_surrogates(rng, rho, DRAWS)
        for shared in SHARES:
            obs = []
            for _ in range(POWER_PAIRS):
                base, a, b = ar1(rng, rho, T, D), ar1(rng, rho, T, D), ar1(rng, rho, T, D)
                obs.append(engine.associate(shared * base + (1 - shared) * a,
                                            shared * base + (1 - shared) * b, far=FAR).z)
            obs = np.array(obs)
            row = {"rho": rho, "shared": shared,
                   "as_stated": float((np.abs(obs) >= cutoff).mean()),
                   "with_the_decoupled_null": float((np.abs((obs - mu) / sd) >= cutoff).mean())}
            power.append(row)
            print(f"{rho:>6.2f}{shared:>8.1f}{row['as_stated']:>19.3f}"
                  f"{row['with_the_decoupled_null']:>17.3f}")
    print("\n  the `shared = 0.0` rows are Arm 1 again and must be refused; the rest must fire.\n")

    # --- Arm 4: where this paper's corpora sit ------------------------------------
    print("=" * 55)
    print("ARM 4 — where this paper's corpora sit on that curve\n")
    corpora = []
    for make in (C.nursery, C.kindergarten, C.grade_two, C.grade_five):
        lesson = make(rng=np.random.default_rng(SEED))
        r = float(np.mean([_row_autocorrelation(v)
                           for v in list(lesson.records.values())[:20]]))
        corpora.append({"corpus": f"curriculum: {lesson.name}", "row_autocorrelation": r})

    # the stored admission verdicts already carry this for the real corpora, so it is read from
    # them rather than re-derived -- a second derivation of the same number is not a check
    for stem, labels in (("exp12", ["synthetic off-axis", "synthetic mid", "synthetic on-axis",
                                    "triple corpus", "alice", "moby-dick"]),
                         ("exp13", ["weather, raw", "weather, national mean removed"]),
                         ("exp14", ["Beijing, raw", "Beijing, centred"]),
                         ("exp19", ["character inference, arm 1", "character inference, arm 2",
                                    "character inference, arm 3", "character inference, arm 4"])):
        path = HERE.parents[1] / "results" / f"{stem}.json"
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8")).get("admission", [])
        for label, row in zip(labels, stored):
            if "row_autocorrelation" in row:
                corpora.append({"corpus": f"{stem}: {label}",
                                "row_autocorrelation": float(row["row_autocorrelation"])})

    # Where the affected regime starts is measured, not declared. The realized rate is binomial
    # under "the level is delivered", so its excess over the stated `far` reads as a z against the
    # cutoff the caller's own level implies -- the same instrument every other verdict here uses.
    # Calling a cell affected above twice the stated level would make that 2 a choice.
    se = (FAR * (1 - FAR) / PAIRS) ** 0.5
    for c in cells:
        c["excess_in_spreads"] = float((c["as_stated"] - FAR) / se) if se > 0 else 0.0
        c["level_is_delivered"] = bool(c["excess_in_spreads"] < cutoff)
    missed = [c["rho"] for c in cells if not c["level_is_delivered"]]
    affected_from = min(missed) if missed else None
    lo = min(c["rho"] for c in cells if c["level_is_delivered"]) if any(
        c["level_is_delivered"] for c in cells) else None

    # each corpus is placed on the curve rather than sorted into a bucket: the realized level at
    # its own autocorrelation, read off the swept cells by interpolation
    xs = [c["rho"] for c in cells]
    ys = [c["as_stated"] for c in cells]
    for c in corpora:
        r = c["row_autocorrelation"]
        c["realised_far_at_its_autocorrelation"] = float(np.interp(max(r, 0.0), xs, ys))
        c["inside_the_affected_regime"] = bool(affected_from is not None and r >= affected_from)
    print(f"{'corpus':>42}{'row autocorrelation':>21}{'level it gets':>15}{'affected':>10}")
    for c in corpora:
        print(f"{c['corpus']:>42}{c['row_autocorrelation']:>+21.3f}"
              f"{c['realised_far_at_its_autocorrelation']:>15.3f}"
              f"{str(c['inside_the_affected_regime']):>10}")

    worst = max(cells, key=lambda c: c["as_stated"])
    out = {"design": {"far": FAR, "seed": SEED, "rows": T, "channels": D,
                      "rhos": list(RHOS), "pairs": PAIRS, "draws": DRAWS, "smoke": SMOKE},
           "cutoff": cutoff,
           "cells": cells, "power": power, "corpora": corpora,
           "stated_far": FAR,
           "realised_far_with_no_autocorrelation": cells[0]["as_stated"],
           "worst_realised_far": worst["as_stated"],
           "worst_realised_far_at_rho": worst["rho"],
           "worst_realised_far_with_the_decoupled_null": max(
               c["with_the_decoupled_null"] for c in cells),
           "lowest_rho_measured": lo,
           "affected_from_rho": affected_from,
           "worst_realised_far_among_the_corpora_read_here": max(
               c["realised_far_at_its_autocorrelation"] for c in corpora),
           "corpora_inside_the_affected_regime": sum(1 for c in corpora
                                                     if c["inside_the_affected_regime"]),
           "corpora_placed": len(corpora),
           "power_kept_at_full_sharing": min(r["with_the_decoupled_null"] for r in power
                                             if r["shared"] == 1.0),
           "power_lost_at_weak_sharing": min(r["with_the_decoupled_null"] for r in power
                                             if r["shared"] == 0.3),
           "seconds": time.time() - started}
    print(f"\n{'=' * 55}")
    print(f"  stated level {FAR}; delivered {cells[0]['as_stated']:.3f} with no autocorrelation "
          f"and {worst['as_stated']:.3f} at rho {worst['rho']}.")
    print(f"  the decoupled null delivers at most "
          f"{out['worst_realised_far_with_the_decoupled_null']:.3f} across the same range,")
    print(f"  and still fires {out['power_kept_at_full_sharing']:.3f} of the time on records that "
          f"share everything.")
    print(f"  {out['corpora_inside_the_affected_regime']} of {len(corpora)} corpora read here sit "
          f"at rho {affected_from} or above.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp45.smoke.json" if SMOKE else "exp45.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
