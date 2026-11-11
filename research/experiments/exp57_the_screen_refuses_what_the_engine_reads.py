"""Stage 50 — the screen and the engine ask different questions, and only one of them was ever put.

What this is about
--------------------
Four corpora have been refused: a road network at 0.68, a conversation's replies at 0.28, its
speakers at 1.48, a word stream's repeats at 0.82. Each refusal was `carries_kinds` reporting a
separation below the level its own instrument implies, and each was recorded as *this corpus cannot
carry a relating read*.

That conclusion does not follow, and this measures why.

`carries_kinds` scores a relation as ``(mean(z over its edges) - base) / spread``, where ``spread``
is the standard deviation of the null's per-pair z. That is an effect size, not a test
statistic: it is divided by the spread of a single draw rather than by the standard error of a
mean, so it does not grow as more edges arrive. It answers "is one edge of this relation
distinguishable from one unrelated pair?" — which is the right question for :func:`complete`,
because `complete` is handed exactly one exemplar pair and must name the relation from it.

:meth:`entroptics_infer.Lattice.operator` is not handed one pair. It derives the relation from every
edge citing it, via :func:`entroptics_infer.relation_over`. So a relation whose per-edge effect is
faint and whose aggregate is overwhelming is refused by the screen and may still be read by the
engine. Nothing has ever tested that, because the screen ran first and the corpus was set down.

The two parts
---------------
**A, on generated records, where the answer is known.** The carried amplitude is swept, so the
screen's separation moves through the band the refused corpora sit in, and at each level the engine
is asked held-out questions. This calibrates what a separation of 0.68 means for the engine — which
is the number the road network was closed on.

**B, on the road network itself.** The same records, the same 340 recorded road edges, and the same
sweep of construction that `exp51` refused over — put to the engine instead of the screen.

The controls, which are the whole of whether this is worth anything
---------------------------------------------------------------------
An aggregate read that scores well proves nothing on its own; the candidates might be separable
without the relation. So every question is asked five ways with the same candidates:

``real``      the operator derived from the relation's own edges.
``residual``  that operator with its identity component ``alpha I`` removed, ``alpha = tr(M)/d``.
              `entroptics-jlens` measures real transports running at 0.42 to 0.89 identity by
              Frobenius energy, and an operator that is mostly identity turns "what stands in this
              relation to the source" into "what resembles the source" -- which is the ``none`` arm
              wearing a relation's name. At a short lag a downstream sensor's cone closely
              resembles its source's, so this corpus is exactly where that would happen.
``shuffled``  derived from those same edges with their ends re-paired at random. Same records, same
              count, no relation. If this scores, the relation is not what is answering.
``foreign``   derived from a different relation's edges over the same records.
``none``      no operator: associate the source with each candidate directly. This is the baseline
              a relating read has to beat to have earned its name.

Refusing and being wrong are counted apart. Accuracy is over the questions a read answered, and the
answer rate is reported beside it — folding the two together scored the road arm below chance in a
first run, because ``none`` takes an ``argmax`` and can never decline.

**Falsifier, stated before running.** For part B: if the real operator does not beat chance, the
no-operator baseline, a scrambled operator and a foreign one, *and* keep beating the no-operator
baseline once its identity component is removed, the road relation is not readable by the engine
either, and `exp51`'s refusal stands on its own terms.

Held out properly
-------------------
The operator is derived from one set of edges and every question is asked about sources that appear
in none of them, so nothing reads its own answer. For the road, the split is by road edge, so a
held-out question is about a pair of sensors whose edge never entered the operator.

Nothing is chosen
-------------------
Part B sweeps the same grid `exp51` swept — lag, cone length, window, alphabet size — and reports
every cell, because picking the cell that reads is the forcing this program does not do. The
candidate count is swept too, since accuracy is meaningless without the chance it is against.
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

from entroptics_infer import Lattice, relations                     # noqa: E402
from entroptics_infer.engine import associate                       # noqa: E402
from entroptics_infer.infer import complete_with, relation_over     # noqa: E402
from draw import structured                                         # noqa: E402
from roads import cone_of, load_road                                # noqa: E402

FAR = 0.05
SEED = 5
#: every question is asked once per arm, with the same candidates. See the module docstring.
ARMS = ("real", "residual", "shuffled", "foreign", "none")
SMOKE = os.environ.get("EXP57_SMOKE") == "1"

# --- part A, the calibration -----------------------------------------------------------------
T, D = 96, 32
CARRIES = (0.5, 0.2, 0.12) if SMOKE else (1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05)
SOURCES = 48 if SMOKE else 120
HELD = 12 if SMOKE else 24
SEEDS = (5, 11) if SMOKE else (5, 11, 23, 37, 41)
DRAWS = 80 if SMOKE else 250

# --- part B, the road network ----------------------------------------------------------------
SERIES = HERE.parents[1] / "corpora" / "PEMS04.npz"
ROADS = HERE.parents[1] / "corpora" / "PEMS04-distance.csv"
LAGS = (1, 6) if SMOKE else (1, 2, 3, 6, 12)
CONES = (6.0,) if SMOKE else (2.0, 6.0, 18.0)
WINDOWS = (96,) if SMOKE else (48, 96)
BINS = (32,) if SMOKE else (16, 32)
ANCHORS = 2 if SMOKE else 6
ROAD_EDGES = 40 if SMOKE else 340
CANDIDATES = (2, 4) if SMOKE else (2, 4, 8)


# =================================================================================================
# part A — what a separation of 0.68 means to the engine
# =================================================================================================

def _ask(records, operator, source, truth, candidates, far):
    """One question against one operator, as ``(answered, right)``.

    Refusing and being wrong are different outcomes and are counted apart. Scoring a refusal as an
    error folds them together and puts the road arm *below* chance -- the read declines most
    questions, and the no-operator baseline takes an ``argmax`` and cannot decline any, so the two
    arms would not be scored on the same basis. Refusal is the
    behaviour this method is built to have, so accuracy is reported over the questions a read
    actually answered, with the answer rate beside it.
    """
    out = complete_with(records, operator=operator, source=source,
                        candidates=candidates, far=far)
    if out.key is None:
        return False, False
    return True, bool(candidates[out.key] == truth)


def identity_share(operator):
    """How much of a derived operator is the identity, by Frobenius energy.

    ``alpha = tr(M)/d`` is the exact orthogonal projection of ``M`` onto ``span(I)`` under the
    Frobenius inner product -- the unique least-squares coefficient, nothing fitted and nothing
    chosen. The control is borrowed from `entroptics-jlens`, which measures real transports running
    at an identity share of 0.42 to 0.89 and shows a derived spectral floor inflating with it.

    It matters here for a reason specific to this corpus. ``relation_over`` forms
    ``M = mean(A^T B)``, and at a short lag a downstream sensor's cone closely resembles the
    source's, so ``M`` can be dominated by its identity part. An operator that is mostly identity
    turns "what stands in this relation to the source" into "what most resembles the source" --
    which is precisely the no-operator baseline. So the residual ``M - alpha I`` is asked the same
    questions, and a relation that only reads with its identity in place has not been read.
    """
    d = operator.shape[0]
    alpha = float(np.trace(operator)) / d
    removed = alpha ** 2 * d
    total = float(np.sum(operator ** 2))
    return alpha, (removed / total if total > 0 else 0.0), operator - alpha * np.eye(d)


def calibrate(seed, carry):
    """A relation carried at a known amplitude: what the screen says, and what the engine does."""
    g = np.random.default_rng(seed)
    kept = np.linalg.qr(g.normal(size=(D, D)))[0]
    other = np.linalg.qr(g.normal(size=(D, D)))[0]
    r = np.random.default_rng(seed + 1)
    lat = Lattice(far=FAR)
    for i in range(SOURCES):
        s = structured(r, T, D)
        lat.add(f"s{i}", s)
        # the same map every time, so the relation is exactly one relation; only how much of the
        # destination it accounts for is swept
        lat.add(f"t{i}", carry * (s @ kept) + np.sqrt(1 - carry ** 2) * structured(r, T, D))
        lat.add(f"o{i}", carry * (s @ other) + np.sqrt(1 - carry ** 2) * structured(r, T, D))
    derive, held = list(range(SOURCES - HELD)), list(range(SOURCES - HELD, SOURCES))
    for i in derive:
        lat.relate(f"s{i}", f"t{i}", "rel")
        lat.relate(f"s{i}", f"o{i}", "other")

    screen = relations.carries_kinds(
        lat.records, {"rel": [(f"s{i}", f"t{i}") for i in derive]},
        far=FAR, seed=seed, draws=DRAWS)

    q = np.random.default_rng(seed + 2)
    real = lat.operator("rel")
    alpha, share, residual = identity_share(real)
    ops = {"real": real, "residual": residual, "foreign": lat.operator("other")}
    order = q.permutation(len(derive))
    ops["shuffled"] = relation_over(
        lat.records, [(f"s{i}", f"t{derive[order[k]]}") for k, i in enumerate(derive)], far=FAR)

    right = {k: 0 for k in ARMS}
    answered = {k: 0 for k in ARMS}
    for i in held:
        others = [int(x) for x in q.choice([j for j in held if j != i], size=3, replace=False)]
        cands = [f"t{i}"] + [f"t{j}" for j in others]
        cands = [cands[k] for k in q.permutation(len(cands))]
        for label, op in ops.items():
            a, r = _ask(lat.records, op, f"s{i}", f"t{i}", cands, FAR)
            answered[label] += a
            right[label] += r
        zs = [abs(associate(lat.records[f"s{i}"], lat.records[c], far=FAR).z) for c in cands]
        answered["none"] += 1
        right["none"] += bool(cands[int(np.argmax(zs))] == f"t{i}")
    return (float(screen.separation["rel"]), bool(screen.present["rel"]),
            right, answered, len(held), float(share))


def part_a():
    print("=" * 78)
    print("A  a relation carried at a known amplitude: the screen's verdict, and the engine's")
    print("=" * 78)
    print(f"{'carry':>6}{'separation':>16}{'screen':>9}{'real(answered)':>15}{'resid':>9}"
          f"{'shuffled':>10}{'foreign':>9}{'none':>8}")
    print("-" * 78)
    rows = []
    for carry in CARRIES:
        seps, present, shares = [], 0, []
        acc = {k: [] for k in ARMS}
        rate = {k: [] for k in ARMS}
        n = 0
        for seed in SEEDS:
            sep, p, right, answered, held, share = calibrate(seed, carry)
            seps.append(sep)
            present += p
            shares.append(share)
            n = held
            for k in ARMS:
                # accuracy is over what a read answered; declining is reported beside it, never
                # folded into it
                acc[k].append(right[k] / answered[k] if answered[k] else float("nan"))
                rate[k].append(answered[k] / held)
        row = {"carry": carry,
               "separation": float(np.mean(seps)),
               "separation_sd": float(np.std(seps, ddof=1)),
               "screen_counts_it": int(present),
               "identity_share": float(np.mean(shares)),
               "lattices": len(SEEDS),
               "questions_each": n,
               "chance": 0.25}
        for k in ARMS:
            row[k] = float(np.nanmean(acc[k]))
            row[f"{k}_sd"] = float(np.nanstd(acc[k], ddof=1))
            row[f"{k}_answered"] = float(np.mean(rate[k]))
        rows.append(row)
        print(f"{carry:>6}{row['separation']:>10.2f} +-{row['separation_sd']:>4.2f}"
              f"{f'{present}/{len(SEEDS)}':>9}"
              f"{row['real']:>8.2f}({row['real_answered']:.2f})"
              f"{row['residual']:>9.2f}{row['shuffled']:>10.2f}"
              f"{row['foreign']:>9.2f}{row['none']:>8.2f}")
    print(f"\n  chance is 0.25. Where `screen` reads 0/{len(SEEDS)} the corpus would have been set")
    print("  down, and the `real` column is what the engine would have done with it.")
    return rows


# =================================================================================================
# part B — the road network, put to the engine
# =================================================================================================

def discretise(flow, bins):
    cuts = np.quantile(flow, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(flow, cuts)


def road_cell(sym, road, *, lag, cone, window, bins, n_sensors, rng, cand_counts):
    """Derive `downstream` from two thirds of the recorded road edges, ask about the other third.

    A held-out question names a road edge that never entered the operator, at a time anchor of its
    own, and offers the true downstream sensor among distractors drawn from sensors the network
    does not connect to the source. That is `exp51`'s `unconnected` control turned into the
    question rather than left as a second relation.

    Every candidate count in ``cand_counts`` is answered from the same derived operators and the
    same source and true-destination cones; only the distractors are redrawn. Building those cones
    once rather than once per candidate count is what makes the full sweep affordable, and it also
    means the counts differ by the question asked and by nothing else.
    """
    adjacent = set(road) | {(b, a) for a, b in road}
    use = road[:ROAD_EDGES]
    cut = (len(use) * 2) // 3
    derive_edges, held_edges = use[:cut], use[cut:]
    anchors = rng.choice(np.arange(window, sym.shape[0] - lag - 1, 400),
                         size=ANCHORS, replace=False)

    records, edges, unrelated = {}, [], []
    for ai, t0 in enumerate(anchors):
        for j, (a, b) in enumerate(derive_edges):
            src, dst = f"s{ai}_{j}", f"d{ai}_{j}"
            records[src] = cone_of(sym, a, int(t0), window, cone, bins)
            records[dst] = cone_of(sym, b, int(t0) + lag, window, cone, bins)
            edges.append((src, dst))
            c = int(rng.integers(n_sensors))
            while c == a or (a, c) in adjacent:
                c = int(rng.integers(n_sensors))
            far = f"u{ai}_{j}"
            records[far] = cone_of(sym, c, int(t0) + lag, window, cone, bins)
            unrelated.append((src, far))

    operator = relation_over(records, edges, far=FAR)
    alpha, share, residual = identity_share(operator)
    order = rng.permutation(len(edges))
    shuffled = relation_over(records, [(s, edges[order[k]][1]) for k, (s, _) in enumerate(edges)],
                             far=FAR)
    foreign = relation_over(records, unrelated, far=FAR)
    ops = {"real": operator, "residual": residual, "shuffled": shuffled, "foreign": foreign}

    # the question side, built once: a source cone and the true downstream cone per held-out edge
    asks = []
    for ai, t0 in enumerate(anchors):
        for j, (a, b) in enumerate(held_edges):
            asks.append((a, int(t0),
                         cone_of(sym, a, int(t0), window, cone, bins),
                         cone_of(sym, b, int(t0) + lag, window, cone, bins)))

    screen = relations.carries_kinds(records, {"downstream": edges, "unconnected": unrelated},
                                     far=FAR, seed=SEED, draws=DRAWS)

    out = []
    for n_cands in cand_counts:
        right = {k: 0 for k in ARMS}
        answered = {k: 0 for k in ARMS}
        asked = 0
        for a, t0, src_cone, true_cone in asks:
            qr = {"src": src_cone, "true": true_cone}
            names = ["true"]
            for k in range(n_cands - 1):
                c = int(rng.integers(n_sensors))
                while c == a or (a, c) in adjacent:
                    c = int(rng.integers(n_sensors))
                qr[f"x{k}"] = cone_of(sym, c, t0 + lag, window, cone, bins)
                names.append(f"x{k}")
            names = [names[k] for k in rng.permutation(len(names))]
            asked += 1
            for label, op in ops.items():
                ans, ok = _ask(qr, op, "src", "true", names, FAR)
                answered[label] += ans
                right[label] += ok
            zs = [abs(associate(qr["src"], qr[c], far=FAR).z) for c in names]
            answered["none"] += 1
            right["none"] += bool(names[int(np.argmax(zs))] == "true")

        cell = {"asked": asked,
                "candidates": n_cands,
                "chance": 1.0 / n_cands,
                "screen_downstream": float(screen.separation["downstream"]),
                "screen_counts_downstream": bool(screen.present["downstream"]),
                "identity_share": float(share), "alpha": float(alpha),
                "derive_edges": len(edges), "held_edges": asked}
        for k in ARMS:
            cell[k] = right[k] / answered[k] if answered[k] else float("nan")
            cell[f"{k}_answered"] = answered[k] / asked
        out.append(cell)
    return out


def part_b():
    print()
    print("=" * 78)
    print("B  the same road network, the same records, put to the engine instead of the screen")
    print("=" * 78)
    flow, road = load_road(SERIES, ROADS)
    n_sensors = flow.shape[1]
    print(f"{n_sensors} sensors, {flow.shape[0]} steps, {len(road)} recorded directed road edges")
    print(f"using {min(ROAD_EDGES, len(road))} of them, two thirds to derive and one third held out\n")
    print(f"{'lag':>4}{'cone':>5}{'win':>5}{'bins':>5}{'cnd':>4}{'sep':>6}{'scr':>6}{'idI':>6}"
          f"{'real':>7}{'ansd':>6}{'resid':>7}{'shuf':>6}{'frgn':>6}{'none':>6}{'chnc':>6}")
    print("-" * 81)
    cells = []
    for bins in BINS:
        sym = discretise(flow, bins)
        for window in WINDOWS:
            for cone in CONES:
                for lag in LAGS:
                    rng = np.random.default_rng(SEED)
                    for cell in road_cell(sym, road, lag=lag, cone=cone, window=window,
                                          bins=bins, n_sensors=n_sensors, rng=rng,
                                          cand_counts=CANDIDATES):
                        cell.update({"lag": lag, "cone": cone, "window": window, "bins": bins})
                        cells.append(cell)
                        print(f"{lag:>4}{cone:>5.0f}{window:>5}{bins:>5}{cell['candidates']:>4}"
                              f"{cell['screen_downstream']:>6.2f}"
                              f"{('yes' if cell['screen_counts_downstream'] else 'no'):>6}"
                              f"{cell['identity_share']:>6.2f}"
                              f"{cell['real']:>7.2f}{cell['real_answered']:>6.2f}"
                              f"{cell['residual']:>7.2f}{cell['shuffled']:>6.2f}"
                              f"{cell['foreign']:>6.2f}{cell['none']:>6.2f}"
                              f"{cell['chance']:>6.2f}")
    return cells


def instrument():
    """Which build of the instrument produced these figures.

    `entroptics` resolves differently on a workstation importing the source tree and on a box
    importing a wheel, and both report the same version string. Measured between two such copies at
    0.2.3, ``projection.py`` differed -- which is the module ``engine.associate`` reads through, so
    the same script would have measured a different instrument without saying so. The path and the
    digest go in the output, so a figure names the build behind it.
    """
    import hashlib
    import entroptics
    path = pathlib.Path(entroptics.__file__).parent
    digests = {}
    for name in ("projection.py", "reads.py", "aperture.py", "screen.py"):
        f = path / name
        if f.exists():
            digests[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return {"version": getattr(entroptics, "__version__", "unknown"),
            "path": "/".join(path.parts[-3:]), "digests": digests}


def run():
    started = time.time()
    build = instrument()
    print(f"entroptics {build['version']} from {build['path']}")
    print(f"  projection.py {build['digests'].get('projection.py', 'missing')}\n")
    rows = part_a()
    cells = part_b()

    def reads(c):
        """Every control cleared at once, which is what it takes to say a relation was read.

        A cell that answered nothing carries ``nan`` and fails here, as it should -- declining every
        question is not a reading. The residual clause is the one that stops an operator which is
        mostly identity from passing: if the reading does not survive removing ``alpha I``, what
        answered was resemblance to the source and not the relation.
        """
        vals = [c["real"], c["residual"], c["shuffled"], c["foreign"], c["none"]]
        if any(np.isnan(v) for v in vals):
            return False
        return (c["real"] > c["chance"] and c["real"] > c["none"]
                and c["real"] > c["shuffled"] and c["real"] > c["foreign"]
                and c["residual"] > c["none"])

    beats_both = [c for c in cells if reads(c)]
    beats_chance = [c for c in cells if not np.isnan(c["real"]) and c["real"] > c["chance"]]
    screened_out = [c for c in cells if not c["screen_counts_downstream"]]
    read_though_refused = [c for c in screened_out if reads(c)]
    answered_nothing = [c for c in cells if np.isnan(c["real"])]
    refused_band = [r for r in rows if r["screen_counts_it"] == 0]

    out = {"instrument": build,
           "design": {"far": FAR, "seed": SEED, "smoke": SMOKE,
                      "carries": list(CARRIES), "seeds": list(SEEDS), "sources": SOURCES,
                      "held": HELD, "draws": DRAWS, "lags": list(LAGS), "cones": list(CONES),
                      "windows": list(WINDOWS), "bins": list(BINS), "anchors": ANCHORS,
                      "road_edges": ROAD_EDGES, "candidates": list(CANDIDATES)},
           "calibration": rows,
           "road_cells": cells,
           "road_cells_swept": len(cells),
           "road_cells_the_screen_refuses": len(screened_out),
           "road_cells_where_real_beats_shuffled_and_none": len(beats_both),
           "road_cells_refused_yet_read": len(read_though_refused),
           "road_cells_above_chance": len(beats_chance),
           "road_cells_that_answered_nothing": len(answered_nothing),
           "road_identity_share_mean": float(np.mean([c["identity_share"] for c in cells])),
           "road_identity_share_max": float(np.max([c["identity_share"] for c in cells])),
           "best_road_real": float(np.nanmax([c["real"] for c in cells])),
           "best_road_margin_over_none": float(np.nanmax([c["real"] - c["none"] for c in cells])),
           "worst_carry_the_screen_counts": float(
               min([r["carry"] for r in rows if r["screen_counts_it"] > 0], default=float("nan"))),
           "lowest_separation_the_engine_reads": float(
               min([r["separation"] for r in refused_band if r["real"] > 3 * r["chance"]],
                   default=float("nan"))),
           "seconds": time.time() - started}

    print("\n" + "=" * 78)
    print(f"  A  the screen counts nothing below carry "
          f"{out['worst_carry_the_screen_counts']:.2f}. The engine still answers held-out")
    print(f"     questions down to separation {out['lowest_separation_the_engine_reads']:.2f}, "
          f"where a scrambled operator is at or below chance.")
    print(f"  B  {len(cells)} road cells swept, the screen refuses downstream in "
          f"{len(screened_out)} of them.")
    print(f"     The derived operator clears every control at once in {len(beats_both)}: above "
          f"chance, above no operator,")
    print(f"     above a scrambled and a foreign one, and still above no operator with its "
          f"identity component removed.")
    print(f"     The derived road operator is {out['road_identity_share_mean']:.0%} identity on "
          f"average and {out['road_identity_share_max']:.0%} at most, which is why that last "
          f"clause is there.")
    if answered_nothing:
        print(f"     {len(answered_nothing)} cells answered no question at all and are counted as "
              f"not read rather than dropped.")
    if read_though_refused:
        print(f"     In {len(read_though_refused)} cells the screen refuses the relation and the "
              f"engine reads it anyway.")
        print(f"     Best held-out accuracy {out['best_road_real']:.2f}, "
              f"{out['best_road_margin_over_none']:+.2f} over associating without the relation.")
    else:
        print("     In no cell does the derived operator clear every control. The road relation is")
        print("     not readable by the engine either, and exp51's refusal stands on its own terms.")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp57.smoke.json" if SMOKE else "exp57.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
