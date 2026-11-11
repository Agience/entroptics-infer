"""Stage 44 — a corpus chosen to satisfy every requirement, screened.

Why this corpus
-----------------
Four requirements are measured rather than guessed (CLAIMS §23.6, and Stages 35 and 36 for the fourth). A
corpus these reads apply to needs a shared basis wide enough to resolve, a shared ordered axis,
several structurally distinct asymmetric relations that are recorded rather than inferred, and
records alike enough that pairs standing in no relation do not read like pairs that do.

PEMS04 was disqualified once in this program's notes for having three channels per sensor, which
makes an operator nearly a scalar. That objection dissolves given CLAIMS Stage 36: an accumulator makes the
alphabet the channels, so a narrow signal becomes a wide record by discretization — a 32-bin
alphabet gives a 96 x 32 record from one channel, which is exactly the shape every generated lesson
runs at. The same construction also supplies the fourth requirement, because every sensor's cone is
assembled by one rule.

What it then has that no corpus here has had:

* **307 sensors** on one five-minute axis, 16,992 steps.
* **340 recorded, directed road edges.** Upstream and downstream are different relations and the
  direction is in the data, not inferred from it.

So this is the first candidate that was chosen against the requirements rather than found and hoped
over.

What is swept, and why it is swept rather than chosen
-------------------------------------------------------
Four things could be set wrongly and each would sink a real relation: how far apart in time two
sensors are compared (`lag`), how long the cone remembers (`cone`), how much axis a record carries
(`window`), and how finely the signal is discretized (`bins`). Choosing any of them to make a
verdict come out is the forcing this program exists to remove, so the whole surface is measured
and reported, and the verdict is whatever the surface says.

The common mode is removed in one arm and not the other, because every sensor on a road network
shares a daily cycle and that alone would relate any two of them.

The control is in the corpus
------------------------------
`unconnected` pairs sensors with no road edge between them, at the same lag. It is the same
question asked where the answer is known to be no. A construction that raises `downstream` and
`unconnected` together has found the daily traffic cycle, not the road.
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

from entroptics_infer import admission, relations                   # noqa: E402
from roads import cone_of, load_road                                # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP51_SMOKE") == "1"
SERIES = HERE.parents[1] / "corpora" / "PEMS04.npz"
ROADS = HERE.parents[1] / "corpora" / "PEMS04-distance.csv"
LAGS = (1, 6) if SMOKE else (1, 2, 3, 6, 12)
CONES = (6.0,) if SMOKE else (2.0, 6.0, 18.0)
WINDOWS = (96,) if SMOKE else (48, 96)
BINS = (32,) if SMOKE else (16, 32)
ANCHORS = 2 if SMOKE else 4
EDGES = 12 if SMOKE else 30
DRAWS = 80 if SMOKE else 250


def discretise(flow, bins, strip_common):
    if strip_common:
        flow = flow - flow.mean(axis=1, keepdims=True)
    cuts = np.quantile(flow, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(flow, cuts)


def build(sym, road, *, lag, cone, window, bins, rng, n_sensors):
    adjacent = set(road) | {(b, a) for a, b in road}
    anchors = rng.choice(np.arange(window, sym.shape[0] - lag - 1, 400),
                         size=ANCHORS, replace=False)
    records, edges = {}, {}
    for ai, t0 in enumerate(anchors):
        for j, (a, b) in enumerate(road[:EDGES]):
            src = f"s{ai}_{j}"
            records[src] = cone_of(sym, a, int(t0), window, cone, bins)
            down = f"d{ai}_{j}"
            records[down] = cone_of(sym, b, int(t0) + lag, window, cone, bins)
            edges.setdefault("downstream", []).append((src, down))
            c = int(rng.integers(n_sensors))
            while c == a or (a, c) in adjacent:
                c = int(rng.integers(n_sensors))
            far = f"u{ai}_{j}"
            records[far] = cone_of(sym, c, int(t0) + lag, window, cone, bins)
            edges.setdefault("unconnected", []).append((src, far))
    return records, edges


def representations(flow_all, road, rng, *, window=96, lag=1, anchors=4, edges=30):
    """The other reading of a refusal: that discretizing one scalar threw away what carries it.

    Three ways of building the record from the same sensors and the same road edges -- the raw three
    channels, all three discretized into bins, and a delay embedding of all three. If the relation is
    there and the representation was hiding it, one of these finds it.
    """
    adjacent = set(road) | {(b, a) for a, b in road}
    K, L = 11, 11
    cuts = [np.quantile(flow_all[:, :, c], np.linspace(0, 1, K + 1)[1:-1]) for c in range(3)]
    sym = np.stack([np.digitize(flow_all[:, :, c], cuts[c]) for c in range(3)], axis=2)

    def raw(s, t):
        return flow_all[t - window:t, s, :]

    def binned(s, t):
        out = np.zeros((window, 3 * K))
        for i, tt in enumerate(range(t - window, t)):
            for c in range(3):
                out[i, c * K + sym[tt, s, c]] = 1.0
        return out

    def delayed(s, t):
        return np.concatenate([flow_all[t - window - k:t - k, s, :] for k in range(L)], axis=1)

    out = []
    for name, rec in (("raw three channels", raw),
                      ("three channels, discretised", binned),
                      ("delay embedding", delayed)):
        anchor = rng.choice(np.arange(window + L + 1, flow_all.shape[0] - lag - 1, 400),
                            size=anchors, replace=False)
        recs, E = {}, {}
        for ai, t0 in enumerate(anchor):
            for j, (a, b) in enumerate(road[:edges]):
                src, down = f"s{ai}_{j}", f"d{ai}_{j}"
                recs[src] = rec(a, int(t0))
                recs[down] = rec(b, int(t0) + lag)
                E.setdefault("downstream", []).append((src, down))
                c = int(rng.integers(flow_all.shape[1]))
                while c == a or (a, c) in adjacent:
                    c = int(rng.integers(flow_all.shape[1]))
                far = f"u{ai}_{j}"
                recs[far] = rec(c, int(t0) + lag)
                E.setdefault("unconnected", []).append((src, far))
        sc = relations.carries_kinds(recs, E, far=FAR, seed=SEED, draws=DRAWS)
        out.append({"representation": name,
                    "channels": int(np.asarray(next(iter(recs.values()))).shape[1]),
                    "downstream": float(sc.separation["downstream"]),
                    "unconnected": float(sc.separation["unconnected"]),
                    "present": bool(sc.present["downstream"])})
    return out


def run():
    started = time.time()
    if not SERIES.exists():
        raise SystemExit(f"{SERIES} is not here; fetch PEMS04 first")
    flow, road = load_road(SERIES, ROADS)
    n_sensors = flow.shape[1]
    print(f"PEMS04: {flow.shape[0]} five-minute steps, {n_sensors} sensors, "
          f"{len(road)} recorded directed road edges\n")
    print(f"{'common mode':>12}{'bins':>6}{'window':>8}{'cone':>7}{'lag':>5}"
          f"{'rho':>7}{'downstream':>12}{'unconnected':>13}{'carried':>9}")
    print("-" * 82)

    cells = []
    for strip in (False, True):
        for bins in BINS:
            sym = discretise(flow, bins, strip)
            for window in WINDOWS:
                for cone in CONES:
                    for lag in LAGS:
                        recs, edges = build(sym, road, lag=lag, cone=cone, window=window,
                                            bins=bins, rng=np.random.default_rng(SEED),
                                            n_sensors=n_sensors)
                        sc = relations.carries_kinds(recs, edges, far=FAR, seed=SEED,
                                                     draws=DRAWS)
                        rho = float(np.mean([admission._row_autocorrelation(v)
                                             for v in list(recs.values())[:30]]))
                        cell = {"common_mode_removed": strip, "bins": bins, "window": window,
                                "cone": cone, "lag": lag, "row_autocorrelation": rho,
                                "records": len(recs),
                                "downstream": float(sc.separation["downstream"]),
                                "unconnected": float(sc.separation["unconnected"]),
                                "downstream_present": bool(sc.present["downstream"]),
                                "unconnected_present": bool(sc.present["unconnected"]),
                                "carried": int(sc.count)}
                        cells.append(cell)
                        print(f"{str(strip):>12}{bins:>6}{window:>8}{cone:>7.0f}{lag:>5}"
                              f"{rho:>7.2f}{cell['downstream']:>12.2f}"
                              f"{cell['unconnected']:>13.2f}{cell['carried']:>9}")

    reps = representations(np.load(SERIES)["data"].astype(float), road,
                           np.random.default_rng(SEED))
    print()
    print("  the other reading: that discretizing one scalar hid it")
    print()
    print(f"  {'representation':>30}{'channels':>10}{'downstream':>12}{'unconnected':>13}")
    for r in reps:
        print(f"  {r['representation']:>30}{r['channels']:>10}{r['downstream']:>12.2f}"
              f"{r['unconnected']:>13.2f}")
    print()

    best = max(cells, key=lambda c: c["downstream"])
    admitted = [c for c in cells if c["downstream_present"]]
    leaked = [c for c in cells if c["unconnected_present"]]
    ordered = sum(1 for c in cells if c["downstream"] > c["unconnected"])
    out = {"design": {"far": FAR, "seed": SEED, "lags": list(LAGS), "cones": list(CONES),
                      "windows": list(WINDOWS), "bins": list(BINS), "anchors": ANCHORS,
                      "edges_per_kind": EDGES, "draws": DRAWS, "smoke": SMOKE},
           "sensors": int(n_sensors), "road_edges": len(road),
           "steps": int(flow.shape[0]),
           "cells": cells, "cells_swept": len(cells),
           "cells_where_the_road_relation_is_present": len(admitted),
           "cells_where_the_control_leaked": len(leaked),
           "cells_where_downstream_beats_unconnected": ordered,
           "best_downstream_separation": best["downstream"],
           "best_cell": {k: best[k] for k in
                         ("common_mode_removed", "bins", "window", "cone", "lag")},
           "unconnected_at_the_best_cell": best["unconnected"],
           "representations": reps,
           "representations_tried": len(reps),
           "representations_that_found_it": sum(1 for r in reps if r["present"]),
           "best_representation_downstream": max(r["downstream"] for r in reps),
           "the_corpus_carries_the_road_relation": bool(admitted),
           "seconds": time.time() - started}
    print("\n" + "=" * 82)
    print(f"  {len(cells)} cells swept; the road relation is present in "
          f"{len(admitted)} of them, and the control leaks in {len(leaked)}.")
    print(f"  Its strongest reading is {best['downstream']:.2f} against the cutoff the level "
          f"implies,")
    print(f"  with the control at {best['unconnected']:.2f} in the same cell.")
    print(f"  downstream reads above unconnected in {ordered} of {len(cells)} cells, so the")
    print("  direction is there and its size is not.")
    if not admitted:
        print("\n  REFUSED. A corpus built to satisfy every stated requirement still does not")
        print("  carry its own recorded relation past the instrument's own level. The road is")
        print("  in the data; it is not in the association between these records.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp51.smoke.json" if SMOKE else "exp51.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
