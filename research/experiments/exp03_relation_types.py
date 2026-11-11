"""Stage 2 — how many kinds of relation does a corpus contain, and can an edge be typed without one?

The question
------------
A relational learner is normally handed the answer. R-GCN's datasets list their relation counts as a
property — WN18: 18, FB15K: 1,345 — and one weight matrix ``W_r`` exists per listed relation. Because
that explodes, ``W_r`` is factorized into ``B`` shared basis matrices, and ``B`` — an estimate of how
many independent modes the relational structure has — is chosen by hand and selected on a held-out
split. Note that ``B`` is not the relation count: R-GCN used two basis functions for FB15k's
1,345 relations. It is the effective dimensionality of the relational structure.

This asks whether that dimensionality can be measured, from edges carrying no type at all.

The construction
----------------
A relation here *is* a matrix, literally. Relation ``k`` owns a source direction ``u_k`` and a target
direction ``w_k``; an edge of relation ``k`` puts one shared latent along ``u_k`` in its source's
record and along ``w_k`` in its target's. The cross-covariance of the two endpoint records then
carries ``|s|² u_k w_kᵀ`` — a matrix determined by *which relation it is*, not by which vertices. So
edges of one kind occupy one direction in the space of ``D × D`` matrices, and the number of
independent directions is a rank, which the instrument counts.

Two regimes, measuring two different things
--------------------------------------------
**Disjoint** — each edge has its own pair of vertices, so a record carries exactly one relation.
The observed matrices then occupy exactly ``R`` directions, and a counted answer can be checked
against a known one. This is the calibration.

**Shared** — edges are drawn over a vertex pool, so a record is the sum of every relation its vertex
takes part in and each observed matrix carries cross-terms from its endpoints' other edges. The
structure genuinely occupies more directions than ``R``. That excess is not error: it is what
relational data looks like when nobody has told you which edge is which type, and it is the reason a
supplied relation index makes the problem easier. The quantity measured here is the same one ``B``
estimates.

What is measured
----------------
**A** — calibration, disjoint. Does the counted number equal ``R``, with ``R`` never supplied?

**B** — unboundedness. R-GCN's ``B`` was 2 on two of three datasets. The count here must not
saturate against anything chosen, so ``R`` is pushed well past any plausible default and the count
is required to keep tracking until the geometry stops it — a limit that is measured, not imposed.

**C** — shared vertices. How the count tracks ``R``, and how large the conflation excess is.

**D** — typing. Does an edge's measured coordinate place it with edges of its own kind? Scored
without ever supplying the number of clusters: within-kind against between-kind coordinate
alignment.

**E** — through the store. The same measurement on edges written to and read back from a real
`agience-mantle` lattice that carries no relation type.

**F** — negative control. Records with no relational structure must read 0 kinds.

What would falsify this stage
-----------------------------
Stated before the run. The stage fails if the counted number does not equal ``R`` in the disjoint
regime; if the count saturates at a value not explained by the geometry; if edges do not separate by
their generating kind above what unrelated edges score; or if the negative control reads other
than 0.

Run
---
    python research/experiments/exp03_relation_types.py
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

from corpus import relation_corpus                                 # noqa: E402
from entroptics_infer import relations, store                      # noqa: E402

T, D = 96, 64
N_VERTICES = 40
EDGES_PER_RELATION = 12
NOISE = 0.10
FAR = 0.05
BASE_SEED = 20260915
DRAWS = 8

RECOVERY_R = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
NOISE_SWEEP = (0.05, 0.10, 0.20, 0.40)
TYPING_R = (2, 4, 6, 8)


def _corpus(seed, R, noise=NOISE, share=True):
    return relation_corpus(seed=seed, n_relations=R, edges_per_relation=EDGES_PER_RELATION,
                           n_vertices=N_VERTICES, T=T, D=D, noise=noise, share_vertices=share)


def part_a_and_b():
    print("\nPARTS A & B — calibration (disjoint vertices): counted kinds against kinds used\n")
    print(f"{'R used':>7}" + "".join(f"{('n=' + str(n)):>10}" for n in NOISE_SWEEP))
    rows = []
    for R in RECOVERY_R:
        line, cells = f"{R:>7}", {}
        for noise in NOISE_SWEEP:
            counts = [
                relations.relation_modes(c.records, c.edges, far=FAR)
                for c in (_corpus(BASE_SEED + d, R, noise, share=False) for d in range(DRAWS))
            ]
            cells[str(noise)] = {"mean": float(np.mean(counts)),
                                 "exact_rate": float(np.mean([k == R for k in counts]))}
            line += f"{np.mean(counts):>10.2f}"
        rows.append({"R": R, "by_noise": cells})
        print(line, flush=True)
    print(f"\n(cell = mean counted kinds over {DRAWS} draws; exact recovery is mean == R used)")
    print(f"the largest R run here is {max(RECOVERY_R)} — nothing caps the count, and the only "
          f"limit is the {D}-dimensional basis, which holds {D // 2} independent role pairs")
    return rows


def part_c():
    print("\nPART C — shared vertices: the count with endpoint conflation present\n")
    print(f"{'R used':>7}{'counted':>10}{'excess':>9}")
    rows = []
    for R in TYPING_R + (12,):
        counts = [relations.relation_modes(c.records, c.edges, far=FAR)
                  for c in (_corpus(BASE_SEED + d, R) for d in range(DRAWS))]
        row = {"R": R, "counted": float(np.mean(counts)), "excess": float(np.mean(counts)) - R}
        rows.append(row)
        print(f"{R:>7}{row['counted']:>10.2f}{row['excess']:>+9.2f}", flush=True)
    print("\n(the excess is endpoint conflation: a record carries every relation its vertex has)")
    return rows


def part_d():
    print("\nPART D — do edges separate by their generating kind?\n")
    print(f"{'R':>3}{'K counted':>11}{'within':>9}{'between':>9}{'separation':>12}")
    rows = []
    for R in TYPING_R:
        within, between, counted = [], [], []
        for d in range(DRAWS):
            c = _corpus(BASE_SEED + d, R)
            X = relations.relation_coordinates(c.records, c.edges, far=FAR)
            counted.append(X.shape[1])
            if X.shape[1] == 0:
                continue
            n = np.linalg.norm(X, axis=1, keepdims=True)
            U = X / np.where(n > 0, n, 1.0)
            G = U @ U.T
            t = np.asarray(c.true_type)
            same = t[:, None] == t[None, :]
            off = ~np.eye(len(t), dtype=bool)
            within.append(float(G[same & off].mean()))
            between.append(float(G[~same & off].mean()))
        row = {"R": R, "K_counted": float(np.mean(counted)),
               "within_kind": float(np.mean(within)), "between_kind": float(np.mean(between))}
        row["separation"] = row["within_kind"] - row["between_kind"]
        rows.append(row)
        print(f"{R:>3}{row['K_counted']:>11.2f}{row['within_kind']:>9.3f}"
              f"{row['between_kind']:>9.3f}{row['separation']:>12.3f}", flush=True)
    print("\n(alignment of edge coordinates; the number of kinds is never supplied to the scoring)")
    return rows


def part_e():
    print("\nPART E — through an agience-mantle lattice that carries no relation type\n")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for R in TYPING_R:
            c = _corpus(BASE_SEED, R)
            with store.open_lattice(pathlib.Path(tmp) / f"r{R}.db", origin="exp03") as lattice:
                lattice.relate_many((s, d, {}) for s, d in c.edges)
                seen = []
                for v in sorted(c.records):
                    seen += [(r.src, r.dst) for r in lattice.relations(v)]
            direct = relations.relation_modes(c.records, c.edges, far=FAR)
            through = relations.relation_modes(c.records, tuple(seen), far=FAR)
            rows.append({"R": R, "edges_written": len(c.edges), "edges_read": len(seen),
                         "modes_direct": direct, "modes_through_store": through})
            print(f"R={R:<3} wrote {len(c.edges):>4}, read {len(seen):>4} back   "
                  f"kinds direct={direct}  kinds via store={through}", flush=True)
    print("\n(the store holds one label for every edge, and no read above consults it)")
    return rows


def part_f():
    print("\nPART F — negative control: records with no relational structure. Truth is 0.\n")
    rng = np.random.default_rng(BASE_SEED)
    counts = []
    for _ in range(DRAWS):
        records = {f"v{i}": rng.normal(size=(T, D)) for i in range(N_VERTICES)}
        pairs = rng.integers(0, N_VERTICES, size=(EDGES_PER_RELATION * 4, 2))
        edges = tuple((f"v{a}", f"v{b}") for a, b in pairs if a != b)
        counts.append(relations.relation_modes(records, edges, far=FAR))
    out = {"draws": DRAWS, "mean_modes": float(np.mean(counts)), "max": int(np.max(counts))}
    print(f"  counted kinds in unrelated records: mean {out['mean_modes']:.2f}, "
          f"max {out['max']}, over {DRAWS} draws  (truth 0)")
    return out


def run():
    started = time.time()
    ab, c, d, e, f = part_a_and_b(), part_c(), part_d(), part_e(), part_f()
    out = {
        "design": {"T": T, "D": D, "n_vertices": N_VERTICES,
                   "edges_per_relation": EDGES_PER_RELATION, "noise": NOISE, "far": FAR,
                   "draws": DRAWS, "base_seed": BASE_SEED, "recovery_R": RECOVERY_R,
                   "noise_sweep": NOISE_SWEEP, "typing_R": TYPING_R},
        "calibration_disjoint": ab, "shared_vertices": c, "typing": d,
        "through_store": e, "negative_control": f,
        "seconds": time.time() - started,
    }
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp03.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp03.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
