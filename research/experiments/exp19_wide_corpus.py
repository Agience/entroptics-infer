"""Stage 13 — a real corpus in the regime where the advantage is supposed to live.

Why this one
-------------
Every other real corpus in this program is narrow — at most twelve channels — which leaves open
whether a wide one behaves differently. This is one. **Forty-five channels.**

`exp18` measures that the edge is a ridge in the ambient width and the noise together, and that
neither carries it alone, so this stage does **not** arrive with a predicted edge to confirm. It
measures a wide real corpus on a task, which no other stage of the program does.

The corpus
----------
UCI Daily and Sports Activities, fetched fresh: nineteen activities, eight subjects, sixty segments
each. A segment is 125 samples at 25 Hz from **five inertial units** — torso, both arms, both legs —
carrying nine channels apiece: three axes each of accelerometer, gyroscope and magnetometer. So the
record is a genuinely wide one on a shared ordered axis, which is the shape everything here needs.

The ground truth is not a statistic
------------------------------------
Which activity a segment is, and which person performed it, were **decided before the recording and
written down by the experimenter**. Nothing about either label is estimated from the signal. That is
the property `exp07`'s text tasks lack and `exp13` is chosen for, and it matters more here than
anywhere, because this is the run the account stands or falls on.

The task, and why both versions are reported
----------------------------------------------
Two matchings, and **both are run rather than one being chosen**, because choosing after seeing the
answer is how a result gets manufactured:

* **same activity** — the true candidate performs the same activity as the query, by a *different*
  person. Everything that distinguishes it is the movement.
* **same subject** — the true candidate is the same person doing a *different* activity. Everything
  that distinguishes it is how that person moves.

Eleven distractors either way, so chance is 1/12.

What would falsify what
------------------------
Stated before the run. It is reported as **no advance** if the association does not beat the best of
the scalar reads. The stage is not the test it claims to be if the corpus is **refused** by the
admission criterion, or if its records turn out not to be wide in the sense that matters — many
channels but almost no resolved structure.

The configuration check is part of the test and its population matters. The claim is that a
*distractor* is louder than the *true key*, which is what `narrow_gain` arranges in the generator and
what puts a magnitude ranking past the crossover of PAPER §5.1, above which it must fail. Counting how many
pairs anywhere in the corpus happen to be narrow and loud answers a different question, so the true
candidate and its own distractors are measured against each other, on the same draws the task uses.

Run
---
    python research/experiments/exp19_wide_corpus.py
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

from corpus import facet_corpus                                     # noqa: E402
from entroptics_infer import admission, engine                      # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
#: Sixty segments exist per activity and subject; this many are read. Enough for the corpus to have
#: every activity and every person in it many times over, and few enough not to read 170 MB of
#: sensor data to answer a question that does not need it.
SEGMENTS = 6
N_CANDIDATES = 12
FAR = 0.05
SEED = 20260915
DRAWS = 3
N_QUERIES = 300


def fetch(into: pathlib.Path):
    into.mkdir(parents=True, exist_ok=True)
    dest = into / "uci-activities.zip"
    if not dest.exists():
        subprocess.run(["curl", "-sS", "-L", "--max-time", "600", URL, "-o", str(dest)], check=True)
    return dest, hashlib.sha256(dest.read_bytes()).hexdigest()


def build(path, *, remove_corpus_mean: bool):
    """One `(samples, channels)` record per segment, keyed by `(activity, subject, segment)`.

    Channels are put on a common footing by the corpus's own spread — an accelerometer's units and a
    magnetometer's are not commensurable — which is the corpus's own statistic and not a number
    chosen here.
    """
    z = zipfile.ZipFile(path)
    wanted = sorted(n for n in z.namelist()
                    if n.endswith(".txt") and int(n.split("/")[-1][1:3]) <= SEGMENTS)
    records, keys = {}, []
    for n in wanted:
        parts = n.split("/")
        activity, subject, segment = parts[-3], parts[-2], parts[-1][:-4]
        key = (activity, subject, segment)
        records[key] = np.loadtxt(io.BytesIO(z.read(n)), delimiter=",")
        keys.append(key)

    X = np.stack([records[k] for k in keys])
    scale = X.reshape(-1, X.shape[2]).std(axis=0)
    scale[scale == 0] = 1.0
    X = X / scale
    if remove_corpus_mean:
        # what every segment does, shared by all of them and no part of what tells one from
        # another — the identity component measured on text, weather and Beijing air alike
        X = X - X.mean(axis=0, keepdims=True)
    return {k: X[i] for i, k in enumerate(keys)}, keys, X.shape[2]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float(0)


def _lead(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Mc @ Vt[0]


def _flat_cos(A, B):
    a, b = A - A.mean(axis=0, keepdims=True), B - B.mean(axis=0, keepdims=True)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.sum(a * b) / (na * nb)) if na > 0 and nb > 0 else float(0)


def _flat_dot(A, B):
    return float(np.sum((A - A.mean(axis=0, keepdims=True))
                        * (B - B.mean(axis=0, keepdims=True))))


METHODS = ("association", "cos", "dot", "mean channel correlation",
           "max channel correlation", "leading-component correlation", "random")


def evaluate(records, keys, *, same, label):
    """Pick the true candidate among twelve. ``same`` is the index of the label that must match."""
    rng = np.random.default_rng(SEED)
    pool = list(keys)
    hits = {m: 0 for m in METHODS}
    per_question = {m: [] for m in METHODS}
    asked = 0

    order = rng.permutation(len(pool))[:N_QUERIES]
    for qi in order:
        q = pool[qi]
        # the true candidate matches on `same` and differs on the other label, so the two tasks
        # cannot be solved by the same accident
        other = 1 - same
        true_pool = [k for k in pool if k[same] == q[same] and k[other] != q[other]]
        false_pool = [k for k in pool if k[same] != q[same]]
        if not true_pool or len(false_pool) < N_CANDIDATES - 1:
            continue
        truth = true_pool[int(rng.integers(len(true_pool)))]
        distractors = [false_pool[i] for i in
                       rng.choice(len(false_pool), N_CANDIDATES - 1, replace=False)]
        cands = [truth] + distractors
        perm = rng.permutation(N_CANDIDATES)
        cands = [cands[i] for i in perm]
        answer = int(np.where(perm == 0)[0][0])

        Q = records[q]
        frames = [records[c] for c in cands]
        picks = {
            "association": int(np.argmax([engine.associate(Q, f, far=FAR).z for f in frames])),
            "cos": int(np.argmax([_flat_cos(Q, f) for f in frames])),
            "dot": int(np.argmax([_flat_dot(Q, f) for f in frames])),
            "random": int(rng.integers(N_CANDIDATES)),
        }
        per_channel = [[abs(_corr(Q[:, j], f[:, j])) for j in range(Q.shape[1])] for f in frames]
        picks["mean channel correlation"] = int(np.argmax([np.mean(r) for r in per_channel]))
        picks["max channel correlation"] = int(np.argmax([np.max(r) for r in per_channel]))
        ql = _lead(Q)
        picks["leading-component correlation"] = int(
            np.argmax([abs(_corr(ql, _lead(f))) for f in frames]))

        for m, pick in picks.items():
            ok = int(pick == answer)
            hits[m] += ok
            # every method answers the SAME question, so a difference between two of them is a
            # paired quantity carried only by the questions they disagree on. A point difference
            # with no error attached cannot say whether one read is behind another.
            per_question[m].append(ok)
        asked += 1

    out = {"arm": label, "task": "same activity" if same == 0 else "same subject",
           "questions": asked, **{m: hits[m] / asked for m in METHODS}}
    scalars = [m for m in METHODS if m not in ("association", "random")]
    best = max(scalars, key=lambda m: out[m])
    out["best_scalar"] = out[best]
    out["best_scalar_read"] = best
    out["edge"] = out["association"] - out["best_scalar"]
    a = np.asarray(per_question["association"])
    b_ = np.asarray(per_question[best])
    assoc_only = int(np.sum((a == 1) & (b_ == 0)))
    scalar_only = int(np.sum((a == 0) & (b_ == 1)))
    disc = assoc_only + scalar_only
    se = (math.sqrt(disc) / asked) if disc and asked else 0.0
    out["association_only"] = assoc_only
    out["scalar_only"] = scalar_only
    out["discordant"] = disc
    out["paired_standard_error"] = se
    out["edge_in_paired_standard_errors"] = (out["edge"] / se) if se > 0 else 0.0
    out["separated_at_two_standard_errors"] = bool(
        abs(out["edge_in_paired_standard_errors"]) >= 2)
    print(f"\n  [{label}] {out['task']}: {asked} questions, chance {1 / N_CANDIDATES:.3f}")
    for m in METHODS:
        print(f"     {m:<32}{out[m]:.3f}")
    print(f"     {'edge over the best scalar':<32}{out['edge']:+.3f}")
    return out


def generated_contrast(draws=60):
    """The same measurement on the generated family, which is what the real one is compared against.

    The claim the comparison rests on is that a generated distractor is **louder than the true key
    by construction** — `narrow_gain` places it past the crossover of section 3.2, above which a
    magnitude ranking must fail. That is the sentence, and it needs a number from a script rather
    than from an argument, so it is measured here the same way the real corpus is.
    """
    t_loud, d_loud, t_broad, d_broad, beaten = [], [], [], [], 0
    for t in range(draws):
        c = facet_corpus(seed=SEED + t, facets=6, n_keys=N_CANDIDATES, T=96, D=32,
                         narrow_gain=2.0, noise=0.10, subset_size=5)
        Q = c.query.frame
        width = Q.shape[1]

        def shape_of(frame):
            r = np.array([abs(_corr(Q[:, ch], frame[:, ch])) for ch in range(width)])
            return (float(r.max()), float(r.sum() / r.max()) if r.max() > 0 else float(0))

        lt, bt = shape_of(c.keys[c.truth].frame)
        ld, bd = zip(*[shape_of(k.frame) for i, k in enumerate(c.keys) if i != c.truth])
        t_loud.append(lt), d_loud.extend(ld)
        t_broad.append(bt), d_broad.extend(bd)
        beaten += int(max(ld) > lt)
    return {"draws": draws,
            "true_peak_agreement": float(np.mean(t_loud)),
            "distractor_peak_agreement": float(np.mean(d_loud)),
            "true_breadth": float(np.mean(t_broad)),
            "distractor_breadth": float(np.mean(d_broad)),
            "questions_with_a_louder_distractor": beaten / draws}


def run():
    started = time.time()
    path, digest = fetch(HERE.parents[1] / "corpora")
    print(f"\n{URL}\n  sha256 {digest[:16]}…")

    verdicts, facets, results, configuration = [], [], [], []
    segments = None
    for label, strip in (("raw", False), ("corpus mean removed", True)):
        records, keys, width = build(path, remove_corpus_mean=strip)
        print(f"\n=== {label} ===")
        # both arms read the same segments: nineteen activities, eight subjects, SEGMENTS each
        segments = len(keys)
        print(f"  {len(keys)} segments, {width} channels, "
              f"{records[keys[0]].shape[0]} samples each")

        # The criterion asks whether *the relation* varies along the shared axis, so the pairs must
        # be pairs the relation holds between. Both tasks are run, so both relations are asked
        # about: two segments of the same activity by different people, and two segments by the
        # same person doing different activities. Random pairs would ask about no relation at all.
        rng = np.random.default_rng(SEED)
        for same, relation in ((0, "same activity"), (1, "same subject")):
            other = 1 - same
            pairs = []
            for i in rng.permutation(len(keys))[:120]:
                q = keys[i]
                partners = [k for k in keys if k[same] == q[same] and k[other] != q[other]]
                if partners:
                    pairs.append((q, partners[int(rng.integers(len(partners)))]))
            v = admission.axis_carries(records, pairs, far=FAR, seed=SEED, draws=DRAWS)
            verdicts.append({"arm": label, "relation": relation, "pairs": len(pairs),
                             "margin_z": v.margin_z, "cutoff": v.cutoff, "admitted": v.admitted,
                             "rows": v.rows, "row_autocorrelation": v.row_autocorrelation})
            print(f"  admission, {relation}: margin_z {v.margin_z:.2f} against cutoff "
                  f"{v.cutoff:.2f} -> {'ADMITTED' if v.admitted else 'REFUSED'}"
                  f"   ({v.rows} rows, lag-1 row correlation {v.row_autocorrelation:+.3f})")

        ks = [engine.rank(records[k], far=FAR) for k in keys]
        ratio = admission.facet_independence(records, keys, far=FAR)
        facets.append({"arm": label, "width": width, "mean_modes": float(np.mean(ks)),
                       "max_modes": int(max(ks)), "modes_per_channel": ratio})
        print(f"  a segment resolves {np.mean(ks):.2f} modes on average, max {max(ks)}, "
              f"of {width} channels")

        # Does this corpus present the configuration the claim is about?
        #
        # The population that question is about is **the true candidate against the distractors in
        # its own question**, and not the corpus's pairs at large. Counting how many pairs anywhere
        # in the corpus happen to be narrow and loud answers a different question, because the
        # claim is that a *distractor* is louder than the *true key* — which is what `narrow_gain`
        # arranges in the generator and what puts a magnitude ranking past the crossover of PAPER §5.1,
        # above which it must fail.
        #
        # So the two are measured against each other, per question, on the same draws the task uses.
        t_loud, d_loud, t_broad, d_broad, beaten, asked = [], [], [], [], 0, 0
        for qi in rng.permutation(len(keys))[:120]:
            q = keys[qi]
            tp = [k for k in keys if k[0] == q[0] and k[1] != q[1]]
            fp = [k for k in keys if k[0] != q[0]]
            if not tp or len(fp) < N_CANDIDATES - 1:
                continue
            truth = tp[int(rng.integers(len(tp)))]
            dis = [fp[i] for i in rng.choice(len(fp), N_CANDIDATES - 1, replace=False)]

            def shape_of(c):
                r = np.array([abs(_corr(records[q][:, ch], records[c][:, ch]))
                              for ch in range(width)])
                return (float(r.max()), float(r.sum() / r.max()) if r.max() > 0 else float(0))

            lt, bt = shape_of(truth)
            ld, bd = zip(*[shape_of(d) for d in dis])
            t_loud.append(lt), d_loud.extend(ld)
            t_broad.append(bt), d_broad.extend(bd)
            beaten += int(max(ld) > lt)
            asked += 1

        shape = {"arm": label, "questions": asked,
                 "true_peak_agreement": float(np.mean(t_loud)),
                 "distractor_peak_agreement": float(np.mean(d_loud)),
                 "true_breadth": float(np.mean(t_broad)),
                 "distractor_breadth": float(np.mean(d_broad)),
                 "questions_with_a_louder_distractor": beaten / asked}
        configuration.append(shape)
        print(f"  peak channel agreement: true {shape['true_peak_agreement']:.3f}, "
              f"distractors {shape['distractor_peak_agreement']:.3f}")
        print(f"  breadth of agreement:   true {shape['true_breadth']:.2f}, "
              f"distractors {shape['distractor_breadth']:.2f}, of {width} channels")
        print(f"  questions where some distractor is louder than the true candidate: "
              f"{100 * shape['questions_with_a_louder_distractor']:.1f}%")

        for same, task in ((0, "same activity"), (1, "same subject")):
            results.append(evaluate(records, keys, same=same, label=label))

    gen = generated_contrast()
    print("\n=== the generated family, measured the same way ===")
    print(f"  peak channel agreement: true {gen['true_peak_agreement']:.3f}, "
          f"distractors {gen['distractor_peak_agreement']:.3f}")
    print(f"  breadth of agreement:   true {gen['true_breadth']:.2f}, "
          f"distractors {gen['distractor_breadth']:.2f}")
    print(f"  questions where some distractor is louder than the true key: "
          f"{100 * gen['questions_with_a_louder_distractor']:.1f}%")

    out = {
        "source": {"url": URL, "sha256": digest},
        "generated_contrast": gen,
        "design": {"segments_per_cell": SEGMENTS, "segments": segments,
                   "candidates": N_CANDIDATES, "far": FAR,
                   "seed": SEED, "draws": DRAWS, "queries": N_QUERIES,
                   "chance": 1 / N_CANDIDATES},
        "admission": verdicts, "facets": facets, "configuration": configuration,
        "results": results,
        "seconds": time.time() - started,
    }
    centred = [r for r in results if r["arm"] == "corpus mean removed"]
    out["edge_same_activity"] = next(r["edge"] for r in centred if r["task"] == "same activity")
    out["edge_same_subject"] = next(r["edge"] for r in centred if r["task"] == "same subject")
    out["width"] = facets[0]["width"]
    out["mean_modes"] = facets[1]["mean_modes"]

    print("\n" + "=" * 76)
    print(f"  {out['width']} channels — `exp18` puts width 48 at edge +1.000, best scalar 0.085")
    print(f"  measured here: same activity {out['edge_same_activity']:+.3f}, "
          f"same subject {out['edge_same_subject']:+.3f}")

    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    (dest / "exp19.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'exp19.json'} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
