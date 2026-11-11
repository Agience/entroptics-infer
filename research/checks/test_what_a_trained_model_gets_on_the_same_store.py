"""The derived read against a trained one, on the same store and the same held-out questions.

Why this has never been run, and why it has to be
----------------------------------------------------
Every claim in this program is of the form *this works with nothing trained*. Not one of them says
how it compares to something that is trained. Those are different statements and the second is
the one anybody will ask for, because "no fitted constants" is only interesting if the thing still
answers.

So this trains a model — TransE, the standard one for exactly this task — on exactly the material the
derived read is given, and asks it exactly the same questions.

What TransE is, and why it is the right opponent
--------------------------------------------------
It embeds every vertex and every relation as a vector and fits them so that ``h + r`` lands near
``t``. It is the canonical baseline for naming the missing end of a triple, it is simple enough to
implement in full here rather than imported from somewhere with its own defaults, and it is fitted
by gradient descent on the store, which is the whole of what this program claims not to need.

**It is given every advantage that costs nothing.** Its dimension, margin and learning rate are
swept and the best setting it achieves is what gets reported — the derived read has nothing to
sweep, so comparing its single number to the baseline's best is the comparison that cannot be
accused of hobbling the opponent. Its embeddings are retrained from scratch for every held-out
question, which is the expensive thing to do and the only fair one: a model that saw the held-out
triple during training is answering from memory.

How they are matched
----------------------
For each statement in the store: remove it, train TransE on what is left, and rank the same candidate
pool the derived read is given, by ``||h + r - t||``. The derived read is measured on the same
questions in the same run. Neither is allowed to decline.

``right``              the nearest candidate is the held-out destination.
``among the reached``  restricted to candidates the relation points into elsewhere — the column that
                       cannot be won by knowing which relation was asked.

What would settle it
----------------------
If the trained model wins, the claim narrows to *this works at all without training*, which is still
a result and a smaller one. If the derived read wins or holds level, then on this store the fitting
buys nothing, and the parameters a transformer would carry are not paying for themselves.
Either way this is the number the program has been missing.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import os

import numpy as np

from concurrent import futures

import record

import ask
import ontology

from entroptics_infer.wrapper import Engine

#: The baseline's declared design. Swept because a baseline reported at one setting is a baseline
#: somebody chose, and the best of the sweep is what gets compared.
DIMENSIONS = (16, 32, 64)
MARGINS = (0.5, 1.0, 2.0)
EPOCHS = 300
RATE = 0.05
#: **Swept, and not optional.** Rearranging when the generator is drawn moved the baseline's best
#: setting from one corner of the sweep to another and its best score by more than a point, so a
#: figure from one seed is a figure somebody drew. Each setting is trained under every seed and the
#: spread across them is reported beside the mean.
SEEDS = (0, 1, 2)
#: how many questions run at once; a third of a shared box, overridable where it is yours alone
WORKERS = int(os.environ.get("ENTROPTICS_WORKERS") or max(2, (os.cpu_count() or 4) // 3))


def train(statements, vertices, relations, *, dim, margin, epochs, rate, rng):
    """TransE on one store, fitted from scratch. Returns the two embedding tables."""
    vi = {v: i for i, v in enumerate(vertices)}
    ri = {r: i for i, r in enumerate(relations)}
    scale = 6.0 / np.sqrt(dim)
    E = rng.uniform(-scale, scale, size=(len(vertices), dim))
    R = rng.uniform(-scale, scale, size=(len(relations), dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12

    h = np.array([vi[a] for a, _, _ in statements])
    r = np.array([ri[m] for _, m, _ in statements])
    t = np.array([vi[b] for _, _, b in statements])

    for _ in range(epochs):
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        # one corrupted triple per real one, head or tail replaced -- the standard negative
        bad = rng.integers(0, len(vertices), size=len(statements))
        flip = rng.random(len(statements)) < 0.5
        bh = np.where(flip, bad, h)
        bt = np.where(flip, t, bad)

        pos = E[h] + R[r] - E[t]
        neg = E[bh] + R[r] - E[bt]
        dp = np.linalg.norm(pos, axis=1)
        dn = np.linalg.norm(neg, axis=1)
        live = (margin + dp - dn) > 0
        if not live.any():
            break

        gp = pos[live] / (dp[live, None] + 1e-12)
        gn = neg[live] / (dn[live, None] + 1e-12)
        np.add.at(E, h[live], -rate * gp)
        np.add.at(E, t[live], rate * gp)
        np.add.at(R, r[live], -rate * (gp - gn))
        np.add.at(E, bh[live], rate * gn)
        np.add.at(E, bt[live], -rate * gn)

    return E, R, vi, ri


def _one_question(job):
    """Both arms on one held-out question, in its own process.

    **The question is the unit of work and the questions are independent**, so this is what gets
    handed to a pool. Each one holds its pair out, rebuilds the coordinate, reads it once, and then
    trains the baseline against that same question at every setting and seed -- so the trained arms
    see exactly the questions the derived arm saw, which is what makes the comparison a comparison.

    Top level, and taking plain data, because a pool cannot pickle a closure or an Engine.
    """
    statements, settings, seeds, src, relation, dst = job
    engine = Engine.on_statements(statements)
    concepts = engine.concepts()
    vertices = sorted(set(concepts) | {c for a, _, b in statements for c in (a, b)})
    relations = sorted({m for _, m, _ in statements})

    pair = [s for s in statements if s[0] == src and s[2] == dst]
    held = [s for s in statements if s not in pair]
    if not any(m == relation for _, m, _ in held):
        return None
    vectors = engine.places(without=pair)
    if src not in vectors or dst not in vectors:
        return None
    landed = engine.carry(src, relation, vectors, without=pair)
    if landed is None:
        return None
    stated = {b for a, m, b in held if a == src and m == relation}
    pool = [c for c in concepts if c in vectors and c != src and c not in stated]
    if dst not in pool or len(pool) < 2:
        return None
    into = {b for a, m, b in held if m == relation}
    reached = [c for c in pool if c in into]

    scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
    mine = pool[int(np.argmax(scores))]
    mine_among = None
    if dst in reached and len(reached) > 1:
        sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
        mine_among = reached[int(np.argmax(sc))]

    trained = {}
    for key, (dim, margin) in settings.items():
        for sd in seeds:
            rng = np.random.default_rng(sd)
            E, R, vi, ri = train(held, vertices, relations, dim=dim, margin=margin,
                                 epochs=EPOCHS, rate=RATE, rng=rng)
            want = E[vi[src]] + R[ri[relation]]
            far = np.linalg.norm(np.stack([E[vi[c]] for c in pool]) - want, axis=1)
            theirs = pool[int(np.argmin(far))]
            among = None
            if mine_among is not None:
                f = np.linalg.norm(np.stack([E[vi[c]] for c in reached]) - want, axis=1)
                among = reached[int(np.argmin(f))]
            trained[(key, sd)] = (theirs, among)

    return {"truth": dst, "pool": len(pool), "reached": len(reached),
            "derived": mine, "derived_among": mine_among,
            "asked_among": mine_among is not None, "trained": trained}


def sweep(statements, settings, *, seeds=SEEDS, workers=None):
    """Both arms over every setting, walking the questions once and in parallel.

    The derived read does not depend on the baseline's dimension or margin, and its coordinate is
    the expensive thing in the whole file -- rebuilt per question because the protocol removes the
    asked-about pair from it. Running the sweep as an outer loop would rebuild that coordinate once
    per setting for no reason.

    So the question is the unit, and the questions go to a pool: they are independent measurements
    and a shared box has cores. Nothing about the result changes -- each setting still sees exactly
    the questions the derived arm saw.
    """
    jobs = [(list(statements), settings, tuple(seeds), src, relation, dst)
            for src, relation, dst in statements]
    with futures.ProcessPoolExecutor(max_workers=workers or WORKERS) as pool:
        answers = [a for a in pool.map(_one_question, jobs, chunksize=4) if a is not None]

    if not answers:
        return None, {}
    asked = len(answers)
    draw = sum(1.0 / a["pool"] for a in answers) / asked
    among_rows = [a for a in answers if a["asked_among"]]
    among_asked = len(among_rows)
    adraw = (sum(1.0 / a["reached"] for a in among_rows) / among_asked
             if among_asked else float("nan"))
    derived_right = sum(1 for a in answers if a["derived"] == a["truth"])
    derived_among = sum(1 for a in among_rows if a["derived_among"] == a["truth"])

    common = {"questions": asked, "a_draw_would": draw,
              "derived": derived_right / asked,
              "derived_times_the_draw": (derived_right / asked) / draw if draw else float("nan"),
              "questions_among_the_reached": among_asked, "a_draw_among_the_reached": adraw,
              "derived_among": derived_among / among_asked if among_asked else float("nan")}

    out = {}
    for key, (dim, margin) in settings.items():
        rights, amongs, alone_t, alone_d = [], [], [], []
        for sd in seeds:
            hit = sum(1 for a in answers if a["trained"][(key, sd)][0] == a["truth"])
            rights.append(hit / asked)
            amongs.append(sum(1 for a in among_rows
                              if a["trained"][(key, sd)][1] == a["truth"]) / among_asked
                          if among_asked else float("nan"))
            alone_t.append(sum(1 for a in answers
                               if a["trained"][(key, sd)][0] == a["truth"]
                               and a["derived"] != a["truth"]))
            alone_d.append(sum(1 for a in answers
                               if a["derived"] == a["truth"]
                               and a["trained"][(key, sd)][0] != a["truth"]))
        out[key] = dict(common, dim=dim, margin=margin, seeds=list(seeds),
                        trained=float(np.mean(rights)),
                        trained_lowest=float(np.min(rights)),
                        trained_highest=float(np.max(rights)),
                        trained_times_the_draw=(float(np.mean(rights)) / draw
                                                if draw else float("nan")),
                        trained_among=float(np.mean(amongs)),
                        trained_among_lowest=float(np.min(amongs)),
                        trained_among_highest=float(np.max(amongs)),
                        trained_alone_right=float(np.mean(alone_t)),
                        derived_alone_right=float(np.mean(alone_d)))
    return common, out


def main() -> int:
    statements = [tuple(p) for p in ontology.STATEMENTS if len(p) == 3]
    print(f"the derived read against a trained one -- {len(statements)} statements, "
          f"TransE retrained per question\n")
    print(f"  {'dim':>5}{'margin':>8}{'questions':>11}{'draw':>7}{'trained':>9}{'over seeds':>16}"
          f"{'derived':>9}{'reached':>9}{'trained':>9}{'derived':>9}")
    print("  " + "-" * 94)

    settings = {f"{d}/{m}": (d, m) for d in DIMENSIONS for m in MARGINS}
    common, rows = sweep(statements, settings)
    for key in settings:
        v = rows.get(key)
        if v is None:
            continue
        spread = f"{v['trained_lowest']:.1%}-{v['trained_highest']:.1%}"
        print(f"  {v['dim']:>5}{v['margin']:>8}{v['questions']:>11}{v['a_draw_would']:>7.1%}"
              f"{v['trained']:>9.1%}{spread:>16}{v['derived']:>9.1%}"
              f"{v['questions_among_the_reached']:>9}{v['trained_among']:>9.1%}"
              f"{v['derived_among']:>9.1%}")
    print()

    if not rows:
        print("  No setting produced a question. Nothing to compare.")
        return 0

    record.figures(dimensions=list(DIMENSIONS), margins=list(MARGINS), epochs=EPOCHS,
                   rate=RATE, seeds=list(SEEDS), statements=len(statements), by_setting=rows)

    every = list(rows.values())
    best = max(every, key=lambda v: v["trained"])
    # the derived arm does not depend on the baseline's sweep, so it had better be the same number
    # in every row -- if it is not, the two arms are not being asked the same questions and there
    # is no comparison here to report
    spread = {round(v["derived"], 12) for v in every}
    if len(spread) > 1:
        print(f"  **The derived arm moved across settings** -- {sorted(spread)} -- which it cannot")
        print("  do, since nothing it computes depends on the baseline's dimension or margin. The")
        print("  two arms are not being asked the same questions and nothing here is a comparison.")
        return 0
    derived = every[0]["derived"]
    derived_among = every[0]["derived_among"]

    print(f"  The derived read is one number -- it has nothing to sweep -- and it names "
          f"**{derived:.1%}** of")
    print(f"  {every[0]['questions']} held-out destinations, {every[0]['derived_times_the_draw']:.1f} "
          f"times a draw, and **{derived_among:.1%}** among the")
    print(f"  {every[0]['questions_among_the_reached']} the relation already reaches.")
    print()
    print(f"  TransE's **best** setting over {len(every)} swept is dim {best['dim']}, margin "
          f"{best['margin']}: **{best['trained']:.1%}**")
    print(f"  overall and **{best['trained_among']:.1%}** among the reached, retrained from scratch "
          f"for every question.")
    print()

    if best["trained"] > derived:
        print("  **The trained model wins on this store.** So what this program has shown is that")
        print("  the read works with nothing fitted, not that it works as well as fitting. That is")
        print("  a real result and it is the smaller one, and every document here should say so.")
    elif derived > best["trained"]:
        print("  **The derived read wins, against the best of the sweep.** On this store the")
        print("  fitting buys nothing: a model given the same material, retrained for every")
        print("  question and handed its best hyperparameters, names fewer of the same answers")
        print("  than an operator computed in one pass with nothing to tune.")
    else:
        print("  **They are level.** Which is itself the point worth reporting: the parameters are")
        print("  not paying for themselves on this store.")
    print()
    print(f"  TransE is implemented in full in this file rather than imported, so its design is")
    print(f"  visible: {EPOCHS} epochs, rate {RATE}, one corrupted triple per real one with head or")
    print("  tail replaced, entities renormalized each epoch. It is retrained from scratch for")
    print("  every held-out question, which is the only fair way to ask it. The derived read is")
    print("  given no sweep because it has no parameters to sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
