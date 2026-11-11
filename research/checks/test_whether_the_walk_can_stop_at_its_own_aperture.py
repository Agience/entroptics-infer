"""The walk stops where it stops seeing: the shipped aperture against the same walk read to its end.

The aperture
--------------
`Engine.places` settles a vertex's neighbourhood largest value first and stops each walk at its own
aperture. The settled values are a spectrum, and their participation number ``2^H`` -- the
entroptics application's "effective count of occupied cells" [0019] -- is how many things are
resolvable from where the walk started. The walk stops once the unsettled tail can no longer change
that count, and keeps the vertices it resolves. Nothing is supplied: a hub in a crowded
neighbourhood and a leaf stop in different places, from their own spectra.

That is the quantity the mass-gap paper carries as ``a = 2^{-h}``, the positive aperture gauge, and
reads as the Shannon entropy of the power marginals; it is what [0003] asks for -- "a single,
deterministic, tuning-free reading whose resolution is matched to the information content of the
particular signal". Same reading, different substrate.

What is compared
------------------
``its own aperture``  `Engine.places`, the shipped coordinate.
``read to its end``   the same walk, `Engine._settle`, read until it has settled every vertex it
                      reaches, accumulated by the same `Engine._row` -- the untruncated coordinate,
                      built from the shipped code with the aperture's stop left out and nothing else
                      changed (`place.read_to_its_end`).

On the read, on what each costs, and on how many vertices each place ends up resolving.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import os
import pathlib
import random
import sys
import time

import numpy as np

import ask
import ontology
import place
import record

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import ingest                            # noqa: E402
from entroptics_infer.wrapper import Engine                    # noqa: E402

from test_whether_more_corpus_closes_the_gap_on_counting import bodies, split  # noqa: E402
from place import walked as walk  # noqa: E402


SENTENCES = int(os.environ.get("ENTROPTICS_SENTENCES") or 120)
QUESTIONS = int(os.environ.get("ENTROPTICS_QUESTIONS") or 300)

SEED = 0


def ask_prose(engine, vectors, asking, tokens):
    right = asked = 0
    for a, b, c in asking:
        if a not in vectors or c not in vectors:
            continue
        landed = engine.carry(a, b, vectors)
        if landed is None:
            continue
        pool = [t for t in tokens if t in vectors and t != a]
        if c not in pool or len(pool) < 2:
            continue
        asked += 1
        scores = ask._along(np.stack([vectors[t] for t in pool]), landed)
        if pool[int(np.argmax(scores))] == c:
            right += 1
    return right, asked


def ask_ontology(engine, statements, arm, candidates):
    """Held out one statement at a time, the way every ontology figure is taken."""
    right = asked = 0
    for index, (src, relation, dst) in enumerate(statements):
        pair = [statements[index]]
        held = [s for i, s in enumerate(statements) if i != index]
        if not any(m == relation for _, m, _ in held):
            continue
        vectors, _ = arm(engine, without=pair)
        if src not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=pair)
        if landed is None:
            continue
        stated = {b for a, m, b in held if a == src and m == relation}
        pool = [c for c in candidates if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        asked += 1
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        if pool[int(np.argmax(scores))] == dst:
            right += 1
    return right, asked


def main() -> int:
    arms = {
        "read to its end": lambda e, **kw: walk(e, aperture=False, **kw),
        "its own aperture": lambda e, **kw: walk(e, **kw),
    }

    print("the walk stops where it stops seeing: the shipped aperture against the walk read to "
          "its end\n")
    rows = {}

    # --- the ontology ---
    statements = tuple(tuple(s) for s in ontology.STATEMENTS)
    engine = Engine.on_statements(statements)
    candidates = engine.concepts()
    print(f"  ontology: {len(statements)} statements, {len(engine.vertices())} vertices\n")
    print(f"  {'stopping at':<20}{'asked':>7}{'named':>9}{'resolved, mean':>16}"
          f"{'of':>7}{'seconds':>10}")
    print("  " + "-" * 69)
    for label, arm in arms.items():
        began = time.perf_counter()
        _, resolved = arm(engine)
        right, asked = ask_ontology(engine, statements, arm, candidates)
        spent = time.perf_counter() - began
        rows[f"ontology/{label}"] = {
            "asked": asked, "named": right / asked if asked else float("nan"),
            "resolved_mean": float(np.mean(list(resolved.values()))),
            "of_vertices": len(engine.vertices()), "seconds": spent}
        r = rows[f"ontology/{label}"]
        print(f"  {label:<20}{asked:>7}{r['named']:>9.1%}{r['resolved_mean']:>16.1f}"
              f"{r['of_vertices']:>7}{spent:>9.1f}s", flush=True)
    print()

    # --- prose, the shipped ingestion's store ---
    names, text = bodies()
    if text:
        train, held = split(text, sentences=SENTENCES)
        prose = Engine.on_statements(train)
        tokens = sorted({t for a, _, c in train for t in (a, c)})
        rng = random.Random(SEED)
        asking = rng.sample(held, min(QUESTIONS, len(held)))
        print(f"  prose: {SENTENCES} sentences, {len(train)} training spans, "
              f"{len(tokens)} tokens, {len(prose.vertices())} vertices\n")
        print(f"  {'stopping at':<20}{'asked':>7}{'named':>9}{'resolved, mean':>16}"
              f"{'of':>7}{'seconds':>10}")
        print("  " + "-" * 69)
        for label, arm in arms.items():
            began = time.perf_counter()
            vectors, resolved = arm(prose, only=set(tokens))
            spent = time.perf_counter() - began
            right, asked = ask_prose(prose, vectors, asking, tokens)
            rows[f"prose/{label}"] = {
                "asked": asked, "named": right / asked if asked else float("nan"),
                "resolved_mean": float(np.mean(list(resolved.values()))),
                "of_vertices": len(prose.vertices()), "seconds": spent}
            r = rows[f"prose/{label}"]
            print(f"  {label:<20}{asked:>7}{r['named']:>9.1%}{r['resolved_mean']:>16.1f}"
                  f"{r['of_vertices']:>7}{spent:>9.1f}s", flush=True)
        print()

    record.figures(sentences=SENTENCES, questions=QUESTIONS, seed=SEED,
                   by_arm=rows)

    for store in ("ontology", "prose"):
        got = {k.split("/")[1]: v for k, v in rows.items() if k.startswith(store + "/")}
        if len(got) < 2:
            continue
        derived, whole = got["its own aperture"], got["read to its end"]
        print(f"  **{store}** -- read to its end the walk names **{whole['named']:.1%}** resolving "
              f"**{whole['resolved_mean']:.0f}** vertices per place;")
        print(f"  stopped at its own aperture it names **{derived['named']:.1%}** resolving "
              f"**{derived['resolved_mean']:.1f}**.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
