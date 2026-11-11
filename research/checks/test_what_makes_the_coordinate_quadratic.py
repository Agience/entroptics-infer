"""Where the quadratic actually lives, before anything is built to remove it.

Why this runs first
---------------------
`test_what_a_read_costs_as_the_store_grows.py` measures that the coordinate grows quadratically and
that it is essentially the whole cost of a read. That says the problem exists; it does not say what
the problem *is*, and the two candidate fixes want opposite facts to be true:

``sparse coordinates``   if a vertex's place touches a small and roughly constant share of the axes,
                         the whole coordinate is mostly zeros and both the memory and the products
                         can be made proportional to what is actually there. This is an
                         engineering fix and changes no number.
``a different coordinate``  if a place touches most of the axes, there is nothing to compress: the
                         coordinate is genuinely dense, the quadratic is in the method rather than in
                         the implementation, and making it scale means changing what is computed --
                         which changes the numbers and invalidates every measurement taken on
                         the current coordinate.

**Guessing between those two would be choosing which piece of work to do by preference.** So this
measures it: how many axes a place touches, as a share of the axes there are, as the store grows.

What is measured
------------------
``axes``          how many the basis carries -- one per (position, rest-of-statement) plus one per
                  statement.
``touched``       how many a place is non-zero on, averaged over vertices, and the widest single one.
``share``         touched over axes. This is the number that decides it. A share that falls as
                  the store grows means the coordinate thins out and sparse wins; a share that holds
                  or rises means it does not.

Both coordinates are measured: the walk read to its end -- the untruncated coordinate, built by
`place.read_to_its_end` from the shipped `_settle` and `_row` with only the aperture's stop left out
-- and the shipped `Engine.places`, which stops each walk at its own aperture. The first says where
the quadratic lives; the second says what the engine that ships carries.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

import lexicon
import place

from entroptics_infer.wrapper import Engine

#: the same seed and sizes the cost measurement uses, so the two tables line up
SEED = "entity.n.01"
BALLS = (150, 300, 600, 1200, 2400)


def at(ball, *, shipped=False):
    """How dense the coordinate is at one store size: read to its end, or as the engine ships it."""
    statements = lexicon.core(lexicon.neighbourhood(ball, SEED)[0])
    if len(statements) < 10:
        return None
    engine = Engine.on_statements(statements)
    axes = len(engine.basis())
    vectors = engine.places() if shipped else place.read_to_its_end(engine)
    if not vectors:
        return None

    touched = np.array([int(np.count_nonzero(v)) for v in vectors.values()])

    return {"ball": ball, "statements": len(statements), "vertices": len(vectors), "axes": axes,
            "touched_mean": float(touched.mean()), "touched_widest": int(touched.max()),
            "touched_narrowest": int(touched.min()),
            "share_mean": float(touched.mean()) / axes,
            "share_widest": float(touched.max()) / axes,
            "dense_entries": int(len(vectors)) * axes,
            "nonzero_entries": int(touched.sum())}


def main() -> int:
    print("where the quadratic lives: how dense a coordinate actually is\n")
    print(f"  {'ball':>6}{'stmts':>7}{'vertices':>10}{'axes':>7}{'touched':>10}{'widest':>9}"
          f"{'share':>8}{'if dense':>12}{'nonzero':>11}")
    print("  " + "-" * 80)

    rows = []
    for ball in BALLS:
        v = at(ball)
        if v is None:
            continue
        rows.append(v)
        print(f"  {v['ball']:>6}{v['statements']:>7}{v['vertices']:>10}{v['axes']:>7}"
              f"{v['touched_mean']:>10.0f}{v['touched_widest']:>9}{v['share_mean']:>8.1%}"
              f"{v['dense_entries']:>12,}{v['nonzero_entries']:>11,}")
    print()

    if len(rows) < 2:
        print("  Fewer than two sizes produced a store; there is no trend to report.")
        return 0

    shipped = [v for v in (at(ball, shipped=True) for ball in BALLS) if v is not None]
    for v in shipped:
        print(f"  shipped, ball {v['ball']}: a place touches {v['touched_mean']:.1f} of "
              f"{v['axes']} axes, {v['share_mean']:.2%}")
    print()
    record.figures(seed=SEED, balls=list(BALLS), by_size=rows, by_size_shipped=shipped)

    first, last = rows[0], rows[-1]
    shares = [v["share_mean"] for v in rows]
    grew = last["statements"] / first["statements"]
    entries_grew = last["nonzero_entries"] / first["nonzero_entries"]

    print(f"  A place is non-zero on **{first['share_mean']:.0%}** of the axes at "
          f"{first['statements']} statements and")
    print(f"  **{last['share_mean']:.0%}** at {last['statements']}. Over that range the store grew "
          f"{grew:.1f} times and the")
    print(f"  number of non-zero entries grew {entries_grew:.1f} times, against {grew * grew:.1f} "
          f"for quadratic.")
    print()

    falling = all(b <= a for a, b in zip(shares, shares[1:]))
    high = min(shares)

    if falling and high < max(shares):
        print(f"  **The share falls as the store grows** -- from {max(shares):.0%} to "
              f"{min(shares):.0%} -- so the coordinate")
        print("  thins out and most of what a dense array holds is zeros that were never written.")
        print("  **That makes the quadratic an implementation fact rather than a method fact**, and")
        print("  a sparse coordinate would compute the same numbers with the work proportional to")
        print("  what is actually there. Which is the piece of work to do.")
    else:
        print(f"  **The share does not fall** -- it runs {min(shares):.0%} to {max(shares):.0%} "
              f"across the range --")
        print("  so the coordinate is genuinely dense and there is nothing in it to compress. **The")
        print("  quadratic is in the method, not in the implementation.** A place is a decayed walk")
        print("  over everything it can reach, and in a connected store that is nearly everything,")
        print("  so making this scale means changing what a coordinate *is* -- which changes every")
        print("  number already measured and cannot be done quietly.")
    print()
    print(f"  At {last['statements']} statements a dense coordinate is "
          f"{last['dense_entries']:,} numbers. The same shape at")
    print("  a hundred thousand statements is what the scale question is actually asking about, and")
    print("  the share above is what decides whether that is storable at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
