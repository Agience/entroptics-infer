"""Does the aperture change how the coordinate *grows*, or only what it costs once?

The question, stated so it can come out either way
----------------------------------------------------
`Engine.places` stops each walk at its own aperture. Read to its end, the same walk settles every
vertex it reaches. Two things could separate them as the store grows:

``a constant factor``   the share of the basis a place touches settles somewhere and stays there.
                        Then the coordinate is still quadratic, cheaper by a constant, and a store
                        a hundred times bigger is still ten thousand times the work.
``a change of growth``  the share keeps falling as the store grows, because where a walk stops is
                        set by what its vantage resolves rather than by the store's size. Then the
                        non-zero entries per place tend to a bound and the coordinate grows like the
                        store rather than like its square.

**Those look identical at one size.** Only a sweep separates them.

What is measured
------------------
For each store size and each coordinate: how many axes there are, how many a place is non-zero on,
the share, and what the whole coordinate costs. The two coordinates are the shipped walk stopped at
its own aperture (`Engine.places`) and the same walk read to its end (`place.read_to_its_end`, the
shipped `_settle` and `_row` with the stop left out). The growth of *non-zero entries* against the
growth of statements is the number that matters for memory.

The read is measured too, at the sizes where it is affordable, because a coordinate that changed
the growth by destroying the read would look like success in the density column alone.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import time

import numpy as np

import record

import ask
import lexicon
import place

from entroptics_infer.wrapper import Engine

SEED = "entity.n.01"
#: pushed further than the cost sweep, because the question is about growth and two points cannot
#: tell a constant share from a falling one
BALLS = (150, 300, 600, 1200, 2400, 4800)
#: the two coordinates: the walk read to its end, and the shipped walk stopped at its own aperture
LEVELS = ("read to its end", "its own aperture")
#: the read is rebuilt per question, so it is only run where that is affordable; the density sweep
#: runs everywhere
READ_UP_TO = 1200


def coordinate(engine, level, *, without=()):
    """The shipped walk at its aperture, or the same walk read to its end."""
    if level == "read to its end":
        return place.read_to_its_end(engine, without=without)
    return engine.places(without=without)


def density_at(statements, below):
    """How dense the coordinate is, and what it costs to build, at one size and one level."""
    engine = Engine.on_statements(statements)
    axes = len(engine.basis())
    started = time.perf_counter()
    vectors = coordinate(engine, below)
    seconds = time.perf_counter() - started
    if not vectors:
        return None
    touched = np.array([int(np.count_nonzero(v)) for v in vectors.values()])
    return {"statements": len(statements), "vertices": len(vectors), "axes": axes,
            "touched_mean": float(touched.mean()), "share": float(touched.mean()) / axes,
            "nonzero_entries": int(touched.sum()), "seconds": seconds}


def read_at(statements, below):
    """The held-out read, so a truncation cannot pass by destroying what it was meant to keep."""
    engine = Engine.on_statements(statements)
    concepts = engine.concepts()
    right = asked = 0
    draw_sum = 0.0
    among = []
    for src, relation, dst in statements:
        held = [s for s in statements if s != (src, relation, dst)]
        vectors = coordinate(engine, below, without=[(src, relation, dst)])
        if src not in vectors or dst not in vectors:
            continue
        landed = engine.carry(src, relation, vectors, without=[(src, relation, dst)])
        if landed is None:
            continue
        stated = {b for a, r, b in held if a == src and r == relation}
        pool = [c for c in concepts if c in vectors and c != src and c not in stated]
        if dst not in pool or len(pool) < 2:
            continue
        into = {b for a, r, b in held if r == relation}
        reached = [c for c in pool if c in into]
        asked += 1
        draw_sum += 1.0 / len(pool)
        scores = ask._along(np.stack([vectors[c] for c in pool]), landed)
        right += pool[int(np.argmax(scores))] == dst
        if dst in reached and len(reached) > 1:
            sc = ask._along(np.stack([vectors[c] for c in reached]), landed)
            among.append((reached[int(np.argmax(sc))] == dst, 1.0 / len(reached)))
    if not asked:
        return None
    acc, draw = right / asked, draw_sum / asked
    got = float(np.mean([a for a, _ in among])) if among else float("nan")
    got_draw = float(np.mean([d for _, d in among])) if among else float("nan")
    return {"questions": asked, "right": acc, "times_the_draw": acc / draw if draw else float("nan"),
            "times_the_draw_among_the_reached": got / got_draw if got_draw else float("nan")}


def main() -> int:
    print("does the aperture change the growth, or only the constant\n")
    print(f"  {'ball':>6}{'stmts':>7}{'axes':>7}{'coordinate':>18}{'touched':>10}{'share':>8}"
          f"{'seconds':>10}{'nonzero':>12}")
    print("  " + "-" * 69)

    table = {}
    for ball in BALLS:
        statements = lexicon.core(lexicon.neighbourhood(ball, SEED)[0])
        if len(statements) < 10:
            continue
        for level in LEVELS:
            v = density_at(statements, level)
            if v is None:
                continue
            v["ball"], v["below"], v["level"] = ball, level, str(level)
            table[f"{ball}/{level}"] = v
            print(f"  {ball:>6}{v['statements']:>7}{v['axes']:>7}{str(level):>18}"
                  f"{v['touched_mean']:>10.0f}{v['share']:>8.1%}{v['seconds']:>10.3f}"
                  f"{v['nonzero_entries']:>12,}")
        print()

    if not table:
        print("  No store was built. Nothing to report.")
        return 0

    print(f"  {'coordinate':>18}{'reads':>7}{'questions':>11}{'right':>8}{'x draw':>9}"
          f"{'among reached':>15}")
    print("  " + "-" * 59)
    reads = {}
    for ball in [b for b in BALLS if b <= READ_UP_TO]:
        statements = lexicon.core(lexicon.neighbourhood(ball, SEED)[0])
        if len(statements) < 10:
            continue
        for level in LEVELS:
            r = read_at(statements, level)
            if r is None:
                continue
            reads[f"{ball}/{level}"] = r
            print(f"  {str(level):>18}{ball:>7}{r['questions']:>11}{r['right']:>8.1%}"
                  f"{r['times_the_draw']:>9.1f}"
                  f"{r['times_the_draw_among_the_reached']:>15.1f}")
    print()

    record.figures(seed=SEED, balls=list(BALLS), levels=[str(l) for l in LEVELS],
                   read_up_to=READ_UP_TO, by_size_and_level=table, reads=reads)

    print(f"  {'coordinate':>18}{'share first':>13}{'share last':>12}{'stmts grew':>12}"
          f"{'entries grew':>14}{'linear':>9}{'quadratic':>11}")
    print("  " + "-" * 80)
    verdicts = {}
    for level in LEVELS:
        mine = [v for v in table.values() if v["level"] == str(level)]
        if len(mine) < 2:
            continue
        mine.sort(key=lambda v: v["statements"])
        a, b = mine[0], mine[-1]
        grew = b["statements"] / a["statements"]
        entries = b["nonzero_entries"] / a["nonzero_entries"]
        verdicts[str(level)] = {"share_first": a["share"], "share_last": b["share"],
                                "statements_grew": grew, "entries_grew": entries,
                                "linear_would_be": grew, "quadratic_would_be": grew * grew,
                                # the measured exponent, which is the ratio re-expressed and not a
                                # fit: comparing 1855 to 91.8 and 8421 on a linear scale calls
                                # almost anything "nearer linear". On the scale these quantities
                                # live on, the exponent is the comparison.
                                "exponent": (float(np.log(entries) / np.log(grew))
                                             if grew > 1.0 and entries > 0 else float("nan")),
                                "shares": [v["share"] for v in mine]}
        print(f"  {str(level):>18}{a['share']:>13.1%}{b['share']:>12.1%}{grew:>12.1f}"
              f"{entries:>14.1f}{grew:>9.1f}{grew * grew:>11.1f}")
    print()

    record.figures(seed=SEED, balls=list(BALLS), levels=[str(l) for l in LEVELS],
                   read_up_to=READ_UP_TO, by_size_and_level=table, reads=reads,
                   growth=verdicts)

    # nothing fitted: each level's entry growth is put beside what linear and quadratic would give
    falling = {k: v["share_last"] < v["share_first"] for k, v in verdicts.items()}
    truncated = {k: v for k, v in verdicts.items() if k != "read to its end"}
    base = verdicts.get("read to its end")
    helped = [k for k, v in truncated.items()
              if base is not None and v["exponent"] < base["exponent"]]

    if base:
        print(f"  Read to its end the share runs {base['share_first']:.0%} to "
              f"{base['share_last']:.0%} and the entries grow")
        print(f"  {base['entries_grew']:.1f} times over a store that grew "
              f"{base['statements_grew']:.1f} -- an exponent of "
              f"**{base['exponent']:.2f}**,")
        print("  where linear is 1 and quadratic is 2.")
        print()

    if helped:
        best = min(helped, key=lambda k: truncated[k]["share_last"])
        v = truncated[best]
        print(f"  **At {best} the share keeps falling** -- {v['share_first']:.0%} down to "
              f"{v['share_last']:.0%} -- and the non-zero")
        print(f"  entries grow {v['entries_grew']:.1f} times where the store grew "
              f"{v['statements_grew']:.1f} -- an exponent of")
        print(f"  **{v['exponent']:.2f}** against **{base['exponent']:.2f}** untruncated, where "
              f"linear is 1 and quadratic is 2.")
        print("  **That is a change of growth and not a constant factor**, which is what the scale")
        print("  question was asking. It is not linear, and calling it that would be reading a")
        print("  ratio on a scale it does not live on. Whether the read")
        print("  survives it is the table above, and whether it holds on other neighbourhoods is")
        print("  not established here.")
    elif any(falling.get(k) for k in truncated):
        print("  **The share falls but the entries still grow like the square.** The aperture is")
        print("  buying a constant factor rather than a different growth, so a store a hundred")
        print("  times bigger is still ten thousand times the work and this is an optimization")
        print("  rather than an answer to scale.")
    else:
        print("  **The aperture did not change the growth.** The coordinate grows")
        print("  as it did, and whatever makes this reach a corpus is not a level on this walk.")
    print()
    print("  The read is rebuilt per question on each coordinate, as the protocol requires. Nothing is")
    print("  fitted: the growth columns are measured ratios beside what linear and quadratic would")
    print("  have produced over the same change in size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
