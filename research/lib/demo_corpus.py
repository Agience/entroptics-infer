"""A corpus you can point the command line at, to see the two readings disagree.

Why this exists
-----------------
The claim that matters is easy to state and hard to believe without running it: a lattice can hold
a relation that no single pair of records could name. A road network does it in 30 of 30 cells; a
public text stream does it on 5 of 5 lattices. Both take hours and a data download.

This writes a corpus in a few seconds where the same thing happens, with the answer known because
the generator put it there. It is a demonstration, not evidence — a generator built to show an
effect cannot establish that the effect occurs in the world, and the corpora above are what do that.

What it writes
----------------
``<out>/records/`` filled with one ``.csv`` per record, and ``<out>/edges.csv`` beside it naming
three relations. The edges sit outside the record directory because a corpus is a directory of
records and every file in it is read as one:

``loud``       ``dst = src @ M``, an exact map. Namable from one pair and held.
``quiet``      the same shape of map, carried at 0.15 amplitude against unrelated structure.
               Held, and *not* namable from one pair — the case this exists to show.
``unrelated``  a destination drawn independently. Neither, which is what makes the other two mean
               something: a reading that grows with edge count would grow on this too if it were
               counting edges rather than finding a relation.

Then
------
    python -m entroptics_infer --far 0.05 lattice <out>/records --edges <out>/edges.csv \\
        --seed 5 --draws 250

    python -m entroptics_infer --far 0.05 lattice <out>/records --edges <out>/edges.csv \\
        --seed 5 --draws 250 --source s07 --relation quiet --candidates loud07 quiet07 none07

The first prints what the lattice holds and what it could name. The second asks a question and
derives the relation from its edges at the moment of asking — there is no weight for it before the
call and none after.

On the amplitude
------------------
0.15 is where this record shape puts the gap, measured rather than picked: at 0.3 the quiet relation
is namable too and there is no gap to see, and at 0.09 it is held but so faintly that the point is
lost in the noise of a small corpus. Change ``--carry`` and watch both columns move.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

from draw import structured

T, D = 96, 32


def write(out: pathlib.Path, *, sources: int, carry: float, seed: int):
    rec = out / "records"
    rec.mkdir(parents=True, exist_ok=True)
    for stale in rec.glob("*.csv"):
        stale.unlink()
    rng = np.random.default_rng(seed)
    loud = np.linalg.qr(rng.normal(size=(D, D)))[0]
    quiet = np.linalg.qr(rng.normal(size=(D, D)))[0]
    edges = []
    for i in range(sources):
        src = structured(rng, T, D)
        np.savetxt(rec / f"s{i:02d}.csv", src, delimiter=",")

        np.savetxt(rec / f"loud{i:02d}.csv", src @ loud, delimiter=",")
        edges.append((f"s{i:02d}", f"loud{i:02d}", "loud"))

        faint = carry * (src @ quiet) + np.sqrt(1 - carry ** 2) * structured(rng, T, D)
        np.savetxt(rec / f"quiet{i:02d}.csv", faint, delimiter=",")
        edges.append((f"s{i:02d}", f"quiet{i:02d}", "quiet"))

        np.savetxt(rec / f"none{i:02d}.csv", structured(rng, T, D), delimiter=",")
        edges.append((f"s{i:02d}", f"none{i:02d}", "unrelated"))

    with (out / "edges.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src", "dst", "relation"])
        writer.writerows(edges)
    return len(edges)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=pathlib.Path, help="directory to write the corpus into")
    ap.add_argument("--sources", type=int, default=60,
                    help="how many source records; each gets three destinations")
    ap.add_argument("--carry", type=float, default=0.15,
                    help="how much of the quiet destination is the relation (default 0.15)")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)

    n = write(args.out, sources=args.sources, carry=args.carry, seed=args.seed)
    print(f"{args.sources * 4} records in {args.out / 'records'} and {n} edges in "
          f"{args.out / 'edges.csv'}")
    print()
    print("Now ask what it holds:")
    print(f"  python -m entroptics_infer --far 0.05 lattice {args.out / 'records'} \\")
    print(f"      --edges {args.out / 'edges.csv'} --seed 5 --draws 250")
    print()
    print("`loud` should be namable and held, `quiet` held and not namable, `unrelated` neither.")
    print("Then ask it a question:")
    print(f"  ... --source s07 --relation quiet --candidates loud07 quiet07 none07")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
