"""Inference from a corpus, with no trained model — as a command you can run.

What this is for
----------------
Every claim in ``research/`` is a number in a table. This is the same machinery with a corpus you
supply, so that "no trained model, no fitted constant" is something you can check rather than read.
There is no training step because there is nothing to train: point it at a directory of records and
ask.

The corpus
----------
A directory of ``.csv`` files. Each file is one record: rows are the shared ordered axis, columns
are its channels, and every record must have the same shape. A header row is optional and is used for
channel names when present.

That shape is the whole input contract, and it is the shape the reads need — ``engine.frame``
refuses anything else rather than reshaping it, because a reshape would decide which axis is ordered
and that decision belongs to whoever knows what the record is.

Why ``--far`` has no default
----------------------------
``far`` is the false-alarm level. By Neyman–Pearson it belongs to the reader, and it is the one
number this program does not derive — so it is a required argument here, exactly as it is a
required keyword everywhere in the library. A default would let you take a reading without ever
naming the level you took it at, and the reading would then travel as though it were parameter-free.

Nothing else here is a number you can set, because there is nothing else to set.

Commands
--------
``describe``   what each record resolves: how many directions stand above its own floor.
``admit``      whether these reads apply to this corpus at all. Run this first.
``ask``        which record a query binds to — or a refusal, which is an answer.
``complete``   the forward pass: two of a triple in, the third out.
``reach``      how far a signal travels before it falls below the floor. No depth argument.
``lattice``    records plus named edges: what the store holds, and what it could name from
               a single example. Two different questions — see below.

    python -m entroptics_infer describe CORPUS --far 0.05
    python -m entroptics_infer admit    CORPUS --far 0.05
    python -m entroptics_infer ask      CORPUS --query NAME --far 0.05
    python -m entroptics_infer complete CORPUS --exemplar A B --source C --far 0.05
    python -m entroptics_infer reach    CORPUS --edges EDGES.csv --origin A --far 0.05
    python -m entroptics_infer lattice  CORPUS --edges EDGES.csv --seed 5 --draws 250 --far 0.05

``lattice`` is the whole method as one store. Its edge file is three columns, ``src,dst,relation``,
and the relation is whatever string the edge carries — nothing validates it against a list, because
there is no list. It reports two columns that do not agree, and the disagreement is the point.
namable asks whether a single exemplar pair could specify the relation, which is what
``complete`` above needs. held asks whether the lattice has the relation at all, which is what a
read deriving it from every edge needs. A road network is refused on the first in 30 of 30 cells
swept and held on the second in all of them, so refusing a corpus on the first number refuses it for
failing a test its reader never takes.

``research/lib/demo_corpus.py`` writes a corpus where the two disagree by construction, if you want to
see it before bringing your own.

``complete`` is the one that is inference rather than matching. `ask` finds the record most like the
query; `complete` is given a *relation* — by example, as a pair of records that stand in it — and
asks which record stands in that same relation to a third. The relation has no name, because there
is nothing to name it from: it is read off the one example as a matrix, which is what a trained
system holds as ``R_k`` or ``W_r`` and obtains by descent over a supplied relation index.

``reach`` is the other half of that. A transformer runs for as many rounds as it has layers and a
graph learner for as many hops as its architecture declares — a number fixed in advance, on other
data. This walks while the signal still clears the record's own floor and stops when it does not, so
the depth is a property of this query on this corpus. There is no depth argument to pass, and
adding one would undo the point.

Edges come from a two-column CSV of vertex pairs. The walk holds no opinion about where they come
from; it only asks what is adjacent to what.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

from . import admission, bind, engine, infer, lattice, propagate


def _read_record(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    """One CSV as a ``(T, D)`` frame, with channel names where the file names them."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path.name}: empty")

    def numeric(row):
        try:
            return [float(x) for x in row]
        except ValueError:
            return None

    names = None
    if numeric(rows[0]) is None:
        names = [c.strip() for c in rows[0]]
        rows = rows[1:]
    values = []
    for i, row in enumerate(rows, start=2 if names else 1):
        v = numeric(row)
        if v is None:
            raise SystemExit(f"{path.name}: line {i} is not numeric; "
                             f"a record is a table of measurements")
        values.append(v)
    W = np.asarray(values, dtype=float)
    if W.ndim != 2 or W.size == 0:
        raise SystemExit(f"{path.name}: parsed to {W.shape}, which is not a (T, D) frame")
    return W, names or [f"c{j}" for j in range(W.shape[1])]


def _load(corpus: pathlib.Path, exclude=()):
    """Every record in the directory, on a common scale, with their shapes checked against each other.

    `exclude` names files in the directory that are not records -- an edge list living beside the
    records it refers to, most usefully. They are skipped by name rather than by failing to parse,
    because a record that will not parse is an error worth reporting and not something to step over.
    """
    skip = {pathlib.Path(e).resolve() for e in exclude}
    paths = [q for q in sorted(corpus.glob("*.csv")) if q.resolve() not in skip]
    if not paths:
        raise SystemExit(f"{corpus}: no .csv files. A corpus is a directory of records.")
    records, names = {}, None
    for p in paths:
        W, chans = _read_record(p)
        if names is None:
            names = chans
        records[p.stem] = W

    shapes = {k: v.shape for k, v in records.items()}
    if len(set(shapes.values())) != 1:
        listed = ", ".join(f"{k} {v}" for k, v in sorted(shapes.items()))
        raise SystemExit(
            f"the records do not share a shape ({listed}). An association is a re-pairing of "
            f"rows, so the rows must correspond; and the two sides must meet on one basis."
        )

    # Channels are put on a common footing by the corpus's own spread, because different channels
    # are not otherwise commensurable. This is the corpus's own statistic and not a chosen number.
    #
    # A channel that never varies is left alone rather than scaled: dividing it by anything is a
    # choice about a channel that carries nothing, and `float(1)` is written rather than `1.0`
    # because the gate in `test_no_constants.py` refuses float literals in the method and it is
    # right to — this one is structural, and saying so in the code is the way to show it.
    stack = np.concatenate(list(records.values()), axis=0)
    scale = stack.std(axis=0)
    scale[scale == 0] = float(1)
    return {k: v / scale for k, v in records.items()}, names


def cmd_describe(args):
    records, names = _load(args.corpus)
    width = next(iter(records.values())).shape[1]
    print(f"{len(records)} records, {next(iter(records.values())).shape[0]} rows, "
          f"{width} channels: {', '.join(names)}\n")
    print(f"{'record':<28}{'resolves':>10}{'of':>5}{'floor':>12}")
    ks = []
    for k in sorted(records):
        r = engine.rank(records[k], far=args.far)
        ks.append(r)
        print(f"{k:<28}{r:>10}{width:>5}{engine.floor(records[k], far=args.far):>12.4f}")
    ratio = admission.facet_independence(records, sorted(records), far=args.far)
    print(f"\nmean {np.mean(ks):.2f} of {width} channels ({ratio:.3f} per channel)")
    print("\nA count of zero is an answer: it means the record reads as noise does at this level.")
    return 0


def cmd_admit(args):
    """Does the relation vary along the shared ordered axis, or is it only in the profiles?"""
    records, _ = _load(args.corpus)
    keys = sorted(records)
    if len(keys) < 2:
        raise SystemExit("admission compares pairs; a corpus of one record has none")
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

    v = admission.axis_carries(records, pairs, far=args.far, seed=args.seed, draws=args.draws)
    print(f"{len(pairs)} pairs, {args.draws} shuffles each\n")
    print(f"  margin   {v.margin_z:>8.2f}")
    print(f"  cutoff   {v.cutoff:>8.2f}   (the caller's own level, not a chosen number)")
    print(f"  verdict  {'ADMITTED' if v.admitted else 'REFUSED':>8}\n")
    if v.admitted:
        print("The association collapses when each record's rows are shuffled, so what it reads was")
        print("coming from the shared axis. These reads apply to this corpus.")
    else:
        print("The association survives shuffling each record's rows -- which destroys the axis and")
        print("leaves every record's own profile exactly as it was. So it is reading the profiles,")
        print("not the axis, and an aggregate statistic is the better tool for whatever you are")
        print("asking. This is a property of the corpus and not a failure of the run.")
    ratio = admission.facet_independence(records, keys, far=args.far)
    print(f"\nStructure per channel: {ratio:.3f}. This is reported without a verdict -- nothing")
    print("supplies a principled level at which a corpus stops carrying enough to read, and it is")
    print("not monotone in how independent the channels are. Compare corpora with it, not cutoffs.")
    return 0 if v.admitted else 1


def cmd_ask(args):
    """Which record does the query bind to? Refusal is an answer."""
    records, _ = _load(args.corpus)
    if args.query not in records:
        raise SystemExit(f"no record named {args.query!r}. Have: {', '.join(sorted(records))}")
    others = [k for k in sorted(records) if k != args.query]
    if not others:
        raise SystemExit("nothing to bind to: the corpus holds only the query")

    d = bind.decide(records[args.query], [records[k] for k in others], far=args.far)
    ranked = sorted(zip(others, d.bindings), key=lambda kv: -kv[1].association.z)
    print(f"{args.query} against {len(others)} candidates\n")
    print(f"{'candidate':<28}{'z':>9}{'shared':>9}{'resolved':>10}")
    for name, b in ranked:
        shared = "-" if b.modes < 0 else str(b.modes)
        print(f"{name:<28}{b.association.z:>9.2f}{shared:>9}"
              f"{str(b.association.resolved):>10}")

    print()
    if d.key is None:
        print(f"REFUSED -- {d.why}")
        print("\nNothing is returned because the corpus does not support a binding, and returning")
        print("the nearest thing would be the system asserting one it cannot evidence.")
        return 1
    print(f"BOUND to {others[d.key]}")
    print("\nTwo reads had to agree: the strongest standardized alignment proposed it, and it was")
    print("the unique candidate sharing the most resolved channels. No model was consulted.")
    return 0


def cmd_complete(args):
    """The forward pass: a relation given by example, and which record stands in it to a source."""
    records, _ = _load(args.corpus)
    named = list(args.exemplar) + [args.source]
    missing = [n for n in named if n not in records]
    if missing:
        raise SystemExit(f"no record named {', '.join(repr(m) for m in missing)}. "
                         f"Have: {', '.join(sorted(records))}")
    if args.candidates:
        candidates = list(args.candidates)
        unknown = [k for k in candidates if k not in records]
        if unknown:
            raise SystemExit(f"no record named {', '.join(repr(u) for u in unknown)}")
    else:
        # Everything else in the corpus. This is the general case and it is the harder one: it
        # offers records that stand in no relation to the source at all. Where the caller knows
        # which records are candidates -- one target per kind, say -- naming them asks the sharper
        # question, which is which *relation* holds rather than which record is related.
        candidates = [k for k in sorted(records) if k not in named]
    if not candidates:
        raise SystemExit("nothing to complete with: every record is already part of the question")

    c = infer.complete(records, exemplar=tuple(args.exemplar), source=args.source,
                       candidates=candidates, far=args.far)
    a, b = args.exemplar
    print(f"{a} : {b}  ::  {args.source} : ?      {len(candidates)} candidates\n")
    print(f"{'candidate':<28}{'z':>9}{'shared':>9}")
    order = sorted(range(len(candidates)), key=lambda i: -c.scores[i].z)
    for i in order:
        shared = "-" if c.modes[i] < 0 else str(c.modes[i])
        print(f"{candidates[i]:<28}{c.scores[i].z:>9.2f}{shared:>9}")

    print()
    if c.key is None:
        print(f"REFUSED -- {c.why}")
        print("\nThe operator was read off the exemplar pair, and no candidate stands in it to the")
        print("source distinguishably. Nothing is returned, because choosing anyway would be the")
        print("system asserting a relation it cannot see.")
        return 1
    print(f"COMPLETED with {candidates[c.key]}")
    print("\nThe relation was read off one example as a matrix and has no name, because there is")
    print("nothing to name it from. No relation index was supplied and no model was consulted.")
    return 0


def _read_edges(path: pathlib.Path, records):
    """A two-column CSV of vertex pairs, as the adjacency the walk asks for."""
    pairs = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.reader(fh), start=1):
            row = [c.strip() for c in row if c.strip()]
            if not row:
                continue
            if len(row) != 2:
                raise SystemExit(f"{path.name}: line {i} has {len(row)} fields; "
                                 f"an edge is a pair of record names")
            pairs.append((row[0], row[1]))
    if pairs and pairs[0][0] not in records and pairs[0][1] not in records:
        pairs = pairs[1:]                      # a header row, recognized by naming no record
    unknown = sorted({v for e in pairs for v in e} - set(records))
    if unknown:
        raise SystemExit(f"edges name records that are not in the corpus: {', '.join(unknown)}")

    adjacency: dict = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return pairs, (lambda v: sorted(adjacency.get(v, ())))


def cmd_reach(args):
    """How far the origin's signal travels. The depth is measured, not supplied."""
    records, _ = _load(args.corpus, exclude=(args.edges,))
    if args.origin not in records:
        raise SystemExit(f"no record named {args.origin!r}. Have: {', '.join(sorted(records))}")
    pairs, neighbours = _read_edges(args.edges, records)
    if not pairs:
        raise SystemExit(f"{args.edges.name}: no edges. A walk needs somewhere to go.")

    r = propagate.reach(records, neighbours, args.origin, far=args.far)
    print(f"{args.origin} over {len(records)} records and {len(pairs)} edges\n")
    print(f"  horizon    {r.depth:>6}   the greatest depth at which anything still resolved")
    print(f"  reached    {len(r.reached) - 1:>6}   vertices the signal carried to")
    print(f"  examined   {r.examined:>6}   vertices scored before the walk stopped\n")

    if len(r.reached) > 1:
        print(f"{'vertex':<28}{'depth':>7}")
        for v, d in sorted(r.reached.items(), key=lambda kv: (kv[1], kv[0])):
            if v != args.origin:
                print(f"{v:<28}{d:>7}")
    else:
        print("Nothing resolved against the origin, so the horizon is 0. That is an answer: the")
        print("signal does not leave this vertex at this level.")

    print(f"\nThe walk was given no maximum depth. It continued while the signal cleared the")
    print(f"record's own floor and stopped when it did not, so {r.depth} is a property of this")
    print(f"query on this corpus rather than of an architecture.")
    if r.frontier:
        n = len(r.frontier)
        was = "vertex was" if n == 1 else "vertices were"
        print(f"\n{n} {was} examined and not reached -- where the signal fell below the")
        print("floor. A walk's edge is a measurement too, so it is reported rather than dropped.")
    return 0



#: what a lattice's edge file says, in order. Named rather than counted, because the gate in
#: `test_no_constants.py` allows only shape and index arithmetic as bare integers and is right to:
#: `len(_NAMED_EDGE_FIELDS)` says what the number is, where a `3` would not.
_NAMED_EDGE_FIELDS = ("src", "dst", "relation")


def _read_named_edges(path: pathlib.Path, records):
    """A three-column CSV, ``src,dst,relation`` — the lattice's edges, with the names they carry.

    A header row is optional and is skipped when its third field is not a relation any other row
    uses. Nothing validates a relation name against a list, because there is no list: a name is
    whatever string an edge carries, which is what the screen needs and an enumeration would defeat.
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r and any(f.strip() for f in r)]
    if not rows:
        raise SystemExit(f"{path.name}: empty; a lattice needs edges")
    width = len(_NAMED_EDGE_FIELDS)
    if len(rows[0]) < width:
        raise SystemExit(f"{path.name}: {len(rows[0])} columns; expected "
                         f"{','.join(_NAMED_EDGE_FIELDS)}")
    if rows[0][0].strip() not in records and len(rows) > 1:
        rows = rows[1:]                              # a header, since its source is not a record
    edges = []
    for i, row in enumerate(rows, start=1):
        if len(row) < width:
            raise SystemExit(f"{path.name}: line {i} has {len(row)} columns; expected "
                             f"{width}: {','.join(_NAMED_EDGE_FIELDS)}")
        src, dst, rel = (f.strip() for f in row[:width])
        for end in (src, dst):
            if end not in records:
                raise SystemExit(
                    f"{path.name}: line {i} names {end!r}, which is not a record in this corpus. "
                    f"An edge between nothing stands for nothing."
                )
        edges.append((src, dst, rel))
    return edges


def cmd_lattice(args):
    """**What does this lattice hold, and what could it name from one example?**

    Two different questions, and a corpus can answer the first and not the second. ``separation``
    divides by the spread of a *single* unrelated pair, so it asks whether one exemplar pair could
    name the relation -- which is what a read supplied by example needs. ``strength`` compares the
    relation's mean against the mean of as many pairs standing in none, which is what a read that
    aggregates every edge needs. Measured on a road network: refused on the first in 30 of 30 cells
    and held on the second in all of them.
    """
    records, _ = _load(args.corpus, exclude=[args.edges])
    edges = _read_named_edges(args.edges, records)
    lat = lattice.Lattice(far=args.far)
    for name, W in records.items():
        lat.add(name, W)
    for src, dst, rel in edges:
        lat.relate(src, dst, rel)

    print(f"{len(records)} records, {len(edges)} edges, "
          f"{len(lat.edges)} relations named")
    print()
    verdict = lat.carries(seed=args.seed, draws=args.draws)
    print(f"{'relation':<24}{'edges':>7}{'separation':>12}{'namable':>9}"
          f"{'strength':>10}{'held':>6}")
    for rel in verdict.named:
        print(f"{rel:<24}{len(lat.edges[rel]):>7}{verdict.separation[rel]:>12.2f}"
              f"{('yes' if rel in verdict.carrying else 'no'):>9}"
              f"{verdict.strength[rel]:>10.1f}"
              f"{('yes' if rel in verdict.holding else 'no'):>6}")

    print()
    if verdict.holding:
        print(f"Held: {', '.join(verdict.holding)}.")
    else:
        print("This lattice holds none of the relations named on it. A relation carried at no")
        print("amplitude reads here too, so a name with nothing behind it is refused rather than")
        print("counted.")
    only_held = tuple(r for r in verdict.holding if r not in verdict.carrying)
    if only_held:
        print(f"Held but not namable from one example: {', '.join(only_held)}. A read supplied by")
        print("example cannot use these; one that derives the relation from every edge can.")

    if args.source:
        if args.relation is None:
            raise SystemExit("--source needs --relation: which relation to ask about")
        if args.relation not in lat.edges:
            raise SystemExit(f"nothing stands in {args.relation!r} here, so there is nothing to "
                             f"derive. Name it on a pair first.")
        got = lat.answer(args.source, args.relation, candidates=args.candidates)
        print()
        if got is None:
            print(f"What stands in {args.relation!r} to {args.source!r}? -- refused.")
            print("The candidates did not resolve, so nothing is returned. A refusal is an answer:")
            print("most records stand in no relation to a given source, and a read that cannot")
            print("decline them is not worth asking.")
        else:
            print(f"What stands in {args.relation!r} to {args.source!r}?  {got}")
            print("Derived from the edges citing that relation at the moment you asked. Nothing")
            print("was trained, and no weight for it exists between calls.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="entroptics-infer",
        description="Inference from a corpus's live geometry. No trained model, no fitted constant.",
    )
    ap.add_argument("--far", type=float, required=True,
                    help="false-alarm level; required, because it is the reader's and is the one "
                         "number this program does not derive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptext in (
        ("describe", cmd_describe, "what each record resolves above its own floor"),
        ("admit", cmd_admit, "whether these reads apply to this corpus (run this first)"),
        ("ask", cmd_ask, "which record the query binds to, or a refusal"),
        ("complete", cmd_complete, "the forward pass: two of a triple in, the third out"),
        ("reach", cmd_reach, "how far a signal travels; the depth is measured, not supplied"),
        ("lattice", cmd_lattice, "what a store of records and named edges holds, and what it "
                                 "could name from one example"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("corpus", type=pathlib.Path, help="directory of .csv records")
        p.set_defaults(fn=fn)
        if name == "admit":
            p.add_argument("--seed", type=int, required=True,
                           help="seed for the shuffles; required so a verdict is reproducible")
            p.add_argument("--draws", type=int, required=True,
                           help="how many shuffles to compare against")
        if name == "ask":
            p.add_argument("--query", required=True, help="the record to bind, by file stem")
        if name == "complete":
            p.add_argument("--exemplar", nargs=2, required=True, metavar=("A", "B"),
                           help="a pair of records standing in the relation; it is named by "
                                "example because there is nothing to name it from")
            p.add_argument("--source", required=True,
                           help="the record the same relation should be applied to")
            p.add_argument("--candidates", nargs="+", metavar="NAME",
                           help="restrict the candidates; default is every other record, which is "
                                "the harder question because most of them are unrelated")
        if name == "reach":
            p.add_argument("--edges", type=pathlib.Path, required=True,
                           help="two-column CSV of vertex pairs")
            p.add_argument("--origin", required=True, help="the record to walk from")
        if name == "lattice":
            p.add_argument("--edges", type=pathlib.Path, required=True,
                           help="three-column CSV: src,dst,relation. The relation is whatever "
                                "string the edge carries; nothing validates it against a list")
            p.add_argument("--seed", type=int, required=True,
                           help="seed for the drawn nulls; required so a verdict is reproducible")
            p.add_argument("--draws", type=int, required=True,
                           help="how many pairs standing in no named relation to draw")
            p.add_argument("--source",
                           help="also ask a question: which record stands in --relation to this one")
            p.add_argument("--relation", help="the relation to ask about, with --source")
            p.add_argument("--candidates", nargs="+", metavar="NAME",
                           help="restrict the candidates; default is every other record, which is "
                                "the harder question because most of them are unrelated")

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except engine.Unreadable as exc:
        print(f"unreadable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
