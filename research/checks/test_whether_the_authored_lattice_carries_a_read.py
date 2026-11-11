"""Is it the lattice that carries the read, or the corpus somebody imported into it.

The question
--------------
`test_whether_a_real_lattice_carries_a_read.py` reads a mantle lattice whole. A lattice can hold an
imported corpus beside what the platform wrote -- a WordNet import is typed ``text/x-wordnet`` -- and
WordNet has already been measured here and carries a read on its own. So a figure on the whole
lattice is consistent with two quite different states of affairs:

``the substrate``   the lattice carries a read because relations recorded by whoever wrote them are
                    the kind of structure this method reads, whatever the content is.
``the import``      the lattice carries a read because there is a lexical taxonomy inside it, and the
                    edges somebody wrote around that taxonomy contribute nothing.

Those matter differently. The first says the platform's own store is the corpus. The second says the
platform's store is a container that happened to be given a good corpus, and the authored part is
still untested.

How the split is decided, and why it is not a judgment
---------------------------------------------------------
Not by knowing which relation names sound lexical. Every vertex carries a content type, written
when it was created, and the import is exactly the vertices typed ``text/x-wordnet``. So a statement
is imported when both its ends are, and authored when at least one end is not. That is read off the
store rather than decided here, and the shares are reported so the split can be disagreed with.

What is measured
------------------
The same three arms on each half: held out, stated (a lookup, and a ceiling), and the permuted stores
that keep every degree and remove the correspondence. Each half is compared to its own null,
because a claim from one half against a null from the other is not a comparison.

What would settle it
----------------------
If the authored half clears its own null, the substrate carries a read and the platform's own
structure is readable. If it does not, what has been measured is WordNet a second time, and the
authored lattice is still waiting for enough of it to exist.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import os
import random

import numpy as np

import record

from entroptics_infer.store import open_lattice

from concurrent import futures

from test_whether_a_real_lattice_carries_a_read import (DRAWS, LEFT_OUT, MOST, ORIGIN,
                                                        WORKERS, _arm, a_connected_part, core,
                                                        measure, permute)

LATTICE = os.environ.get("ENTROPTICS_LATTICE")
#: the content type the import carries. Read off the store, not decided here.
IMPORTED = "text/x-wordnet"


def halves(path):
    """``(imported, authored, how)`` -- statements split by what their ends are typed as.

    Everything here comes through :class:`entroptics_infer.store.Lattice`: the labels from mantle's
    own counters, the statements from ``src_ids_by_label`` and ``edges_of``, and what each vertex
    is from ``get_many``. Nothing opens the file.
    """
    with open_lattice(path, origin=ORIGIN) as lattice:
        if not lattice.edge_labels_measured():
            lattice.measure_labels()
        labels = [l for l in lattice.labels() if l not in LEFT_OUT]
        rows = [st for l in labels for st in lattice.statements(label=l)]
        kind = lattice.kinds(sorted({v for st in rows for v in (st[0], st[2])}))

    pairs = {(s, l, d) for s, l, d in rows}
    labels = collections.Counter(l for _, l, _ in pairs)
    reciprocity = {}
    for label in labels:
        mine = [(s, d) for s, l, d in pairs if l == label]
        reciprocity[label] = sum(1 for s, d in mine if (d, label, s) in pairs) / len(mine)
    kept = {l for l, share in reciprocity.items() if share == 0.0}

    statements = sorted({(s, l, d) for s, l, d in pairs if l in kept and s != d})
    imported = [st for st in statements
                if kind.get(st[0]) == IMPORTED and kind.get(st[2]) == IMPORTED]
    authored = [st for st in statements if st not in set(imported)]

    by_label = {}
    for label in sorted(kept):
        mine = [st for st in statements if st[1] == label]
        if not mine:
            continue
        n = sum(1 for st in mine if st in set(imported))
        by_label[label] = {"statements": len(mine), "both_ends_imported": n / len(mine)}

    return imported, authored, {"labels": dict(labels), "reciprocity": reciprocity,
                                "dropped": sorted(set(labels) - kept), "by_label": by_label,
                                "imported": len(imported), "authored": len(authored)}


def arms(statements, label):
    """Held out, stated, and the permuted nulls, on one half."""
    store = core(a_connected_part(core(statements), MOST))
    if len(store) < 10:
        print(f"  {label}: core is {len(store)} statements; nothing to ask.")
        return None
    print(f"  {label}: {len(statements)} statements, core read {len(store)}, "
          f"{len({c for a, _, b in store for c in (a, b)})} vertices, "
          f"{dict(collections.Counter(l for _, l, _ in store).most_common())}")
    # the arms are independent measurements, so they run at once -- see the companion file for why
    # this is the difference between minutes and an hour, and why it changes no number
    work = ([(store, True), (store, False)]
            + [(permute(store, random.Random(d)), True) for d in range(DRAWS)])
    with futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        done = list(pool.map(_arm, work))
    held, stated = done[0], done[1]
    shuffled = [v for v in done[2:] if v is not None]
    return {"read": len(store), "held_out": held, "stated": stated, "shuffled": shuffled}


def verdict(name, arm):
    """Whether this half clears its own null, at the resolution the draws allow."""
    if not arm or not arm["held_out"]:
        return None
    got = arm["held_out"]["times_the_draw_among_the_reached"]
    mine = [v["times_the_draw_among_the_reached"] for v in arm["shuffled"]
            if v["times_the_draw_among_the_reached"] == v["times_the_draw_among_the_reached"]]
    if got != got or not mine:
        return None
    over = sum(1 for m in mine if m >= got)
    floor = 1.0 / (len(mine) + 1)
    return {"half": name, "claim": got, "null_worst": max(mine), "draws": len(mine),
            "reached_or_beat": over, "share": (over + 1) / (len(mine) + 1), "floor": floor,
            "right": arm["held_out"]["right"], "questions": arm["held_out"]["questions"],
            "times_the_draw": arm["held_out"]["times_the_draw"],
            "clears": (over + 1) / (len(mine) + 1) <= floor}


def main() -> int:
    if not LATTICE or not os.path.exists(LATTICE):
        print("  Skipped: this check reads a private mantle lattice, named by ENTROPTICS_LATTICE, and")
        print("  none is set (or the file is absent). Nothing was measured.")
        return 0

    imported, authored, how = halves(LATTICE)
    print("the lattice, split by what its vertices are typed as\n")
    print(f"  {'label':>22}{'statements':>12}{'both ends imported':>21}")
    print("  " + "-" * 55)
    for label, v in sorted(how["by_label"].items(), key=lambda kv: -kv[1]["statements"]):
        print(f"  {label:>22}{v['statements']:>12}{v['both_ends_imported']:>21.0%}")
    print(f"\n  dropped for holding both directions: {', '.join(how['dropped']) or 'none'}")
    print(f"  imported {how['imported']}, authored {how['authored']}\n")

    out = {}
    for name, statements in (("imported", imported), ("authored", authored)):
        out[name] = arms(statements, name)
        print()

    print(f"  {'half':>12}{'read':>7}{'questions':>11}{'right':>8}{'x draw':>9}"
          f"{'among reached':>15}{'null worst':>12}{'share':>8}{'clears':>8}")
    print("  " + "-" * 90)
    verdicts = {}
    for name, arm in out.items():
        v = verdict(name, arm)
        if v is None:
            print(f"  {name:>12}   no question survived")
            continue
        verdicts[name] = v
        print(f"  {name:>12}{arm['read']:>7}{v['questions']:>11}{v['right']:>8.1%}"
              f"{v['times_the_draw']:>9.1f}{v['claim']:>15.1f}{v['null_worst']:>12.1f}"
              f"{v['share']:>8.3f}{str(v['clears']):>8}")
    print()

    record.figures(imported_type=IMPORTED, most=MOST, draws=DRAWS,
                   how=how, arms=out, verdicts=verdicts)

    a = verdicts.get("authored")
    i = verdicts.get("imported")
    if a and a["clears"]:
        print(f"  **The authored half carries a read on its own.** {a['right']:.1%} of "
              f"{a['questions']} held-out")
        print(f"  statements named, {a['times_the_draw']:.1f} times a draw over the pool and "
              f"{a['claim']:.1f} times among the")
        print(f"  candidates the label already reaches, where no permuted store got past "
              f"{a['null_worst']:.1f} in")
        print(f"  {a['draws']} draws. **So it is not the import doing the work**: edges written by")
        print("  the platform about its own artifacts carry the same kind of structure, and the")
        print("  substrate is the corpus.")
    elif a:
        print(f"  **The authored half does not clear its own null.** It reads "
              f"{a['claim']:.1f} times a draw among")
        print(f"  the reached candidates against a permuted store's {a['null_worst']:.1f}, share "
              f"{a['share']:.3f} at a floor of")
        print(f"  {a['floor']:.3f}. **So what the lattice result measured is the corpus inside it**,")
        print("  and the authored structure is not yet enough to read on its own. That is a fact")
        print("  about how much has been written, not necessarily about the method.")
    if i:
        print()
        print(f"  The imported half reads {i['claim']:.1f} times a draw among the reached, "
              f"{'clearing' if i['clears'] else 'not clearing'} its own")
        print(f"  null at {i['share']:.3f}. It is a lexical taxonomy and this program has already")
        print("  measured one, so it is here as the comparison rather than as a result.")
    print()
    print("  The split is read off each vertex's recorded content type rather than decided here.")
    print("  Each half is compared to its own null, and the lattice is read on a copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
