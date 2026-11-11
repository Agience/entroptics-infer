"""Whether the shipped ingestion's prose store shares its coordinate the way hand-written triples do.

The question
--------------
A place is built on one axis per ``(position, rest-of-statement)``, and two vertices meet on an axis
only when they stand at the same position in statements whose remainder is identical. In an
ontology that happens constantly: ``dog is_a mammal`` and ``cat is_a mammal`` put ``dog`` and ``cat``
on one axis, and that is how the coordinate knows they are alike. An axis occupied by one vertex
cannot make anything close to anything, so a store whose axes are almost all occupied once carries
no comparison however large it is.

What is compared
------------------
``ontology``   the hand-written triples, the known-good case.
``prose``      the store `entroptics_infer.ingest` writes from prose: each position cut at the
               extent its own middle earns.

For each: how many axes, how many are occupied by more than one vertex, and what share of a vertex's
axes are shared ones. The last is the number that says whether the store can be read at all.
The axes are the shipped engine's own basis, read off `Engine._adjacency`.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import pathlib
import sys

import numpy as np

import record

import ontology

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer import ingest                             # noqa: E402
from entroptics_infer.wrapper import Engine                     # noqa: E402

from test_whether_the_read_predicts_the_next_token import body  # noqa: E402

#: CHOSEN: how much prose to read. The sharing is a property of the text, and more of it moves the
#: counts without moving the share, which is what is being compared.
SENTENCES = 400


def sharing(statements):
    """How much of the coordinate two vertices can actually meet on, on the engine's own basis."""
    engine = Engine.on_statements(statements)
    on_axis = collections.defaultdict(set)
    for vertex, edges in engine._adjacency(()).items():
        for _other, axis in edges:
            if axis[0] != "in":
                on_axis[axis].add(vertex)
    per_vertex = collections.Counter()
    shared_per_vertex = collections.Counter()
    for axis, vertices in on_axis.items():
        for v in vertices:
            per_vertex[v] += 1
            if len(vertices) > 1:
                shared_per_vertex[v] += 1

    axes = len(on_axis)
    shared_axes = sum(1 for vs in on_axis.values() if len(vs) > 1)
    vertices = sorted(per_vertex)
    share = [shared_per_vertex[v] / per_vertex[v] for v in vertices if per_vertex[v]]
    return {"statements": len(statements), "vertices": len(vertices), "axes": axes,
            "axes_shared": shared_axes,
            "share_of_axes_shared": shared_axes / axes if axes else float("nan"),
            "a_vertexs_axes_that_are_shared": float(np.mean(share)) if share else float("nan"),
            "widest_axis": max((len(v) for v in on_axis.values()), default=0)}


def main() -> int:
    name, text = body()
    if text is None:
        print("  No cached prose body found under corpora/. Nothing to measure.")
        return 0
    spans = ingest.spans(text, sentences=SENTENCES)

    print(f"whether the shipped prose store shares its coordinate -- {SENTENCES} sentences from "
          f"{name}\n")
    print(f"  {'store':>14}{'stmts':>8}{'vertices':>10}{'axes':>9}{'shared':>9}"
          f"{'of axes':>9}{'of a vertex':>13}{'widest':>8}")
    print("  " + "-" * 80)
    rows = {"ontology": sharing([tuple(p) for p in ontology.STATEMENTS]),
            "prose": sharing(spans)}
    for label, v in rows.items():
        print(f"  {label:>14}{v['statements']:>8}{v['vertices']:>10}{v['axes']:>9}"
              f"{v['axes_shared']:>9}{v['share_of_axes_shared']:>9.1%}"
              f"{v['a_vertexs_axes_that_are_shared']:>13.1%}{v['widest_axis']:>8}")
    print()
    record.figures(body=name, sentences=SENTENCES, by_store=rows)

    known = rows["ontology"]["a_vertexs_axes_that_are_shared"]
    got = rows["prose"]["a_vertexs_axes_that_are_shared"]
    print(f"  On the hand-written triples **{known:.1%}** of a vertex's axes are ones another "
          f"vertex also stands on;")
    print(f"  on the shipped prose store, **{got:.1%}**. An axis is a position plus the *rest* of "
          f"the statement,")
    print("  so a store meets on its axes where its remainders repeat, and that is what this counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
