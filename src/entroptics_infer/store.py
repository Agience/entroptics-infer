"""The one seam onto ``agience-mantle``. Nothing else in this package imports the store.

Why the store is mantle and not a dictionary
--------------------------------------------
The program's subject is inference from a corpus's relational structure, and the thing that
makes that more than a graph library is that mantle's edges carry identity, provenance and version
history inside themselves. An experiment run against a dictionary would be measuring a method on
data with no history; run against the lattice it is measuring the method on the substrate the
platform actually has. The store is used here exactly as a node uses it — a local lattice file,
opened fresh, with no server in the path.

No relation vocabulary, at either end
-------------------------------------
``mantle.entities.relation`` holds no enumeration. An edge records what was observed about it —
an open, uninterpreted string, or nothing — and the store classifies none of it. A fixed set of
relation kinds is a decision made in advance about which relationships are possible, and it fails in
both directions: relations that do not fit are flattened into the nearest member at write time where
nothing downstream can recover the distinction, and a vocabulary larger than the corpus asserts
structure that is not there.

``LatticeGraphStore`` still requires a ``label`` on each edge row, and that column is an open
vocabulary of its own — a corpus build routinely writes tens of distinct labels and
``count_edges_by_label`` indexes whatever is there without knowing the set in advance. This seam
writes one label for every edge it creates. One label is not a vocabulary: it carries no
information, and no read in this package consults it.
``test_the_label_carries_no_information`` walks the package's syntax tree and fails if any read ever
starts to.

What kind a relation is, is measured — see :mod:`entroptics_infer.relations`, where an edge's type
is its position among the kinds the corpus turns out to contain, and how many there are is a rank.

What is *not* here
------------------
No authorization, no grants, no encryption path, no mesh. Those are real and load-bearing in mantle
and irrelevant to this measurement; importing them would put a large surface behind a research
result that does not exercise it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

__all__ = ["StoreUnavailable", "Relation", "Lattice", "open_lattice", "UNTYPED"]

#: The single label every edge carries. Not a type, not a member of a vocabulary, and never read.
#: See the module docstring: the column is required and this is what goes in it.
UNTYPED = "relation"


class StoreUnavailable(RuntimeError):
    """``agience-mantle`` is not importable, so there is no lattice to open.

    Raised instead of falling back to an in-memory substitute. A silent fallback would let an
    experiment report a result about "the lattice" that was measured on something else.
    """


def _graph_store():
    try:
        from mantle.db.edge import LatticeGraphStore
    except Exception as exc:                                  # pragma: no cover - import-shape only
        raise StoreUnavailable(
            "agience-mantle is not importable, so there is no lattice to open. Install it with "
            "`pip install -e path/to/agience-mantle` or `pip install 'entroptics-infer[store]'`. "
            "This raises rather than substituting a dictionary, because a result measured on a "
            "substitute is not a result about the store."
        ) from exc
    return LatticeGraphStore


@dataclass(frozen=True)
class Relation:
    """One observed relation between two vertices.

    Deliberately has no ``kind`` field. What kind of relation this is, is not something the
    writer knew and wrote down; it is something a reader measures from the geometry of the records
    at either end. Adding a ``kind`` here would reintroduce the vocabulary by the back door.
    """

    src: str
    dst: str
    props: dict


class Lattice:
    """A local lattice, opened fresh. Edges in, relations out, and no type anywhere between."""

    def __init__(self, path: str, *, origin: str) -> None:
        self._path = str(path)
        self._store = _graph_store()(self._path, origin=str(origin))
        self._store.ensure_schema()
        self.origin = str(origin)
        #: opened only if something asks what a vertex is, because a store may have no artifact side
        self._artifact_store = None

    def relate(self, src: str, dst: str, **props: Any) -> None:
        """Record that two vertices are related, without saying how.

        ``props`` carries whatever was *observed* about the relation — a count, a position, a
        timestamp. It must not carry a name for the relation's kind, and nothing in this package
        reads one.
        """
        self._store.add_edge(str(src), str(dst), UNTYPED, dict(props))

    def relate_many(self, edges: Iterable[tuple[str, str, dict]]) -> int:
        """Bulk form of :meth:`relate`; returns the number handled."""
        return int(
            self._store.add_edges((s, d, UNTYPED, dict(p or {})) for s, d, p in edges)
        )

    def relations(self, vertex: str, *, direction: str = "out") -> list[Relation]:
        """Every relation touching a vertex, as observations rather than as typed edges."""
        rows = self._store.edges_of(str(vertex), direction=direction)
        return [Relation(src=r["src"], dst=r["dst"], props=dict(r.get("props") or {})) for r in rows]

    def neighbours(self, vertex: str, *, direction: str = "out") -> list[str]:
        """The vertices one relation away, in the order the store returns them."""
        return list(self._store.neighbors(str(vertex), direction=direction))

    def count(self) -> int:
        return int(self._store.count_edges())

    # ---- reading a store somebody else wrote ---------------------------------------------------

    def labels(self) -> list[str]:
        """Every label the store has edges under, as mantle counts them.

        Read through ``labels_with_edges``, which is the store's own index, rather than by scanning
        the table. Nothing in this package attaches meaning to a label -- what kind a relation is,
        is measured (see :mod:`entroptics_infer.relations`) -- but a reader of somebody else's
        lattice has to know what is in it before it can read any of it, and asking the store is how.

        ``[]`` where the store has not measured its labels, which is a real state and not an error:
        the counters are backfilled, and an unbackfilled store answers "I do not know" rather than
        "none". :meth:`Lattice.edge_labels_measured` is the question to ask about that.
        """
        return list(self._store.labels_with_edges() or [])

    def edge_labels_measured(self) -> bool:
        """Whether the store's label counters have been backfilled, so :meth:`labels` can answer."""
        return bool(self._store.edge_labels_measured())

    def measure_labels(self) -> dict:
        """Fill in the store's label counters so :meth:`labels` can answer. This writes.

        A lattice that was never backfilled reports ``edge_labels_measured() is False`` and
        :meth:`labels` returns nothing -- which is the store saying it does not know, not that there
        are none. mantle's own ``backfill_edge_label_counters`` is what settles it, and this is a
        thin pass-through to it rather than a scan of the table written here.

        It is a separate method, and named for what it does, because a reader should not silently
        write to somebody else's store. Read a copy if the original matters, which is what
        ``research/`` does.
        """
        return dict(self._store.backfill_edge_label_counters())

    def count_by_label(self, label: str) -> int:
        """How many edges the store holds under one label, from its own counter."""
        return int(self._store.count_edges_by_label(str(label)) or 0)

    def statements(self, *, label: str | None = None, limit: int | None = None) -> list[tuple]:
        """``[(src, label, dst), ...]`` -- the store read as statements, through mantle's own API.

        **This is the method that stops a reader hand-rolling SQL.** A lattice is a file, and it is
        always tempting to open it and select from ``edge``; doing that reads a schema this package
        does not own, past whatever the store does about ordering, partial results and its own
        counters. Every row here comes from ``src_ids_by_label`` and ``edges_of``, which are the
        store's own way of being read.

        A ``label`` reads that one; ``None`` reads every label the store reports. ``limit`` caps
        the statements returned, and is applied per label so a store dominated by one relation
        does not return only that one.

        Note what a statement is here: a ``(src, label, dst)`` triple where the label sits in the
        middle, which is the shape :class:`entroptics_infer.wrapper.Engine` reads. The label is what
        the writer recorded; this package does not interpret it and no read consults its spelling.
        """
        wanted = [str(label)] if label is not None else self.labels()
        out: list[tuple] = []
        for one in wanted:
            seen = set()
            for src in self._store.src_ids_by_label(one):
                for row in self._store.edges_of(str(src), label=one, direction="out"):
                    st = (row["src"], one, row["dst"])
                    if st in seen:
                        continue
                    seen.add(st)
                    out.append(st)
                    if limit is not None and len(seen) >= limit:
                        break
                if limit is not None and len(seen) >= limit:
                    break
        return out

    def _artifacts(self):
        """The store's artifact side, opened on the same file, lazily and through mantle's API."""
        if getattr(self, "_artifact_store", None) is None:
            try:
                from mantle.db.vertex import LatticeArtifactStore
            except Exception as exc:                          # pragma: no cover - import-shape only
                raise StoreUnavailable(
                    "this build of agience-mantle has no artifact store, so what a vertex IS "
                    "cannot be asked of it. The edges are still readable."
                ) from exc
            self._artifact_store = LatticeArtifactStore(self._path, origin=self.origin)
        return self._artifact_store

    def kinds(self, vertices: Iterable[str]) -> dict:
        """``{vertex: content type}`` -- what each vertex is, as its writer recorded it.

        Asked through ``get_many`` rather than read off a column, for the same reason the edges are.
        A vertex missing from the answer is one the artifact side does not hold, and it is left out
        rather than given a type this package invented.

        No read in this package branches on a content type. It is here because a reader measuring
        somebody else's lattice needs to say which part of it a result is about -- a store that
        had a corpus imported into it is two stores, and reporting a figure over both without
        saying so would be reporting the import.
        """
        got = self._artifacts().get_many([str(v) for v in vertices])
        return {k: v.get("content_type") or v.get("ct") for k, v in got.items() if v}

    def of_kind(self, content_type: str, *, limit: int | None = None) -> list[str]:
        """Every artifact of one content type, as the store lists them."""
        rows = self._artifacts().list_artifacts(content_type=str(content_type), limit=limit)
        return [r["id"] for r in rows]

    def close(self) -> None:
        """Release the underlying file handle.

        Needed wherever the lattice lives somewhere that will be removed — a temporary directory on
        Windows cannot be deleted while the file is open, and an experiment that leaves handles
        behind fails in its teardown rather than in its measurement, which is a confusing place to
        discover it.
        """
        conn = getattr(self._store, "db", None)
        closer = getattr(conn, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "Lattice":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError(
            "a lattice is not enumerable through this seam. A corpus-wide read would need a scan "
            "the store does not owe us, and every read in this program starts from a vertex."
        )


def open_lattice(path: str, *, origin: str) -> Lattice:
    """Open (or create) a lattice file.

    ``origin`` has no default. mantle's own constructor refuses a generated one because it would
    change on restart and fork the node's proper time, and this seam does not paper over that.
    """
    return Lattice(path, origin=origin)
