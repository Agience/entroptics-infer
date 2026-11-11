"""The store seam: a real lattice, and no relation type anywhere in it.

These tests write to a temporary file and never touch anything already in the workspace. The data
policy is not only about corpora — a test that opened an existing lattice would be reading data it
did not generate.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from entroptics_infer import store

mantle = pytest.importorskip("mantle", reason="agience-mantle not installed; the store seam is optional")

#: The two places that read labels, and why neither is a read of the corpus. Both are about the
#: store, not about any relation in it: ``survey`` measures whether a store still states a held-out
#: answer backwards or under another label, which it can only do by grouping statements by label;
#: ``Lattice.statements`` returns what the writer recorded, one label at a time when the caller
#: asks for one. Neither returns a kind, and no inference read calls them for one.
STORE_DIAGNOSTICS = {("survey.py", "survey"), ("store.py", "statements")}


@pytest.fixture
def lattice(tmp_path):
    return store.open_lattice(tmp_path / "probe.db", origin="test-origin")


def test_a_relation_round_trips_without_a_type(lattice):
    lattice.relate("a", "b", count=3)
    lattice.relate("a", "c", count=1)

    rels = lattice.relations("a")
    assert {r.dst for r in rels} == {"b", "c"}
    assert sorted(lattice.neighbours("a")) == ["b", "c"]
    assert lattice.count() == 2

    for r in rels:
        assert not hasattr(r, "kind"), (
            "a Relation grew a kind field. What kind a relation is must be measured from the "
            "records at either end, never written down by whoever created the edge"
        )


def test_observations_survive_but_carry_no_kind(lattice):
    lattice.relate("x", "y", count=7, position=2)
    (r,) = lattice.relations("x")
    assert r.props == {"count": 7, "position": 2}


def test_the_label_carries_no_information():
    """No read in the package may branch on an edge label.

    The seam writes one label because the column requires one. The moment a read starts comparing
    against it, that single label has become a vocabulary of size one and the forcing is back. This
    walks the package for any comparison or branch involving the label.
    """
    pkg = pathlib.Path(store.__file__).parent
    offenders = []
    seen_exempt = set()
    for path in sorted(pkg.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and (path.name, fn.name) in STORE_DIAGNOSTICS:
                seen_exempt.add((path.name, fn.name))
                skip |= {id(n) for n in ast.walk(fn)}
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if isinstance(node, ast.Compare):
                names = {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                } | {
                    n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                }
                if {"UNTYPED", "label"} & names:
                    offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value == "label":
                    offenders.append(f"{path.name}:{node.lineno} reads the label column")
    assert not offenders, (
        "the package reads or branches on an edge label:\n  " + "\n  ".join(offenders)
    )
    assert seen_exempt == STORE_DIAGNOSTICS, (
        f"a store diagnostic named in STORE_DIAGNOSTICS no longer exists, so its exemption guards "
        f"nothing: {sorted(STORE_DIAGNOSTICS - seen_exempt)}"
    )


def test_a_missing_store_raises_rather_than_substituting(monkeypatch):
    """No silent in-memory fallback. A result measured on a substitute is not a result."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith("mantle"):
            raise ImportError("simulated: agience-mantle absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(store.StoreUnavailable):
        store.open_lattice("unused.db", origin="test-origin")


def test_the_lattice_is_not_enumerable_through_this_seam(lattice):
    with pytest.raises(NotImplementedError):
        iter(lattice)
