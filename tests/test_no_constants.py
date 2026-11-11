"""The constant gate: no fitted number may enter the method, and the check is an AST walk.

Why an AST and not a grep
-------------------------
A regex over the source has to enumerate the spellings a constant can take — ``0.05``, ``5e-2``,
``.05``, ``1/20``, a value assembled at import time — and it will always miss one. The parser
enumerates them for us: every numeric literal in the file is an ``ast.Constant`` node whatever it
is spelled like, and a walk over the tree cannot be evaded by formatting.

What is banned, and why the line is where it is
-----------------------------------------------
**Every float literal is banned, without exception.** A float in this method could only be a
threshold, a tolerance, a temperature, a learning rate or a fitted coefficient — there is no
structural quantity in the package that is naturally fractional. This is the rule that matters and
it has no allowlist.

**Integer literals are allowed only in** ``STRUCTURAL``. These are shape and index arithmetic:
``W.ndim != 2`` asks whether a frame is a frame, ``x[0]`` takes a first element. They decide
nothing about the data. Any other integer — a head count, a hop cap, a maximum rank, a corpus size
— is exactly the kind of number this program exists to remove, so it is refused and must be
derived or passed in by the caller.

What this gate deliberately does not cover
------------------------------------------
``research/experiments/`` is outside the gate. An experiment must be free to state its own design:
how large a corpus it generates, which seeds it runs, what range it sweeps α over. Those are the
experiment's declared parameters, published with the result, and hiding them behind a derivation
would make the design *less* legible rather than more. The gate covers the method — everything
a claim rests on — and the experiments are read as design, not as method.

``far`` is not caught here because it is never written in the package: every function requires it
from the caller. ``test_far_is_never_defaulted`` is what enforces that, and the two tests together
are the whole of rule 2.
"""
from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "entroptics_infer"

#: Integers that are shape and index arithmetic rather than decisions about data.
#: ``0`` and ``1`` are the additive and multiplicative identities and the first two indices;
#: ``2`` is the dimensionality of the field the instrument reads, which is a property of the
#: instrument and not a choice made here.
STRUCTURAL = frozenset({0, 1, 2})


def _sources():
    return sorted(PACKAGE.rglob("*.py"))


def _offending(tree: ast.AST) -> list[tuple[int, object]]:
    """Every numeric literal in ``tree`` that is not structural, with its line number.

    ``bool`` is excluded before the numeric check: ``True`` is an ``ast.Constant`` whose value is
    an ``int`` subclass, and a flag is not a magic number.
    """
    out: list[tuple[int, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float, complex)):
            continue
        if isinstance(v, int) and v in STRUCTURAL:
            continue
        out.append((node.lineno, v))
    return out


def test_the_package_carries_no_fitted_number():
    """Rule 2, over the whole method surface."""
    found = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [(path.name, line, value) for line, value in _offending(tree)]
    assert not found, (
        "numeric literals outside the structural set are refused in the method — "
        "derive the value or require it from the caller:\n"
        + "\n".join(f"  {name}:{line}  {value!r}" for name, line, value in found)
    )


@pytest.mark.parametrize(
    "planted",
    [
        "TEMPERATURE = 0.05",                       # a bare float threshold
        "X = 5e-2",                                 # the same number, spelled to dodge a grep
        "X = .05",                                  # and again
        "MAX_HOPS = 3",                             # a non-structural integer: a cap
        "def f(x):\n    return x * 1.4826",         # a fitted coefficient inside an expression
        "def f(x):\n    return x if x > 0.5 else 0",  # a threshold in a comparison
    ],
)
def test_the_gate_rejects_a_planted_constant(planted):
    """The gate must be shown to reject, or its passing means nothing.

    A check that has never failed is not known to be a check at all: it may be walking an empty
    list, parsing the wrong tree, or comparing against itself. Each case here is a constant a
    real implementation would plausibly acquire, in a spelling chosen to defeat a text search.
    """
    tree = ast.parse(textwrap.dedent(planted))
    assert _offending(tree), f"the gate accepted a planted constant: {planted!r}"


def test_the_gate_accepts_structural_arithmetic():
    """And it must be shown to accept, or it is a gate that refuses everything.

    Every line here decides nothing about data: they ask about shape, take an index, or use an
    identity. If the gate rejected these it would be unusable and the allowlist would grow until
    it meant nothing.
    """
    tree = ast.parse(
        textwrap.dedent(
            """
            def f(W, xs):
                if W.ndim != 2:
                    raise ValueError
                first = xs[0]
                rest = xs[1:]
                last = xs[-1]
                return first, rest, last, W.shape[0] * W.shape[1]
            """
        )
    )
    assert not _offending(tree)


def test_far_is_never_defaulted():
    """No function in the method may supply its own false-alarm level.

    A default here would let a reading be taken without the level ever being named, and that
    reading would then travel as though it were parameter-free — which is precisely the failure
    the program is built to avoid. ``far`` is the reader's input and the caller is the reader.
    """
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            named = [a.arg for a in args.kwonlyargs]
            for arg, default in zip(named, args.kw_defaults):
                if arg == "far" and default is not None:
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
            positional = [a.arg for a in args.posonlyargs + args.args]
            padded = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
            for arg, default in zip(positional, padded):
                if arg == "far" and default is not None:
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, "far must be required, never defaulted:\n  " + "\n  ".join(offenders)
