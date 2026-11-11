"""What numbers does the answer path still carry that nothing derived?

The claim this checks
-----------------------
The paper claims the derivation costs no fitted constant, so the thing that would
falsify the program is any fitted constant in the answer path. The paper carries one admitted exception — ``far``, the false-alarm level, argued to
belong to the reader by Neyman-Pearson. Every other number is supposed to be read off the store.

**A claim like that has to be checked by enumeration, not by memory.** A constant enters a read path
quietly: in a default argument nobody looks at again, in a tolerance written while debugging, in a
threshold that seems obvious at the time. `below` is the example -- read as a truncation level it is
argued for nowhere and every prose figure is taken at 1e-2, and it is in fact an aperture with a
derivation.

How it is found
-----------------
**By walking the syntax tree, never by grepping.** A number reaches code as a literal in an
expression, a default in a signature, a keyword at a call site, a module-level name, a comparison
bound, or a value in a dict or tuple, and a pattern that catches five of those is a pattern that
reports a clean answer path with the sixth still in it.

What counts, and what does not
--------------------------------
A number is structural when it says what the arithmetic is rather than what to expect of the
data: ``0`` and ``1`` as identities, ``2`` in a square or a halving, indices and positions, the base
of a logarithm. Those are not constants in the sense that matters -- nothing about the corpus would
change them.

A number is a supplied constant when a different value would give a different answer and only
judgment chose it. Those are what this counts, and it prints every one with its file, line and the
code around it, so a reader disagrees with a listing rather than with an assurance.

The answer path is what a read touches
----------------------------------------
Only the package is walked, never `research/`. An experiment declares sweep points and sizes and
seeds, and those are the design of a measurement rather than part of a read: `research` exists to
choose numbers and report what they buy. The claim is about what a caller gets when they ask the
engine a question.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import ast
import pathlib
import sys

import record

HERE = pathlib.Path(__file__).resolve().parents[1]   # research/
PACKAGE = HERE.parent / "src" / "entroptics_infer"

#: Numbers that state what the arithmetic is, not what to expect of the data. A halving is a
#: halving on every corpus; a squaring is a squaring. Nothing here is a judgment about a store.
STRUCTURAL = frozenset({0, 1, 2, -1, 0.0, 1.0, 2.0, -1.0, 0.5})

#: Names whose value is a number but whose meaning is a position or a size read off the store.
#: Kept as a set rather than a pattern because the check is about what a name *is*.
NOT_A_LEVEL = frozenset({"POSITION", "_WHOLE"})


class Numbers(ast.NodeVisitor):
    """Every numeric literal in a module, with where it stands and what it stands in."""

    def __init__(self, path, source):
        self.path = path
        self.lines = source.splitlines()
        self.found = []
        self._where = []

    # -- context, so a number can be reported as what it is rather than as a value ---------------

    def visit_FunctionDef(self, node):
        self._where.append(("signature", node.name))
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
            self._look(default, "a default argument")
        self.generic_visit(node)
        self._where.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Compare(self, node):
        for comparator in node.comparators:
            self._look(comparator, "a comparison bound")
        self.generic_visit(node)

    def visit_Call(self, node):
        for keyword in node.keywords:
            if keyword.arg:
                self._look(keyword.value, f"passed as {keyword.arg}=")
        self.generic_visit(node)

    def visit_Assign(self, node):
        named = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if named and named[0] not in NOT_A_LEVEL:
            self._look(node.value, f"assigned to {named[0]}")
        self.generic_visit(node)

    def visit_Constant(self, node):
        self._look(node, "in an expression")

    # -- the one place a number is judged --------------------------------------------------------

    def _look(self, node, how):
        if not isinstance(node, ast.Constant):
            return
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if value in STRUCTURAL:
            return
        line = node.lineno
        self.found.append({
            "file": self.path.name, "line": line, "value": float(value), "how": how,
            "code": self.lines[line - 1].strip() if line <= len(self.lines) else ""})


def sweep(package):
    """Every non-structural number in the package, in one list."""
    out = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        seen = Numbers(path, source)
        seen.visit(ast.parse(source))
        out.extend(seen.found)
    # one report per line, keeping the most specific description of what the number is doing
    best = {}
    for got in out:
        key = (got["file"], got["line"], got["value"])
        if key not in best or best[key]["how"] == "in an expression":
            best[key] = got
    return [best[k] for k in sorted(best)]


def main() -> int:
    if not PACKAGE.is_dir():
        print(f"  No package at {PACKAGE}. Nothing to walk.")
        return 0

    found = sweep(PACKAGE)
    files = len(list(PACKAGE.rglob("*.py")))

    print("what numbers does the answer path still carry that nothing derived?\n")
    print(f"  {files} modules walked by syntax tree, "
          f"{len(STRUCTURAL)} structural values allowed through\n")

    if not found:
        print("  **Nothing.** No numeric literal in the package states an expectation about a "
              "store.")
    else:
        print(f"  {'file':<22}{'line':>6}{'value':>14}  what it is doing")
        print("  " + "-" * 92)
        for got in found:
            print(f"  {got['file']:<22}{got['line']:>6}{got['value']:>14.6g}  {got['how']}")
            print(f"  {'':<42}{got['code'][:70]}")
    print()

    record.figures(modules=files, structural_allowed=sorted(STRUCTURAL),
                   numbers_left=len(found),
                   by_file={f"{g['file']}:{g['line']}": g for g in found})

    if not found:
        print("  **The answer path carries no supplied constant.** Every number a read depends on "
              "is")
        print("  either an identity, an index, or read off the store at the moment of the question.")
    else:
        print(f"  **{len(found)}** number{'' if len(found) == 1 else 's'} in the answer path "
              f"came from judgment rather than from a store.")
        print(f"  Each is listed with the line it stands on, so it can be derived, argued for, or")
        print(f"  removed — and until one of those happens, the paper's claim is standing on "
              f"it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
