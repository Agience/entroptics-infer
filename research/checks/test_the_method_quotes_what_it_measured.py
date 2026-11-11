"""Every figure the reader's own docstrings quote, checked against what the tests recorded.

The gap
---------
`prose_check.py` checks five documents. The method's docstrings are not among them, and they are
full of figures: `ask.py` alone quotes 25%, 9%, 3%, 26%, 82% and more, each one a claim about what
this reader does. Nothing else checks them.

They are exactly where a figure goes stale unnoticed, because a docstring is written beside the code
it describes and then left there while the code moves. A design justification can go stale the same
way -- citing, say, a walk from `salmon` to `paw` that the reader does not take.

What is checked, and what cannot be
-------------------------------------
Every number in a docstring, against every record any test wrote. A figure that appears in none of
them is a figure nothing measured.

**Only docstrings, not code.** A literal in an expression is not a claim -- `0.5 ** (1.0 / n)` quotes
nothing -- so the docstrings are taken from the syntax tree rather than by reading lines, and code
is never looked at. The constants gate is what covers literals in code, and it covers them under a
different rule.

**Pooled, not scoped.** `prose_check.py` checks a figure against the record of the script named on
its own line, and falls back to the whole set otherwise. In a docstring that sharp check never
fires: a paragraph states its figures in one sentence and names the script that produced them in
the next, so no line carries both. Scoping by paragraph was measured as a repair and is not one --
it adds 22 figures of coverage across all five documents and flags a paragraph that cites another
experiment for its *control* rather than its numbers, which is how these documents are written.

So this is the weaker check, and it is worth having because it is the only one on the method's
docstrings.

What it cannot catch
----------------------
A stale reason. "Scoring against the frontier walked from `salmon` to `paw`" cites no number and
passes every gate here while being false. That class is found by reading, and the mitigation is that
a justification should name the script that would show it -- a convention, not something this
enforces.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))
# and the prose gate, which is a tool at research/ top
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import ast
import pathlib

import prose_check as pc

HERE = pathlib.Path(__file__).resolve().parents[1]   # research/
SRC = HERE.parent / "src"

#: The method: what the reader imports, followed from ``ask.py``, and every module of the shipped
#: package, which is the method whether or not the reader reaches it.
ENTRIES = ("ask",) + tuple(f"entroptics_infer.{p.stem}"
                           for p in sorted((SRC / "entroptics_infer").glob("*.py")))

#: Where an imported name is looked for, in the order the reader's own path puts them.
ROOTS = (HERE / "lib", HERE / "checks", HERE, SRC)


def _module_path(dotted: str):
    """The file a module name resolves to under :data:`ROOTS`, or ``None`` if it is not here."""
    parts = dotted.split(".")
    for root in ROOTS:
        for path in (root.joinpath(*parts).with_suffix(".py"),
                     root.joinpath(*parts) / "__init__.py"):
            if path.is_file():
                return path
    return None


def method_files(entries: tuple = ENTRIES) -> list:
    """Every module the method reaches, followed transitively from its entry points.

    A module resolves under ``research/lib``, ``research/checks``, ``research/`` or ``src/``, and a
    package import (``from entroptics_infer import ingest``) also reaches each named submodule.
    Modules outside the repository -- numpy, the instrument -- are not followed.
    """
    seen, queue = {}, [_module_path(entry) for entry in entries]
    while queue:
        path = queue.pop()
        if path is None:
            continue
        name = path.relative_to(HERE.parent).as_posix()
        if name in seen:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen[name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue += [_module_path(a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                queue.append(_module_path(node.module))
                queue += [_module_path(f"{node.module}.{a.name}") for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                stem = base.relative_to(SRC).as_posix().replace("/", ".") \
                    if base.is_relative_to(SRC) else None
                if stem is None:
                    continue
                if node.module:
                    queue.append(_module_path(f"{stem}.{node.module}"))
                else:
                    queue += [_module_path(f"{stem}.{a.name}") for a in node.names]
    return sorted(seen.items())


def docstrings(tree) -> list:
    """``(line, text)`` for the module's docstring and every function's and class's."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                out.append((getattr(node, "lineno", 1), text))
    return out


def main() -> int:
    _, pooled = pc.measured()
    files = method_files()
    print(f"the method is what `ask.py` and `src/entroptics_infer` reach: "
          f"{', '.join(n for n, _ in files)}\n")
    print(f"  {'file':<44}{'figures':>9}{'backed':>8}{'unbacked':>10}")
    print("  " + "-" * 73)

    unbacked = []
    total = 0
    for name, tree in files:
        here = 0
        bad = 0
        for start, text in docstrings(tree):
            for offset, line in enumerate(text.splitlines()):
                if pc.SKIP_LINE.search(line):
                    continue
                stripped = pc.NOT_A_FIGURE.sub(" ", line)
                stripped = pc.SECTION_REF.sub(" ", stripped)
                # a percentage is a share, checked as one, and taken out before the plain numbers
                # so that "87.6%" is not also read as the number 87.6
                percents = pc.PLAIN_PCT.findall(stripped)
                without = pc.PLAIN_PCT.sub(" ", stripped)
                tokens = ([(t, pc._normalise(t)) for t in
                           pc.NUMBER.findall(without) + pc.BOLD_INT.findall(without)]
                          + [(t + "%", pc._normalise_percent(t)) for t in percents])
                for token, wanted in tokens:
                    here += 1
                    if not (wanted & pooled):
                        bad += 1
                        unbacked.append((name, start + offset, token, line.strip()[:62]))
        total += here
        print(f"  {name:<44}{here:>9}{here - bad:>8}{bad:>10}")
    print("  " + "-" * 73)
    print(f"  {'total':<44}{total:>9}")
    print()

    if not unbacked:
        print("  Every figure the method quotes about itself appears in something a test recorded.")
        print("  That is the weaker of the two checks -- see the docstring above for why the sharp")
        print("  one cannot fire here -- and it is the only check on the method's docstrings.")
        return 0

    print("  **Figures the method quotes that nothing recorded:**")
    for name, line, token, text in unbacked:
        print(f"    {name}:{line}  {token}")
        print(f"      {text}")
    print()
    print("  A figure in a docstring is a claim about what this code does. One that nothing")
    print("  measured is either stale or was never true, and both want the same fix: measure it")
    print("  or remove it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
