"""The reader's method, walked for numbers nobody derived.

Why this exists
-----------------
`tests/test_no_constants.py` walks the package for numeric literals and refuses every float and
every integer outside shape arithmetic. It covers ``src/entroptics_infer/``. The reader's method is
not in the package — it is `ask.py`, `place.py` and `operators.py` in this directory — so the
package gate never walks it. Two numbers of the kind this refuses:

* a cap such as ``[:3]`` in `ask.walk` on what each operator may offer. Load-bearing, and three is
  not even the best value of it: `test_the_cap_on_offers.py` has the table.
* a sample size such as ``draws=200`` in `ask._floor`, which does work -- at forty candidates the
  sampled floor reads 0.70 against a converged 0.75.

The method carries neither, and this is what keeps the next one out.

What counts as the method
---------------------------
**Whatever the reader imports**, followed transitively from `ask.py`. Listing the files by hand
would be the same mistake in a different place: a new module would join the method without joining
the gate. A module that `ask.py` cannot reach is not the method, however useful it is.

What is allowed, and why each
-------------------------------
``0`` ``1`` ``2``       shape and index arithmetic, as in the package gate: the additive and
                        multiplicative identities and the first two positions.
``0.0`` ``1.0``         the same two identities written as floats, which is what they are in a
                        norm guard and an initial carried weight.

Everything else must be registered below with a reason, and the reason has to say why the number
is not a choice about the data. A number that cannot be justified that way is a forcing, and the
answer is to derive it or remove it rather than to add a row here.

Display is not method
-----------------------
A literal inside ``if __name__ == "__main__"`` prints a table for a person to read. It decides
nothing, and the modules here all carry one.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]   # research/

#: The identities. Same argument as the package gate, plus their float spellings.
STRUCTURAL: frozenset = frozenset({0, 1, 2, 0.0, 1.0})

#: Every other literal the method may carry, with why it is not a choice about the data.
#: A row here is a claim, and an unregistered literal fails the gate.
REGISTERED: dict = {
    ("ask.py", 0.5): "the definition of a median, in the floor's closed form: the best of n has "
                     "P(best <= t) = F(t)**n, whose median is the quantile at 0.5 ** (1/n). Not a "
                     "level chosen for this store -- any other value would be a different "
                     "statistic, not a tuned version of this one.",
}


def method_files(entry: str = "ask.py") -> list:
    """Every module the reader reaches, followed transitively from its entry point."""
    seen, queue = {}, [entry]
    while queue:
        name = queue.pop()
        # the reader's modules live in research/lib/, beside the tools at research/ top
        path = next((d / name for d in (HERE / "lib", HERE) if (d / name).is_file()), HERE / name)
        if name in seen or not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen[name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue += [f"{a.name}.py" for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                queue.append(f"{node.module}.py")
    return sorted(seen.items())


def _display_lines(tree: ast.AST) -> set:
    """Line numbers inside an ``if __name__ == "__main__"`` block."""
    out: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"):
            for inner in ast.walk(node):
                line = getattr(inner, "lineno", None)
                if line is not None:
                    out.add(line)
    return out


def literals(name: str, tree: ast.AST) -> list:
    """``(line, value)`` for every numeric literal in the live path of this module."""
    display = _display_lines(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float, complex)):
            continue
        if v in STRUCTURAL or node.lineno in display:
            continue
        out.append((node.lineno, v))
    return sorted(out)


def main() -> int:
    files = method_files()
    print(f"the method is what `ask.py` reaches: {', '.join(n for n, _ in files)}\n")
    print(f"  {'module':<16}{'live literals':>15}{'registered':>12}{'unregistered':>14}")
    print("  " + "-" * 57)

    failures = []
    for name, tree in files:
        found = literals(name, tree)
        reg = [(ln, v) for ln, v in found if (name, v) in REGISTERED]
        bad = [(ln, v) for ln, v in found if (name, v) not in REGISTERED]
        failures += [(name, ln, v) for ln, v in bad]
        print(f"  {name:<16}{len(found):>15}{len(reg):>12}{len(bad):>14}")
    print("  " + "-" * 57)
    print()

    if REGISTERED:
        print("  registered, with the reason each is not a choice about the data:")
        for (name, v), why in REGISTERED.items():
            print(f"    {name} {v!r}")
            for line in (why[i:i + 84] for i in range(0, len(why), 84)):
                print(f"      {line}")
        print()

    if failures:
        print("  **Unregistered numbers in the method:**")
        for name, ln, v in failures:
            print(f"    {name}:{ln}  {v!r}")
        print()
        print("  Each is either derivable from the store, removable, or a forcing. Registering one")
        print("  without a reason that survives reading is how a threshold gets in.")
        return 1

    print("  No unregistered number in the method. Every literal is an identity, an index, inside")
    print("  a display block, or registered above with why it is not a choice about the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
