"""Count what the Lean development actually proves, so the figure in the paper has a producer.

Why this exists
-----------------
The theorem count was maintained by hand and drifted: one place in `PAPER.md` said thirty while two
others said thirty-four, and nothing could be run to settle it. A number in the ledger may cite only
a script, and this is that script.

What it counts, exactly
-------------------------
Declarations beginning a line with ``theorem`` or ``lemma``, per file, under ``Infer/``. An
attribute line such as ``@[simp]`` sits above its declaration and is not counted separately.
``def``, ``structure`` and ``abbrev`` are definitions rather than results and are not counted --
they are reported separately so the difference is visible rather than assumed.

It also reports what would make the count meaningless: any ``sorry``, and any ``axiom`` the
development declares of its own. Neither should ever be non-zero, and the count is printed beside
them so the two are never read apart.

This does not compile anything. Compiling needs Mathlib and a toolchain; see `check.sh`. A count
from this script says what is written, not that it builds -- run both.
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

RESULT = re.compile(r"^(theorem|lemma)\s+(\S+)")
DEFINITION = re.compile(r"^(def|structure|abbrev|inductive)\s+(\S+)")


def count(root: pathlib.Path):
    files = sorted(root.glob("Infer/*.lean"))
    rows, total, defs, sorries, axioms = [], 0, 0, 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        names = [m.group(2) for m in (RESULT.match(ln) for ln in text.splitlines()) if m]
        dnames = [m.group(2) for m in (DEFINITION.match(ln) for ln in text.splitlines()) if m]
        # `sorry` inside a word (`sorryAx`, or prose about not using one) still counts as a hit
        # worth showing, because a count that quietly excused some of them would be worthless
        s = len(re.findall(r"\bsorry\b", text))
        a = len([ln for ln in text.splitlines() if ln.startswith("axiom ")])
        rows.append({"file": f.name, "results": len(names), "definitions": len(dnames),
                     "sorry": s, "axiom": a, "names": names})
        total += len(names)
        defs += len(dnames)
        sorries += s
        axioms += a
    return rows, total, defs, sorries, axioms


def main(argv):
    root = pathlib.Path(argv[1]).resolve() if len(argv) > 1 else HERE
    rows, total, defs, sorries, axioms = count(root)
    if not rows:
        print(f"no Lean files under {root / 'Infer'}", file=sys.stderr)
        return 2

    # The count is written where the other results live, so `prose_check.py` can check the figure
    # in the paper against it. A theorem count quoted in prose and maintained by hand is exactly
    # the drift this repository checks for everywhere else -- one document said thirty while two
    # said thirty-four, and nothing could be run to settle it.
    dest = HERE.parents[0] / "results"
    if dest.is_dir():
        import json
        (dest / "lean.json").write_text(json.dumps(
            {"theorems": total, "definitions": defs, "sorry": sorries, "declared_axioms": axioms,
             "files": len(rows),
             "per_file": {r["file"]: r["results"] for r in rows},
             "names": {r["file"]: r["names"] for r in rows}},
            indent=2), encoding="utf-8")
    print(f"{'file':<24}{'theorems':>10}{'definitions':>13}{'sorry':>7}{'axiom':>7}")
    print("-" * 61)
    for r in rows:
        print(f"{r['file']:<24}{r['results']:>10}{r['definitions']:>13}"
              f"{r['sorry']:>7}{r['axiom']:>7}")
    print("-" * 61)
    print(f"{f'{len(rows)} files':<24}{total:>10}{defs:>13}{sorries:>7}{axioms:>7}")
    print()
    print(f"{total} theorems, {sorries} sorry, {axioms} declared axioms.")
    if sorries or axioms:
        print("A development with a `sorry` or a declared axiom proves less than its count suggests.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
