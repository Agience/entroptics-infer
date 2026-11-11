"""The ledger's structure, checked: unique addresses, resolvable citations, scripts that exist.

What the figure gates do not see
----------------------------------
`prose_check.py` checks every figure against the record of the script a row cites, and
`verify_records.py` checks every record against a fresh run of that script. Between them, a number
in this ledger cannot be stale.

Neither looks at the ledger as a structure. Two rows can carry the same address and both gates stay
green, because an address is not a figure. A stage numbered from an address another stage already
holds makes a citation such as `§44.2–§44.4` point at two different pairs of rows with nothing to
say which.

What is checked
-----------------
Four invariants, all mechanical, none needing judgment:

``unique``       no two rows share an address.
``one stage``    no address prefix is used by two stages. In this ledger a stage's number and its
                 rows' prefix are independent — Stage 43 carries 46.x, Stage 41 carries 44.x —
                 so the stage number cannot be taken as the prefix.
``resolvable``   every ``§N.N`` cited in a document is a row that exists.
``runnable``     every script a row cites is a file that exists, which is the ledger's own rule:
                 *a row may not cite a document*.
``counted``      where a document states how many of these tests there are, that is how many there
                 are. A sentence counting the files goes stale as soon as files are added and
                 the sentence is not. No figure gate can catch this --
                 a count of files is an integer, and the loose check asks only whether an integer
                 appears in some result somewhere, which every small integer does.

What is deliberately not checked
----------------------------------
Whether a row's claim is true, whether its figures match — those have their own gates — and whether
a stage's rows are numbered in sequence. A gap in a stage's numbering is not a defect: rows get
removed when what they claimed stops being reproduced, and closing the gap afterwards would
renumber addresses other documents cite.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent          # research/checks/
ROOT = HERE.parents[1]

LEDGER = HERE.parent / "CLAIMS.md"
CITING = ("research/PAPER.md", "README.md")

ROW = re.compile(r"^\|\s*(\d+\.\d+[a-z]?)\s*\|")
STAGE = re.compile(r"^##\s+(Stage\s+\S+.*)")
CITE = re.compile(r"§\s*(\d+\.\d+[a-z]?)")
SCRIPT = re.compile(r"\]\(([^)]+\.py)\)")

#: the line a document uses to say "run all of them", and the anchor for the count check
RUN_ALL = "for f in research/checks/test_*.py"


def rows():
    """``[(address, stage, line number, the row)]`` for every row in the ledger."""
    out, stage = [], "(before any stage)"
    for n, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), start=1):
        m = STAGE.match(line)
        if m:
            stage = m.group(1)
        m = ROW.match(line)
        if m:
            out.append((m.group(1), stage, n, line))
    return out


def main() -> int:
    found = rows()
    addresses = [a for a, _, _, _ in found]
    print(f"{len(found)} rows across {len({s for _, s, _, _ in found})} stages\n")

    faults: dict = collections.defaultdict(list)

    # unique addresses
    for address, count in collections.Counter(addresses).items():
        if count > 1:
            where = [(n, s) for a, s, n, _ in found if a == address]
            faults["two rows share an address"].append(
                f"{address} at " + ", ".join(f"line {n} ({s[:32]})" for n, s in where))

    # one stage per prefix
    by_prefix: dict = collections.defaultdict(set)
    for address, stage, _, _ in found:
        by_prefix[address.split(".")[0]].add(stage)
    for prefix, stages in sorted(by_prefix.items(), key=lambda kv: int(kv[0])):
        if len(stages) > 1:
            faults["one prefix, two stages"].append(
                f"{prefix}.x in " + " and ".join(sorted(s[:40] for s in stages)))

    # every citation resolves
    known = set(addresses)
    for doc in CITING:
        path = ROOT / doc
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for cited in CITE.findall(line):
                # a section reference and a claim reference look identical; only those written as
                # a claim of this ledger are checked, which is how the citing documents mark them
                if "CLAIMS" not in line:
                    continue
                if cited not in known:
                    faults["cited but not a row"].append(f"{doc}:{n} cites §{cited}")

    # the reproducing block counts the tests, so the count has to be the count
    tests = sorted(p.name for p in HERE.glob("test_*.py"))
    for doc in CITING:
        path = ROOT / doc
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if RUN_ALL not in text:
            continue
        # the paragraph that runs them all, and the one after it: where a count would be written
        start = text.index(RUN_ALL)
        block = text[start:start + 1200]
        if str(len(tests)) not in re.findall(r"\d+", block):
            faults["a count of the tests is not the count"].append(
                f"{doc} runs every test_*.py and states no count equal to the "
                f"{len(tests)} on disk")

    # every cited script exists
    for address, _, n, line in found:
        for script in SCRIPT.findall(line):
            if "://" in script:
                # a script in another public repository, cited by URL: not on this disk to check
                continue
            target = (LEDGER.parent / script).resolve()
            if not target.is_file():
                faults["a row cites a script that is not there"].append(
                    f"{address} (line {n}) cites {script}")

    import record
    record.figures(rows=len(found), stages=len({s for _, s, _, _ in found}),
                   **{name.replace(" ", "_"): len(items) for name, items in faults.items()},
                   shared_addresses=len(faults["two rows share an address"]),
                   shared_prefixes=len(faults["one prefix, two stages"]),
                   unresolved_citations=len(faults["cited but not a row"]),
                   missing_scripts=len(faults["a row cites a script that is not there"]),
                   miscounted_tests=len(faults["a count of the tests is not the count"]),
                   tests_on_disk=len(tests))

    print(f"  {'invariant':<44}{'faults':>8}")
    print("  " + "-" * 52)
    for name in ("two rows share an address", "one prefix, two stages",
                 "cited but not a row", "a row cites a script that is not there",
                 "a count of the tests is not the count"):
        print(f"  {name:<44}{len(faults[name]):>8}")
    print("  " + "-" * 52)
    print()

    if not any(faults.values()):
        print("  Every address is unique, every prefix belongs to one stage, every citation into")
        print("  this ledger resolves, every script a row names is a file that exists, and where a")
        print("  document counts these tests it counts the ones that are here.")
        return 0

    for name, items in faults.items():
        if not items:
            continue
        print(f"  **{name}**")
        for item in items[:12]:
            print(f"    {item}")
        if len(items) > 12:
            print(f"    ... and {len(items) - 12} more")
        print()
    print("  An address is not a figure, so neither `prose_check.py` nor `verify_records.py` will")
    print("  report any of this. That is what this file is for.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
