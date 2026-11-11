"""Every figure quoted in prose, checked against the results it claims to come from.

Why this exists
---------------
[`verify.py`](verify.py) re-runs the experiments and checks the figures registered in its own
table. That covers the claims someone thought to register. It does not cover the far larger number
of figures that appear in a sentence — "the edge is +0.067", "12 of 132 pairs" — and those are
where a figure is most likely to be wrong.

A figure in prose can go stale in a way a table cannot: the experiment is re-run, its JSON changes,
`verify.py` is updated, and a sentence three sections away still quotes the old number. Nothing
fails. The claim is simply wrong from then on.

What this does
--------------
It reads every number written in a document, and asks whether that number appears anywhere in
any experiment's results. A figure that appears in no result is not necessarily wrong — it may be a
design parameter, a count of sections, a year — so this reports rather than asserts, and the
judgment stays with a person.

Two strengths of check, and why there are two
----------------------------------------------
**Where a line names an experiment** — every row of `CLAIMS.md` does, and much of the paper's prose
does — its figures are checked against *that* experiment's results. This is the sharp check, and it
catches the case the loose one cannot: a figure that is stale for the claim it is attached to while
still appearing somewhere in the twenty-odd other result files. A ledger row can have every figure
in it stale that way, and membership in the whole set lets all of them through.

**Where a line names no experiment**, membership in the whole result set is what remains. It is
weaker but sound: a number that appears in none of it is a number nothing measured. Guessing an
owner by proximity would produce confident nonsense, so it is not attempted.

Run
---
    python research/prose_check.py
    python research/prose_check.py --show-unmatched      # every suspect, not just the summary
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

#: Documents whose prose makes claims.
DOCUMENTS = ("research/PAPER.md", "research/CLAIMS.md", "README.md")

#: A number as it appears in a claim: optionally signed, with a decimal point. Bare integers are
#: excluded because almost every integer in a document is a section number, a year, a count of files
#: or a figure in a URL, and flagging them would bury the signal this is looking for.
NUMBER = re.compile(r"(?<![\w.])([+-]?\d+\.\d+)(?![\w.])")

#: A **bolded** integer is the exception. In these documents bold is how a measured count is written
#: -- "**134** registered figures", "**12 of 132** pairs" -- so it is a claim even without a decimal
#: point. Without this, a stale count of experiments passes unchecked.
BOLD_INT = re.compile(r"\*\*([+-]?\d+)\*\*")

#: And a **bolded percentage**, which is how a measured share is written here -- "**97%** of those
#: kept", "**+37%**". Neither pattern above sees one: `NUMBER` wants a decimal point and `BOLD_INT`
#: wants the bold to hold nothing but digits, so `**97%**` matches neither.
#:
#: Most figures written that way are shares measured about the reader. Planting a wrong figure and
#: watching whether the gate refuses it is how a gate like this one is tested.
#:
#: The sign is kept out of the capture: a record holds the share, not its direction, so `+37%` and
#: `37%` are the same claim about the same measurement.
BOLD_PCT = re.compile(r"\*\*[+-]?(\d+(?:\.\d+)?)%\*\*")

#: Lines that are not claims: headings, shell commands and fenced code.
#:
#: Filenames are not a reason to skip a line. Skipping anything containing `.py` would silently drop
#: most of `CLAIMS.md` — every ledger row links to the script it cites, so the most claim-dense
#: document in the repo would go almost entirely unchecked. Link targets and paths are stripped from
#: the line instead, so their digits cannot be mistaken for figures.
SKIP_LINE = re.compile(r"^\s*(#|python|pip|sh|\$|>|```)|https?://")

#: A markdown link target, an inline path, or a code span holding a filename: stripped, not skipped.
NOT_A_FIGURE = re.compile(
    r"\]\([^)]*\)|`[^`]*\.(?:py|json|md|lean|csv)[^`]*`|\b[\w./-]+\.(?:py|json|md|lean|csv)\b")

#: A ledger row opens with its own claim number — `| 1.1 | ...`. That is an address, like a section
#: reference, and stripping it is the difference between reading the ledger and drowning in it.
CLAIM_ID = re.compile(r"^\s*\|\s*\d+\.\d+[a-z]?\s*\|")

#: A cross-reference to a section, not a measurement. `§9.6` is an address.
SECTION_REF = re.compile(r"[§S]\s*\d+\.\d+")

#: A ledger row cited in prose -- "claims 7.2, 7.2a-c and 7.3" -- is an address, and is read here
#: as a figure. It is measured and left alone: applying the gate's own stripping first, the four
#: documents carry **2** such tokens between them, because they otherwise write addresses with a
#: section mark.
#:
#: No pattern strips it. `\bclaims?\s+(N.N...)+` also strips "the paper claims 0.85 accuracy" and
#: "it claims 1.000 precision", which are figures -- so it would close two tokens worth of address
#: and open a silent hole for every figure in a sentence using that verb. A fix that hides real
#: figures is worse than the defect it removes.

#: The experiment a line names, if it names one. Every `CLAIMS.md` row does, by its own rule.
NAMES_EXP = re.compile(r"\bexp(\d{2})\b")

#: A row may cite a script by name rather than by an `expNN` tag — the reader's tests are named for
#: what they ask. Those citations appear only as link targets and inline paths, which `NOT_A_FIGURE`
#: strips, so the name has to be read from the raw line before stripping or the row falls back to
#: the pooled set. Pooled matching is the weak check this file exists to replace: a
#: figure usually appears somewhere among thirty-odd result files whether or not the cited script
#: produced it.
NAMES_SCRIPT = re.compile(r"\b(test_[a-z0-9_]+)\.py\b")

#: **A paper that stands alone names no script in its body**, so the line-by-line citation above
#: finds nothing there and every figure would fall back to the pooled set. Such a paper carries the
#: mapping in one place instead: a table near its end whose rows open with the section a result is
#: in and name the check behind it -- ``| §4.2 the bare operator | ...test_x.py |``. Every line under
#: that section is then checked against the records of the checks its section maps to, or its
#: parent section's. A section mapping to several checks is checked against their records together:
#: still that section's own checks, and not the whole result set. A row names a check either by its
#: script, as above, or by its record -- ``results/exp42.json`` -- since an experiment's script name
#: carries a suffix after its id and its record does not.
SECTION_ROW = re.compile(r"^\|\s*§\s*(\d+(?:\.\d+)*)\b")
HEADING = re.compile(r"^(#{2,4})\s+(.*)$")
NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")

#: **Documents written with bold in headings only.** Their measured shares and counts are plain
#: text, so ``BOLD_PCT`` and ``BOLD_INT`` see none of them. In these every percentage is a figure,
#: and so is every integer of two or more digits that is not an address, a year or part of a
#: grouped number. Inline mathematics is not read, and neither is the references list, which
#: carries years and volume numbers.
PLAIN = frozenset({"research/PAPER.md", "README.md"})
PLAIN_PCT = re.compile(r"(?<![\w.])[+-]?(\d+(?:\.\d+)?)%")
PLAIN_INT = re.compile(r"(?<![\w.,$^{_-])(\d{2,})(?![\w.,%])")
YEAR = re.compile(r"^(19|20)\d\d$")
MATH = re.compile(r"\$[^$]*\$")


def _numbers_in(obj, out: set):
    """Every float anywhere in a results document, at the precision it would be quoted at."""
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        v = float(obj)
        # A figure is quoted rounded, so record the roundings a writer would use -- and record the
        # precision with it. Matching "93.3" against a value that merely rounds to 93 at zero places
        # is how a coincidence passes for a citation.
        # places 0 lets a bolded integer match an integral value; the pairing is on
        # (precision, text), so a token written to one decimal still only matches a
        # value recorded at one decimal, and the coincidence this guards against stays shut
        for places in (0, 1, 2, 3, 4):
            # At zero places a value is only allowed to back a bare integer when it is one.
            # Rounding every float to zero places would let any value from 8.5 to 9.4 stand behind
            # a claim of "9", and a results file holding thirty floats backs most small integers by
            # coincidence. Counts are exactly what the reader's rows are made of.
            for scaled in (v, v * 100):
                # a results file may hold a NaN or an infinity; neither backs any written figure
                if not math.isfinite(scaled):
                    continue
                if places == 0 and scaled != int(scaled):
                    continue
                out.add((places, f"{scaled:.{places}f}"))
                out.add((places, f"{-scaled:.{places}f}"))

        # A percentage is a different kind of claim from a count, and needs a different rule.
        # "82%" is how a share of 0.8244 is written -- rounding is the notation, not a loss -- while
        # a bolded count of "9" must not be backed by 8.7. The strict rule above is right for the
        # count and refuses the share, so the share is recorded separately, under its own key, and
        # only a token written with a % sign can reach it.
        if math.isfinite(v):
            for places in (0, 1, 2):
                out.add(("%", places, f"{abs(v) * 100:.{places}f}"))
                out.add(("%", places, f"{abs(v):.{places}f}"))


def measured() -> tuple:
    """Every number each experiment produced, per experiment and pooled."""
    results = HERE / "results"
    if not results.is_dir():
        raise SystemExit(f"{results} does not exist; run the experiments before checking prose")
    per: dict = {}
    pooled: set = set()
    for path in sorted(results.glob("*.json")):
        here: set = set()
        _numbers_in(json.loads(path.read_text(encoding="utf-8")), here)
        per[path.stem] = here
        pooled |= here
    return per, pooled


def _normalise_percent(token: str) -> set:
    """A bolded percentage, at the precision it was written to, against the percent key."""
    places = len(token.split(".")[-1]) if "." in token else 0
    return {("%", places, f"{abs(float(token)):.{places}f}")}


def _normalise(token: str) -> set:
    """The figure as written, at the precision it was written to.

    The precision is part of the claim. A sentence saying "93.3" is asserting three significant
    figures, and a result that rounds to 93 does not support it.
    """
    written = token.split(".")[-1]
    places = len(written) if "." in token else 0
    v = float(token)
    return {(places, f"{v:.{places}f}")}


def scan(show_unmatched: bool) -> int:
    per, pooled = measured()
    total = matched = 0
    scoped = scoped_bad = 0
    unmatched: list = []

    for doc in DOCUMENTS:
        path = ROOT / doc
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").splitlines()
        # the section -> checks table, if the document carries one
        sections: dict = {}
        for line in text:
            m = SECTION_ROW.match(line)
            if m:
                sections.setdefault(m.group(1), set()).update(
                    n for n in NAMES_SCRIPT.findall(line) + [f"exp{e}" for e in NAMES_EXP.findall(line)]
                    if n in per)
        here, reading = None, True
        for lineno, line in enumerate(text, start=1):
            h = HEADING.match(line)
            if h:
                title = h.group(2).strip()
                n = NUMBERED.match(title)
                if n:
                    here = n.group(1)
                elif len(h.group(1)) == 2:
                    here = None
                reading = title.lower() != "references"
                continue
            if not reading or SKIP_LINE.search(line):
                continue
            stripped = CLAIM_ID.sub("| ", line)
            stripped = NOT_A_FIGURE.sub(" ", stripped)
            stripped = SECTION_REF.sub(" ", stripped)
            # a line naming one experiment is checked against that experiment alone; naming several
            # leaves it ambiguous, so it falls back to the pooled set rather than guessing
            named = {f"exp{n}" for n in NAMES_EXP.findall(stripped)}
            # read from the raw line: the citation is a link target, and stripping removes it
            named |= {n for n in NAMES_SCRIPT.findall(line) if n in per}
            owner = per.get(next(iter(named))) if len(named) == 1 else None
            if owner is None and not named and sections:
                key = here
                while key is not None and key not in sections:
                    key = key.rsplit(".", 1)[0] if "." in key else None
                if key is not None and sections[key]:
                    named = {f"§{key}"}
                    owner = set().union(*(per[s] for s in sections[key]))
            have = owner if owner is not None else pooled
            plain = NUMBER.findall(stripped) + BOLD_INT.findall(stripped)
            percents = BOLD_PCT.findall(stripped)
            if doc in PLAIN:
                bare = MATH.sub(" ", stripped)
                percents = PLAIN_PCT.findall(bare)
                without = PLAIN_PCT.sub(" ", bare)
                plain = NUMBER.findall(without) + [
                    n for n in PLAIN_INT.findall(without) if not YEAR.match(n)]
            for token, kind in [(t, "n") for t in plain] + [(t, "%") for t in percents]:
                total += 1
                if owner is not None:
                    scoped += 1
                wanted = _normalise_percent(token) if kind == "%" else _normalise(token)
                if wanted & have:
                    matched += 1
                else:
                    unmatched.append((doc, lineno, token, stripped.strip(),
                                      next(iter(named)) if owner is not None else None))
                    if owner is not None:
                        scoped_bad += 1

    print(f"{total} figures quoted across {len(DOCUMENTS)} documents")
    print(f"{matched} appear in some experiment's results")
    print(f"{len(unmatched)} do not\n")

    print(f"of those, {scoped} are on a line naming one experiment and checked against it alone; "
          f"{scoped_bad} of those do not appear in it\n")

    by_doc: dict = {}
    for doc, lineno, token, line, owner in unmatched:
        by_doc.setdefault(doc, []).append((lineno, token, line, owner))
    for doc in sorted(by_doc):
        print(f"  {doc}: {len(by_doc[doc])}")
    if show_unmatched:
        print()
        for doc in sorted(by_doc):
            print(f"\n=== {doc} ===")
            for lineno, token, line, owner in by_doc[doc]:
                # the documents are UTF-8 and a console may not be, so the excerpt is made safe
                safe = line[:92].encode("ascii", "replace").decode("ascii")
                tag = f"[{owner}]" if owner else "[pooled]"
                print(f"  {lineno:>5}  {token:>9}  {tag:>9}  {safe}")
    else:
        print("\n  --show-unmatched lists them")

    print("\nA figure appearing in no result is not automatically wrong: declared design parameters")
    print("(a false-alarm level, a noise amplitude, a version number) are written down rather than")
    print("measured. What it means is that nothing in the data backs it up, which is the question")
    print("worth asking of every number in a claim.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show-unmatched", action="store_true",
                    help="list every figure that appears in no result")
    sys.exit(scan(ap.parse_args().show_unmatched))
