"""What the generated stores grade, counted -- because it is not what the ontology grades.

Why this needs checking rather than stating
---------------------------------------------
The generated structures in this line answer most of their questions correctly at the denser
shares, and the ontology answers about a fifth of its own. Those look like the same measurement
disagreeing, and they are not the same measurement.

In a generated structure the truth is a class: a relation sends a group to a group, and a read
is right when what it names has the licensed value of the property. Every concept with that value is
an equally true answer. In the ontology the truth is a statement: `dog is_a mammal` names one
destination, and `bird` is wrong however similar it is.

So the generated figures measure whether the read recovers a class and cannot measure whether it
recovers an individual -- not because nobody tried, but because their ground truth has no
individuals in it. That is a limit on what every generated result in this stage can mean, and it
should be countable rather than asserted.

What is counted
-----------------
For every question a generated structure can put, how many concepts the grader would accept, and
the same for the ontology's own questions. Where a grader accepts several, a read is right for
naming any of them and the figure is a class-level figure; where it accepts one, the figure is
about the individual.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import collections

import numpy as np

import record

import ontology
from test_inference_on_a_partly_stated_structure import observed, structure

DRAWS = 3
SHARE = 0.4


def main() -> int:
    accepted, group_sizes = [], []
    for seed in range(DRAWS):
        rng = np.random.default_rng(7000 + seed)
        concepts, relations, group, sends = structure(rng)
        statements = observed(concepts, relations, group, sends, SHARE, rng)
        stated: dict = {}
        for a, r, b in statements:
            stated.setdefault((a, r), set()).add(b)
        members = collections.Counter(group.values())
        for a in concepts:
            for r in relations:
                unstated = [c for c in concepts if c not in stated.get((a, r), ()) and c != a]
                want = [c for c in unstated if group[c] == sends[r][group[a]]]
                if not want or len(want) == len(unstated):
                    continue
                accepted.append(len(want))
                group_sizes.append(members[sends[r][group[a]]])

    real = collections.defaultdict(set)
    for parts in ontology.STATEMENTS:
        if len(parts) == 3:
            real[(parts[0], parts[1])].add(parts[2])
    ontology_accepts = [len(v) for v in real.values()]

    print("how many concepts each grader would accept as a right answer\n")
    print(f"  {'grader':<22}{'questions':>11}{'least':>8}{'median':>9}{'most':>7}")
    print("  " + "-" * 57)
    print(f"  {'a generated store':<22}{len(accepted):>11}{min(accepted):>8}"
          f"{float(np.median(accepted)):>9.0f}{max(accepted):>7}")
    print(f"  {'the ontology':<22}{len(ontology_accepts):>11}{min(ontology_accepts):>8}"
          f"{float(np.median(ontology_accepts)):>9.0f}{max(ontology_accepts):>7}")
    print()

    one = sum(1 for n in ontology_accepts if n == 1) / len(ontology_accepts)
    # not "is the accepted set the whole group" -- it never is, because the destinations the
    # store already states are off the table, so it is whatever is left of the group. The
    # countable thing is whether more than one answer is accepted at all, which is the difference
    # between grading a class and grading a statement.
    several = sum(1 for n in accepted if n > 1) / len(accepted)
    record.figures(
        generated_questions=len(accepted),
        generated_accepts_median=float(np.median(accepted)),
        generated_accepts_least=min(accepted), generated_accepts_most=max(accepted),
        generated_accepts_more_than_one=several,
        ontology_questions=len(ontology_accepts),
        ontology_accepts_median=float(np.median(ontology_accepts)),
        ontology_accepts_most=max(ontology_accepts),
        ontology_accepts_exactly_one=one)

    print(f"  On the generated store {several:.0%} of questions accept more than one answer.")
    print(f"  On the ontology {one:.0%} accept exactly one.")
    print()
    print("  **The two are not the same measurement.** A generated grader accepts a class, so a")
    print("  read is right when it names any member — which is what a relation carries, and")
    print("  §47.86 shows the read has that completely. An ontology grader accepts a statement, so")
    print("  a read is right only when it names the individual, which §47.86 shows it does not.")
    print()
    print("  **So every inference figure on a generated store in this stage is a class-level")
    print("  figure**, and none of them says the method recovers an individual. Not because the")
    print("  question was ducked, but because a structure whose truth is a group has no")
    print("  individuals in it to recover. The ontology is where the identity question is asked,")
    print("  and there the answer is the one in §47.84.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
