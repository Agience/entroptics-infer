"""Stage 36 — letting the read choose its own backing, and why that is not allowed.

What `exp42` left open
------------------------
`exp42` measured that a relation vertex should not be derived from every edge that cites it:
averaging suppresses noise and suppresses resolvable structure with it, so the best backing is
neither one edge nor all of them, and where it sits moves with how well the relation holds.

That is a sweep, and a sweep is not an inference engine. Something has to choose, and if the choice
is a number carried in from a sweep then the system has a fitted parameter in it after all, chosen
on data that will not be the data it runs on. The program does not allow that.

The rule that looked right
----------------------------
**The read already knows when an aggregate has collapsed**: `complete` returns nothing when nothing
resolves against the operator or when two candidates tie. That refusal is a property of the read,
not of the answer — it is available at the moment of asking with the truth unknown. So the backing
need not be chosen in advance: derive the relation at every backing the lattice allows, ask at each,
and let the refusals mark where the aggregate stopped being usable.

Two rules, neither containing a number:

``most``   — the answer from the largest backing that still resolves.
``agreed`` — the answer most of the resolving backings agree on.

On the noisy lesson both match or beat the best fixed backing at every noise level, and the best
fixed backing was chosen in hindsight with the answers in hand. That is the result this experiment
was built to report, and it is wrong.

Why it is wrong
-----------------
A rule that retries until something resolves is a search, and a refusal is calibrated for one
question. Asking twenty-three erodes the level it was calibrated at, and no amount of accuracy on
relations that are real will show it — every relation in the noisy lesson is a real one, merely
obscured, so a search that answers more is answering more of something that is there.

The control that finds it is a relation that is not one. `grade three` names a relation whose
edges join unrelated vertices, and asking for it by example is asking for something that is not
there. A single edge answers it 15 times in 60 — the read's own false-alarm level, doing its job.
Both rules answer it 46 and 47 times in 60.

So the finding is a negative, and it is the useful kind: the backing cannot be chosen by retrying
until the read stops refusing. Refusal is the instrument's verdict at a stated level and spending
it twenty-three times is spending it. `exp42`'s sweep is not replaced, and the question it leaves
open — how a relation vertex chooses its own backing without a supplied number — stays open, with
one more answer ruled out.

Both arms are reported, because the first without the second is the mistake this records.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parents[2] / "src")]

import curriculum as C                                              # noqa: E402
from entroptics_infer import complete_with, engine, operator_over    # noqa: E402

FAR = 0.05
SEED = 5
SMOKE = os.environ.get("EXP43_SMOKE") == "1"
N = 8 if SMOKE else 24
BLURS = (0.3, 0.85) if SMOKE else (0.0, 0.3, 0.6, 0.75, 0.85, 0.9)
QUESTIONS = 20 if SMOKE else 100


def one_question(lesson, src, kind, cands, pool, rng):
    """Ask at every backing the pool allows. Returns each backing's answer, or None for a refusal."""
    order = [pool[int(i)] for i in rng.permutation(len(pool))]
    out = {}
    for k in range(1, len(order) + 1):
        op = operator_over(lesson.records, order[:k])
        res = complete_with(lesson.records, operator=op, source=src,
                            candidates=cands, far=FAR)
        out[k] = cands[res.key] if res.key is not None else None
    return out


def most(answers):
    """The largest backing that still resolves."""
    live = [k for k, v in answers.items() if v is not None]
    return answers[max(live)] if live else None


def most_backing(answers):
    """Which backing `most` landed on, so the rule's choice can be read rather than assumed."""
    live = [k for k, v in answers.items() if v is not None]
    return max(live) if live else None


def agreed(answers):
    """What most of the resolving backings say; a tie is a refusal, as everywhere else here."""
    live = [v for v in answers.values() if v is not None]
    if not live:
        return None
    counts = Counter(live).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def run_level(lesson, rng):
    by = lesson.named_edges()
    kinds = sorted(by)
    targets = {}
    for k, pairs in by.items():
        for s, d in pairs:
            targets.setdefault(s, {})[k] = d
    sources = [s for s, m in targets.items() if len(m) >= 2]

    per_backing: dict[int, dict] = {}
    rules = {"most": {"asked": 0, "answered": 0, "right": 0},
             "agreed": {"asked": 0, "answered": 0, "right": 0}}
    landed = []
    for _ in range(QUESTIONS):
        src = sources[int(rng.integers(len(sources)))]
        kind = kinds[int(rng.integers(len(kinds)))]
        truth = targets[src][kind]
        cands = [targets[src][k] for k in sorted(targets[src])]
        pool = [(a, b) for a, b in by[kind] if a != src and b not in cands]
        if len(cands) < 2 or not pool:
            continue
        answers = one_question(lesson, src, kind, cands, pool, rng)

        for k, v in answers.items():
            cell = per_backing.setdefault(k, {"asked": 0, "answered": 0, "right": 0})
            cell["asked"] += 1
            cell["answered"] += int(v is not None)
            cell["right"] += int(v == truth)
        k_used = most_backing(answers)
        if k_used is not None:
            landed.append(k_used)
        for name, pick in (("most", most), ("agreed", agreed)):
            v = pick(answers)
            rules[name]["asked"] += 1
            rules[name]["answered"] += int(v is not None)
            rules[name]["right"] += int(v == truth)

    def scored(cell):
        return {**cell,
                "answered_share": cell["answered"] / cell["asked"] if cell["asked"] else 0.0,
                "accuracy": (cell["right"] / cell["answered"]) if cell["answered"]
                else float("nan"),
                "share_of_all_asked": cell["right"] / cell["asked"] if cell["asked"] else 0.0}

    fixed = {k: scored(v) for k, v in sorted(per_backing.items())}
    best_k = max(fixed, key=lambda k: fixed[k]["share_of_all_asked"])
    return {"fixed": fixed,
            "best_fixed_backing_in_hindsight": best_k,
            "best_fixed": fixed[best_k],
            "rules": {n: scored(v) for n, v in rules.items()},
            "backing_most_landed_on": {
                "median": float(np.median(landed)) if landed else float("nan"),
                "mean": float(np.mean(landed)) if landed else float("nan"),
                "lowest": int(min(landed)) if landed else None,
                "highest": int(max(landed)) if landed else None,
                "share_at_the_very_top": (sum(1 for k in landed if k == max(fixed)) / len(landed))
                if landed else float("nan")}}


def a_relation_that_is_not_one(rng, questions):
    """The control. A name whose edges join unrelated vertices, asked for by example.

    A single edge answers this at the read's own false-alarm level and no more, which is the
    instrument working. Any rule that answers it more often has bought its extra answers by
    spending that level, and the noisy lesson cannot show it: every relation there is real.
    """
    lesson = C.grade_three(rng=np.random.default_rng(SEED), n=N)
    by = lesson.named_edges()
    targets = {}
    for k, pairs in by.items():
        for s, d in pairs:
            targets.setdefault(s, {})[k] = d
    sources = sorted(targets)

    out = {}
    for kind in sorted(by):
        tal = {"asked": 0, "one_edge": 0, "most": 0, "agreed": 0}
        for _ in range(questions):
            src = sources[int(rng.integers(len(sources)))]
            cands = [targets[src][k] for k in sorted(targets[src])]
            pool = [(a, b) for a, b in by[kind] if a != src and b not in cands]
            if len(cands) < 2 or not pool:
                continue
            answers = one_question(lesson, src, kind, cands, pool, rng)
            tal["asked"] += 1
            tal["one_edge"] += int(answers[1] is not None)
            tal["most"] += int(most(answers) is not None)
            tal["agreed"] += int(agreed(answers) is not None)
        if tal["asked"]:
            out[kind] = {**tal,
                         "one_edge_share": tal["one_edge"] / tal["asked"],
                         "most_share": tal["most"] / tal["asked"],
                         "agreed_share": tal["agreed"] / tal["asked"]}
    return out


def run():
    started = time.time()
    levels = []
    print(f"{'relation holds':>15}{'rule':>10}{'answered':>10}{'accuracy':>10}"
          f"{'right of all':>14}{'vs best fixed':>15}")
    print("-" * 76)
    for blur in BLURS:
        lesson = C.grade_five(rng=np.random.default_rng(SEED), n=N, blur=blur)
        r = run_level(lesson, np.random.default_rng(SEED + 1))
        r["blur"] = blur
        r["holds"] = 1 - blur
        levels.append(r)
        bf = r["best_fixed"]["share_of_all_asked"]
        print(f"{1 - blur:>13.0%}  {'best fixed':>10}"
              f"{r['best_fixed']['answered_share']:>10.3f}{r['best_fixed']['accuracy']:>10.3f}"
              f"{bf:>14.3f}"
              f"{'  (at ' + str(r['best_fixed_backing_in_hindsight']) + ' edges)':>15}")
        for name in ("most", "agreed"):
            c = r["rules"][name]
            print(f"{'':>15}{name:>10}{c['answered_share']:>10.3f}{c['accuracy']:>10.3f}"
                  f"{c['share_of_all_asked']:>14.3f}"
                  f"{c['share_of_all_asked'] - bf:>+15.3f}")
        L = r["backing_most_landed_on"]
        print(f"{'':>15}{'landed on':>10}  median {L['median']:.0f} edges, "
              f"range {L['lowest']}-{L['highest']}, "
              f"at the very top {L['share_at_the_very_top']:.0%} of the time")
        print()

    def beats(name):
        return sum(1 for lv in levels
                   if lv["rules"][name]["share_of_all_asked"]
                   >= lv["best_fixed"]["share_of_all_asked"])

    def worst(name):
        return min(lv["rules"][name]["share_of_all_asked"]
                   - lv["best_fixed"]["share_of_all_asked"] for lv in levels)

    control = a_relation_that_is_not_one(np.random.default_rng(SEED + 3),
                                        QUESTIONS if not SMOKE else 20)
    print("=" * 76)
    print("  THE CONTROL — the same rules, asked for a relation that is not one")
    print()
    print(f"  {'relation':>10}{'asked':>8}{'one edge':>10}{'most':>8}{'agreed':>9}")
    for kind, c in sorted(control.items()):
        print(f"  {kind:>10}{c['asked']:>8}{c['one_edge_share']:>10.3f}"
              f"{c['most_share']:>8.3f}{c['agreed_share']:>9.3f}")
    hollow = min(control.values(), key=lambda c: c["one_edge_share"])
    hollow_name = min(control, key=lambda k: control[k]["one_edge_share"])
    print()
    print(f"  '{hollow_name}' is the name with nothing behind it. One edge answers it"
          f" {hollow['one_edge_share']:.3f} of the time —")
    print(f"  the read's own level. The rules answer it {hollow['most_share']:.3f} and"
          f" {hollow['agreed_share']:.3f}.")
    print("  Retrying until something resolves spends the level the refusal was calibrated at,")
    print("  so neither rule is admissible however well it does where the relation is real.")
    print()

    out = {"design": {"far": FAR, "seed": SEED, "sources": N, "blurs": list(BLURS),
                      "questions": QUESTIONS, "smoke": SMOKE},
           "levels": levels,
           "control": control,
           "the_name_with_nothing_behind_it": hollow_name,
           "hollow_answered_from_one_edge": hollow["one_edge_share"],
           "hollow_answered_by_most": hollow["most_share"],
           "hollow_answered_by_agreed": hollow["agreed_share"],
           "either_rule_is_admissible": bool(
               hollow["most_share"] <= hollow["one_edge_share"]
               or hollow["agreed_share"] <= hollow["one_edge_share"]),
           "noise_levels": len(levels),
           "levels_where_most_matches_or_beats_the_best_fixed": beats("most"),
           "levels_where_agreed_matches_or_beats_the_best_fixed": beats("agreed"),
           "worst_shortfall_of_most": worst("most"),
           "worst_shortfall_of_agreed": worst("agreed"),
           "lowest_accuracy_of_most": min(lv["rules"]["most"]["accuracy"] for lv in levels),
           "lowest_accuracy_of_agreed": min(lv["rules"]["agreed"]["accuracy"] for lv in levels),
           "median_backing_landed_on_by_level": [lv["backing_most_landed_on"]["median"]
                                                 for lv in levels],
           "share_at_the_very_top_by_level": [lv["backing_most_landed_on"]["share_at_the_very_top"]
                                              for lv in levels],
           "seconds": time.time() - started}
    print("=" * 76)
    print(f"  the best fixed backing was chosen in hindsight, knowing the answers; the rules")
    print(f"  were not told it and were not told the noise level.")
    print(f"    'largest that still resolves' matches or beats it at "
          f"{out['levels_where_most_matches_or_beats_the_best_fixed']} of {len(levels)} levels, "
          f"worst shortfall {out['worst_shortfall_of_most']:+.3f}")
    print(f"    'what most backings agree on'  matches or beats it at "
          f"{out['levels_where_agreed_matches_or_beats_the_best_fixed']} of {len(levels)} levels, "
          f"worst shortfall {out['worst_shortfall_of_agreed']:+.3f}")
    print(f"  lowest accuracy over every level: most {out['lowest_accuracy_of_most']:.3f}, "
          f"agreed {out['lowest_accuracy_of_agreed']:.3f}")
    print()
    print(f"  ADMISSIBLE: {out['either_rule_is_admissible']} — and every line above this one")
    print("  is what it looks like when a search is mistaken for an inference.")
    dest = HERE.parents[1] / "results"
    dest.mkdir(exist_ok=True)
    name = "exp43.smoke.json" if SMOKE else "exp43.json"
    (dest / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest / name} in {out['seconds']:.1f}s")
    return out


if __name__ == "__main__":
    run()
