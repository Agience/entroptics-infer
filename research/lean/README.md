# The machine-checked core

What is formalized here is the part of the method that is a **mathematical identity** rather than a
measurement: the statistic the engine scores with, the construction of the operator, and the two
bounds the reader's refusal and reach rest on. Everything else in this program is measured, and
measurements live in `research/CLAIMS.md` with the script that produced them. These do not need a
script — they are either true or not, and Lean says which.

`#print axioms` on every theorem below returns only `propext`, `Classical.choice` and `Quot.sound`,
which are Lean's own. **No `sorry`, no `admit`, and no axiom of ours.** `axioms.lean` is the file
that asks, and it is the thing to run first if you do not want to read the proofs.

## What is proved

### `Infer/Carry.lean` — the operator is a sum, and it never has to be built

| theorem | what it says |
|---|---|
| `carry_identity` | `u ⬝ (AᵀB) = (A ⬝ u) ⬝ B`. Carrying a location through a relation does not require forming the operator. The left side builds a `basis × basis` matrix; the right side never does. |
| `operator_append` | a statement arriving adds **one rank-one term** and changes nothing already there. This is the whole of learning here: no retraining, no update rule, no rate. |
| `operator_append_sub` | and withdrawing is the same fact backwards, so an edge is as cheap to remove as to add. |
| `operator_perm` | the operator does not depend on the **order** the statements arrived in. Whatever an ingest order buys, it does not buy it here — any effect of ordering lives in the coordinate, not in the weight. |
| `operator_nil` | a relation nothing uses does not act. |

`carry_identity` is the one the shipped engine stands on: `entroptics_infer.wrapper.Engine.carry`
computes the right-hand side, and
`research/checks/test_whether_the_shipped_engine_reads_what_the_research_one_reads.py` measures the
difference against the left-hand side on a real store. The theorem says the difference is zero; the
measurement says what floating point makes of that.

### `Infer/Permutation.lean`, `Infer/PairSum.lean`, `Infer/SecondMoment.lean` — the exact null

The score that replaces `softmax(q·k/√d)` is an alignment standardized by the **exact permutation
null** of re-pairing two records' rows. "Exact" means the moments are closed-form rather than
sampled, and these three files are why that word is allowed.

| theorem | what it says |
|---|---|
| `alignment_sum_over_perms_eq_zero` | on centred records the alignment has mean zero over all permutations — the null's first moment, with nothing estimated |
| `sum_perm_alignment_sq`, `sum_perm_S_sq` | the second moment in closed form |
| `sum_perm_offDiagonal_centred`, `weighted_collapse`, `n_mul_P_gap` | the pair-sum decomposition the second moment is assembled from |

No temperature appears in any of them, which is the point: a softmax needs `√d` supplied, and this
statistic needs nothing supplied.

### `Infer/RelationCount.lean` — how many kinds of relation a corpus can carry

`roleMaps_linearIndependent` and `two_mul_le_finrank_of_roles` bound the number of distinguishable
relation directions by the dimension available to them. The program *counts* relation kinds rather
than declaring them, and this is the ceiling that counting runs into.

### `Infer/Horizon.lean` — reach is finite, and not because anyone capped it

`below_floor_eventually`, `horizon_exists` and `constant_signal_never_falls`. A signal decaying past
the floor stops; a signal that does not decay does not. There is no hop cap anywhere in the method,
and this is the statement that one is not needed.

### `Infer/Arrival.lean` — what a read owes the store

`answer_congr`, `answer_arrive`, and `not_all_reads_are_stable` — the last of which is a **negative**
result and is here on purpose: not every read is stable under a store gaining statements, so
stability is a property to be checked per read rather than assumed of the design.

`answerable_arrive` and `arrival_keeps_every_answer` state the other half: an arrival never takes
a question away, and everything answerable before is answerable after with the same answer.
`arrival_can_add_an_answerable_question` exhibits an arrival that makes a new question answerable,
so the pair cannot be satisfied by a store that learns nothing.

## Checking it

Lean `v4.31.0` (`lean-toolchain`) against the Mathlib commit pinned in `lakefile.toml` and
`lake-manifest.json`. From this directory, with elan installed:

```sh
lake exe cache get        # Mathlib's prebuilt oleans
lake build                # the root and every file under Infer/
lake env lean axioms.lean # print what every headline theorem depends on
python count.py           # theorems, definitions, sorry and axioms, per file
```

`check.sh` compiles a single file against a Mathlib that is already built, without going through
lake — by default the one `lake exe cache get` puts under this directory's `.lake/`, or any other
built tree of the same Mathlib named by `MATHLIB`:

```sh
./check.sh Infer/Permutation.lean -o Infer/Permutation.olean
```
