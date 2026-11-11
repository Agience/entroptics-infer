import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.List.Lemmas
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

/-!
# The operator is a sum, and it never has to be built

The engine derives a relation's operator from the statements that use it: stack the placed sides
either way round and take the cross-covariance `M = AᵀB`. Three facts about that construction are
what make the engine work, and all three are consequences of `M` being a **sum over statements**
rather than a fitted object.

`carry_identity`
:   `u ⬝ (AᵀB) = (A ⬝ u) ⬝ B`. Carrying a location through a relation does not require forming the
    operator. The left side builds a `|basis| × |basis|` matrix; the right side never does, and the
    two are equal. This is what `Engine.carry` computes, and `research/` measures the difference
    between the two paths at the size floating point makes it.

`operator_append`
:   a statement arriving adds **one rank-one term** and changes nothing already there. This is the
    whole of learning in this system: no retraining, no update rule, no rate. The store gains a
    statement and the operator gains that statement's contribution, because the operator *is* the
    sum.

`operator_perm`
:   the operator does not depend on the order the statements arrived in. That one is worth stating
    because the programme is about to ask what a curriculum buys: whatever an ingest order does, it
    does **not** do it through the operator, so any measured effect of ordering has to live in the
    coordinate the statements are placed on.

Nothing here has a parameter, so there is nothing in these proofs to tune. They are about the
construction, not about any store.
-/

namespace Infer

open Finset

-- `Fintype` is required only where a sum over the basis is taken, so it is asked for there rather
-- than declared over the whole section. Nothing here needs decidable equality on the basis.
variable {κ : Type*}

/-- A statement as the operator sees it: the placed vertex before the relation, and the one after.

The engine reads roles from **position**, never from a name, so a statement reaches this point as an
ordered pair of coordinates and nothing recording what kind of relation produced it. -/
abbrev Sided (κ : Type*) := (κ → ℝ) × (κ → ℝ)

/-- `M = AᵀB`, entry by entry: the cross-covariance of the two sides over the statements.

Observed rather than fitted. There is no target, no residual and nothing iterated -- the entry at
`(d, e)` is a sum over the statements of the product of their two sides' coordinates. -/
def operator (S : List (Sided κ)) (d e : κ) : ℝ :=
  (S.map (fun p => p.1 d * p.2 e)).sum

/-- The same read with the operator never formed: each statement's *before* side is contracted
against `u` first, giving one number per statement, and those weight the *after* sides. -/
def carried [Fintype κ] (S : List (Sided κ)) (u : κ → ℝ) (e : κ) : ℝ :=
  (S.map (fun p => (∑ d, u d * p.1 d) * p.2 e)).sum

/-- **The operator is never built.** `u ⬝ (AᵀB) = (A ⬝ u) ⬝ B`.

The cost is the point. Forming `operator` takes `|κ|²` entries each summing over the statements;
`carried` takes one inner product per statement and no matrix at all. They are the same number. -/
theorem carry_identity [Fintype κ] (S : List (Sided κ)) (u : κ → ℝ) (e : κ) :
    (∑ d, u d * operator S d e) = carried S u e := by
  induction S with
  | nil => simp [operator, carried]
  | cons p S ih =>
      have hsplit : ∀ d : κ, u d * operator (p :: S) d e
          = u d * (p.1 d * p.2 e) + u d * operator S d e := by
        intro d
        simp [operator, mul_add]
      calc (∑ d, u d * operator (p :: S) d e)
          = ∑ d, (u d * (p.1 d * p.2 e) + u d * operator S d e) := by
            exact Finset.sum_congr rfl (fun d _ => hsplit d)
        _ = (∑ d, u d * (p.1 d * p.2 e)) + ∑ d, u d * operator S d e := by
            exact Finset.sum_add_distrib
        _ = (∑ d, u d * p.1 d) * p.2 e + carried S u e := by
            rw [ih, Finset.sum_mul]
            exact congrArg (· + carried S u e)
              (Finset.sum_congr rfl (fun d _ => by ring))
        _ = carried (p :: S) u e := by simp [carried]

/-- **Learning is one addition.** A statement arriving contributes a single rank-one term and
disturbs nothing already in the operator.

There is no update rule here and no rate: the operator is the sum, so a write enters it by being
one of the things summed. An operator derived *after* the write differs from one derived before by
exactly `a d * b e` and in no other way. -/
theorem operator_append (S : List (Sided κ)) (a b : κ → ℝ) (d e : κ) :
    operator (S ++ [(a, b)]) d e = operator S d e + a d * b e := by
  simp [operator]

/-- And withdrawing is the same fact read backwards: what a statement contributed is recoverable
without rebuilding anything, which is why an edge can be removed as cheaply as it was added. -/
theorem operator_append_sub (S : List (Sided κ)) (a b : κ → ℝ) (d e : κ) :
    operator (S ++ [(a, b)]) d e - a d * b e = operator S d e := by
  rw [operator_append]; ring

/-- **The order statements arrived in does not reach the operator.**

Stated because the next question this programme asks is what an ingest order buys. Whatever a
curriculum does, it does not do it here: the operator over a reordering of the same statements is
the same operator, so any measured effect of ordering lives in the coordinate the statements are
placed on, not in the weight derived from them. -/
theorem operator_perm {S T : List (Sided κ)} (h : S.Perm T) (d e : κ) :
    operator S d e = operator T d e := by
  simp only [operator]
  exact (h.map (fun p => p.1 d * p.2 e)).sum_eq

/-- The empty store has the zero operator: a relation nothing uses does not act.

Trivial, and worth having as a theorem rather than a convention, because the engine returns
*nothing* there rather than a small matrix, and that distinction is load-bearing. -/
theorem operator_nil (d e : κ) : operator ([] : List (Sided κ)) d e = 0 := by
  simp [operator]

end Infer
