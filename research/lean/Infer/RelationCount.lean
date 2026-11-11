/-
  The relation count, without a vocabulary.

  Every system this work positions against is handed the number of relations with the dataset.
  R-GCN's Table 3 lists relation counts as dataset properties; RESCAL is given `k`. Here the count is
  supposed to be **read off the corpus**, and `exp03` measures that it can be. This is why it can.

  A relation in this construction owns two directions: a **source role** `u k` and a **target role**
  `w k`. An edge of relation `k` between records `a` and `b` places one shared latent along `u k` in
  `a` and along `w k` in `b`, so the cross-covariance of the pair carries the rank-one map

      x ↦ ⟪w k, x⟫ • u k

  and the relation *is* that map. Two things follow, and together they are the counting argument.

  **The maps are linearly independent** whenever the target roles are orthonormal and no source role
  is zero. So the span of the edge set has dimension exactly `R`: counting resolved dimensions
  recovers the number of relations, and nothing had to supply it. That is the statement that makes a
  measured count meaningful rather than coincidental.

  **The count is bounded by half the basis**, because `R` relations need `2R` orthonormal directions.
  `exp23` measures that this bound is *attained* at every width tested — 8 of 8, 16 of 16, 32 of 32,
  64 of 64 — so the limit is the basis and not a cap, a vocabulary, or a saturating counter.

  What is deliberately not claimed: that a corpus's relations must be orthonormal. They are in the
  generator, which is why the generator can say how many it made. The theorems below say what follows
  *given* that structure, which is what a counting argument is for.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.LinearAlgebra.FiniteDimensional.Defs
import Mathlib.LinearAlgebra.Dimension.Constructions

namespace Infer

open Finset

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The relation as the map it is: a rank-one operator from a target role onto a source role.

This is `u k ⟪w k, ·⟫`, the same object RESCAL holds as `R_k` and a transformer holds inside
`W_Q`/`W_K`. The difference is entirely in where it comes from. -/
noncomputable def roleMap (u w : E) : E →ₗ[ℝ] E :=
  LinearMap.smulRight ((innerSL ℝ w).toLinearMap) u

@[simp]
theorem roleMap_apply (u w x : E) : roleMap u w x = (inner ℝ w x) • u := rfl

/-- **The relation maps are linearly independent.**

Given orthonormal target roles and no zero source role, the `R` relation maps span an `R`-dimensional
space. So the dimension of the edge set's span **is** the relation count, and reading that dimension
off the corpus recovers a number nothing supplied.

The proof is the one line the construction affords: apply a vanishing combination to the target role
`w j`, and orthonormality collapses the sum to `c j • u j`. -/
theorem roleMaps_linearIndependent {R : ℕ} (u w : Fin R → E)
    (hw : Orthonormal ℝ w) (hu : ∀ j, u j ≠ 0) :
    LinearIndependent ℝ (fun k => roleMap (u k) (w k)) := by
  classical
  rw [linearIndependent_iff']
  intro s c hs j hj
  -- evaluate the vanishing combination at the target role `w j`
  have happ := congrArg (fun f : E →ₗ[ℝ] E => f (w j)) hs
  simp only [LinearMap.coe_sum, Finset.sum_apply, LinearMap.smul_apply, roleMap_apply,
    LinearMap.zero_apply] at happ
  -- every term but `k = j` dies on orthonormality
  have hterm : ∀ k ∈ s, c k • ((inner ℝ (w k) (w j)) • u k)
      = if k = j then c j • u j else 0 := by
    intro k _
    by_cases h : k = j
    · subst h
      simp [hw.1 k]
    · simp [orthonormal_iff_ite.mp hw, h]
  rw [Finset.sum_congr rfl hterm, Finset.sum_ite_eq' s j] at happ
  simp only [if_pos hj] at happ
  exact (smul_eq_zero.mp happ).resolve_right (hu j)

/-- **The count is bounded by half the basis, and by nothing else.**

`R` relations own `2R` orthonormal directions, so `2R ≤ dim E`. This is the whole ceiling: there is
no vocabulary, no cap and no saturation short of it, and `exp23` measures the bound being attained at
every width it tests. -/
theorem two_mul_le_finrank_of_roles [FiniteDimensional ℝ E] {R : ℕ} (u w : Fin R → E)
    (h : Orthonormal ℝ (Sum.elim u w)) :
    2 * R ≤ Module.finrank ℝ E := by
  have hcard : Fintype.card (Fin R ⊕ Fin R) = 2 * R := by simp [two_mul]
  have := h.linearIndependent.fintype_card_le_finrank
  omega

end Infer
