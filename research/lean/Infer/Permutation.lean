/-
  The exact permutation null.

  Every score in this development is an alignment standardised by the null of re-pairing two records'
  rows. That null is what makes the score free of any supplied constant — there is no temperature and
  no fitted scale, only a statistic and the distribution it would have if the rows were paired at
  random. The whole claim rests on the moments of that distribution being *exact* rather than
  estimated, so they are worth proving rather than citing.

  For two records `A` and `B` sharing an ordered axis, the statistic is

      S π = ∑ t, ⟪A (π t), B t⟫

  over a permutation `π` of the axis.

  What is here:

  * **The re-pairing is uniform.** Summing any function of `π t` over all permutations gives the same
    total whichever index is asked about, and that total is the function's own sum scaled by how many
    permutations share each value. This is the statement that `π t` is uniformly distributed, and it
    is what a second moment would be built on.
  * **The first moment is zero** whenever `A` is centred. Centring is not a convenience: it is
    precisely the hypothesis that makes the null mean vanish, and the implementation centres both
    frames before every read for exactly this reason.

  The proofs avoid counting how many permutations carry one index to another — the usual route, and
  the tedious one. Instead: a permutation-sum is the *same* at every index, because the permutations
  act transitively; and the total over all indices is a whole multiple of the plain sum, because each
  permutation is a reindexing. Those two facts together give both results.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Logic.Equiv.Basic

namespace Infer

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-! ### The two facts everything reduces to -/

/-- Summing a function of `π t` over every permutation gives the same total whichever index is asked
about.

This is transitivity of the permutation action, used as a reindexing: any index can be carried to any
other by a transposition, and right-multiplying by that transposition permutes the permutations. -/
theorem sum_perm_apply_const (f : ι → ℝ) (u v : ι) :
    (∑ ρ : Equiv.Perm ι, f (ρ v)) = ∑ ρ : Equiv.Perm ι, f (ρ u) := by
  have hτ : (Equiv.swap u v) u = v := Equiv.swap_apply_left u v
  calc (∑ ρ : Equiv.Perm ι, f (ρ v))
      = ∑ ρ : Equiv.Perm ι, f ((ρ * Equiv.swap u v) u) := by
        refine Finset.sum_congr rfl fun ρ _ => ?_
        simp [Equiv.Perm.mul_apply, hτ]
    _ = ∑ σ : Equiv.Perm ι, f (σ u) :=
        Fintype.sum_equiv (Equiv.mulRight (Equiv.swap u v)) _ _ (fun ρ => rfl)

/-- The total over every index and every permutation is a whole multiple of the plain total — each
permutation contributes the same sum, being a reindexing of the axis. -/
theorem sum_index_sum_perm (f : ι → ℝ) :
    (∑ v : ι, ∑ ρ : Equiv.Perm ι, f (ρ v))
      = (Fintype.card (Equiv.Perm ι)) • ∑ v : ι, f v := by
  rw [Finset.sum_comm]
  rw [Finset.sum_congr rfl (fun ρ _ => Equiv.sum_comp ρ f)]
  simp [Finset.sum_const, Finset.card_univ]

/-! ### Uniformity of the re-pairing -/

/-- **The re-pairing is uniform.** `card ι • (∑_ρ f (ρ t)) = card (Perm ι) • ∑_u f u`.

Dividing through by the two counts gives `E_π[f (π t)] = (1/|ι|) ∑_u f u`, which is what a uniform
`π t` means. The counts are kept as scalar multiples rather than divisions, so nothing has to be
shown nonzero and no field structure is assumed.

This is the general form. The first moment below is the special case where the sum on the right is
zero, and a second moment would be built on this. -/
theorem card_smul_sum_perm_apply (f : ι → ℝ) (t : ι) :
    (Fintype.card ι) • (∑ ρ : Equiv.Perm ι, f (ρ t))
      = (Fintype.card (Equiv.Perm ι)) • ∑ u : ι, f u := by
  classical
  rw [← sum_index_sum_perm f,
    Finset.sum_congr rfl (fun v _ => sum_perm_apply_const f t v),
    Finset.sum_const, Finset.card_univ]

/-- **A centred record sums to zero over every permutation, at every index.**

The first moment reduces to this: a nonzero count times the constant is zero, so the constant is. -/
theorem sum_perm_apply_eq_zero [Nonempty ι] {f : ι → ℝ} (hf : ∑ u : ι, f u = 0) (t : ι) :
    (∑ ρ : Equiv.Perm ι, f (ρ t)) = 0 := by
  classical
  have h := card_smul_sum_perm_apply f t
  rw [hf, smul_zero] at h
  have hcard : (Fintype.card ι) ≠ 0 := Fintype.card_ne_zero
  simpa [nsmul_eq_mul, hcard] using h

/-! ### The statistic and its first moment -/

variable {κ : Type*} [Fintype κ]

/-- The alignment of two records under a re-pairing of the rows of the first.

Written coordinate-wise rather than through an inner-product space, so that nothing but finite sums
of reals is involved and the statement is the one the implementation computes. -/
def alignment (A B : ι → κ → ℝ) (π : Equiv.Perm ι) : ℝ :=
  ∑ t, ∑ d, A (π t) d * B t d

omit [Fintype κ] in
/-- Centring is load-bearing, not decoration. This isolates the step where it enters: the
permutation-sum factors out of the alignment, and what is left is exactly the sum that centring
kills. A score standardised against a null whose mean was *not* zero would carry an offset nobody
supplied. -/
theorem alignment_factors_through_perm_sum (A B : ι → κ → ℝ) (t : ι) (d : κ) :
    (∑ π : Equiv.Perm ι, A (π t) d * B t d)
      = (∑ π : Equiv.Perm ι, A (π t) d) * B t d := by
  rw [← Finset.sum_mul]

/-- **The first moment of the exact permutation null is zero.**

`∑ π, alignment A B π = 0` whenever `A` is centred — so the mean of the statistic under a uniformly
random re-pairing is zero, with no scale supplied and nothing estimated. This is the `E_π[S] = 0`
the standardised score divides into. -/
theorem alignment_sum_over_perms_eq_zero [Nonempty ι] (A B : ι → κ → ℝ)
    (hA : ∀ d : κ, ∑ u : ι, A u d = 0) :
    (∑ π : Equiv.Perm ι, alignment A B π) = 0 := by
  classical
  unfold alignment
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun d _ => ?_
  rw [alignment_factors_through_perm_sum A B t d, sum_perm_apply_eq_zero (hA d) t, zero_mul]

end Infer
