/-
  The pair statistic: what the null's second moment is built from.

  The first moment needed one fact about a re-pairing — that `π t` is uniform. The second needs the
  joint behaviour of `π t` and `π s` at two *distinct* indices, because squaring the alignment
  produces a term for every ordered pair of rows. Those terms split into a diagonal, where `t = s`
  and the first moment's machinery applies unchanged, and an off-diagonal, which is what this file is
  about.

  Two facts, mirroring the single-index case exactly:

  * **The joint permutation-sum is the same at every pair of distinct indices.** Where the first
    moment used transitivity — any index can be carried to any other — this uses *2-transitivity*:
    any ordered pair of distinct indices can be carried to any other by a permutation, built here
    from two transpositions rather than cited, so the construction is visible.
  * **The off-diagonal total is a whole multiple of a plain expression in the two records.** Each
    permutation contributes the same amount, being a reindexing of the axis.

  Together these give the off-diagonal contribution in closed form, which is the remaining
  ingredient for `Var_π[S]`. The variance itself is not assembled here — that is stated openly in the
  paper rather than implied by having its parts.
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

/-! ### Carrying one pair of distinct indices to another -/

/-- A permutation taking `t'` to `t` and `s'` to `s`, for any two pairs of distinct indices.

Built from two transpositions rather than cited, because the construction is short and the
side-conditions are the content: the first swap puts `t'` where `t` is, and the second moves the
image of `s'` onto `s` without disturbing `t`. That it does not disturb `t` is exactly where `t ≠ s`
and `t' ≠ s'` are used. -/
noncomputable def pairMover (t s t' s' : ι) : Equiv.Perm ι :=
  Equiv.swap ((Equiv.swap t' t) s') s * Equiv.swap t' t

omit [Fintype ι] in
/-- The first index lands where it should. -/
theorem pairMover_fst {t s t' s' : ι} (hts : t ≠ s) (hts' : t' ≠ s') :
    pairMover t s t' s' t' = t := by
  classical
  have h1 : (Equiv.swap t' t) t' = t := Equiv.swap_apply_left t' t
  -- the image of `s'` is not `t`: the swap is injective and already sends `t'` there, so if it
  -- sent `s'` there too the two indices would coincide, which they do not
  have hne : (Equiv.swap t' t) s' ≠ t := fun h =>
    hts' ((Equiv.swap t' t).injective (h1.trans h.symm))
  have hfix : (Equiv.swap ((Equiv.swap t' t) s') s) t = t :=
    Equiv.swap_apply_of_ne_of_ne (Ne.symm hne) hts
  simp [pairMover, Equiv.Perm.mul_apply, h1, hfix]

omit [Fintype ι] in
/-- The second index lands where it should, and this half needs no side-condition: the second swap
is chosen to move exactly the image of `s'`. -/
theorem pairMover_snd (t s t' s' : ι) :
    pairMover t s t' s' s' = s := by
  classical
  simp [pairMover, Equiv.Perm.mul_apply, Equiv.swap_apply_left]

/-! ### The joint permutation-sum is the same at every pair of distinct indices -/

/-- **2-transitivity, as a reindexing.** The joint sum over all permutations does not depend on
*which* pair of distinct indices is asked about.

This is the second-moment analogue of `sum_perm_apply_const`, and the proof has the same shape:
carry one pair onto the other, then absorb the carrier by right-multiplication. -/
theorem sum_perm_pair_const (f g : ι → ℝ) {t s t' s' : ι} (hts : t ≠ s) (hts' : t' ≠ s') :
    (∑ π : Equiv.Perm ι, f (π t) * g (π s))
      = ∑ π : Equiv.Perm ι, f (π t') * g (π s') := by
  classical
  set τ := pairMover t s t' s' with hτ
  have ht : τ t' = t := pairMover_fst hts hts'
  have hs : τ s' = s := pairMover_snd t s t' s'
  calc (∑ π : Equiv.Perm ι, f (π t) * g (π s))
      = ∑ π : Equiv.Perm ι, f ((π * τ) t') * g ((π * τ) s') := by
        refine Finset.sum_congr rfl fun π _ => ?_
        simp [Equiv.Perm.mul_apply, ht, hs]
    _ = ∑ σ : Equiv.Perm ι, f (σ t') * g (σ s') :=
        Fintype.sum_equiv (Equiv.mulRight τ) _ _ (fun π => rfl)

/-! ### The off-diagonal total -/

/-- The off-diagonal contribution at a single permutation, written without filtering a `Finset`.

`∑_{t ≠ s} f(π t) g(π s)` equals the full product minus its diagonal, and saying it that way keeps
every sum over the whole index type — which is what makes the reindexing below a one-line step
instead of an argument about subsets. -/
def offDiagonal (f g : ι → ℝ) (π : Equiv.Perm ι) : ℝ :=
  (∑ t, f (π t)) * (∑ s, g (π s)) - ∑ t, f (π t) * g (π t)

omit [DecidableEq ι] in
/-- **Every permutation contributes the same off-diagonal amount**, because each is a reindexing of
the axis and all three sums run over the whole of it. -/
theorem offDiagonal_eq (f g : ι → ℝ) (π : Equiv.Perm ι) :
    offDiagonal f g π = (∑ u, f u) * (∑ v, g v) - ∑ u, f u * g u := by
  unfold offDiagonal
  rw [Equiv.sum_comp π f, Equiv.sum_comp π g, Equiv.sum_comp π (fun u => f u * g u)]

/-- **The off-diagonal total in closed form.**

`∑_π offDiagonal f g π = card (Perm ι) • ((∑ f)(∑ g) − ∑ f·g)`. Combined with
`sum_perm_pair_const`, this pins the joint permutation-sum at any single pair of distinct indices:
the total is known and every pair contributes equally. That is the remaining ingredient of the
second moment. -/
theorem sum_perm_offDiagonal (f g : ι → ℝ) :
    (∑ π : Equiv.Perm ι, offDiagonal f g π)
      = (Fintype.card (Equiv.Perm ι)) • ((∑ u, f u) * (∑ v, g v) - ∑ u, f u * g u) := by
  rw [Finset.sum_congr rfl (fun π _ => offDiagonal_eq f g π)]
  simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_sub]

/-- Where both records are **centred**, the off-diagonal total collapses to a multiple of the
diagonal alone — the product term vanishes with the centring.

This is the step that turns the second moment into a statement about `∑ f·g` and nothing else, which
is where the cross-Gram of the two records comes from. -/
theorem sum_perm_offDiagonal_centred {f g : ι → ℝ}
    (hf : ∑ u : ι, f u = 0) :
    (∑ π : Equiv.Perm ι, offDiagonal f g π)
      = (Fintype.card (Equiv.Perm ι)) • (-∑ u, f u * g u) := by
  rw [sum_perm_offDiagonal f g, hf, zero_mul, zero_sub]

end Infer
