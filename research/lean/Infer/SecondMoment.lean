/-
  The second moment of the exact permutation null.

  `Permutation.lean` proves the first moment is zero. This is the other half, and it is the half the
  implementation actually divides by: the score is an alignment standardised by the spread of the
  null, and that spread is claimed to be *exact* — closed form, no sampling, no fitted scale.

  For two records `a` and `b` on a shared ordered axis of `n + 1` rows, the statistic under a
  re-pairing `π` is

      S π = ∑ t, a (π t) * b t

  and the claim is

      n • ∑_π (S π)^2 = card (Perm ι) • (∑ a²)(∑ b²)

  which says `Var_π[S] = (∑ a²)(∑ b²) / (card ι − 1)` once both sides are divided by the number of
  permutations. That is the Pitman–Hoeffding second moment, and it is what `entroptics.reads.coupling`
  divides into the alignment.

  **The statement is division-free on purpose.** Writing `card ι − 1` as a natural subtraction, or
  the variance as a quotient, would put an arithmetic side-condition in the way of the content. Here
  the axis is `n + 1` rows by hypothesis and the factor is `n`, so nothing is subtracted and nothing
  is divided.

  The route avoids counting permutations, as `Permutation.lean` does. Two facts carry it:

  * the joint sum `∑_π a(π t) a(π s)` is the **same at every pair of distinct indices**, because the
    permutations act 2-transitively (`PairSum.sum_perm_pair_const`), and
  * its **total over all distinct pairs** is known in closed form
    (`PairSum.sum_perm_offDiagonal_centred`).

  A constant whose total is known is determined, and that is the whole of the off-diagonal. The
  diagonal comes from `Permutation.card_smul_sum_perm_apply` the same way.
-/
import Infer.Permutation
import Infer.PairSum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity

namespace Infer

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-! ### The statistic and the joint sum it decomposes into -/

/-- The scalar statistic: one record's rows re-paired against the other's.

This is `Permutation.alignment` with a single feature coordinate. The second moment is proved here
in the scalar case because that is where the arithmetic lives; the matrix case adds a sum over
coordinates and no new content. -/
def S (a b : ι → ℝ) (π : Equiv.Perm ι) : ℝ := ∑ t, a (π t) * b t

/-- The joint permutation-sum of two records at a pair of indices.

Everything below is a statement about this quantity: it takes one value on the diagonal and one
value off it, and both are pinned in closed form.

It is stated for **two** functions rather than one because the matrix case needs the coordinate `d`
of a record against its coordinate `e`, and that is the only difference between the scalar second
moment and the matrix one. -/
def P (f g : ι → ℝ) (t s : ι) : ℝ := ∑ π : Equiv.Perm ι, f (π t) * g (π s)

/-- **Expanding a product of two statistics.** It is a weighted sum of the joint sums.

Stated for two statistics rather than one square, because the matrix case needs coordinate `d`
against coordinate `e` and the square is the case `d = e`. Nothing is used beyond swapping finite
sums and pulling the weights out of the sum over permutations, which they do not depend on. -/
theorem sum_perm_S_mul_eq (f g p q : ι → ℝ) :
    (∑ π : Equiv.Perm ι, S f p π * S g q π) = ∑ t, ∑ s, P f g t s * (p t * q s) := by
  classical
  have expand : ∀ π : Equiv.Perm ι,
      S f p π * S g q π = ∑ t, ∑ s, (f (π t) * g (π s)) * (p t * q s) := by
    intro π
    unfold S
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun s _ => by ring
  calc (∑ π : Equiv.Perm ι, S f p π * S g q π)
      = ∑ π : Equiv.Perm ι, ∑ t, ∑ s, (f (π t) * g (π s)) * (p t * q s) :=
        Finset.sum_congr rfl fun π _ => expand π
    _ = ∑ t, ∑ π : Equiv.Perm ι, ∑ s, (f (π t) * g (π s)) * (p t * q s) := Finset.sum_comm
    _ = ∑ t, ∑ s, ∑ π : Equiv.Perm ι, (f (π t) * g (π s)) * (p t * q s) :=
        Finset.sum_congr rfl fun t _ => Finset.sum_comm
    _ = ∑ t, ∑ s, P f g t s * (p t * q s) := by
        refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun s _ => ?_
        unfold P
        rw [← Finset.sum_mul]

/-- The square, as the case where both statistics are the same. -/
theorem sum_perm_S_sq_eq (a b : ι → ℝ) :
    (∑ π : Equiv.Perm ι, (S a b π) ^ 2) = ∑ t, ∑ s, P a a t s * (b t * b s) := by
  simpa [sq] using sum_perm_S_mul_eq a a b b

/-! ### The diagonal -/

/-- **The diagonal value, division-free.** `card ι • P a a t t = card (Perm ι) • ∑ a²`.

This is `card_smul_sum_perm_apply` at `f = a²`: the re-pairing is uniform, so the sum of `a(π t)²`
over all permutations is the record's own sum of squares, scaled by how many permutations share each
value. It does not depend on `t`, which is the content. -/
theorem card_smul_P_diag (f g : ι → ℝ) (t : ι) :
    (Fintype.card ι) • P f g t t
      = (Fintype.card (Equiv.Perm ι)) • ∑ u, f u * g u := by
  unfold P
  exact card_smul_sum_perm_apply (fun u => f u * g u) t

/-! ### The off-diagonal -/

/-- Grouping the off-diagonal by pair or by permutation gives the same total.

`∑_t ∑_{s ≠ t} P a a t s` and `∑_π offDiagonal a a π` are the same sum of `a(π t) a(π s)` over all
permutations and all ordered pairs of distinct indices, counted two ways. -/
theorem sum_offDiag_P (f g : ι → ℝ) :
    (∑ t, ∑ s ∈ Finset.univ.erase t, P f g t s)
      = ∑ π : Equiv.Perm ι, offDiagonal f g π := by
  classical
  have perπ : ∀ π : Equiv.Perm ι,
      (∑ t, ∑ s ∈ Finset.univ.erase t, f (π t) * g (π s)) = offDiagonal f g π := by
    intro π
    have step : ∀ t : ι, (∑ s ∈ Finset.univ.erase t, f (π t) * g (π s))
        = (∑ s, f (π t) * g (π s)) - f (π t) * g (π t) :=
      fun t => Finset.sum_erase_eq_sub (Finset.mem_univ t)
    calc (∑ t, ∑ s ∈ Finset.univ.erase t, f (π t) * g (π s))
        = ∑ t, ((∑ s, f (π t) * g (π s)) - f (π t) * g (π t)) :=
          Finset.sum_congr rfl fun t _ => step t
      _ = (∑ t, ∑ s, f (π t) * g (π s)) - ∑ t, f (π t) * g (π t) := by
          rw [Finset.sum_sub_distrib]
      _ = offDiagonal f g π := by
          unfold offDiagonal
          rw [Finset.sum_mul_sum]
  calc (∑ t, ∑ s ∈ Finset.univ.erase t, P f g t s)
      = ∑ t, ∑ π : Equiv.Perm ι, ∑ s ∈ Finset.univ.erase t, f (π t) * g (π s) := by
        refine Finset.sum_congr rfl fun t _ => ?_
        unfold P
        exact Finset.sum_comm
    _ = ∑ π : Equiv.Perm ι, ∑ t, ∑ s ∈ Finset.univ.erase t, f (π t) * g (π s) := Finset.sum_comm
    _ = ∑ π : Equiv.Perm ι, offDiagonal f g π := Finset.sum_congr rfl fun π _ => perπ π

/-! ### Both values in closed form -/

variable {n : ℕ}

/-- The diagonal value, as a real equation on an axis of `n + 1` rows. -/
theorem P_diag_mul (f g : ι → ℝ) (t : ι) (hn : Fintype.card ι = n + 1) :
    ((n : ℝ) + 1) * P f g t t
      = (Fintype.card (Equiv.Perm ι) : ℝ) * ∑ u, f u * g u := by
  have h := card_smul_P_diag f g t
  rw [hn] at h
  have := congrArg (fun x : ℝ => x) h
  simpa [nsmul_eq_mul, Nat.cast_add, Nat.cast_one, add_comm] using h

/-- **Every off-diagonal entry is the same value**, because the permutations act 2-transitively. -/
theorem P_offDiag_const (f g : ι → ℝ) {t s t' s' : ι} (hts : t ≠ s) (hts' : t' ≠ s') :
    P f g t s = P f g t' s' := sum_perm_pair_const f g hts hts'

/-- The off-diagonal total, counted: `n + 1` rows each contributing `n` equal entries. -/
theorem sum_offDiag_count (f g : ι → ℝ) {t₀ s₀ : ι} (h : t₀ ≠ s₀)
    (hn : Fintype.card ι = n + 1) :
    (∑ t, ∑ s ∈ Finset.univ.erase t, P f g t s)
      = ((n : ℝ) + 1) * ((n : ℝ) * P f g t₀ s₀) := by
  classical
  have card_erase : ∀ t : ι, (Finset.univ.erase t).card = n := by
    intro t
    rw [Finset.card_erase_of_mem (Finset.mem_univ t), Finset.card_univ, hn]
    simp
  have inner : ∀ t : ι, (∑ s ∈ Finset.univ.erase t, P f g t s) = (n : ℝ) * P f g t₀ s₀ := by
    intro t
    have : ∀ s ∈ Finset.univ.erase t, P f g t s = P f g t₀ s₀ := by
      intro s hs
      exact P_offDiag_const f g (Ne.symm (Finset.ne_of_mem_erase hs)) h
    rw [Finset.sum_congr rfl this, Finset.sum_const, card_erase t, nsmul_eq_mul]
  rw [Finset.sum_congr rfl fun t _ => inner t, Finset.sum_const, Finset.card_univ, hn,
    nsmul_eq_mul]
  push_cast
  ring

/-- **The off-diagonal value in closed form**, where the record is centred. -/
theorem P_offDiag_mul (f g : ι → ℝ) {t₀ s₀ : ι} (h : t₀ ≠ s₀)
    (hn : Fintype.card ι = n + 1) (hf : ∑ u, f u = 0) :
    ((n : ℝ) + 1) * ((n : ℝ) * P f g t₀ s₀)
      = (Fintype.card (Equiv.Perm ι) : ℝ) * (-∑ u, f u * g u) := by
  rw [← sum_offDiag_count f g h hn, sum_offDiag_P f g, sum_perm_offDiagonal_centred hf]
  simp [nsmul_eq_mul]

/-! ### The second moment -/

/-- **The second moment of the exact permutation null.**

`n • ∑_π (S π)^2 = card (Perm ι) • (∑ a²)(∑ b²)` on an axis of `n + 1` rows, with both records
centred. Dividing by the number of permutations reads it as

    Var_π[S] = (∑ a²)(∑ b²) / (card ι − 1)

which is the Pitman–Hoeffding second moment, and — with the first moment zero — the whole of what
the standardised score divides by. No constant is supplied, nothing is sampled, and nothing is
estimated: the spread of the null is this expression and not an approximation to it.

Both centring hypotheses are load-bearing and neither is decoration. `a` centred kills the
off-diagonal's product term; `b` centred is what turns `∑_{s ≠ t} b s` into `−b t` and collapses the
whole off-diagonal onto the diagonal's shape. -/
theorem sum_perm_S_sq (hn : Fintype.card ι = n + 1) (hn1 : 1 ≤ n)
    {a b : ι → ℝ} (ha : ∑ u, a u = 0) (hb : ∑ u, b u = 0) :
    (n : ℝ) * (∑ π : Equiv.Perm ι, (S a b π) ^ 2)
      = (Fintype.card (Equiv.Perm ι) : ℝ) * ((∑ u, a u ^ 2) * (∑ v, b v ^ 2)) := by
  classical
  have hcard : 1 < Fintype.card ι := by rw [hn]; omega
  obtain ⟨t₀, s₀, hts⟩ := Fintype.exists_pair_of_one_lt_card hcard
  set N : ℝ := (Fintype.card (Equiv.Perm ι) : ℝ) with hN
  set A : ℝ := ∑ u, a u ^ 2 with hA
  set D : ℝ := P a a t₀ t₀ with hDdef
  set O : ℝ := P a a t₀ s₀ with hOdef
  have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hne : ((n : ℝ) + 1) ≠ 0 := ne_of_gt hpos

  -- the diagonal takes one value, at every index
  have hdiag : ∀ t : ι, P a a t t = D := fun t =>
    mul_left_cancel₀ hne ((P_diag_mul a a t hn).trans (P_diag_mul a a t₀ hn).symm)

  -- the weighted sum collapses: centring `b` turns the off-diagonal into the diagonal's shape
  have key : (∑ π : Equiv.Perm ι, (S a b π) ^ 2) = (D - O) * ∑ v, b v ^ 2 := by
    rw [sum_perm_S_sq_eq]
    have per_t : ∀ t : ι, (∑ s, P a a t s * (b t * b s)) = (D - O) * b t ^ 2 := by
      intro t
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ t)]
      have hoff : ∀ s ∈ Finset.univ.erase t, P a a t s * (b t * b s) = O * (b t * b s) :=
        fun s hs => by
          rw [P_offDiag_const a a (Ne.symm (Finset.ne_of_mem_erase hs)) hts]
      have hsum : (∑ s ∈ Finset.univ.erase t, O * (b t * b s)) = O * (b t * -b t) := by
        rw [← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_erase_eq_sub (Finset.mem_univ t), hb]
        ring
      rw [Finset.sum_congr rfl hoff, hsum, hdiag t]
      ring
    rw [Finset.sum_congr rfl fun t _ => per_t t, ← Finset.mul_sum]

  -- the two closed forms, combined
  -- the generalised lemmas speak of `∑ f * g`; at `f = g = a` that is the sum of squares
  have hD : ((n : ℝ) + 1) * D = N * A := by
    have := P_diag_mul a a t₀ hn
    simpa [hA, sq] using this
  have hO : ((n : ℝ) + 1) * ((n : ℝ) * O) = N * (-A) := by
    have := P_offDiag_mul a a hts hn ha
    simpa [hA, sq] using this
  have hDO : (n : ℝ) * (D - O) = N * A := by
    apply mul_left_cancel₀ hne
    linear_combination (n : ℝ) * hD - hO

  rw [key, ← mul_assoc, hDO, mul_assoc]

/-! ### The two steps, extracted -/

/-- **The pair value, division-free.** `n * (diagonal − off-diagonal) = card (Perm ι) * ∑ f g`.

Both closed forms combined, with the `n + 1` cancelled. This is the whole arithmetic content of the
second moment: everything else is bookkeeping over which indices and coordinates are being summed. -/
theorem n_mul_P_gap (f g : ι → ℝ) {t₀ s₀ : ι} (h : t₀ ≠ s₀)
    (hn : Fintype.card ι = n + 1) (hf : ∑ u, f u = 0) :
    (n : ℝ) * (P f g t₀ t₀ - P f g t₀ s₀)
      = (Fintype.card (Equiv.Perm ι) : ℝ) * ∑ u, f u * g u := by
  have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  apply mul_left_cancel₀ (ne_of_gt hpos)
  linear_combination (n : ℝ) * P_diag_mul f g t₀ hn - P_offDiag_mul f g h hn hf

/-- **The weighted sum collapses onto the diagonal**, because the weight `q` is centred.

`∑_t ∑_s P f g t s * (p t * q s)` keeps only `(diagonal − off-diagonal) * ∑ p q`. The step that does
it is `∑_{s ≠ t} q s = −q t`, which is exactly what centring `q` buys — and it is why the second
moment needs *both* records centred rather than one. -/
theorem weighted_collapse (f g p q : ι → ℝ) {t₀ s₀ : ι} (h : t₀ ≠ s₀)
    (hn : Fintype.card ι = n + 1) (hq : ∑ s, q s = 0) :
    (∑ t, ∑ s, P f g t s * (p t * q s))
      = (P f g t₀ t₀ - P f g t₀ s₀) * ∑ t, p t * q t := by
  classical
  have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hne := ne_of_gt hpos
  have hdiag : ∀ t : ι, P f g t t = P f g t₀ t₀ := fun t =>
    mul_left_cancel₀ hne ((P_diag_mul f g t hn).trans (P_diag_mul f g t₀ hn).symm)
  have per_t : ∀ t : ι, (∑ s, P f g t s * (p t * q s))
      = (P f g t₀ t₀ - P f g t₀ s₀) * (p t * q t) := by
    intro t
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ t)]
    have hoff : ∀ s ∈ Finset.univ.erase t,
        P f g t s * (p t * q s) = P f g t₀ s₀ * (p t * q s) := fun s hs => by
      rw [P_offDiag_const f g (Ne.symm (Finset.ne_of_mem_erase hs)) h]
    have hsum : (∑ s ∈ Finset.univ.erase t, P f g t₀ s₀ * (p t * q s))
        = P f g t₀ s₀ * (p t * -q t) := by
      rw [← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_erase_eq_sub (Finset.mem_univ t), hq]
      ring
    rw [Finset.sum_congr rfl hoff, hsum, hdiag t]
    ring
  rw [Finset.sum_congr rfl fun t _ => per_t t, ← Finset.mul_sum]

/-! ### The matrix case -/

variable {κ : Type*} [Fintype κ]

/-- **The second moment of the matrix statistic**, which is what the implementation computes.

`Permutation.alignment` sums over a feature axis as well as the ordered one, and its second moment is

    n • ∑_π (alignment A B π)^2 = card (Perm ι) • tr (Cᴬ Cᴮ)

with `Cᴬ d e = ∑_u A u d * A u e` the record's own cross-Gram. Divided by the number of permutations
that reads `Var_π[S] = tr(Cᴬ Cᴮ) / (card ι − 1)` — the quantity `entroptics.reads.coupling`
standardises by, and the reason the score's normaliser is a **matrix functional** where a cosine's is
a pair of magnitudes.

The proof adds nothing to the scalar case beyond summing it over coordinate pairs, which is the
precise sense in which the matrix case "adds a sum over coordinates and no new content". Both records
must be centred **in every coordinate**, and the two hypotheses do the different jobs named above. -/
theorem sum_perm_alignment_sq (hn : Fintype.card ι = n + 1) (hn1 : 1 ≤ n)
    {A B : ι → κ → ℝ}
    (hA : ∀ d : κ, ∑ u : ι, A u d = 0) (hB : ∀ d : κ, ∑ t : ι, B t d = 0) :
    (n : ℝ) * (∑ π : Equiv.Perm ι, (alignment A B π) ^ 2)
      = (Fintype.card (Equiv.Perm ι) : ℝ)
        * ∑ d : κ, ∑ e : κ, (∑ u, A u d * A u e) * (∑ t, B t d * B t e) := by
  classical
  have hcard : 1 < Fintype.card ι := by rw [hn]; omega
  obtain ⟨t₀, s₀, hts⟩ := Fintype.exists_pair_of_one_lt_card hcard

  -- the alignment is the sum over coordinates of a scalar statistic
  have split : ∀ π : Equiv.Perm ι,
      alignment A B π = ∑ d : κ, S (fun u => A u d) (fun t => B t d) π := by
    intro π
    unfold alignment S
    exact Finset.sum_comm

  -- so its square is a double sum over coordinate pairs
  have sq_split : ∀ π : Equiv.Perm ι, (alignment A B π) ^ 2
      = ∑ d : κ, ∑ e : κ,
          S (fun u => A u d) (fun t => B t d) π * S (fun u => A u e) (fun t => B t e) π := by
    intro π
    rw [sq, split π, Finset.sum_mul_sum]

  -- one coordinate pair, by the scalar route
  have per_pair : ∀ d e : κ,
      (n : ℝ) * (∑ π : Equiv.Perm ι,
          S (fun u => A u d) (fun t => B t d) π * S (fun u => A u e) (fun t => B t e) π)
        = (Fintype.card (Equiv.Perm ι) : ℝ)
          * ((∑ u, A u d * A u e) * (∑ t, B t d * B t e)) := by
    intro d e
    rw [sum_perm_S_mul_eq (fun u => A u d) (fun u => A u e) (fun t => B t d) (fun t => B t e),
      weighted_collapse (fun u => A u d) (fun u => A u e) (fun t => B t d) (fun t => B t e)
        hts hn (hB e),
      ← mul_assoc,
      n_mul_P_gap (fun u => A u d) (fun u => A u e) hts hn (hA d),
      mul_assoc]

  calc (n : ℝ) * (∑ π : Equiv.Perm ι, (alignment A B π) ^ 2)
      = (n : ℝ) * ∑ π : Equiv.Perm ι, ∑ d : κ, ∑ e : κ,
          S (fun u => A u d) (fun t => B t d) π * S (fun u => A u e) (fun t => B t e) π := by
        rw [Finset.sum_congr rfl fun π _ => sq_split π]
    _ = ∑ d : κ, ∑ e : κ, (n : ℝ) * (∑ π : Equiv.Perm ι,
          S (fun u => A u d) (fun t => B t d) π * S (fun u => A u e) (fun t => B t e) π) := by
        rw [Finset.sum_comm, Finset.mul_sum]
        refine Finset.sum_congr rfl fun d _ => ?_
        rw [Finset.sum_comm, Finset.mul_sum]
    _ = ∑ d : κ, ∑ e : κ, (Fintype.card (Equiv.Perm ι) : ℝ)
          * ((∑ u, A u d * A u e) * (∑ t, B t d * B t e)) :=
        Finset.sum_congr rfl fun d _ => Finset.sum_congr rfl fun e _ => per_pair d e
    _ = (Fintype.card (Equiv.Perm ι) : ℝ)
          * ∑ d : κ, ∑ e : κ, (∑ u, A u d * A u e) * (∑ t, B t d * B t e) := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun d _ => (Finset.mul_sum _ _ _).symm

end Infer
