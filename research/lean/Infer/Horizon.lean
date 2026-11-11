/-
  The horizon terminates.

  Propagation in this development continues while a signal still clears the floor and stops when it
  does not, and nothing anywhere accepts a maximum depth. That is only safe if the walk is bound to
  stop, so this is the statement that it is.

  Two facts do the work, and they are independent.

  * A finite aperture band-limits, so the correlation it carries decays at a positive rate — the
    reading of Yang–Mills in `entroptics-mass-gap`, where the mass gap *is* `‖C(τ)‖ ≤ M e^{−Δτ}` with
    `Δ > 0`. Here that hypothesis appears as `z n ≤ M * r ^ n` with `r < 1`, which is the same
    statement with `r = e^{−Δ}`.
  * The floor is strictly positive. A floor of zero would be no floor at all, and a walk against it
    could run forever without contradiction.

  Given both, the signal is below the floor from some step onward, and the horizon — the last step at
  which anything still resolves — exists. The point of proving it rather than asserting it is that the
  implementation has no depth argument to fall back on: if this were false, the walk would be a loop
  with no termination condition.
-/
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Order.Filter.AtTopBot.Basic

namespace Infer

open Filter

/-- A geometrically decaying signal falls below any strictly positive floor, and stays below it.

`z` is the signal at each step, `M * r ^ n` its geometric envelope with `r < 1`, and `floor` the
level below which nothing is reported. The conclusion is not merely that some step is quiet but that
**every** step from `N` on is, which is what lets a walk stop rather than merely pause.

No sign condition on `M` is needed and none is assumed: the envelope tends to zero whichever way `M`
points, so the theorem is slightly stronger than the physical reading that motivates it. -/
theorem below_floor_eventually
    {M r floor : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) (hfloor : 0 < floor)
    (z : ℕ → ℝ) (hz : ∀ n, z n ≤ M * r ^ n) :
    ∃ N : ℕ, ∀ n, N ≤ n → z n < floor := by
  -- `M * r ^ n → 0`, so it is eventually below the floor; the envelope carries `z` with it.
  have henv : Tendsto (fun n : ℕ => M * r ^ n) atTop (nhds 0) := by
    have := tendsto_pow_atTop_nhds_zero_of_lt_one hr0 hr1
    simpa using this.const_mul M
  have : ∀ᶠ n in atTop, M * r ^ n < floor :=
    henv.eventually (eventually_lt_nhds hfloor)
  obtain ⟨N, hN⟩ := eventually_atTop.mp this
  exact ⟨N, fun n hn => lt_of_le_of_lt (hz n) (hN n hn)⟩

/-- The horizon exists: there is a last step at which the signal still clears the floor.

Stated as the set of resolving steps being bounded above. A walk that expands only from resolving
steps therefore has finitely many to expand from, which is termination. -/
theorem horizon_exists
    {M r floor : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) (hfloor : 0 < floor)
    (z : ℕ → ℝ) (hz : ∀ n, z n ≤ M * r ^ n) :
    ∃ N : ℕ, ∀ n, floor ≤ z n → n < N := by
  obtain ⟨N, hN⟩ := below_floor_eventually hr0 hr1 hfloor z hz
  refine ⟨N, fun n hn => ?_⟩
  by_contra hcon
  exact absurd hn (not_le.mpr (hN n (not_lt.mp hcon)))

/-- Without a positive rate there is no guarantee: a signal that does not decay can sit above the
floor forever. This is the converse standing as a warning rather than a theorem to lean on — the
decay hypothesis is load-bearing and is discharged by the aperture argument, not assumed for
convenience. -/
theorem constant_signal_never_falls
    {c floor : ℝ} (h : floor ≤ c) :
    ∀ n : ℕ, floor ≤ (fun _ : ℕ => c) n := fun _ => h

end Infer
