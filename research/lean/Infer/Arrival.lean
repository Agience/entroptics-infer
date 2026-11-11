/-
  An answer cannot move when something arrives, and a separation can.

  `exp55` measures a lattice that grows: four relationships turn up one at a time, and the same
  eighty questions are re-asked verbatim at every step. Not one answer already given changes. The
  paper calls that **structural** rather than fortunate, and this is the statement behind the word.

  The reason is that a completion is a **local** read. It looks up the source, the two records of
  the exemplar, and the candidates, and it sees nothing else — there is no fitted state carrying
  information about the rest of the lattice, because there is no fitted state at all. So a store
  that grows elsewhere is, from that question's point of view, the same store.

  That is `answer_congr` below, and `answer_arrive` is the form it takes for an arrival: adding
  records to a store leaves every answer whose records were already present exactly where it was.
  Not approximately — the outcome is literally the same value.

  The same file carries the other half, because half of this on its own would be misleading.
  `exp55` also measures that an already-present relation's **separation** moves a great deal as the
  lattice grows: 7.36 to 4.78, by as much as 4.62 as the records go from 48 to 80. That is not a
  defect and it does not contradict the theorem — a separation is not a local read. `carries_kinds`
  scores every relation against a baseline built from pairs standing in none, so it consults the
  whole store, and arriving records join that baseline.

  `not_all_reads_are_stable` makes that precise by exhibiting one: a read over the whole store that
  does change under an arrival. So locality is exactly what buys the stability, and a read that
  gives it up gives the stability up with it. A reader who takes the first theorem to mean *nothing*
  moves when a lattice grows would be wrong, and the second theorem is what stops them.

  What is deliberately not claimed: anything about whether an answer is *correct*. These theorems
  are about invariance under arrival and say nothing about what the read returns.
-/
import Mathlib.Data.List.Basic
import Mathlib.Data.Option.Basic
import Mathlib.Data.Finset.Card

namespace Infer

universe u v w

variable {Name : Type u} {Record : Type v} {Outcome : Type w}

/-- A store is what the lattice offers a read: a name, and the record held under it if any. -/
def Store (Name : Type u) (Record : Type v) : Type max u v := Name → Option Record

/-- The records one question names. Nothing else is looked at, and that is the whole point. -/
structure Question (Name : Type u) where
  source : Name
  exemplarFrom : Name
  exemplarTo : Name
  candidates : List Name

/-- Every name a question mentions, in one list. -/
def Question.named (q : Question Name) : List Name :=
  q.source :: q.exemplarFrom :: q.exemplarTo :: q.candidates

/-- Two stores agree wherever a question looks. -/
def AgreeOn (s t : Store Name Record) (ns : List Name) : Prop :=
  ∀ n ∈ ns, s n = t n

/-- A **local** read: it consumes the records that were looked up and nothing else.

`complete` is of this shape. It takes the exemplar's two records, the source's, and the
candidates', forms association matrices between them and compares those — and at no point does it
consult a record it was not handed, because there is no parameter anywhere that could carry one. -/
def Read (Record : Type v) (Outcome : Type w) : Type max v w :=
  Option Record → Option Record → Option Record → List (Option Record) → Outcome

/-- Asking a question of a store: look the names up, hand the results to the read. -/
def answer (r : Read Record Outcome) (s : Store Name Record) (q : Question Name) : Outcome :=
  r (s q.source) (s q.exemplarFrom) (s q.exemplarTo) (q.candidates.map s)

/-- **Stores agreeing where a question looks give that question the same answer.**

The content of the theorem is the modelling: that a completion is a function of its lookups. Once
that is written down the rest is a congruence, which is as it should be — the claim is not that
some clever invariance holds, it is that there is nothing present for a change elsewhere to act on. -/
theorem answer_congr (r : Read Record Outcome) {s t : Store Name Record} {q : Question Name}
    (h : AgreeOn s t q.named) : answer r s q = answer r t q := by
  have hs : s q.source = t q.source := h _ (by simp [Question.named])
  have hf : s q.exemplarFrom = t q.exemplarFrom := h _ (by simp [Question.named])
  have ht : s q.exemplarTo = t q.exemplarTo := h _ (by simp [Question.named])
  have hc : q.candidates.map s = q.candidates.map t := by
    apply List.map_congr_left
    intro n hn
    exact h n (by simp [Question.named, hn])
  simp only [answer, hs, hf, ht, hc]

/-- An arrival: records turn up, and what was already held keeps its place.

This is how a lattice grows. Nothing is overwritten — an edge or a record that was there is there
still, and `extra` is only consulted where the store was empty. -/
def arrive (s extra : Store Name Record) : Store Name Record :=
  fun n => (s n).orElse fun _ => extra n

@[simp]
theorem arrive_of_some {s extra : Store Name Record} {n : Name} {x : Record}
    (h : s n = some x) : arrive s extra n = some x := by
  simp [arrive, h]

/-- **An answer whose records were already present does not move when a lattice grows.**

This is `exp55`'s measurement as a theorem: 0 of 80 answers moved, and they could not have. -/
theorem answer_arrive (r : Read Record Outcome) (s extra : Store Name Record)
    (q : Question Name) (h : ∀ n ∈ q.named, (s n).isSome) :
    answer r (arrive s extra) q = answer r s q := by
  apply answer_congr
  intro n hn
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp (h n hn)
  simp [hx]

/-- A read that consults the whole store rather than the records a question names.

A separation is of this shape: `carries_kinds` reads a relation against a baseline drawn from pairs
standing in no named relation, so it ranges over everything the store holds. -/
def GlobalRead (Name : Type u) (Record : Type v) (Outcome : Type w) : Type max u v w :=
  Store Name Record → Outcome

/-- **And a read over the whole store need not be stable under an arrival.**

Exhibited rather than asserted: a store holding one record, an arrival bringing a second, and a read
that reports whether some given name is occupied. It answers differently before and after, which is
the behaviour `exp55` measures in a separation and which the locality theorem above does not forbid
— because that read is not local.

So the two theorems together say what the measurement says: a completion is invariant under arrival
because it is local, and a separation moves because it is not. -/
theorem not_all_reads_are_stable :
    ∃ (g : GlobalRead ℕ ℕ Bool) (s extra : Store ℕ ℕ), g (arrive s extra) ≠ g s := by
  refine ⟨fun st => (st 1).isSome, fun _ => none, fun _ => some 0, ?_⟩
  simp [arrive]

/-! ### The other half: what a lattice can answer only grows

`answer_arrive` says an answer already given does not move. On its own that is the statement of a
system that does not learn, and it would be satisfied by one that ignores everything arriving. The
claim the programme actually makes is both at once: **what could be answered still can be, with the
same answer, and more can be answered than before.**
-/

/-- A question this store can answer: every name it mentions is held.

This is the hypothesis `answer_arrive` already takes, named so it can be reasoned about. Note what
it does *not* say — nothing about the answer being right, only that the lookups succeed. -/
def Answerable (s : Store Name Record) (q : Question Name) : Prop :=
  ∀ n ∈ q.named, (s n).isSome

/-- **Arrival never takes a question away.**

A record that was held is held still, so a question whose names all resolved still resolves. There
is no eviction, no capacity and no forgetting in `arrive` -- which is why this holds and why a
store with any of those would need a different theorem. -/
theorem answerable_arrive {s extra : Store Name Record} {q : Question Name}
    (h : Answerable s q) : Answerable (arrive s extra) q := by
  intro n hn
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp (h n hn)
  simp [arrive, hx]

/-- **Learning without disturbance, as one statement.**

Everything answerable before is answerable after, and every such answer is the same value. This is
the pair of properties a trained system cannot have together: adding a relation there means new
parameters and a fresh descent, and the old answers move because the weights moved. Here there are
no weights to move. -/
theorem arrival_keeps_every_answer (r : Read Record Outcome) (s extra : Store Name Record)
    (q : Question Name) (h : Answerable s q) :
    Answerable (arrive s extra) q ∧ answer r (arrive s extra) q = answer r s q :=
  ⟨answerable_arrive h, answer_arrive r s extra q h⟩

/-- **And arrival can add a question that could not be answered before.**

Exhibited rather than asserted, so "it does not disturb" cannot be satisfied vacuously by a store
that learns nothing: an empty store answers nothing, and after an arrival it answers. Together with
the theorem above this is the whole of what an arrival does -- strictly more, and nothing moved. -/
theorem arrival_can_add_an_answerable_question :
    ∃ (s extra : Store ℕ ℕ) (q : Question ℕ),
      ¬ Answerable s q ∧ Answerable (arrive s extra) q := by
  refine ⟨fun _ => none, fun _ => some 0, ⟨0, 0, 0, []⟩, ?_, ?_⟩
  · intro h
    have := h 0 (by simp [Question.named])
    simp at this
  · intro n _
    simp [arrive]

end Infer
