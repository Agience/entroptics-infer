# Related work — what this is, against what exists

Every claim about a cited paper on this page was read out of that paper, and
the page says where. Where a detail is quoted, it is quoted because the program's positioning
rests on it and a paraphrase would be doing work the source does not do.

---

## 1. The line this work sits on

The idea that a relation between two things is a matrix rather than a scalar is not new, and
this program does not claim it. What is new here is that the matrix is measured at read time
rather than fitted by descent, and that the set of relations is counted rather than supplied.

### Tensor product representations — Smolensky (1990)

> Smolensky, P. *Tensor product variable binding and the representation of symbolic structures in
> connectionist systems.* Artificial Intelligence 46(1–2):159–216, 1990.

The canonical neurosymbolic binding mechanism: a value/variable pair is represented by the outer
product of a vector for the role and a vector for the filler, and a structure is the sum of such
products. It is where the claim that a distributed representation can carry symbolic structure comes
from, and it is why a triplet is the natural unit rather than a pair — a binding needs a role, a
filler, and the product that joins them.

**What this program takes:** the triplet as the unit, and the outer product as the join.
What it changes: Smolensky's role and filler vectors are supplied by the modeller. Here both
sides are records the corpus already holds, and the product is a measurement of them.

### RESCAL — Nickel, Tresp & Kriegel (2011)

> Nickel, M., Tresp, V. & Kriegel, H-P. *A Three-Way Model for Collective Learning on
> Multi-Relational Data.* ICML 2011.

Factorizes a three-way tensor so each frontal slice is `X_k ≈ A R_k Aᵀ`, where `A` holds the entity
latent components and `R_k` is an asymmetric r × r matrix specifying how those components
interact under relation `k`. This is the clearest statement in the literature that a relation is a
matrix, not a scalar weight — and it is the direct ancestor of what this program measures.

**Two things are supplied and neither is measured.** The latent dimension `r` is chosen. The
relation index `k` ranges over a vocabulary given with the dataset.

### DistMult and its descendants

R-GCN's own §4 states the lineage compactly: in DistMult *"every relation `r` is associated with a
diagonal matrix `R_r ∈ ℝ^{d×d}` and a triple `(s, r, o)` is scored as `f(s,r,o) = e_sᵀ R_r e_o`"*.
DistMult is RESCAL's matrix restricted to a diagonal; ComplEx and HolE generalize it again. The
whole family scores a triplet as a bilinear form with a trained, per-relation matrix.

### Fast weight programmers — Schlag, Irie & Schmidhuber (2021)

> Schlag, I., Irie, K. & Schmidhuber, J. *Linear Transformers Are Secretly Fast Weight Programmers.*
> ICML 2021. arXiv:2102.11174.

Shows the formal equivalence between linearized self-attention and the fast weight controllers of
the early 1990s: a slow net learns to program the fast weights of another through additive outer
products of keys and values. Attention, read this way, is already a matrix-valued memory rather
than a table of scalars — and the paper identifies a capacity limit in the purely additive form,
replacing it with a delta-rule instruction.

**What this program takes:** the reading of attention as an outer-product memory, and the warning
that additive accumulation has a capacity limit that shows up as interference.
What it changes: the slow net. Here nothing programs the fast weights; they are read.

### R-GCN — Schlichtkrull, Kipf, Bloem, van den Berg, Titov & Welling (2018)

> *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018. arXiv:1703.06103
> (v1 17 Mar 2017, v4 26 Oct 2017).

The closest existing system to what Stage 2 builds, and the sharpest illustration of the two
forcings this program removes. Read out of the paper directly:

**The relation vocabulary is an input.** Table 3 lists relation counts as a property of each
dataset — WN18: 18, FB15K: 1,345, FB15k-237: 237. A per-relation weight matrix `W_r` exists because
`r` was enumerated in advance.

**The number of relation modes is guessed, then selected on a validation set.** Because per-relation
matrices explode in parameters, §2 factorizes them — basis decomposition (Eq. 3) as a linear
combination of `B` shared basis matrices, block decomposition (Eq. 4) as a direct sum
`W_r^{(l)} = ⊕_{b=1}^{B} Q_{br}^{(l)}`. The paper's own words on how `B` is set: *"For FB15k and
WN18, we report results using basis decomposition (Eq. 3) with two basis functions … For
FB15k-237, we found block decomposition (Eq. 4) to perform best, using two layers with block
dimension 5 × 5."* — with *"20% of the training set"* set aside *"as a validation set for
hyperparameter tuning"*.

`B` is an estimate of how many independent modes the relational structure actually has. It is
the quantity this program counts with the instrument. R-GCN guesses it and selects on held-out
accuracy; here it is a measurement taken before any task is scored.

**And the authors name the defect themselves.** §5.1, discussing where R-GCN underperforms:

> *"We conjecture that the fixed choice of normalization constant for the aggregation of messages
> from neighboring nodes is partly to blame for this behavior, which can be particularly problematic
> for nodes of high degree. A potential way to overcome this limitation is to introduce an attention
> mechanism, i.e. to replace the normalization constant `1/c_{i,r}` with data-dependent attention
> weights `a_{ij,r}`, where `Σ_{j,r} a_{ij,r} = 1`. We expect this to be a promising avenue for
> future research."*

This is the program's positioning in one paragraph, written by someone else. A fixed normalization
constant is the disease; their proposed cure replaces it with attention weights that are
data-dependent but still constrained to sum to one. That constraint is itself a forcing, and it
is the one that makes refusal impossible: a row that sums to one always answers, even when every
term in it is noise. The construction here removes both — the weight is standardized against an
exact permutation null, and nothing normalizes the row, so a query whose keys all read as noise
returns zeros.

### Attention — Vaswani et al. (2017)

The baseline the whole program is measured against. Two constants sit in
`A = softmax(QKᵀ/√d)`: the temperature `√d`, and the softmax's own normalization. Note that the
temperature does not affect an argmax, softmax being monotone — so where this program compares
*ordering* it compares against the dot product and cosine directly, and the temperature is compared
where it actually acts, in the aggregation.

---

### Surrogate data — Theiler, Eubank, Longtin, Galdrikian & Farmer (1992)

Not a representation-learning method, and it is here because one measurement in this program rests
on it. Asking whether a corpus's channels move together needs a null in which they do not, and
inventing one would put a constant where an observation should be.

The surrogate construction supplies it from the data: randomize each series' Fourier phases
independently, and its power spectrum — and so its entire autocorrelation — is preserved *exactly*
while its alignment with the other series is destroyed. Applied per channel, it gives the same
corpus with the cross-channel dependence removed and nothing else changed.

It is the null of the admission test, which `exp12` runs on every corpus the program uses. The
method is parameter-free, which is why it is admissible here.

## 2. What is different here, stated so it can be disagreed with

| | RESCAL / DistMult / R-GCN | here |
|---|---|---|
| relation is | a matrix | a matrix |
| that matrix comes from | gradient descent on a training objective | the cross-covariance of two records, measured |
| the number of relations | an input (`R` in the dataset table) | counted from the data, unbounded |
| the number of relation modes | `B`, guessed and validation-selected | counted by the instrument |
| the entity coordinate | `r`, chosen | `K`, counted |
| row normalization | `Σ a = 1`, imposed | none — the row may sum to zero |
| what happens with no evidence | the row still sums to one, so something is returned | zeros; refusal is arithmetic, not a branch |
| null hypothesis for a weight | none stated | exact permutation null, closed-form moments |

**The one-line difference.** Every system above answers the question *"what weight best explains the
training data?"*. This one answers *"what association is present in this corpus, and is it more than
a re-pairing of the rows would produce?"*. The first question needs a fit; the second needs a
measurement and a null.

---

## 3. The triplet is a transistor, and that is load-bearing

A transistor is a three-terminal device in which the flow between two terminals is modulated by the
third, and it is the minimal unit from which both amplification and switching compose. The
platform's `(context, content, operator)` triple has that shape: the operator modulates what passes
between content and context.

This is not a decorative analogy — it fixes two design decisions.

**A transistor's gain is not a stored number.** It is a property of the device *at its operating
point*, and it changes when the bias changes. That is exactly what "weights generated in realtime
from the ontology edges" has to mean: the association between two records is not a value looked up
from a fitted table but a quantity measured under the current context. A trained `W_r` is a gain
soldered in at the factory.

**Three terminals, not two, is what makes the relation a matrix.** A two-terminal association
between two records is a scalar. A third terminal that modulates it turns the association into a map
— the relation slice `R_k` in RESCAL, the role⊗filler product in Smolensky, and here the
cross-covariance of two frames *under a third*. The triplet is why the matrix is the right object
and the scalar is the wrong one, and it is the same reason in all three literatures.

---

## 4. The prior work inside this workspace

`agience-pharos/research/paper-2-knowledge-without-weights.md`
(https://github.com/Agience/agience-pharos), *Knowledge without weights*, is the direct predecessor, and it is
read here as prior art with known defects rather than as a foundation: it carries fitting and
supplied constants that this program's rules disallow, and it says so about itself in places. Three
of its results are nonetheless binding, because each is a null that rules out a design:

1. **§70** — a coordinate keyed on a name is orthogonal across id families by
   construction. *Nothing here may key a coordinate on an identifier.*
2. **§65** — `resolved_modes = 0` on raw ontology descent frames; a sparse coordinate reads to the
   aperture exactly as noise does. Refitting in a corpus basis resolved 5–7 modes.
   *Every read here is taken in a basis derived from the corpus.*
3. **§65** — a fitted propagator on the conversational axis is indistinguishable from the same
   concepts scrambled (paired `t = 0.13`). *No propagator is fitted to a concept sequence here.*

And from `entroptics-jlens`: a derived floor is inflated by an identity component, 2× to 21×
across ten models. *Any transport this program reads a rank off has its identity share removed and
reported first.*

---

## Sources

- [Smolensky (1990), *Artificial Intelligence* 46(1–2):159–216](https://www.sciencedirect.com/science/article/abs/pii/000437029090007M) · [PDF](http://www.lscp.net/persons/dupoux/teaching/AT1_2014/papers/Smolensky_1990_TensorProductVariableBinding.AI.pdf) · [dblp](https://dblp.org/rec/journals/ai/Smolensky90.html)
- [Nickel, Tresp & Kriegel (2011), ICML — RESCAL](http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf) · [reference implementation](https://github.com/mnick/rescal.py) · [survey treatment](https://www.dbs.ifi.lmu.de/~tresp/papers/nectar.pdf)
- [Schlag, Irie & Schmidhuber (2021), ICML — arXiv:2102.11174](https://arxiv.org/abs/2102.11174) · [PMLR](https://proceedings.mlr.press/v139/schlag21a.html)
- [Schlichtkrull et al. (2018), ESWC — arXiv:1703.06103](https://arxiv.org/abs/1703.06103) · [Springer](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38)
- [Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), *Physica D* 58(1-4):77-94 — surrogate data](https://doi.org/10.1016/0167-2789(92)90102-S)

### Corpora

Every real corpus is fetched fresh by the experiment that uses it, and its SHA-256 is recorded in that experiment's results.

- [UCI 360 — Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) (De Vito et al.)
- [UCI 357 — Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection) (Candanedo & Feldheim)
- [UCI 235 — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (Hebrail & Berard)
- [UCI 240 — Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (Anguita et al.)
- [UCI 501 — Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) (Zhang et al.)
- [CORGIS Weather](https://corgis-edu.github.io/corgis/csv/weather/) (NOAA, via the CORGIS collection)
