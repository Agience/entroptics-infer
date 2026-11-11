# entroptics-infer

[![PyPI](https://img.shields.io/pypi/v/entroptics-infer)](https://pypi.org/project/entroptics-infer/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24143174.svg)](https://doi.org/10.5281/zenodo.24143174)
[![Licence: AGPL-3.0-only or commercial](https://img.shields.io/badge/licence-AGPL--3.0--only%20or%20commercial-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Instrument](https://img.shields.io/badge/instrument-entroptics-4B0082)](https://github.com/Agience/entroptics)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

Paper: [research/PAPER.pdf](research/PAPER.pdf) (two-column; source in [research/PAPER.md](research/PAPER.md),
built by `research/typeset/build_paper.py`). Every published number has a row in
[`research/CLAIMS.md`](research/CLAIMS.md) naming the script that produced it.

Inference from a corpus's live geometry: attention weights measured from relational structure at
read time, with no trained model and no fitted constant.

Pre-alpha. The aim is a reader that does what a language model does, from a store, with nothing
trained. Today it answers relational questions over a store of statements, refuses when the
evidence does not clear the level it is asked at, and learns by being told one more statement.

---

## The claim

A transformer computes its attention pattern at read time, from the data: `A = softmax(QKᵀ/√d)`.
The parts below are derived from the corpus at read time, with no fitted constant.

| a transformer trains | this derives |
|---|---|
| embedding table | the corpus's own resolved basis; its dimension K is counted |
| `W_Q`/`W_K`/`W_V` | resolved sub-bases of screens over the same corpus — nothing fitted by descent |
| `softmax(q·k/√d)` | a measured association against an exact permutation null — no temperature, no forced normalization |
| a relation vocabulary | relation type as a direction in a resolved edge basis; the count is measured, not supplied |
| layer count | propagation runs until the signal falls below the derived floor — depth is measured per query |
| a fallback answer | refusal, as arithmetic: nothing above the floor means nothing out |

The one number that is not derived is the false-alarm level `far` (α). By Neyman–Pearson it belongs
to the reader, and no function in the package defaults it, so a reading cannot be taken without
naming the level it was taken at.

## Install

Python 3.11 or later:

```bash
pip install entroptics-infer              # the engine; pulls entroptics>=0.2.12 and numpy
pip install "entroptics-infer[store]"     # adds agience-mantle, for a persistent lattice
```

The import is `entroptics_infer`. Importing it does not import torch; `DerivedAttention` needs torch
installed separately.

## Using it as a library

### The data model: a statement is a vertex

The store is statements, and a statement is a tuple of vertex names of any width:

```python
("dog", "is_a", "mammal")
("wheel", "part_of", "bicycle")
```

Three things about that are load-bearing, and code that does not respect them gets wrong answers
rather than errors.

**Roles come from position, not from names.** Nothing anywhere says that the middle slot is "the
relation". A relation is found wherever it sits between two other parts, and what precedes and
follows it are read off from there. So a store written in another order works without anything being
added, and statements of different widths coexist.

**A relation is a vertex like any other.** Anything used between two other parts has an operator by
the same rule, and something never used that way has none. There is no relation vocabulary to
declare, and adding a relation is writing a statement that uses it.

**Nothing is stored on a vertex.** No embedding, no type, no label. A vertex's coordinate is
computed from the edges when a read asks for one, and is discarded. See
[What it costs](#what-it-costs-and-how-to-keep-it-cheap) before caching it.

### Building a store

In memory, the form to use in a test or a notebook:

```python
from entroptics_infer import Engine

engine = Engine.on_statements([
    ("dog", "is_a", "mammal"),
    ("cat", "is_a", "mammal"),
    ("mammal", "is_a", "animal"),
])
```

Backed by mantle (`pip install "entroptics-infer[store]"`), the form to use for anything that
outlives the process:

```python
from entroptics_infer.store import open_lattice

with open_lattice("lattice.db", origin="a-name-for-this-node") as lattice:
    engine.write_to(lattice)          # statements in
    back = Engine.on_lattice(lattice, seed="statement:0", size=10_000)
```

`origin` has no default. mantle refuses a generated one because it would change on restart and fork
the node's proper time.

**What `write_to` puts in the lattice**: each statement becomes a vertex, and each of its parts an
edge carrying only the position it arrives at. No relation name is written as a type. A statement
vertex is recognized on the way back by what it *is* — a vertex whose edges carry positions — not by
anything declaring it.

**`on_lattice` reads a connected region, not the whole store.** It walks outward from `seed` until it
has `size` vertices. A random sample of a large store is almost entirely disjoint, and every question
would then ask about a vertex sharing no structure with any other. `size` is your declared reach, and
the cost of a read grows with it.

### Reading a lattice somebody else wrote

Reading an existing lattice is a different job from reading one you built, and the seam has the
methods for it. Use them rather than opening the file: a lattice is a sqlite database, and selecting
from `edge` directly reads a schema this package does not own, past whatever the store does about
ordering, partial results and its own counters.

```python
with open_lattice(path, origin="your-reader") as lattice:
    if not lattice.edge_labels_measured():
        lattice.measure_labels()          # writes; see below
    for label in lattice.labels():
        print(label, lattice.count_by_label(label))

    statements = lattice.statements(label="part_of")   # or label=None for every label
    kinds = lattice.kinds([v for st in statements for v in (st[0], st[2])])
```

| method | what it asks mantle |
|---|---|
| `labels()` | `labels_with_edges()` — every label the store has edges under |
| `count_by_label(l)` | `count_edges_by_label(l)` — from the store's own counter |
| `statements(label=…)` | `src_ids_by_label()` then `edges_of()`, as `(src, label, dst)` |
| `kinds(vertices)` | the artifact store's `get_many()` — what each vertex is |
| `measure_labels()` | `backfill_edge_label_counters()` — this writes |

**`labels()` returning nothing means "I don't know", not "there are none."** The counters are
backfilled, and a store that was never backfilled cannot answer. `edge_labels_measured()` is the
question to ask, and `measure_labels()` settles it. It is named for the fact that it writes, so that
a reader never silently modifies somebody else's store. Read a copy if the original matters.

### Asking

```python
answer = engine.infer("dog", "is_a")

answer.named     # the vertex it names, or None: a refusal, not an error
answer.scores    # the attention row over the candidates, in the order offered
answer.stated    # whether the store, as read, still contains this statement
answer.pool      # how many candidates were offered
answer.reached   # how many of them this relation points into elsewhere
```

**`infer` holds out by default.** Any statement of that relation from that source is removed from the
operator *and* from the coordinate before anything is computed. Keep that on while evaluating: it is
what makes an accuracy figure mean anything. Turn it off with `hold_out=False` when serving a
question whose answer the store does not hold, because then there is nothing to hold out and
rebuilding the coordinate is pure cost.

**Check `answer.stated`.** If it is true, the store still contains the statement and you have measured
a lookup.

**Use `answer.reached` as your baseline, not `answer.pool`.** Naming one of 65 sounds impressive
until the relation only ever points at 12 things; the number to beat is a draw among those 12.

The lower-level calls, when you want the pieces:

```python
vectors = engine.places()                              # the coordinate: the embedding table
M       = engine.operator("is_a", vectors)             # the relation's weight matrix
landed  = engine.carry("dog", "is_a", vectors)         # where the relation takes a vertex
row     = engine.attend("dog", "is_a", candidates, vectors=vectors)   # the attention row
```

Pass `vectors` explicitly whenever you make more than one call. Every method that takes it builds
its own otherwise, and that is the expensive part.

**`operator` returns `None` on a relation used fewer than twice.** A cross-covariance over one row is
that row and would predict that row and nothing else, so a matrix there would report a lookup as an
operator. Handle the `None`; it is not a failure.

**`attend` does not normalize.** Rows do not sum to one, weights can be negative, and a query that
resolves nothing returns zeros. Downstream code that assumes a probability distribution will misread
a refusal as a uniform belief.

### Learning

```python
engine = engine.relate("penguin", "is_a", "bird")
```

That is the whole of it. A new engine is returned rather than the old one mutated, so code comparing
before and after compares two stores rather than one store at two times.

Nothing is retrained and no coordinate is updated, because none was stored. The next read derives a
different operator because the new statement is in the sum it is derived from;
`operator_append` in [`research/lean/Infer/Carry.lean`](research/lean/Infer/Carry.lean) proves the
operator gains exactly that statement's rank-one term and nothing else moves.

Two measured things to expect
([`test_whether_a_fact_arrives_without_retraining.py`](research/checks/test_whether_a_fact_arrives_without_retraining.py)):

- **A statement arrives in a few milliseconds** on a small store — write, coordinate rebuilt,
  operator rederived, question answered: a median of 4.3 ms on the reference machine.
- **A relation needs a handful of uses, not most of itself.** First correct answer comes at a mean of
  3.5 statements. On the measured store 15 of 26 questions were never right however many were
  added, so writing more of the same relation does not rescue a question that is not landing.

### What it costs, and how to keep it cheap

Measured on WordNet cores through this engine, each figure the median of repeated calls as stored in
the record
([`test_what_a_read_costs_as_the_store_grows.py`](research/checks/test_what_a_read_costs_as_the_store_grows.py)):

| statements | coordinate | carry | held-out question | served question |
|---|---|---|---|---|
| 172 | 0.0107 s | 0.0003 s | 0.0114 s | 0.0004 s |
| 363 | 0.0309 s | 0.0014 s | 0.0343 s | 0.0014 s |

The seconds belong to the machine and move from run to run; the proportions and the growth are the
finding. The coordinate is nearly the whole cost. Applying the operator, the inference itself, is
about a twentieth of the coordinate at 363 statements. Step by step the coordinate's time grows
faster than the store and nearer linear than quadratic: ×3.96 and ×2.88 where the store grew ×2.87
and ×2.11.

So the one thing that decides throughput is how often the coordinate is rebuilt:

```python
vectors = engine.places()                     # once, when the store last changed
for q in questions:                           # ~0.0014 s each at 363 statements
    engine.attend(q, relation, candidates, vectors=vectors)
```

**Rebuild it when the store changes, not per question.** The exception is evaluation under
hold-out, which must rebuild per question, because a cached coordinate would be answering from
material that contains the answer. That is about 25× more expensive, and it is the price of a sound
measurement, not the price of running.

**Where the walk stops.** A place is built by a walk that settles every vertex at the largest value
that reaches it, largest first. Read to its end that walk reaches the whole connected region, and a
place is non-zero on 100% of the axes at every size measured, from 50 to 705 statements: the
coordinate is quadratic in the method, with the non-zero entries growing 185.9 times over a store
that grew 14.1, an exponent of 1.975.

`places()` does not read it to its end. Each walk stops at its own aperture: the participation
number `2^H` of the values it has settled, read as they arrive, and the walk stops once the unsettled
tail can no longer change that count. Nothing is passed in and there is no level to choose; a hub
and a leaf stop in different places, from their own spectra.

On WordNet cores walked out from `entity.n.01`, a place touches 14.1% of the axes at 50 statements
and 10.4% at 705, and the non-zero entries grow at an exponent of 1.860. On 172 held-out questions
the read names 26.2% at its aperture against 18.6% read to its end, 19.0 times a draw among the
reached against 13.2. On the ontology a place resolves 4.1 vertices of 150 and names 36.8% of 68
held-out statements against 32.4% read to its end.

The aperture changes the growth, but 1.860 is not 1. Nothing here shows a 100k-statement store works.

### As a torch layer

```python
from entroptics_infer.torch_layer import DerivedAttention, has_nothing_to_train

layer = DerivedAttention(far=0.01)
has_nothing_to_train(layer)     # {'parameters': 0, 'parameter_elements': 0, ...}
out, weights = layer(query, key, value)
```

`far` is the false-alarm level, it is required, and nothing in this package defaults it.

**The shapes differ from `nn.MultiheadAttention`:**

```
nn.MultiheadAttention   query (Q, D)      key (K, D)      -> weights (Q, K)
DerivedAttention        query (Q, S, D)   key (K, S, D)   -> weights (Q, K)
```

A token here is a frame, an ordered axis and a feature axis, not a vector. The statistic is the
alignment of two records' cross-covariance standardized by a re-pairing of their rows, and a
re-pairing of one row is not a null. Both sides must carry the same `S`.

**If your model's tokens are vectors with nothing ordered inside them, this layer does not apply**,
and adapting it will not help, because the thing it reads is not there. If a token is a span, a
window, a sequence of measurements, or a document's sentences, it applies directly.

Expect from the weights: rows that do not sum to one, weights that can be negative, and no gradient
through the decision. The zeroing is a threshold derived from the record's own entropy and is not
differentiable. This is a layer for models that read, not for models that train through it.

### Where it does not apply

- **Tokens with no internal order.** There is nothing for the null to re-pair (see the torch layer
  above).
- **Relations that are symmetric.** The operator carries a source's directions onto a target's
  *different* directions; a symmetric relation is represented perfectly well by a scalar and this
  buys nothing.
- **Relations stored in both directions, under two names.** If a store holds `(a, hypernym, b)` *and*
  `(b, hyponym, a)`, holding one out leaves the other, and a lookup scores as inference. Keep one
  direction of each inverse pair.
- **Relations stored in both directions under the same name.** A relation can be its own inverse —
  WordNet's `attribute` and `also_see` are — and then there is no second name to leave out. A
  symmetric label left in states every held-out fact backwards. Measure the reciprocity per label,
  which `survey()` does, and drop any relation that has any.
- **The same pair recorded under two labels.** If a store holds `(a, aligns, b)` *and*
  `(a, colimit_of, b)`, holding out one leaves the other stating the same destination, and the read
  only has to notice that `a` and `b` are related at all. Neither label is the other's inverse, and
  nothing about their names suggests they overlap; the tell is labels with exactly equal statement
  counts. Either hold out every statement joining the two vertices, or measure the overlap and say
  what you did.
- **Vertices named once.** A vertex mentioned in a single statement cannot be held out — removing its
  only statement removes the vertex — so it is unaskable by construction. In a taxonomy walked
  outward that is most of a store. Drop vertices named once, repeatedly, until every vertex left is
  named at least twice (the 2-core).
- **Provenance labels.** A label every vertex carries (`created`, or similar) is a property of the
  vertex, not a relation between two pieces of content, and it swamps everything else. Leave it out.
- **Large stores.** See [What it costs](#what-it-costs-and-how-to-keep-it-cheap).

The first three failures are the ones most likely to produce an impressive wrong result, and all are
cheap to detect. `survey()` checks every one of them in a second:

```python
from entroptics_infer import survey

s = survey(your_statements)
s["labels_holding_both_directions"]   # drop these relations entirely
s["pairs_under_two_labels"]           # hold out the whole pair where this is not zero
s["combinations"]                     # which labels travel together
s["not_askable"]                      # statements no held-out question can be put to
```

Run it on a store you did not build and on one you did.

## The command line

A corpus is a directory of `.csv` files. Each file is one record: rows are the shared ordered axis,
columns are its channels, and every record has the same shape.

```
entroptics-infer --far 0.05 describe CORPUS
entroptics-infer --far 0.05 admit    CORPUS --seed 1 --draws 3
entroptics-infer --far 0.05 ask      CORPUS --query NAME
entroptics-infer --far 0.05 complete CORPUS --exemplar A B --source C
entroptics-infer --far 0.05 reach    CORPUS --edges EDGES.csv --origin A
entroptics-infer --far 0.05 lattice  CORPUS --edges EDGES.csv --seed 5 --draws 250
```

(`python -m entroptics_infer` is the same command.) `--far` is required and has no default, and no
other threshold or level is set: `--seed` and `--draws` only say which shuffles to compare against,
so that a verdict is reproducible. `admit` runs first and answers whether these reads apply to the
corpus at all. `ask` either binds, showing the candidates and why it decided, or returns `REFUSED`
with a non-zero exit status: refusal is an answer and reaches the exit code. `complete` is the
forward pass, given a relation by example as a pair of records that stand in it. `reach` walks an
edge list until the signal falls below the floor, so depth is a property of the query. `lattice`
reads the records together with named edges (`src,dst,relation`, the relation being whatever string
the edge carries) and reports two things that need not agree: whether the store holds each relation,
and whether a single example pair could name it. With `--source`, `--relation` and `--candidates` it
also asks which record stands in that relation to the source.

In a clone of this repository, [`research/lib/demo_corpus.py`](research/lib/demo_corpus.py) writes a
corpus to try it on: 240 records under `demo/records/` and their edges in `demo/edges.csv`.

```bash
python research/lib/demo_corpus.py demo
entroptics-infer --far 0.05 describe demo/records
entroptics-infer --far 0.05 admit    demo/records --seed 1 --draws 3
entroptics-infer --far 0.05 lattice  demo/records --edges demo/edges.csv --seed 5 --draws 250
entroptics-infer --far 0.05 lattice  demo/records --edges demo/edges.csv --seed 5 --draws 250 \
    --source s07 --relation quiet --candidates loud07 quiet07 none07
```

## The research

### Layout

```
src/entroptics_infer/    the package: the engine, the scores, relation types, propagation, the CLI
tests/                   the package test suite (pytest)
research/                the paper (Markdown, LaTeX, PDF), the claims ledger, related work, and the
                         gates (verify.py, verify_records.py, prose_check.py)
research/experiments/    one script per experiment (expNN_*.py)
research/checks/         one script per research question (test_*.py), run as programs
research/results/        the raw output every ledger row cites
research/lib/            the shared helpers the experiments and checks import: the generated
                         stores and corpora, the hand-written ontology, the reader (ask.py), the
                         curriculum and its conversational shell (tutor.py), the record writer
research/lean/           the machine-checked core (Lean 4 + Mathlib)
research/typeset/        the two-column build of the paper (pandoc + XeLaTeX)
```

`research/corpora/` is created by the experiments that need it and is git-ignored: corpora are
built fresh from a stated process and seed, or fetched by the experiment that reads them, which
records the SHA-256 of what it received.

### From source

```bash
git clone https://github.com/Agience/entroptics-infer
cd entroptics-infer
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"           # pulls entroptics>=0.2.12 and numpy from PyPI
pytest                            # the package suite under tests/
```

`.[dev,store]` adds [`agience-mantle`](https://github.com/Agience/agience-mantle), the store seam;
without it the store tests skip and the instrument half still runs. `conftest.py` lets the suite
import the package from `src/` without an install.

### Run the research checks

The research checks are programs, not pytest tests: each prints its table, decides its own verdict,
exits non-zero on failure, and records what it measured under `research/results/`. Running one
rewrites its record. Some need more than the package: `torch` for the trained baselines; `nltk` with
its WordNet corpus (`python -m nltk.downloader wordnet`) for the WordNet reads; the prose bodies under
`research/corpora/`, which the text experiments (`research/experiments/exp07_real_text.py` and those
after it) fetch; and, for the reads of a real lattice, `agience-mantle` and a private mantle lattice
named by `ENTROPTICS_LATTICE`. Without the lattice those checks say they skipped, exit 0 and write no
record; v0.1.0 reports no figure from a real lattice.

The fast gates, in seconds:

```bash
python research/checks/test_the_ledger_is_consistent.py    # the ledger's addresses and citations
python research/checks/test_the_method_carries_no_constant.py
python research/checks/test_what_constants_are_left_in_the_answer_path.py
python research/prose_check.py                             # every figure in prose against its script
python research/verify.py --paths                          # every registered figure resolves
python research/lean/count.py                              # what the Lean development proves
```

Re-measuring everything:

```bash
python research/verify.py                                  # re-run every experiment, report drift
python research/verify_records.py                          # re-run every research check, compare records
for f in research/checks/test_*.py; do python "$f"; done  # each of the 55 research checks
```

One check takes a single question; for example

```bash
python research/lib/ask.py salmon lives_in
python research/lib/ask.py where does a salmon live
python research/lib/ask.py what is quantum chromodynamics
```

asks the reader a question of the hand-written ontology in `research/lib/ontology.py`: the first names a
relation, the second names none so the reader chooses one and says which, and the third names
nothing the store holds, so the reader refuses.

### Build the Lean

The formal development is in [`research/lean/`](research/lean/), pinned to Lean `v4.31.0` and the
Mathlib commit in `lakefile.toml` / `lake-manifest.json`. With [elan](https://github.com/leanprover/elan)
installed:

```bash
cd research/lean
lake exe cache get        # download Mathlib's prebuilt oleans
lake build                # compile Infer and its seven files
lake env lean axioms.lean # print what every headline theorem depends on
```

`#print axioms` on every headline theorem returns only `propext`, `Classical.choice` and
`Quot.sound`. [`research/lean/README.md`](research/lean/README.md) states what each file proves.

## The three rules, and how each is enforced

1. Entroptics is load-bearing — it decides, without a supplied constant.
   [`test_entroptics_is_loadbearing.py`](tests/test_entroptics_is_loadbearing.py) substitutes a
   blind instrument and requires every downstream read to change.
2. No constants, thresholds, fitting or forcing — an AST walk over the package
   ([`test_no_constants.py`](tests/test_no_constants.py)) refuses every float literal and every
   integer outside shape and index arithmetic, and planted constants in spellings chosen to defeat
   a text search are checked to be rejected.
3. Every claim is backed by data — [`research/CLAIMS.md`](research/CLAIMS.md) carries the
   experiment and the design it ran at for every published number, and a row may cite only a
   runnable script.
   [`research/verify.py`](research/verify.py) re-runs the experiments and checks the registered
   figures against what comes back.

## Data policy

No corpus is committed. A corpus is either generated or fetched, and nothing else is read.

- Generated corpora are built fresh by the experiment that consumes them, from a stated process and
  a stated seed. They are small, and the generator is the record.
- Real corpora are fetched by the experiment that reads them: public-domain books from Project
  Gutenberg, datasets from the UCI Machine Learning Repository, the CORGIS weather table, and the
  PEMS04 traffic set, which `exp51` expects in `research/corpora/` before it runs. The WordNet reads
  use NLTK's WordNet corpus. A fetch records the SHA-256 of what it received.

`research/corpora/` is a cache of what was fetched and is git-ignored.

## Where it sits in the literature

A relation being a matrix rather than a scalar is not new, and this does not claim it. RESCAL,
DistMult, R-GCN, Smolensky's tensor-product binding and linearized attention all carry one. In every
one of them the matrix is obtained by gradient descent and the relation index is supplied with the
dataset. Here it is measured and the index does not exist.
[`research/RELATED-WORK.md`](research/RELATED-WORK.md) states the difference in a form that can be
disagreed with.

## Citing

[`CITATION.cff`](CITATION.cff) carries the citation metadata.

## Licence

AGPL-3.0-only, or a commercial licence — see [`LICENSE`](LICENSE) and
[`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Patent pending — see [`PATENTS.md`](PATENTS.md).
