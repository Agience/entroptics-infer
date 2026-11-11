"""Inference from a corpus's live geometry, with no trained model and no fitted constant.

What this package is
--------------------
A transformer computes its attention pattern at read time, from the data. What it *trains* is the
embedding table, the projections ``W_Q``/``W_K``/``W_V``/``W_O``, and the MLP. This package is the
test of a single claim: that everything a transformer trains can instead be derived from the
corpus at read time, and that the derivation costs no fitted constant.

The association between two records is a matrix — their cross-covariance — not a scalar. That
much is not new; RESCAL, DistMult, R-GCN and linearized attention all carry a matrix-valued
relation. What is new is that the matrix is *measured* rather than fitted, and that the number of
relations is *counted* rather than supplied with the dataset. ``research/RELATED-WORK.md`` states
the difference in a form that can be disagreed with.

The three rules
---------------
1. **Entroptics is load-bearing** — it decides, without a supplied constant. Checked by
   ``tests/test_entroptics_is_loadbearing.py``, which neuters the instrument and requires every
   downstream read to change.
2. **No constants, thresholds, fitting or forcing** — checked by an AST walk over this package in
   ``tests/test_no_constants.py``. Every float literal is refused; integers are refused outside
   shape and index arithmetic.
3. **Every claim is backed by data** — ``research/CLAIMS.md`` carries the experiment and the design for
   each published number, and nothing enters ``research/PAPER.md`` without a row there.

``far`` — the one number that is not derived
--------------------------------------------
The false-alarm level belongs to the reader, by Neyman–Pearson, and this program does not derive
it. No function here defaults it, so a reading cannot be taken without naming the level it was
taken at, and every published claim is a sweep across it rather than a value at one point.

Layout
------
``engine``   the one seam onto ``entroptics``. Nothing else in the package imports the instrument.
``bind``     the association as a matrix, and what is counted off it to decide a binding.
``relations`` relation type as a measured position among the kinds a corpus turns out to contain.
``propagate`` how far a signal reaches — depth measured per query, with nowhere to put a hop cap.
``infer``     the forward pass: two of a triple in, the third out, the operator given by example.
``category``  colimits and limits over records — a set read as one object, with no aggregation chosen.
``admission`` does this corpus admit these reads at all — run it before scoring any task.
``store``    the one seam onto ``agience-mantle``. Nothing else in the package imports the store.

Read ``research/PAPER.md`` first: it states the method and what is
measured, and ``research/CLAIMS.md`` names the check behind every figure.
"""
from __future__ import annotations

from . import (admission, bind, category, engine, infer, ingest, propagate, relations, store,
               survey as survey_module, wrapper)
from .bind import (BOUND, INDISTINGUISHABLE, NOTHING_RESOLVED, NO_SHARED_CHANNEL,
                   Binding, Decision, association_matrix, attend, decide, select, shared_modes)
from .admission import Admission, admits, axis_carries, facet_independence
from .category import Diagram, colimit, diagram, inside, limit
from .engine import Association, Unreadable, associate, basis, floor, frame, rank
from .infer import (Completion, complete, complete_with, operator_of,
                    operator_over, relation_over)
from .lattice import Lattice, Verdict
from .propagate import Reach, horizon, reach
from .wrapper import Answer, Engine
from .survey import core, survey
from .ingest import corpus, memberships, spans
from .relations import (Screen, carries_kinds, relation_basis, relation_coordinates,
                        relation_modes, signature,
                        signatures)

__all__ = [
    # prose in, statements out -- the step between a corpus and anything else here
    "ingest", "corpus", "spans", "memberships",
    "Engine",
    "survey",
    "core",
    "Answer",
    "wrapper",
    "store",
    "bind",
    "engine",
    "relations",
    "propagate",
    "infer",
    "admission",
    "Admission",
    "admits",
    "axis_carries",
    "facet_independence",
    "category",
    "Diagram",
    "colimit",
    "diagram",
    "inside",
    "limit",
    "Completion",
    "complete",
    "complete_with",
    "operator_of",
    "operator_over",
    "relation_over",
    "Lattice",
    "Verdict",
    "Reach",
    "horizon",
    "reach",
    "Screen",
    "carries_kinds",
    "relation_basis",
    "relation_coordinates",
    "relation_modes",
    "signature",
    "signatures",
    "Binding",
    "Decision",
    "association_matrix",
    "attend",
    "decide",
    "select",
    "shared_modes",
    "BOUND",
    "INDISTINGUISHABLE",
    "NOTHING_RESOLVED",
    "NO_SHARED_CHANNEL",
    "Association",
    "Unreadable",
    "associate",
    "basis",
    "floor",
    "frame",
    "rank",
]
