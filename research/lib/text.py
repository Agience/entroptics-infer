"""A real text, turned into records the instrument can read — with nothing trained anywhere.

What a word's record is
-----------------------
A word type's record is a ``(segments, features)`` field. The **ordered axis** is position in the
text, cut into segments, and the **feature axis** is co-occurrence with the commonest word types.
Row ``i`` is what surrounded that word in segment ``i``.

Every word's record shares the same ordered axis — the same segments, in the same order — which is
what lets two of them be compared at all: the association is a re-pairing of rows, so the rows have
to correspond to the same thing.

Nothing here is trained. There is no embedding table, no distillation, no matrix fitted to a
corpus. A record is a count of what was observed next to what, and the basis it is read in is
derived from the corpus by the instrument.

Why the feature axis is co-occurrence and not something cleverer
-----------------------------------------------------------------
Because anything cleverer is trained. `paper-2` (*Knowledge without weights*,
``research/paper-2-knowledge-without-weights.md`` in https://github.com/Agience/agience-pharos) §70 rules out the other obvious option: a coordinate
keyed on an identifier is orthogonal across id families by construction. Raw
co-occurrence is the coordinate a corpus supplies about itself, and `paper-2` §65's finding — that a
sparse ontology coordinate reads to the aperture exactly as noise does — is why the feature axis is
restricted to types common enough for the rows to be populated.

The parameters here are the **experiment's declared design**, not method constants: how many
segments, how wide a window, how many feature types, how often a word must occur. They are reported
with every result and swept where they could matter. The method itself — everything in
``src/entroptics_infer`` — still carries none.

Data policy
-----------
The text is fetched fresh, into a git-ignored directory, by the script that consumes it. Its URL and
the SHA-256 of what arrived are recorded with the result, so a later run can tell whether it read the
same bytes.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Text", "TextRecords", "fetch", "tokenise", "build_records", "cooccurrence"]

#: Public-domain, small, and fetched rather than carried in the repository.
ALICE = "https://www.gutenberg.org/files/11/11-0.txt"

_WORD = re.compile(r"[a-z]+")
_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)


@dataclass(frozen=True)
class Text:
    """A fetched text and what it was."""

    url: str
    sha256: str
    characters: int
    tokens: tuple


@dataclass(frozen=True)
class TextRecords:
    """Word records over a shared segment axis, and the design they were built at."""

    records: dict                      # word -> (segments, features) frame
    vocabulary: tuple                  # the words that have records, most frequent first
    features: tuple                    # the word types the feature axis counts
    segments: int
    window: int
    counts: dict = field(repr=False)   # word -> occurrences, for reporting


def fetch(url: str = ALICE, *, into: pathlib.Path) -> Text:
    """Fetch a text, or reuse the copy already fetched into ``into``.

    Re-reading a file this function itself fetched is not a breach of the data policy: the policy is
    against starting from data the program did not generate or fetch under its own stated process.
    The SHA-256 is returned so a run can say which bytes it read.
    """
    into.mkdir(parents=True, exist_ok=True)
    dest = into / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".txt")
    if not dest.exists():
        subprocess.run(["curl", "-sS", "-L", "--max-time", "60", url, "-o", str(dest)], check=True)
    raw = dest.read_bytes()
    body = raw.decode("utf-8", errors="replace")
    # Strip the licence wrapper, which is boilerplate shared by every Gutenberg text and would
    # otherwise be the strongest co-occurrence structure in a small corpus.
    start = _START.search(body)
    end = _END.search(body)
    if start:
        body = body[start.end():]
    if end:
        body = body[: end.start()] if not start else body[: _END.search(body).start()]
    return Text(url=url, sha256=hashlib.sha256(raw).hexdigest(),
                characters=len(body), tokens=tuple(tokenise(body)))


def tokenise(body: str) -> list:
    """Lowercase alphabetic tokens. No stemming, no stopword list, no tagger."""
    return _WORD.findall(body.lower())


def cooccurrence(tokens, *, window: int, features: tuple) -> dict:
    """Total co-occurrence counts of every token type with each feature type."""
    index = {w: i for i, w in enumerate(features)}
    totals: dict = {}
    n = len(tokens)
    for p, w in enumerate(tokens):
        row = totals.setdefault(w, np.zeros(len(features)))
        lo, hi = max(0, p - window), min(n, p + window + 1)
        for q in range(lo, hi):
            if q == p:
                continue
            j = index.get(tokens[q])
            if j is not None:
                row[j] += 1.0
    return totals


def build_records(text: Text, *, segments: int, window: int, n_features: int,
                  min_count: int, remove_baseline: bool = True) -> TextRecords:
    """Word records over a shared segment axis.

    ``min_count`` keeps out types too rare for their rows to be populated — a record that is mostly
    empty reads to the aperture as noise does, which `paper-2` §65 measured and which would put
    unresolvable frames into the comparison rather than telling us anything.

    ``remove_baseline`` subtracts the corpus's **own average context profile** from every record.
    Every word in a corpus sits next to the same common words, so every record carries the corpus
    unigram distribution as a large shared component that says nothing about which word it is. That
    is the identity share `entroptics-jlens` measures inflating a derived floor by 2× to 21×.

    Measured on Moby-Dick's first half at 48 segments, the subtraction lowers what is readable:
    records resolving anything at all go from **302 of 302** raw to **193 of 302** with it, and the
    mean resolved rank from **1.46 to 0.76**, with a maximum of 4 against 7
    (`research/experiments/exp11_text_diagnostics.py`, part A).

    The baseline is a **measured** quantity, the mean over this corpus's own records, not a constant
    brought from anywhere. Each row is first put on a common footing as a distribution over
    features, so that a word's frequency is not mistaken for its character.
    """
    tokens = list(text.tokens)
    counts = Counter(tokens)
    features = tuple(w for w, _ in counts.most_common(n_features))
    index = {w: i for i, w in enumerate(features)}

    vocabulary = tuple(w for w, c in counts.most_common() if c >= min_count)
    slot = {w: i for i, w in enumerate(vocabulary)}

    per_segment = max(1, len(tokens) // segments)
    frames = {w: np.zeros((segments, len(features))) for w in vocabulary}

    n = len(tokens)
    for p, w in enumerate(tokens):
        i = slot.get(w)
        if i is None:
            continue
        seg = min(segments - 1, p // per_segment)
        row = frames[w][seg]
        lo, hi = max(0, p - window), min(n, p + window + 1)
        for q in range(lo, hi):
            if q == p:
                continue
            j = index.get(tokens[q])
            if j is not None:
                row[j] += 1.0

    if remove_baseline and vocabulary:
        order = list(vocabulary)
        stack = np.stack([frames[w] for w in order])            # (words, segments, features)
        share = stack.sum(axis=2, keepdims=True)
        share[share == 0] = 1.0
        profile = stack / share                                  # each row a distribution
        baseline = profile.mean(axis=0, keepdims=True)           # the corpus's own average context
        centred = profile - baseline
        frames = {w: centred[i] for i, w in enumerate(order)}

    return TextRecords(records=frames, vocabulary=vocabulary, features=features,
                       segments=int(segments), window=int(window),
                       counts={w: counts[w] for w in vocabulary})


def profiles(code, V, split, window):
    """Each symbol's decayed co-occurrence profile over the first ``split`` positions of ``code``.

    ``V`` symbols plus one for out-of-vocabulary make an ``(V + 1, V + 1)`` count, each pair within
    ``window`` positions of each other weighted by ``exp(-gap / window)`` and counted both ways round.
    Every row is scaled to unit length; a row that never occurred is left at zero.
    """
    F = V + 1
    P = np.zeros((F, F))
    decay = np.exp(-np.arange(1, window + 1) / window)
    for p in range(split):
        c = code[p]
        for j, q in enumerate(range(p - 1, max(-1, p - window - 1), -1)):
            P[c, code[q]] += decay[j]
            P[code[q], c] += decay[j]
    n = np.linalg.norm(P, axis=1, keepdims=True)
    return P / np.where(n > 0, n, 1.0)
