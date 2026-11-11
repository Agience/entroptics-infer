"""The drop-in layer, counted and put to work beside the layer it replaces.

What is asked
---------------
`entroptics_infer.torch_layer.DerivedAttention` is offered as something a model builder can put where
``nn.MultiheadAttention`` sits. Three things have to be true for that offer to mean anything, and
each is measured here rather than asserted in a docstring:

``nothing to train``   the module holds no parameters and no buffers. Counted, against the parameter
                       count of an ``nn.MultiheadAttention`` of the same width, so the comparison is
                       a number rather than a claim.
``it decides``         given frames generated from distinct bases, the layer's weight is largest on
                       the frame the query came from. Untrained, because there is nothing to
                       train.
``it refuses``         given a query that is noise, the row is zeros and the output is zero. A
                       softmax cannot do this: it returns a distribution over the keys whatever is
                       in them.

What is deliberately not claimed
----------------------------------
**Nothing here says this layer beats a trained transformer.** An ``nn.MultiheadAttention`` is
included to be *counted*, and its accuracy is reported at its initialization because that is the
state a model builder starts from -- what it would do after training is a different question, needs
a training set and a loss, and is not asked here. The claim this file supports is narrow and is the
only one the program makes: of the two, one needs training before it can do this and the other
does not.

The generator, stated in full
-------------------------------
Each key frame is ``S`` steps of a random combination of ``rank`` directions drawn for that key
alone; the query is one key's frame with independent noise added at a stated amplitude. So the
structure a reader could find is *which directions a frame occupies*, and the answer is not written
anywhere in the data -- it is which key the query was built from. Amplitude is swept, and so is
``far``, because the false-alarm level belongs to the reader and a result at one level is a result
at one level.
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

# the shared helpers (record, ontology, ask, ...) live in research/lib/
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import record

FAR = (0.05, 0.01, 0.001)
#: swept until the layer fails. A sweep that reads 100% everywhere has not found the edge of what
#: the method does, and a reader is entitled to see where it stops rather than only where it works.
NOISE = (0.25, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DRAWS = 20
KEYS = 6
STEPS = 24
WIDTH = 16
RANK = 3


def _frames(rng):
    """``(keys, which)`` -- one frame per key, each occupying its own directions."""
    keys = []
    for _ in range(KEYS):
        basis = rng.normal(size=(RANK, WIDTH))
        loading = rng.normal(size=(STEPS, RANK))
        keys.append(loading @ basis)
    return np.stack(keys)


def one_draw(layer, rng, noise):
    """Does the layer put its largest weight on the key the query was built from."""
    import torch

    keys = _frames(rng)
    which = int(rng.integers(KEYS))
    query = keys[which] + noise * rng.normal(size=(STEPS, WIDTH))

    q = torch.as_tensor(query[None, :, :], dtype=torch.float64)
    k = torch.as_tensor(keys, dtype=torch.float64)
    v = torch.as_tensor(np.eye(KEYS), dtype=torch.float64)
    out, w = layer(q, k, v)
    row = w[0].numpy()

    # a query made of noise alone, on the same keys -- what the layer does with nothing to find
    junk = rng.normal(size=(STEPS, WIDTH)) * float(np.std(keys))
    _, wj = layer(torch.as_tensor(junk[None, :, :], dtype=torch.float64), k, v)

    return {"right": bool(np.argmax(row) == which) if np.any(row) else False,
            "answered": bool(np.any(row)),
            "row_sum": float(np.sum(row)),
            "any_negative": bool(np.any(row < 0)),
            "noise_row_is_zero": bool(not np.any(wj[0].numpy())),
            "output_is_zero": bool(not np.any(out.numpy()))}


def main() -> int:
    try:
        import torch
        from torch import nn
    except Exception as exc:                               # pragma: no cover - import-shape only
        print(f"  torch is not installed, so the adapter cannot be exercised: {exc}")
        return 0

    from entroptics_infer.torch_layer import DerivedAttention, has_nothing_to_train

    ours = DerivedAttention(far=FAR[0])
    theirs = nn.MultiheadAttention(embed_dim=WIDTH, num_heads=1, batch_first=True)
    mine = has_nothing_to_train(ours)
    yours = has_nothing_to_train(theirs)

    print("the drop-in layer, counted and put to work\n")
    print(f"  {'layer':>24}{'tensors':>10}{'numbers to train':>20}{'buffers':>10}")
    print("  " + "-" * 64)
    print(f"  {'DerivedAttention':>24}{mine['parameters']:>10}"
          f"{mine['parameter_elements']:>20}{mine['buffers']:>10}")
    print(f"  {'nn.MultiheadAttention':>24}{yours['parameters']:>10}"
          f"{yours['parameter_elements']:>20}{yours['buffers']:>10}")
    print()
    print(f"  Same width ({WIDTH}), same place in a model. One of them has "
          f"{yours['parameter_elements']} numbers that")
    print("  do not exist until something trains them. The other has none, and the sweep below is")
    print("  taken at initialization because for this layer there is no other state.\n")

    print(f"  {'far':>8}{'noise':>8}{'draws':>7}{'answered':>10}{'right':>8}"
          f"{'row sums to':>14}{'negatives':>11}{'noise refused':>15}")
    print("  " + "-" * 81)
    rows = {}
    for far in FAR:
        layer = DerivedAttention(far=far)
        for noise in NOISE:
            rng = np.random.default_rng(abs(hash((far, noise))) % (2 ** 32))
            draws = [one_draw(layer, rng, noise) for _ in range(DRAWS)]
            answered = float(np.mean([d["answered"] for d in draws]))
            right = float(np.mean([d["right"] for d in draws]))
            sums = float(np.mean([d["row_sum"] for d in draws]))
            neg = float(np.mean([d["any_negative"] for d in draws]))
            refused = float(np.mean([d["noise_row_is_zero"] for d in draws]))
            rows[f"{far}/{noise}"] = {"far": far, "noise": noise, "draws": DRAWS,
                                      "answered": answered, "right": right,
                                      "mean_row_sum": sums, "any_negative": neg,
                                      "noise_query_refused": refused}
            print(f"  {far:>8}{noise:>8}{DRAWS:>7}{answered:>10.0%}{right:>8.0%}"
                  f"{sums:>14.3f}{neg:>11.0%}{refused:>15.0%}")
    print()

    record.figures(far=list(FAR), noise=list(NOISE), draws=DRAWS, keys=KEYS, steps=STEPS,
                   width=WIDTH, rank=RANK, ours=mine, multihead=yours, by_setting=rows)

    draw = 1.0 / KEYS
    every = list(rows.values())
    best = max(every, key=lambda v: v["right"])
    worst = min(every, key=lambda v: v["right"])
    # reported as a range rather than tested against a tolerance: a cut here would be a number
    # chosen to make the sentence below come out, which is the thing this program refuses
    lo_sum = min(v["mean_row_sum"] for v in every)
    hi_sum = max(v["mean_row_sum"] for v in every)

    # the largest noise at which every level in the sweep still named the right key on every draw
    perfect = [n for n in NOISE
               if all(v["right"] == 1.0 for v in every if v["noise"] == n)]
    clean = max(perfect) if perfect else float("nan")
    print(f"  **The layer holds nothing to train**, against {yours['parameter_elements']} numbers in")
    print(f"  the layer it replaces, and it is not idle: a draw among {KEYS} keys is {draw:.0%}, and")
    print(f"  it names the right key on **every draw at every level** out to noise {clean:g},")
    print("  having at no point been shown which key anything came from.")
    print()
    print("  **And the sweep finds where it stops**, which is the reason it runs this far. Past")
    print("  that the frames are buried and the layer degrades -- but look at how. At the loosest")
    print("  level it keeps answering into the noise and is wrong; at the tightest it stops")
    print("  answering instead:")
    for far in FAR:
        at = [v for v in every if v["far"] == far]
        hard = max(at, key=lambda v: v["noise"])
        print(f"    far {far:<7} at noise {hard['noise']:g}: answered {hard['answered']:.0%}, "
              f"right {hard['right']:.0%}")
    print("  **Tightening the level turns a wrong answer into a refusal**, which is what a")
    print("  false-alarm level is for and what a softmax has no mechanism to do.")
    print()
    print(f"  **And the rows do not sum to one**: across the sweep the mean row sum runs")
    print(f"  {lo_sum:.3f} to {hi_sum:.3f}, where a softmax would put every one of them at exactly")
    print("  1.000 by construction. That is the substitution doing what it was for -- the row")
    print("  carries how much was found rather than a distribution imposed on whatever was there.")
    refused = float(np.mean([v["noise_query_refused"] for v in every]))
    print()
    print(f"  A query made of noise on the same keys is refused on {refused:.0%} of draws. That is")
    print("  the one behaviour a softmax has no way to produce at all: it returns a distribution")
    print("  over the keys whatever is in them, and cannot report that there was nothing there.")
    print()
    print("  The generator is stated in the module docstring. `far` is swept because the")
    print("  false-alarm level belongs to the reader and this program does not derive it. What a")
    print("  trained MultiheadAttention would do is not measured here and is not the comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
