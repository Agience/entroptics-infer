"""An attention layer with nothing to train, for dropping into a torch model.

What this is
--------------
:class:`DerivedAttention` is an ``nn.Module`` that occupies the place
``nn.MultiheadAttention`` occupies and holds no parameters at all. There is no ``W_Q``, no
``W_K``, no ``W_V``, no ``W_O`` and no temperature. ``layer.parameters()`` is empty, and
``research/checks/test_whether_the_torch_layer_has_anything_to_train.py`` asserts that rather than asking
you to take it on trust.

torch is imported here and nowhere else in this package. The engine is framework-free; this
module is an adapter over it, so a model that wants tensors gets tensors and the answer path picks
up no dependency on autograd.

The shape is different, and pretending otherwise would be the whole mistake
----------------------------------------------------------------------------
Standard attention scores a query *vector* against a key *vector* with ``q·k/√d``. The statistic
here is not a function of two vectors. It is the alignment of two records' cross-covariance,
standardized by the exact permutation null of re-pairing their rows -- and a re-pairing of one row
is not a null. So a "token" in this layer is a frame: an ordered axis and a feature axis, shape
``(S, D)``, not a vector of length ``D``.

    nn.MultiheadAttention        query (Q, D)      key (K, D)      -> weights (Q, K)
    DerivedAttention             query (Q, S, D)   key (K, S, D)   -> weights (Q, K)

``S`` is the shared ordered axis the two sides meet on, and both sides must carry the same ``S``
because the null re-pairs their rows. That is a real constraint on what can be asked, not a
formality: a caller who cannot put two records on one ordered axis does not have an association to
measure, and :class:`entroptics_infer.engine.Unreadable` says so rather than reshaping something
into working.

**A model builder wanting to swap this in has to have frames.** Where the model's tokens are
vectors with nothing ordered inside them, this layer does not apply and no amount of adapting will
make it -- the thing it reads is not there. Where a token is a span, a window, a sequence of
measurements, a document's sentences, or any other object with an internal order, it applies
directly.

What the weights are, and what they are not
---------------------------------------------
``weights[i, j]`` is the signed strength of the association between query frame ``i`` and key frame
``j``, zeroed where the instrument does not resolve it. Consequences a caller should expect,
because they are the point rather than side effects:

* **Rows do not sum to one.** A softmax forces one whether or not anything resolved; a query whose
  keys all read as noise gets a row of zeros here, and the output is zero. That is the layer
  refusing, and it has no other way of refusing.
* **Weights can be negative.** Two records can anti-resolve, and a softmax cannot say so.
* **There is no gradient through the decision.** The zeroing is a threshold the instrument derives
  from the record's own entropy; it is not differentiable and is not meant to be. This layer is for
  models that read, not for models that train through it.

``far`` is required and has no default, here as everywhere in this package: the false-alarm level
belongs to the reader, and a layer that defaulted it would let a model take readings without ever
naming the level they were taken at.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import torch
from torch import nn

from .engine import associate

#: A frame is what the instrument reads: one ordered axis and one feature axis.
FRAME_AXES = 2
#: A stack of frames carries one axis more -- the token index the layer attends over.
STACK_AXES = FRAME_AXES + 1


class DerivedAttention(nn.Module):
    """Attention whose weights are measured from the data rather than trained.

    ``far`` is the reader's false-alarm level and is required.

    ``signed`` chooses what the weight carries. With it true the weight is the signed strength, so
    an anti-resolving pair pulls the output away from that value; with it false the weight is the
    standardized score's magnitude. Neither is a tuned quantity -- both come straight off the
    instrument, and the choice is about what the surrounding model wants to do with a negative.
    """

    def __init__(self, *, far: float, signed: bool = True) -> None:
        super().__init__()
        self.far = float(far)
        self.signed = bool(signed)

    def extra_repr(self) -> str:
        return f"far={self.far}, signed={self.signed}, parameters=0"

    def weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """``(Q, K)`` measured associations between every query frame and every key frame.

        Computed outside autograd. The instrument reads numpy arrays and returns a decision, and a
        decision has no derivative -- attaching one would be inventing a gradient for a threshold
        the data set.
        """
        q = query.detach().cpu().numpy().astype(float)
        k = key.detach().cpu().numpy().astype(float)
        out = np.zeros((q.shape[0], k.shape[0]))
        for i in range(q.shape[0]):
            for j in range(k.shape[0]):
                a = associate(q[i], k[j], far=self.far)
                out[i, j] = a.strength if self.signed else abs(a.z) * float(a.resolved)
        return torch.as_tensor(out, dtype=query.dtype, device=query.device)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(output, weights)`` -- the same pair ``nn.MultiheadAttention`` returns.

        ``query`` is ``(Q, S, D)``, ``key`` is ``(K, S, D)`` and ``value`` is ``(K, Dv)``. The
        output is ``(Q, Dv)`` and is not a convex combination of the values: the weights are not
        normalized, so a query that resolves nothing returns a zero row rather than an average of
        things it has no reason to average.
        """
        if query.ndim != STACK_AXES or key.ndim != STACK_AXES:
            raise ValueError(
                f"a token here is a frame with an ordered axis and a feature axis, so query and "
                f"key are (Q, S, D) and (K, S, D); got {tuple(query.shape)} and "
                f"{tuple(key.shape)}. If the model's tokens are vectors there is no ordered axis "
                f"for the null to re-pair, and this layer has nothing to read."
            )
        if query.shape[1] != key.shape[1]:
            raise ValueError(
                f"the null is a re-pairing of rows, so the two sides must meet on one ordered "
                f"axis; got {query.shape[1]} and {key.shape[1]} steps."
            )
        if key.shape[0] != value.shape[0]:
            raise ValueError(
                f"there must be one value per key; got {key.shape[0]} keys and "
                f"{value.shape[0]} values."
            )
        w = self.weights(query, key)
        return w @ value, w


def has_nothing_to_train(module: nn.Module) -> dict[str, Any]:
    """What a caller can check for themselves before trusting the claim in the docstring.

    Returns the counts rather than a verdict, because a function that answered ``True`` would be
    one more thing to trust.
    """
    params = list(module.parameters())
    buffers = list(module.buffers())
    return {"parameters": len(params),
            "parameter_elements": int(sum(p.numel() for p in params)),
            "buffers": len(buffers),
            "buffer_elements": int(sum(b.numel() for b in buffers)),
            "requires_grad": any(p.requires_grad for p in params)}
