"""The PEMS04 road network, read as a flow series and the road edges between its sensors.

``load_road`` reads the series and the edges from the paths its caller passes. ``cone_of`` turns one
sensor's discretized flow into a light-cone record: a decayed count of the symbols seen, one row per
position in the window.
"""
from __future__ import annotations

import csv

import numpy as np


def load_road(series, roads):
    """``(flow, road)``: the flow channel of ``series`` as floats, and ``roads``' edges as pairs."""
    flow = np.load(series)["data"][:, :, 0].astype(float)
    road = [(int(r["from"]), int(r["to"]))
            for r in csv.DictReader(open(roads, encoding="utf-8"))]
    return flow, road


def cone_of(sym, sensor, end, window, cone, bins):
    """The ``(window, bins)`` light cone of ``sensor`` ending at ``end``, decaying at ``cone``."""
    acc = np.zeros(bins)
    out = np.zeros((window, bins))
    decay = float(np.exp(-1.0 / cone))
    for i, t in enumerate(range(end - window, end)):
        acc *= decay
        acc[sym[t, sensor]] += 1.0
        out[i] = acc
    return out
