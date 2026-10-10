"""F03 physical overlap accounting, separate from historical bound protocols.

CPU arithmetic only. Inputs are chronological relative log-close paths; final
zero is the current-price coordinate, not a reconstructed current bar. The caller
must verify contract, period, snapshot, row identity and causal input windows.
"""

import numpy as np

from .history_price_truth import _matrix

SHIFTS = (1, 16, 64)


def segments(shift):
    """A ends at t, B at t+d; both decoders predict only ages 1..127.

    Common predictions number 127-d, not 128-d. A's current close is only a
    coordinate; B can reconstruct it later, so it belongs to B's entered region.
    The leaving region is measured in A only: it is outside B's declared memory.
    """
    if type(shift) is not int or shift not in SHIFTS:
        raise ValueError("Fixed physical row shifts 1, 16 or 64 required")
    return {
        "common_a": slice(shift, 127),
        "common_b": slice(0, 127 - shift),
        "leaving_a": slice(0, shift),
        "entered_b": slice(127 - shift, 127),
    }


def compare(a, b, raw_a, raw_b, shift):
    """Score two completed reconstructions against independently obtained raw truth.

    Center each common predicted/true path by its own common mean. This removes
    the arbitrary endpoint-price origin without correcting predictions with true
    prices. Retain endpoint-anchored errors and offset errors separately.

    E = consensus_error + disagreement / 4 is an accounting identity, not a causal
    explanation or proof of what a consistency loss will do. The consensus uses
    B's later observation and MUST NOT be used as an online result at A's time.
    """
    regions = segments(shift)
    arrays = [
        _matrix(x, label)
        for x, label in zip(
            (a, b, raw_a, raw_b),
            ("prediction_a", "prediction_b", "truth_a", "truth_b"),
            strict=True,
        )
    ]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Matched paired rows required")
    if any(np.any(x[:, -1] != 0) for x in arrays):
        raise ValueError("All paths must have exact current-close zero")
    a, b, raw_a, raw_b = arrays
    ca, cb = regions["common_a"], regions["common_b"]
    pa, pb, ya, yb = a[:, ca], b[:, cb], raw_a[:, ca], raw_b[:, cb]

    def center(x):
        return x - x.mean(axis=1, keepdims=True)

    pa0, pb0, ya0, yb0 = map(center, (pa, pb, ya, yb))
    if not np.allclose(ya0, yb0, atol=1e-12, rtol=1e-10):
        raise ValueError("Raw common physical prices differ after removing their origins")

    def mse(x):
        return np.mean(x**2, axis=1)

    ea, eb = mse(pa0 - ya0), mse(pb0 - ya0)
    disagreement = mse(pa0 - pb0)
    consensus_error = mse((pa0 + pb0) / 2 - ya0)
    mean_error = (ea + eb) / 2
    output = {
        "common_count": 127 - shift,
        "entered_count": shift,
        "leaving_count": shift,
        "shape_a_mse": ea,
        "shape_b_mse": eb,
        "disagreement_mse": disagreement,
        "mean_shape_mse": mean_error,
        "consensus_shape_mse": consensus_error,
        "identity_residual": mean_error - consensus_error - disagreement / 4,
        "common_anchored_a_mse": mse(pa - ya),
        "common_anchored_b_mse": mse(pb - yb),
        "common_offset_a_mse": np.mean(pa - ya, axis=1) ** 2,
        "common_offset_b_mse": np.mean(pb - yb, axis=1) ** 2,
        "leaving_a_mse": mse((a - raw_a)[:, regions["leaving_a"]]),
        "entered_b_mse": mse((b - raw_b)[:, regions["entered_b"]]),
        "flat_shape_baseline_mse": mse(ya0),
        "flat_leaving_baseline_mse": mse(raw_a[:, regions["leaving_a"]]),
        "flat_entered_baseline_mse": mse(raw_b[:, regions["entered_b"]]),
    }
    if not all(np.isfinite(v).all() for v in output.values()):
        raise ValueError("Nonfinite overlap metrics")
    return output
