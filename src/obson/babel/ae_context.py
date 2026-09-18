"""Stage-2 fixed causal EMA context. No centered smoothing or target changes."""

import numpy as np
import pandas as pd

from .ar_codec import FEATURES, INITIAL_SIGMA, SIGMA_FLOOR, encode_frame

CONTEXT = ("close_minus_ema8", "ema8_minus_ema32", "ema8_slope", "ema32_slope")


def feature_names(mode):
    if mode not in ("none", "ema8_32"):
        raise ValueError("Unknown history context mode")
    return tuple(FEATURES) + (CONTEXT if mode == "ema8_32" else ())


def encode_context(frame, period, mode="none"):
    result = encode_frame(frame, period)
    feature_names(mode)
    if mode == "none":
        return result
    price = pd.Series(np.log(frame.close.to_numpy()))
    fast = price.ewm(span=8, adjust=False).mean().to_numpy()
    slow = price.ewm(span=32, adjust=False).mean().to_numpy()
    sigma = np.sqrt(np.maximum(np.r_[INITIAL_SIGMA ** 2, result["variance"][:-1]], SIGMA_FLOOR ** 2))
    context = np.column_stack((price.to_numpy() - fast, fast - slow,
                               np.diff(fast, prepend=fast[0]), np.diff(slow, prepend=slow[0])))
    context = np.arcsinh(context / sigma[:, None]).astype(np.float32)
    if not np.isfinite(context).all():
        raise ValueError("Nonfinite causal EMA context")
    result["x"] = np.column_stack((result["x"], context))
    return result
