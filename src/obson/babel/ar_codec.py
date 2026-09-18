"""Positive-price causal codec shared by historical encoding and free rollout."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = (
    "gap_log_percent", "body_log_percent", "upper_log_percent", "lower_log_percent",
    "gap_vol_units", "body_vol_units", "upper_vol_units", "lower_vol_units",
    "log_prior_sigma", "log_volume", "log_oi", "oi_available", "log_period", "log_extra_interval",
)
TARGETS = ("gap_asinh", "body_asinh", "upper_root", "lower_root", "log_volume", "log_oi", "log_extra_interval")
ALPHA = 2 / 65
SIGMA_FLOOR = 1e-4
INITIAL_SIGMA = .01


@dataclass
class State:
    close: float
    variance: float = INITIAL_SIGMA ** 2

    @property
    def sigma(self):
        return max(self.variance ** .5, SIGMA_FLOOR)


def geometry(ohlc, previous_close):
    o, h, l, c = np.asarray(ohlc, dtype=float)
    if not np.isfinite([o, h, l, c, previous_close]).all() or min(o, h, l, c, previous_close) <= 0:
        raise ValueError("AR codec requires finite positive prices")
    if h < max(o, c, l) or l > min(o, c, h):
        raise ValueError("Invalid OHLC geometry")
    # Subtract logs instead of dividing extreme prices.
    return np.array([np.log(o) - np.log(previous_close), np.log(c) - np.log(o),
                     max(0., np.log(h) - np.log(max(o, c))),
                     max(0., np.log(min(o, c)) - np.log(l))])


def target(g, sigma, volume, oi, elapsed_ratio):
    if min(volume, oi) < 0 or not np.isfinite([volume, oi, elapsed_ratio]).all() or elapsed_ratio < 1:
        raise ValueError("Invalid volume/OI or sub-period timestamp interval")
    z = g / sigma
    return np.array([np.arcsinh(z[0]), np.arcsinh(z[1]), np.sqrt(z[2]), np.sqrt(z[3]),
                     np.log1p(volume) / 10, np.log1p(oi) / 10, np.log(elapsed_ratio)], np.float32)


def consume(state, ohlc, volume, oi, oi_available, period, elapsed_ratio=1.):
    """Encode one observed/generated bar and update only from that bar."""
    g = geometry(ohlc, state.close)
    sigma = state.sigma
    y = target(g, sigma, volume, oi, elapsed_ratio)
    x = np.r_[np.arcsinh(g * 100), np.arcsinh(g / sigma), np.log(sigma),
              np.log1p(volume) / 10, np.log1p(oi) / 10 if oi_available else 0.,
              float(oi_available), np.log1p(period) / 6, np.log(elapsed_ratio)].astype(np.float32)
    state.variance = (1 - ALPHA) * state.variance + ALPHA * float((g[0] + g[1]) ** 2)
    state.close = float(ohlc[3])
    return x, y, g


def decode(state, y, oi_available=True):
    """Invert model coordinates without any future reference prices or ATR.

    Overflow fails explicitly; never clip an exploding price and call it valid.
    Folded channels are nonnegative in the model distribution.
    """
    y = np.asarray(y, float)
    if y.shape != (7,) or not np.isfinite(y).all() or (y[2:] < 0).any():
        raise ValueError("Invalid generated coordinates")
    with np.errstate(over="raise", invalid="raise"):
        try:
            gap, body = np.sinh(y[:2]) * state.sigma
            upper, lower = y[2:4] ** 2 * state.sigma
            log_o = np.log(state.close) + gap
            log_c = log_o + body
            logs = np.array([log_o, max(log_o, log_c) + upper, min(log_o, log_c) - lower, log_c])
            if not np.isfinite(logs).all() or np.max(np.abs(logs)) > 600:
                raise ValueError("Generated price overflow/underflow")
            ohlc = np.exp(logs)
            volume = np.expm1(y[4] * 10)
            oi = np.expm1(y[5] * 10) if oi_available else 0.
            elapsed = np.exp(y[6])
        except FloatingPointError as exc:
            raise ValueError("Generated value overflow") from exc
    return ohlc, float(volume), float(oi), float(elapsed)


def encode_frame(frame, period):
    prices = frame[["open", "high", "low", "close"]].to_numpy()
    elapsed = frame.datetime.diff().dt.total_seconds().fillna(period * 60).to_numpy() / (period * 60)
    if not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("AR codec requires finite positive prices")
    o, h, l, c = prices.T
    volume, oi = frame.volume.to_numpy(), frame.oi.to_numpy()
    if (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any():
        raise ValueError("Invalid OHLC geometry")
    if (elapsed < 1).any() or not np.isfinite(elapsed).all() or min(volume.min(), oi.min()) < 0:
        raise ValueError("Invalid volume/OI or sub-period timestamp interval")
    available = frame.oi_available.to_numpy(bool)
    previous = np.r_[o[0], c[:-1]]
    g = np.column_stack((np.log(o) - np.log(previous), np.log(c) - np.log(o),
                         np.maximum(0, np.log(h) - np.log(np.maximum(o, c))),
                         np.maximum(0, np.log(np.minimum(o, c)) - np.log(l))))
    # Vectorized equivalent of consume(), including the explicit initial state.
    variance = pd.Series(np.r_[INITIAL_SIGMA ** 2, (g[:, 0] + g[:, 1]) ** 2]).ewm(alpha=ALPHA, adjust=False).mean().to_numpy()[1:]
    sigma = np.sqrt(np.maximum(np.r_[INITIAL_SIGMA ** 2, variance[:-1]], SIGMA_FLOOR ** 2))
    current_sigma = np.sqrt(np.maximum(variance, SIGMA_FLOOR ** 2))
    extras = np.column_stack((np.log1p(volume) / 10, np.log1p(oi) / 10, np.log(elapsed)))

    def coordinates(scale):
        z = g / scale[:, None]
        return np.column_stack((np.arcsinh(z[:, :2]), np.sqrt(z[:, 2:]), extras)).astype(np.float32)

    x = np.column_stack((np.arcsinh(g * 100), np.arcsinh(g / sigma[:, None]), np.log(sigma),
                         extras[:, 0], np.where(available, extras[:, 1], 0), available,
                         np.full(len(g), np.log1p(period) / 6), extras[:, 2])).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite AR features")
    return {"x": x, "y": coordinates(sigma), "variance": variance,
            "persistence": coordinates(current_sigma), "available": available}
