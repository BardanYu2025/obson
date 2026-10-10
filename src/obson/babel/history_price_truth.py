"""Raw-close physical truth for F11; separate from bound float32-feature replay.

No model imports, feature inversion, fitting or future labels. CSV timestamps are
bar starts in Asia/Shanghai (naive local time); callers must supply a locked as-of.
This is a CPU component, not the independent V13 evaluation runner.
"""

import csv
import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SCHEMA = "babel-raw-history-price-truth-v1"
HORIZONS = (16, 64)
FLOOR = 1e-8
LOW_RMS = 1e-4


def _matrix(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "ifu" or array.ndim != 2 or array.shape[1] != 128:
        raise ValueError(f"{name}: numeric N x 128 required")
    array = array.astype(np.float64, copy=True)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name}: nonempty finite input required")
    return array


def summarize_path(path):
    """Unchanged descriptor definitions, applied to explicit log-price paths.

    A horizon of 16 uses 16 closes and 15 returns. Ratio floors are the historical
    numerical guards, not a market tick/noise model. Tiny real movements can have
    high efficiency; neither efficiency nor the 0/1/2 class predicts a future bar.
    """
    path = _matrix(path, "relative_log_close")
    if np.any(path[:, -1] != 0):
        raise ValueError("Path must be anchored to its current close (exact zero)")
    values, rms_values = [], []
    for horizon in HORIZONS:
        prices = path[:, -horizon:]
        returns = np.diff(prices, axis=1)
        travel = np.abs(returns).sum(axis=1)
        efficiency = np.divide(
            returns.sum(axis=1), travel, out=np.zeros(len(path)), where=travel > 1e-12
        )
        centered = prices - prices.mean(axis=1, keepdims=True)
        time = np.arange(horizon) - (horizon - 1) / 2
        denominator = (centered**2).sum(axis=1) * (time**2).sum()
        linearity = np.divide(
            (centered @ time) ** 2,
            denominator,
            out=np.zeros(len(path)),
            where=denominator > 1e-24,
        )
        rms = np.sqrt(np.mean(returns**2, axis=1))
        values.extend((efficiency, np.clip(linearity, 0, 1), np.log(np.maximum(rms, FLOOR))))
        rms_values.append(rms)
    descriptors = np.column_stack(values)
    rms = np.column_stack(rms_values)
    efficiency = descriptors[:, [0, 3]]
    return {
        "descriptors": descriptors,
        "rms_log_return": rms,
        "rms_bps": rms * 10000,
        "direction": np.where(efficiency < -0.2, 0, np.where(efficiency > 0.2, 2, 1)),
        "stratum": np.where(rms <= FLOOR, 0, np.where(rms <= LOW_RMS, 1, 2)),
    }


def from_raw_closes(closes):
    """Positive original closes, never inverse-normalized float32 features.

    Reject an already float32 array: casting it back cannot recover lost price
    precision. Python numeric sequences and original float64 CSV arrays work.
    """
    source = np.asarray(closes)
    if source.dtype.kind == "f" and source.dtype.itemsize < 8:
        raise ValueError("Use original float64 closes, not float32 or restored features")
    prices = _matrix(source, "raw_closes")
    if np.any(prices <= 0):
        raise ValueError("Raw closes must be positive")
    anchor = prices[:, -1:]
    # log1p preserves near-flat changes; log differences avoid extreme ratios
    # overflowing or rounding a tiny positive ratio to zero. Flat is exactly zero.
    path = np.log(prices) - np.log(anchor)
    difference = prices - anchor
    near = np.abs(difference) <= anchor * 0.5
    ratio = np.divide(difference, anchor, out=np.zeros_like(prices), where=near)
    path[near] = np.log1p(ratio[near])
    path[:, -1] = 0
    return {
        "schema": SCHEMA,
        "source": "original_raw_closes_not_model_output",
        "relative_log_close": path,
        **summarize_path(path),
    }


def audit_prediction(predicted_path, raw_closes):
    """Score a completed prediction; raw truth is never passed to a model.

    Keeps legacy observed_audit unchanged. The caller retains model/row identities
    alongside these arrays; the CSV adapter below verifies physical row identity.
    """
    truth = from_raw_closes(raw_closes)
    predicted = _matrix(predicted_path, "prediction")
    if predicted.shape != truth["relative_log_close"].shape:
        raise ValueError("Prediction and raw truth rows must match")
    estimates = summarize_path(predicted)
    return {
        "schema": SCHEMA,
        "truth": truth,
        "descriptor_error": estimates["descriptors"] - truth["descriptors"],
        "rms_error": estimates["rms_log_return"] - truth["rms_log_return"],
        "direction_correct": estimates["direction"] == truth["direction"],
        "path_rmse": np.column_stack(
            [
                np.sqrt(
                    np.mean((predicted[:, -h:] - truth["relative_log_close"][:, -h:]) ** 2, axis=1)
                )
                for h in HORIZONS
            ]
        ),
    }


def _local_time(value):
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is not None:
        raise ValueError("Use naive Asia/Shanghai bar-start/as-of timestamps")
    return parsed


class RawCloseWindows:
    """Read hash-bound single-contract CSVs; exact endpoint rows, closed bars only.

    inventory is a caller-locked mapping key -> original CSV SHA256. Bytes are
    verified before parsing and cached as a snapshot. This verifies price rows,
    not feature preparation, train/test separation or V13 data independence.
    """

    def __init__(self, root, inventory):
        self.root = Path(root).resolve()
        self.inventory = dict(inventory)
        self.cache = {}

    def window(self, row, *, asof):
        key = row["key"]
        symbol, period, contract = key.split("/")
        if period not in ("15", "30", "60") or any(
            not name or name in (".", "..") or "\\" in name for name in (symbol, contract)
        ):
            raise ValueError("Supported single-contract key required")
        path = (self.root / symbol / f"{contract}_{period}m.csv").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("CSV must remain inside raw root")
        if key not in self.cache:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != self.inventory[key]:
                raise ValueError(f"Original CSV fingerprint mismatch: {key}")
            records = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
            times = [_local_time(r["datetime"]) for r in records]
            if any(left >= right for left, right in zip(times, times[1:], strict=False)):
                raise ValueError("Raw bar starts must be strictly increasing")
            closes = np.array([float(r["close"]) for r in records], dtype=np.float64)
            self.cache[key] = (times, closes)
        times, closes = self.cache[key]
        index = row["row"]
        if type(index) is not int or not 127 <= index < len(closes):
            raise ValueError("An exact row with 128 closes is required")
        if times[index] != _local_time(row["end"]):
            raise ValueError("Raw row and endpoint timestamp differ")
        if times[index] + timedelta(minutes=int(period)) > _local_time(asof):
            raise ValueError("Endpoint bar has not closed by as-of")
        result = closes[index - 127 : index + 1].copy()
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Physical window requires positive finite closes")
        return result
