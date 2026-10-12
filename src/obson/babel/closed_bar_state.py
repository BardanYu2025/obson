"""F11 causal raw-bar adapter; no neural execution, fitting or window-local EMA.

Timestamps are naive Asia/Shanghai bar starts. A caller must supply observations
from the declared contract-file origin, or restore a snapshot of that stream.
"""

import copy
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .ar_codec import State, consume

SCHEMA = "babel-closed-bar-state-v1"
WINDOW = 128
FEATURES = "causal28_ema8_32_activity_v1"


def _token(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _time(value):
    try:
        value = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected naive Asia/Shanghai timestamp") from exc
    if pd.isna(value) or value.tzinfo is not None:
        raise ValueError("Expected naive Asia/Shanghai timestamp")
    return value


def _ewm(previous, current, span):
    # pandas adjust=False does not recompute the average when values are equal.
    if previous is None or previous == current:
        return float(current)
    alpha = 2 / (span + 1)
    return float((1 - alpha) * previous + alpha * current)


@dataclass(frozen=True)
class ClosedBar:
    key: str
    period: int
    datetime: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float = 0.0
    oi_available: bool = False
    open_oi: float | None = None
    closed: bool = True


def _canonical_bar(bar, key, period):
    if not isinstance(bar, ClosedBar):
        raise ValueError("ClosedBar required")
    if (
        bar.key != key
        or type(bar.period) is not int
        or bar.period != period
        or bar.closed is not True
    ):
        raise ValueError("Expected the same contract/period and an explicitly closed bar")
    if not isinstance(bar.oi_available, (bool, np.bool_)):
        raise ValueError("OI availability must be explicit boolean")
    values = np.asarray([bar.open, bar.high, bar.low, bar.close, bar.volume, bar.oi], float)
    if not np.isfinite(values).all() or (values[:4] <= 0).any() or (values[4:] < 0).any():
        raise ValueError("Finite positive OHLC and nonnegative volume/OI required")
    o, h, lo, c, v, oi = values.tolist()
    if h < max(o, lo, c) or lo > min(o, h, c):
        raise ValueError("Invalid OHLC geometry")
    t = _time(bar.datetime)
    # Original activity_features coerces invalid opening OI to missing and falls
    # back only when the previous bar is consecutive and has valid closing OI.
    try:
        opening = float(bar.open_oi) if bar.open_oi is not None else None
    except (TypeError, ValueError):
        opening = None
    if opening is not None and not np.isfinite(opening):
        opening = None
    return ClosedBar(key, period, str(t), o, h, lo, c, v, oi, bool(bar.oi_available), opening)


def frame_to_bars(frame, key, period):
    """Consume validate_frame output without re-inferring OI availability.

    Missing OI columns and present-but-zero OI remain different in the original
    18 channels. Both lack a valid positive OI base for activity ratios.
    """
    required = {"datetime", "open", "high", "low", "close", "volume", "oi", "oi_available"}
    if not required.issubset(frame.columns):
        raise ValueError("Expected validated frame with explicit oi and oi_available")
    for row in frame.to_dict("records"):
        yield ClosedBar(
            key=key,
            period=period,
            datetime=row["datetime"],
            **{name: row[name] for name in ("open", "high", "low", "close", "volume", "oi")},
            oi_available=row["oi_available"],
            open_oi=row.get("open_oi"),
        )


class PreparedBarStream:
    """One contract/period, complete-prefix EMAs, rolling128 normalized features.

    push rejects before committing any state. Returned arrays and metadata are
    copies. Reset requires a new object; night/session gaps never reset EMAs.
    """

    def __init__(self, key, period, statistics, feature_history_origin):
        parts = key.split("/") if isinstance(key, str) else []
        if (
            type(period) is not int
            or period not in (15, 30, 60)
            or len(parts) != 3
            or not all(parts)
            or parts[1] != str(period)
        ):
            raise ValueError("Explicit single-contract key and supported period required")
        if not isinstance(feature_history_origin, str) or not feature_history_origin.strip():
            raise ValueError("Explicit feature_history_origin required")
        try:
            stats = json.loads(json.dumps(statistics, allow_nan=False))
            mean, scale = np.asarray(stats["x_mean"], float), np.asarray(stats["x_scale"], float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid fixed source statistics") from exc
        if (
            mean.shape != (28,)
            or scale.shape != (28,)
            or not np.isfinite([mean, scale]).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Expected finite28-channel training mean and positive scale")
        self.key, self.period = key, period
        self.feature_history_origin = feature_history_origin
        self._statistics, self._mean, self._scale = stats, mean, scale
        self.statistics_signature = _token(stats)
        self.input_signature = _token(
            {"schema": "babel-history-state-api-v1", "statistics": stats, "features": FEATURES}
        )
        self.count = 0
        self.codec = None
        self.fast = self.slow = self.volume_ema = self.previous_log_volume = None
        self.last_time = self.first_time = None
        self.previous_oi = 0.0
        self.previous_oi_valid = False
        self._features = deque(maxlen=WINDOW)
        self._bars = deque(maxlen=WINDOW)

    @property
    def row(self):
        return self.count - 1

    @property
    def statistics(self):
        return copy.deepcopy(self._statistics)

    def push(self, bar, asof):
        """Return None during127-bar warmup, then a reader-ready feature window."""
        bar = _canonical_bar(bar, self.key, self.period)
        t, observed_at = _time(bar.datetime), _time(asof)
        if t + pd.Timedelta(minutes=self.period) > observed_at:
            raise ValueError("Bar is not closed as of the supplied observation time")
        elapsed = (
            1.0
            if self.last_time is None
            else (t - self.last_time).total_seconds() / (self.period * 60)
        )
        if not np.isfinite(elapsed) or elapsed < 1:
            raise ValueError("Duplicate, out-of-order or sub-period bar")
        codec = copy.copy(self.codec) if self.codec is not None else State(bar.open)
        sigma = codec.sigma
        base, _, _ = consume(
            codec,
            [bar.open, bar.high, bar.low, bar.close],
            bar.volume,
            bar.oi,
            bar.oi_available,
            self.period,
            elapsed,
        )
        price = np.log(bar.close)
        fast, slow = _ewm(self.fast, price, 8), _ewm(self.slow, price, 32)
        context = np.arcsinh(
            np.array(
                [
                    price - fast,
                    fast - slow,
                    0.0 if self.fast is None else fast - self.fast,
                    0.0 if self.slow is None else slow - self.slow,
                ]
            )
            / sigma
        ).astype(np.float32)
        lv = np.log1p(bar.volume)
        prior = lv if self.volume_ema is None else self.volume_ema
        previous_lv = lv if self.previous_log_volume is None else self.previous_log_volume
        current_valid = bar.oi_available and bar.oi > 0
        use_open = current_valid and bar.open_oi is not None and bar.open_oi > 0
        use_previous = not use_open and current_valid and self.previous_oi_valid and elapsed == 1
        valid = use_open or use_previous
        safe = (bar.open_oi if use_open else self.previous_oi) if valid else 1.0
        delta = bar.oi - safe if valid else 0.0
        ratio_valid = valid and bar.volume > 0
        activity = np.array(
            [
                lv - prior,
                lv - previous_lv,
                np.arcsinh(100 * delta / safe),
                np.arcsinh(delta / max(bar.volume, 1.0)) if ratio_valid else 0.0,
                np.arcsinh(100 * bar.volume / safe) if valid else 0.0,
                valid,
                ratio_valid,
                valid,
                use_open,
                self.count > 0,
            ],
            np.float32,
        )
        features = np.concatenate((base, context, activity)).astype(np.float32)
        normalized = ((features - self._mean) / self._scale).astype(np.float32)
        if not np.isfinite(features).all() or not np.isfinite(normalized).all():
            raise ValueError("Nonfinite prepared features; state unchanged")
        # Commit only after all validation/arithmetic has succeeded.
        self.codec, self.fast, self.slow = codec, fast, slow
        self.volume_ema, self.previous_log_volume = _ewm(self.volume_ema, lv, 32), float(lv)
        self.previous_oi, self.previous_oi_valid = bar.oi, bool(current_valid)
        self.last_time = t
        if self.first_time is None:
            self.first_time = t
        self.count += 1
        self._features.append(features)
        self._bars.append(asdict(bar))
        return self.current()

    def current(self):
        if self.count < WINDOW:
            return None
        raw = np.stack(self._features)
        bars = copy.deepcopy(list(self._bars))
        end = {
            "key": self.key,
            "period": self.period,
            "end": str(self.last_time),
            "row": self.row,
            "closed": True,
            "feature_history_origin": self.feature_history_origin,
            "feature_first_bar": str(self.first_time),
            "available_at": str(self.last_time + pd.Timedelta(minutes=self.period)),
        }
        return {
            "windows": ((raw - self._mean) / self._scale).astype(np.float32)[None],
            "endpoints": [end],
            "raw_closes": np.asarray([[b["close"] for b in bars]], np.float64),
            "raw_features": raw[None].copy(),
            "raw_bars": bars,
            "row": self.row,
            "count": self.count,
            "current_close": bars[-1]["close"],
            "time": str(self.last_time),
            "input_signature": self.input_signature,
        }

    def snapshot(self):
        payload = {
            "schema": SCHEMA,
            "key": self.key,
            "period": self.period,
            "feature_history_origin": self.feature_history_origin,
            "statistics_signature": self.statistics_signature,
            "count": self.count,
            "first_time": None if self.first_time is None else str(self.first_time),
            "last_time": None if self.last_time is None else str(self.last_time),
            "codec": None if self.codec is None else asdict(self.codec),
            "fast": self.fast,
            "slow": self.slow,
            "volume_ema": self.volume_ema,
            "previous_log_volume": self.previous_log_volume,
            "previous_oi": self.previous_oi,
            "previous_oi_valid": self.previous_oi_valid,
            "features": [x.tolist() for x in self._features],
            "bars": copy.deepcopy(list(self._bars)),
        }
        return {"payload": payload, "sha256": _token(payload)}

    @classmethod
    def restore(cls, snapshot, statistics, feature_history_origin):
        """Restore JSON round-tripped state; corrupt or differently bound input fails."""
        try:
            value = snapshot["payload"]
            if _token(value) != snapshot["sha256"] or value["schema"] != SCHEMA:
                raise ValueError("Snapshot integrity/schema mismatch")
            obj = cls(value["key"], value["period"], statistics, feature_history_origin)
            if (
                value["statistics_signature"] != obj.statistics_signature
                or value["feature_history_origin"] != feature_history_origin
            ):
                raise ValueError("Snapshot statistics or feature origin mismatch")
            n = value["count"]
            if type(n) is not int or n < 0 or len(value["bars"]) != min(n, WINDOW):
                raise ValueError("Invalid snapshot count/ring")
            if n == 0:
                if value != obj.snapshot()["payload"]:
                    raise ValueError("Invalid empty snapshot")
                return obj
            features = np.asarray(value["features"], np.float32)
            if (
                features.shape != (min(n, WINDOW), 28)
                or not np.isfinite(features).all()
                or not np.isfinite((features - obj._mean) / obj._scale).all()
            ):
                raise ValueError("Invalid snapshot feature ring")
            bars = [_canonical_bar(ClosedBar(**b), obj.key, obj.period) for b in value["bars"]]
            times = [_time(b.datetime) for b in bars]
            first, last = _time(value["first_time"]), _time(value["last_time"])
            if (
                last != times[-1]
                or first > times[0]
                or (n <= WINDOW and first != times[0])
                or any(
                    (b - a).total_seconds() < obj.period * 60
                    for a, b in zip(times, times[1:], strict=False)
                )
            ):
                raise ValueError("Invalid snapshot timestamp order")
            codec = State(**value["codec"])
            numeric = [
                codec.close,
                codec.variance,
                value["fast"],
                value["slow"],
                value["volume_ema"],
                value["previous_log_volume"],
                value["previous_oi"],
            ]
            if (
                not np.isfinite(numeric).all()
                or codec.variance < 0
                or codec.close != bars[-1].close
                or value["previous_oi"] != bars[-1].oi
                or type(value["previous_oi_valid"]) is not bool
                or value["previous_oi_valid"] != (bars[-1].oi_available and bars[-1].oi > 0)
                or value["previous_log_volume"] != float(np.log1p(bars[-1].volume))
            ):
                raise ValueError("Invalid snapshot recursive state")
            obj.count, obj.first_time, obj.last_time = n, first, last
            obj.codec = codec
            for key in (
                "fast",
                "slow",
                "volume_ema",
                "previous_log_volume",
                "previous_oi",
                "previous_oi_valid",
            ):
                setattr(obj, key, value[key])
            obj._features.extend(x.copy() for x in features)
            obj._bars.extend(asdict(b) for b in bars)
            return obj
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid or incompatible stream snapshot: {exc}") from exc


class ClosedBarHistoryReader:
    """Atomic closed-bar preparation plus an already-frozen HistoryStateReader.

    No loader or model choice is hidden here. The caller opens its explicitly
    selected real reader on CUDA; synthetic/mock readers support CPU tests.
    """

    def __init__(self, reader, key, period, feature_history_origin):
        self.reader = reader
        self.stream = PreparedBarStream(key, period, reader.statistics, feature_history_origin)
        if self.stream.input_signature != reader.input_signature:
            raise ValueError("Reader and raw adapter source normalization signatures differ")

    def push(self, bar, asof):
        before = self.stream.snapshot()
        prepared = self.stream.push(bar, asof)
        if prepared is None:
            return None
        try:
            estimate = self.reader.read_prepared(
                prepared["windows"],
                prepared["endpoints"],
                input_signature=prepared["input_signature"],
                audit_observed=False,
            )
        except BaseException:
            # No neural state is carried between windows. Restore the causal
            # preprocessing cursor so the same closed bar can be retried.
            self.stream = PreparedBarStream.restore(
                before, self.reader.statistics, self.stream.feature_history_origin
            )
            raise
        return {"prepared": prepared, "estimate": estimate}
