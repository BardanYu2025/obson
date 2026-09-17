"""Independent Babel data contract: validated contracts, causal features, global splits."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .progress import progress
from .structure import annotate

FEATURES = (
    "open_delta",
    "high_delta",
    "low_delta",
    "close_delta",
    "volume_z",
    "oi_delta",
    "oi_z",
    "oi_available",
    "time_sin",
    "time_cos",
    "period",
    "elapsed",
)


@dataclass
class Series:
    code: str
    period: int
    contract: str
    frame: pd.DataFrame
    labels: dict
    x: np.ndarray
    main: np.ndarray
    sessions: np.ndarray
    source_hash: str

    @property
    def key(self):
        return f"{self.code}/{self.period}/{self.contract}"

    @property
    def ends(self):
        return (self.frame.datetime + pd.Timedelta(minutes=self.period)).to_numpy("datetime64[ns]")


def validate_frame(df, name="input"):
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df.datetime, errors="raise")
    if df.datetime.dt.tz is not None:
        df["datetime"] = df.datetime.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if (
        df.datetime.isna().any()
        or df.datetime.duplicated().any()
        or not df.datetime.is_monotonic_increasing
    ):
        raise ValueError(f"{name}: timestamps must be unique, valid and increasing")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[required[1:]].to_numpy()).all():
        raise ValueError(f"{name}: nonfinite OHLCV")
    if (
        (df.high < df[["open", "close", "low"]].max(axis=1)).any()
        or (df.low > df[["open", "close", "high"]].min(axis=1)).any()
        or (df.volume < 0).any()
    ):
        raise ValueError(f"{name}: invalid OHLC geometry/negative volume")
    oi_col = next((c for c in ("close_oi", "open_interest") if c in df), None)
    df["oi_available"] = oi_col is not None
    df["oi"] = pd.to_numeric(df[oi_col], errors="raise") if oi_col else 0.0
    if not np.isfinite(df.oi).all() or (df.oi < 0).any():
        raise ValueError(f"{name}: invalid open interest")
    return df.reset_index(drop=True)


def features(df, a, period):
    """No window-end statistics. Each token only uses present/past raw observations."""
    past_atr = np.r_[a[0], a[:-1]]
    prev_close = df.close.shift(1).fillna(df.open.iloc[0]).to_numpy()
    price = (df[["open", "high", "low", "close"]].to_numpy() - prev_close[:, None]) / past_atr[
        :, None
    ]

    def log_z(values):
        v = np.log1p(values)
        past = v.shift(1).rolling(64, min_periods=2)
        return ((v - past.mean()) / past.std(ddof=0).clip(lower=0.1)).fillna(0).to_numpy()

    minutes = df.datetime.dt.hour * 60 + df.datetime.dt.minute
    angle = minutes.to_numpy() / 1440 * 2 * np.pi
    oi_delta = np.log1p(df.oi).diff().fillna(0).to_numpy() * 100
    elapsed = df.datetime.diff().dt.total_seconds().fillna(period * 60).to_numpy() / (period * 60)
    x = np.column_stack(
        [
            price,
            log_z(df.volume),
            oi_delta,
            log_z(df.oi),
            df.oi_available.to_numpy(),
            np.sin(angle),
            np.cos(angle),
            np.full(len(df), np.log1p(period) / 6),
            np.log1p(elapsed),
        ]
    )
    return np.clip(x, -20, 20).astype(np.float32)


def session_dates(dt, daytime_dates):
    """Night/early-morning bars map to next observed daytime session.

    Uses the union of supplied contracts' daytime dates, not weekday arithmetic
    over holidays. Trailing nights without a known next session stay ineligible.
    This is a data-derived session calendar, not an exchange calendar service.
    """
    dt = pd.Series(pd.to_datetime(dt))
    nominal = (dt + pd.Timedelta(hours=6)).to_numpy("datetime64[D]")
    daytime = (dt.dt.hour >= 8) & (dt.dt.hour < 18)
    nominal[daytime] = dt[daytime].to_numpy("datetime64[D]")
    cal = np.sort(np.unique(np.asarray(daytime_dates, dtype="datetime64[D]")))
    out = np.full(len(dt), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if len(cal):
        ix = np.searchsorted(cal, nominal)
        valid = ix < len(cal)
        out[valid] = cal[ix[valid]]
    return out


def load_series(root="data/contracts", symbols=None, periods=(60, 30), asof=None):
    """Use previous session's volume leader among supplied contracts.

    Never use the legacy same-day volume roll calendar: its selection knows the
    day's final volume. Selection is lagged here and shared across periods.
    """
    root = Path(root)
    symbols = sorted(set(symbols or [p.name for p in root.iterdir() if p.is_dir()]))
    periods = sorted(set(periods))
    started = time.monotonic()
    progress(f"data: reading {root}; symbols={symbols}, periods={periods}")
    raw, diagnostics = [], []
    bars = 0
    for code in symbols:
        for period in periods:
            for f in sorted((root / code).glob(f"*_{period}m.csv")):
                df = validate_frame(pd.read_csv(f), str(f))
                if asof is not None:
                    df = df[
                        df.datetime + pd.Timedelta(minutes=period) <= pd.Timestamp(asof)
                    ].reset_index(drop=True)
                if len(df) < 2:
                    continue
                contract = f.name.removesuffix(f"_{period}m.csv")
                digest = hashlib.sha256(
                    pd.util.hash_pandas_object(df, index=False).values.tobytes()
                ).hexdigest()
                raw.append((code, period, contract, df, digest))
                bars += len(df)
                if len(raw) == 1 or len(raw) % 50 == 0:
                    progress(
                        f"data: read {len(raw)} files / {bars:,} bars; elapsed={time.monotonic() - started:.0f}s"
                    )
    if not raw:
        raise ValueError("No valid contract data found")
    progress(
        f"data: read complete, {len(raw)} files / {bars:,} bars; building sessions and lagged contract selection"
    )
    calendar = np.unique(
        np.concatenate(
            [
                df.loc[
                    (df.datetime.dt.hour >= 8) & (df.datetime.dt.hour < 18), "datetime"
                ].to_numpy("datetime64[D]")
                for _, _, _, df, _ in raw
            ]
        )
    )
    sessions = [session_dates(df.datetime, calendar) for _, _, _, df, _ in raw]
    active = {}
    for code in symbols:
        # One frequency drives contract selection; don't double count mixed periods.
        available = [p for c, p, _, _, _ in raw if c == code]
        if not available:
            diagnostics.append(f"{code}: no data")
            continue
        reference = max(available)
        daily = []
        for (c, p, contract, df, _), sess in zip(raw, sessions, strict=True):
            if c == code and p == reference:
                z = pd.DataFrame(
                    {"session": sess, "volume": df.volume.to_numpy(), "contract": contract}
                ).dropna()
                daily.append(z.groupby(["session", "contract"], as_index=False).volume.sum())
        table = (
            pd.concat(daily)
            .pivot_table(index="session", columns="contract", values="volume", aggfunc="sum")
            .sort_index()
        )
        # Winner for session D is selected from D-1, which has already ended.
        winners = table.fillna(0).idxmax(axis=1).shift(1)
        active[code] = dict(zip(winners.index.to_numpy("datetime64[D]"), winners, strict=True))
    result = []
    progress(
        "data: generating causal labels and features on CPU (GPU may be idle during this stage)"
    )
    last_report = time.monotonic()
    for (code, period, contract, df, digest), sess in zip(raw, sessions, strict=True):
        labels = annotate(df)
        main = np.array([active[code].get(day) == contract for day in sess], dtype=bool)
        result.append(
            Series(
                code,
                period,
                contract,
                df,
                labels,
                features(df, labels["atr"], period),
                main,
                sess,
                digest,
            )
        )
        if not df.oi_available.all():
            diagnostics.append(
                f"{code}/{contract}/{period}: OI absent, explicit availability flag is 0"
            )
        if len(result) == 1 or len(result) == len(raw) or time.monotonic() - last_report >= 10:
            progress(
                f"data: labels/features {len(result)}/{len(raw)} contracts; current={code}/{period}/{contract}; elapsed={time.monotonic() - started:.0f}s"
            )
            last_report = time.monotonic()
    diagnostics.append(
        "Contract selection uses previous-session volume among local files; coverage is not guaranteed complete."
    )
    diagnostics.append(
        "Session calendar inferred from observed daytime bars; trailing nights without a daytime session are excluded."
    )
    progress(
        f"data: ready, eligible anchors={sum(int(s.main.sum()) for s in result):,}; elapsed={time.monotonic() - started:.0f}s"
    )
    return result, diagnostics


def split_boundaries(series):
    days = np.sort(np.unique(np.concatenate([s.sessions[s.main] for s in series])))
    days = days[~np.isnat(days)]
    if len(days) < 10:
        raise ValueError("At least 10 observed sessions are required for train/validation/test")
    return {
        "train_until": str(days[int(len(days) * 0.7) - 1]),
        "val_until": str(days[int(len(days) * 0.85) - 1]),
        "test_until": str(days[-1]),
    }


def split_mask(s, boundaries, split):
    tr, va = np.datetime64(boundaries["train_until"]), np.datetime64(boundaries["val_until"])
    if split == "train":
        mask = s.sessions <= tr
    elif split == "val":
        mask = (s.sessions > tr) & (s.sessions <= va)
    elif split == "test":
        mask = (s.sessions > va) & (s.sessions <= np.datetime64(boundaries["test_until"]))
    else:
        raise ValueError(f"Unknown split {split}")
    return mask & s.main & ~np.isnat(s.sessions)


def manifest(series, boundaries=None):
    return {
        "boundaries": boundaries,
        "features": list(FEATURES),
        "calendar": "observed-daytime-next-session",
        "contract_selection": "lagged-session-volume",
        "timestamps": "Asia/Shanghai bar start; available at start+period",
        "sources": [
            {
                "key": s.key,
                "sha256": s.source_hash,
                "rows": len(s.frame),
                "start": str(s.frame.datetime.iloc[0]),
                "end": str(s.frame.datetime.iloc[-1]),
            }
            for s in series
        ],
    }
