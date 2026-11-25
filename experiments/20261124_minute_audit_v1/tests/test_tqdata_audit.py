import json

import numpy as np
import pandas as pd
import pytest

from tqdata_audit import audit, inspect_frame


def frame(times=("2026-09-12 03:00", "2026-09-12 03:01", "2026-09-14 09:00")):
    dt = pd.DatetimeIndex(times)
    result = pd.DataFrame(
        {"datetime": dt.astype(str), "datetime_nano": dt.tz_localize("Asia/Shanghai").asi8}
    )
    for col in ("open", "high", "low", "close", "open_oi", "close_oi"):
        result[col] = 100.0
    result["volume"] = 0.0
    return result


def test_closed_minute_clock_cutoff_and_breaks():
    result = inspect_frame(frame())
    assert not any(result["checks"].values())
    assert result["after_old_cutoff_rows"] == 2  # Strictly after, not equal.
    assert result["max_contiguous_flat_zero_minutes"] == 2  # Weekend resets.


def test_corruption_is_counted_not_silently_filtered():
    data = frame()
    data.loc[0, "datetime_nano"] += 8 * 3600 * 10**9
    data.loc[0, "high"] = 99.0
    data.loc[1, "volume"] = -1
    data.loc[2, "close_oi"] = np.nan
    result = inspect_frame(data)
    assert result["rows"] == 3
    assert result["checks"]["timestamp_clock_mismatch"] == 1
    assert result["checks"]["invalid_ohlc"] == 1
    assert result["checks"]["negative_volume_or_oi_rows"] == 1
    assert result["checks"]["nonfinite_numeric_rows"] == 1


def test_duplicates_and_backward_not_removed():
    data = frame(("2026-01-06 09:02", "2026-01-06 09:01", "2026-01-06 09:01"))
    result = inspect_frame(data)
    assert result["checks"]["duplicate_timestamps"] == 1
    assert result["checks"]["backward_steps"] == 1


def test_missing_oi_does_not_become_zero():
    with pytest.raises(ValueError, match="Missing columns"):
        inspect_frame(frame().drop(columns="close_oi"))


def test_deliveries_separate_families_and_ignore_samples_and_archives(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "source"
    (root / "main").mkdir(parents=True)
    (root / "index").mkdir()
    (root / "month_contracts/bars/MA").mkdir(parents=True)
    (root / "month_contracts/_sample").mkdir(parents=True)
    frame().to_parquet(root / "main/rb.parquet", index=False)
    frame().to_parquet(root / "index/rb.parquet", index=False)
    frame().to_parquet(
        root / "month_contracts/bars/MA/CZCE_MA609_2026-01-01_2026-09-16.parquet", index=False
    )
    frame().to_parquet(root / "month_contracts/_sample/rb.parquet", index=False)
    output = tmp_path / "result"
    result = audit(root, output)
    assert result["groups"]["main"]["rows"] == 3
    assert result["groups"]["month"]["rows"] == 3
    assert result["groups"]["index"]["rows"] == 3
    assert result["missing_month_products_vs_main"] == ["rb"]
    assert all(r["stable_during_read"] for r in json.loads((output / "inventory.json").read_text()))
    with pytest.raises(FileExistsError):
        audit(root, output)
    with pytest.raises(ValueError, match="outside"):
        audit(root, root / "result")
