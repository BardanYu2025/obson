"""Raw synthetic alignment tests only; no model inference or optimizer updates."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obson.babel import cross_period_audit as audit


def fine_bars(n=8):
    close = np.arange(n, dtype=float) + 100
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-06 09:00", periods=n, freq="15min"),
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.arange(n, dtype=float) + 1,
            "open_oi": np.arange(n, dtype=float) + 1000,
            "close_oi": np.arange(n, dtype=float) + 1001,
        }
    )


def coarse_bars(fine, ratio=2):
    rows = []
    for start in range(0, len(fine), ratio):
        block = fine.iloc[start : start + ratio]
        rows.append(
            {
                "datetime": block.datetime.iloc[0],
                "open": block.open.iloc[0],
                "high": block.high.max(),
                "low": block.low.min(),
                "close": block.close.iloc[-1],
                "volume": block.volume.sum(),
                "open_oi": block.open_oi.iloc[0],
                "close_oi": block.close_oi.iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def validated(frame, period):
    return audit.read_bars(frame.to_csv(index=False).encode(), period)


@pytest.mark.parametrize("ratio", [2, 4])
def test_perfect_raw_aggregation_and_oi_endpoints(ratio):
    fine = fine_bars()
    coarse = coarse_bars(fine, ratio)
    counts = audit.compare(fine, coarse, 15, ratio * 15)["counts"]
    assert counts["full_ohlcv_and_oi_match"] == len(coarse)
    assert counts["full_ohlcv_mismatch"] == 0
    coarse.loc[0, "close_oi"] = fine.close_oi.iloc[:ratio].sum()
    report = audit.compare(fine, coarse, 15, ratio * 15)
    assert report["counts"]["full_oi_mismatch"] == 1
    assert report["counts"]["full_ohlcv_match"] == len(coarse)


def test_partial_exact_match_is_not_eligible():
    fine = fine_bars(4).drop(index=1).reset_index(drop=True)
    first, last = coarse_bars(fine.iloc[:1]), coarse_bars(fine.iloc[1:])
    coarse = pd.concat([first, last], ignore_index=True)
    report = audit.compare(fine, coarse, 15, 30)
    assert report["counts"]["partial_coverage"] == 1
    assert report["counts"]["partial_ohlcv_match_not_eligible"] == 1
    assert report["counts"]["full_price_match"] == 1


def test_missing_child_changes_volume_and_coverage():
    fine = fine_bars()
    coarse = coarse_bars(fine)
    fine = fine.drop(index=1).reset_index(drop=True)
    report = audit.compare(fine, coarse, 15, 30)
    assert report["counts"]["partial_coverage"] == 1
    assert report["counts"]["partial_ohlcv_match_not_eligible"] == 0
    assert report["counts"]["full_ohlcv_match"] == len(coarse) - 1


def test_shifted_grid_is_not_positionally_paired():
    fine = fine_bars()
    coarse = coarse_bars(fine)
    fine["datetime"] += pd.Timedelta(minutes=5)
    counts = audit.compare(fine, coarse, 15, 30)["counts"]
    assert counts["full_coverage"] == 0
    assert counts["straddled_intervals"] == len(coarse)
    assert counts["contained_ohlcv_match"] == 0


def test_coverage_edges_and_empty_ranges():
    fine = fine_bars(4)
    coarse = coarse_bars(fine_bars(8))
    counts = audit.compare(fine, coarse, 15, 30)["counts"]
    assert counts["full_coverage"] == 2
    assert counts["empty_coverage"] == 2
    assert sum(counts[k] for k in ("full_coverage", "partial_coverage", "empty_coverage")) == 4


def test_separate_price_volume_and_missing_oi():
    fine = fine_bars()
    coarse = coarse_bars(fine)
    coarse.loc[0, "volume"] += 1
    coarse.loc[1, "high"] += 1
    coarse = validated(coarse.drop(columns=["open_oi"]), 30)
    report = audit.compare(fine, coarse, 15, 30)
    assert report["counts"]["full_price_match"] == 3
    assert report["counts"]["full_ohlcv_match"] == 2
    assert report["counts"]["full_oi_comparable"] == 0
    assert report["counts"]["full_ohlcv_and_oi_match"] == 0
    assert report["fields"]["volume"]["full_mismatches"] == 1
    assert report["fields"]["high"]["full_max_abs_difference"] == 1


def test_absolute_tolerance_not_relative_large_price():
    fine = fine_bars()
    for name in audit.FIELDS[:4]:
        fine[name] += 1e7
    coarse = coarse_bars(fine)
    coarse.loc[0, "high"] += 0.001
    assert audit.compare(fine, coarse, 15, 30)["counts"]["full_price_mismatch"] == 1


@pytest.mark.parametrize(
    "problem", ["duplicate", "reverse", "overlap", "nat", "geometry", "volume", "price"]
)
def test_invalid_inputs_fail(problem):
    frame = fine_bars()
    if problem == "duplicate":
        frame.loc[1, "datetime"] = frame.datetime.iloc[0]
    if problem == "reverse":
        frame = frame.iloc[::-1]
    if problem == "overlap":
        frame.loc[1, "datetime"] -= pd.Timedelta(minutes=1)
    if problem == "nat":
        frame.loc[1, "datetime"] = pd.NaT
    if problem == "geometry":
        frame.loc[1, "high"] = 1
    if problem == "volume":
        frame.loc[1, "volume"] = -1
    if problem == "price":
        frame.loc[1, "close"] = np.inf
    with pytest.raises(ValueError):
        validated(frame, 15)


def test_timezone_and_oi_unknowns():
    frame = fine_bars()
    frame["datetime"] = frame.datetime.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    frame.loc[0, "close_oi"] = -1
    result = validated(frame, 15)
    assert result.datetime.iloc[0] == pd.Timestamp("2025-01-06 09:00")
    assert np.isnan(result.close_oi.iloc[0])


def test_128_context_requires_both_periods():
    fine = fine_bars(260)
    result = audit.compare(fine, coarse_bars(fine), 15, 30)
    assert result["counts"]["price_matches_with_128_rows_each"] == 3


def write_contract(root, contract, periods=(15, 30, 60)):
    folder = root / "x"
    folder.mkdir(parents=True, exist_ok=True)
    fine = fine_bars()
    for period in periods:
        frame = fine if period == 15 else coarse_bars(fine, period // 15)
        frame.to_csv(folder / f"{contract}_{period}m.csv", index=False)


def test_inventory_is_contract_specific_and_hash_bound(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    write_contract(root, "EX.x1", (15,))
    write_contract(root, "EX.x2", (30, 60))
    result = audit.run(root, out)
    assert result["comparisons"]["15->30"]["contracts"] == 0
    assert result["comparisons"]["30->60"]["counts"]["full_ohlcv_match"] == 2
    assert len(result["unavailable_comparisons"]) == 5
    completion = json.loads((out / "completion.json").read_text())
    assert completion["training_pairs_certified"] is False
    for name, digest in completion["report_sha256"].items():
        assert audit.sha256_bytes((out / name).read_bytes()) == digest
    with pytest.raises(ValueError, match="not empty"):
        audit.run(root, out)
    with pytest.raises(ValueError, match="separate"):
        audit.run(root, root / "out")
    with pytest.raises(ValueError, match="separate"):
        audit.run(root, tmp_path)


def test_invalid_file_writes_failure_and_partial_inventory(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    write_contract(root, "EX.x1")
    (root / "x/EX.x1_15m.csv").write_text("broken\n1\n")
    with pytest.raises(ValueError, match="invalid input"):
        audit.run(root, out)
    assert (out / "failure.json").exists()
    assert not (out / "completion.json").exists()
    result = json.loads((out / "alignment_metrics.json").read_text())
    assert result["status"] == "invalid_inputs"
    assert len(result["invalid_files"]) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_wrapper_exports_success_and_failure(tmp_path, fail):
    root, out, download = tmp_path / "raw", tmp_path / "out", tmp_path / "download"
    write_contract(root, "EX.x1")
    if fail:
        (root / "x/EX.x1_15m.csv").write_text("broken\n1\n")
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        BABEL_DATA_ROOT=str(root),
        BABEL_CROSS_PERIOD_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(download),
        BABEL_CROSS_PERIOD_LOG=str(tmp_path / "missing.log"),
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_cross_period_audit_autodl.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == (1 if fail else 0), result.stderr
    archive = download / "out_reports.tar.gz"
    assert archive.is_file()
    with tarfile.open(archive) as tar:
        status = tar.extractfile("out/run_status.txt").read().decode()
        assert f"run_status={'failed' if fail else 'complete'}" in status
        assert "out/alignment_metrics.json" in tar.getnames()
        assert not any(n.endswith(".csv") for n in tar.getnames())


def test_zero_volume_diagnostic_does_not_repair_eligibility():
    fine = fine_bars(4)
    fine.loc[0, list(audit.FIELDS[:4])] = 200
    fine.loc[0, "volume"] = 0
    coarse = coarse_bars(fine)
    # Coarse feed ignores the zero-volume placeholder, fine feed retains it.
    coarse.loc[0, ["open", "high", "low"]] = fine.loc[1, ["open", "high", "low"]].to_numpy()
    result = audit.compare(fine, coarse, 15, 30)["counts"]
    assert result["full_mismatch_first_child_zero_volume"] == 1
    assert result["full_mismatch_active_only_prices_match_diagnostic"] == 1
    assert result["full_price_match"] == 1
    assert result["full_price_mismatch"] == 1


def test_vectorized_aggregation_matches_scalar_oracle():
    rng = np.random.default_rng(317)
    for _ in range(24):
        fine = fine_bars(40)
        coarse = coarse_bars(fine, 4)
        fine = fine.loc[rng.random(len(fine)) > 0.15].reset_index(drop=True)
        fine.loc[rng.integers(len(fine)), "high"] += 100
        counts = audit.compare(fine, coarse, 15, 60)["counts"]
        expected = dict.fromkeys(("full_coverage", "partial_coverage", "empty_coverage", "contained_ohlcv_match", "full_ohlcv_match"), 0)
        for row in coarse.itertuples():
            block = fine[
                (fine.datetime >= row.datetime)
                & (fine.datetime < row.datetime + pd.Timedelta(minutes=60))
            ]
            full = len(block) == 4
            expected[
                "full_coverage" if full else "partial_coverage" if len(block) else "empty_coverage"
            ] += 1
            if not len(block):
                continue
            aggregate = [
                block.open.iloc[0],
                block.high.max(),
                block.low.min(),
                block.close.iloc[-1],
                block.volume.sum(),
            ]
            equal = np.allclose(
                aggregate,
                [row.open, row.high, row.low, row.close, row.volume],
                atol=audit.ATOL,
                rtol=0,
            )
            expected["contained_ohlcv_match"] += int(equal)
            expected["full_ohlcv_match"] += int(equal and full)
        for key, value in expected.items():
            assert counts[key] == value
