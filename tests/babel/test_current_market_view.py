"""Physical alignment and offline delivery checks; no neural execution."""

import copy
import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from obson.babel import current_market_view as view


def raw_fixture():
    start = datetime(2026, 1, 1)
    records = [
        {
            "datetime": str(start + timedelta(minutes=15 * i)),
            "open": 100 + i,
            "high": 102 + i,
            "low": 99 + i,
            "close": 101 + i,
            "volume": 12,
            "close_oi": 300,
        }
        for i in range(140)
    ]
    row = {
        "key": "X/15/TEST",
        "symbol": "X",
        "period": 15,
        "row": 130,
        "end": records[130]["datetime"],
    }
    return records, row, str(start + timedelta(minutes=15 * 131))


def test_exact_window_ends_at_closed_bar_and_ignores_future():
    records, row, cutoff = raw_fixture()
    actual = view.bar_window(records, row, cutoff)
    assert np.asarray(actual["ohlc"])[:, 3].tolist() == list(range(104, 232))
    assert actual["closed_at"] == cutoff
    records[131]["close"] = float("nan")
    records[139]["datetime"] = "invalid future"
    assert view.bar_window(records, row, cutoff) == actual
    with pytest.raises(ValueError, match="not closed"):
        view.bar_window(records, row, row["end"])
    with pytest.raises(ValueError, match="naive"):
        view.bar_window(records, row, cutoff + "+08:00")


@pytest.mark.parametrize(
    "field,value",
    [("symbol", "Y"), ("period", 60), ("row", 126), ("row", True), ("end", "2026-01-01 00:00:00")],
)
def test_wrong_identity_is_not_silently_nearest_matched(field, value):
    records, row, cutoff = raw_fixture()
    with pytest.raises(ValueError):
        view.bar_window(records, {**row, field: value}, cutoff)


def test_raw_quality_failure_and_missing_activity():
    records, row, cutoff = raw_fixture()
    records[3]["volume"] = ""
    records[4]["close_oi"] = -1
    actual = view.bar_window(records, row, cutoff)
    assert actual["volume"][0] is None and actual["open_interest"][1] is None
    records[3]["high"] = 1
    with pytest.raises(ValueError, match="ordering"):
        view.bar_window(records, row, cutoff)
    records, row, cutoff = raw_fixture()
    records[3]["datetime"] = records[4]["datetime"]
    with pytest.raises(ValueError, match="increasing"):
        view.bar_window(records, row, cutoff)


def test_price_error_uses_physical_positions_and_excludes_zero_anchor():
    # Every historical prediction has a 5bp error; known current price has zero.
    raw = 100 * np.exp(np.arange(128)[None, :] * 0.001)
    expected = (np.arange(128) - 127)[None, :] * 0.001
    prediction = expected.copy()
    prediction[:, :-1] += 0.0005
    before = prediction.copy()
    _, _, errors = view.score(raw, dict.fromkeys(view.MODELS, prediction))
    np.testing.assert_allclose(errors["macro_s42"]["rmse_bps"], [[5, 5, 5]], atol=1e-11)
    np.testing.assert_array_equal(prediction, before)
    _, _, exact = view.score(raw, dict.fromkeys(view.MODELS, expected))
    assert exact["macro_s42"]["max_abs_bps"].max() < 1e-10
    with pytest.raises(ValueError, match="anchored"):
        view.score(raw, {"bad": np.ones((1, 128))})


def test_flat_failure_is_preserved_and_models_scored_separately():
    raw = np.full((1, 128), 24425.0)
    fake = (np.arange(128) - 127)[None, :] * 0.001
    _, summaries, errors = view.score(raw, {"macro_s42": fake, "macro_s43": np.zeros_like(fake)})
    assert summaries["raw"]["direction"].tolist() == [[1, 1]]
    assert summaries["raw"]["rms_bps"].tolist() == [[0, 0]]
    assert summaries["macro_s42"]["direction"].tolist() == [[2, 2]]
    result = view.cohort_summary(summaries, errors)
    assert result["macro_s42"]["classification"][0]["accuracy"] == 0
    assert result["macro_s43"]["classification"][0]["accuracy"] == 1
    assert result["macro_s42"]["mean_rmse_bps"] > 0


def make_archive(tmp_path, monkeypatch, files, extra=None):
    done = json.dumps(
        {
            "source_unchanged": True,
            "files": {key: hashlib.sha256(value).hexdigest() for key, value in files.items()},
        }
    ).encode()
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {"completion.json": done, **files}.items():
            info = tarfile.TarInfo(view.PREFIX + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            tar.addfile(extra)
    monkeypatch.setattr(view, "ARCHIVE_SHA256", view.sha256(path))
    monkeypatch.setattr(view, "COMPLETION_SHA256", hashlib.sha256(done).hexdigest())
    return path


def test_archive_and_consumed_member_fingerprints(tmp_path, monkeypatch):
    path = make_archive(tmp_path, monkeypatch, {"x.json": b"{}"})
    report = view.ReviewedArchive(path)
    try:
        assert report.read("x.json") == b"{}"
        report.done["files"]["x.json"] = "wrong"
        with pytest.raises(ValueError, match="fingerprint"):
            report.read("x.json")
    finally:
        report.close()
    assert report.tar.closed
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="reviewed"):
        view.ReviewedArchive(path)


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "symlink"])
def test_unsafe_archives_rejected_without_extraction(tmp_path, monkeypatch, kind):
    name = {
        "traversal": view.PREFIX + "../escape",
        "duplicate": view.PREFIX + "x.json",
        "symlink": view.PREFIX + "link",
    }[kind]
    extra = tarfile.TarInfo(name)
    if kind == "symlink":
        extra.type = tarfile.SYMTYPE
        extra.linkname = "/tmp/escape"
    path = make_archive(tmp_path, monkeypatch, {"x.json": b"{}"}, extra)
    with pytest.raises(ValueError, match="Unsafe"):
        view.ReviewedArchive(path)


def test_build_keeps_every_row_and_both_fixed_versions(tmp_path, monkeypatch):
    records, row, cutoff = raw_fixture()
    files = {}
    window = view.bar_window(records, row, cutoff)
    original = view.truth.from_raw_closes(np.asarray(window["ohlc"])[None, :, 3])[
        "relative_log_close"
    ]
    for split, count in {"test": 2, "cross_research": 1}.items():
        files[split + "_rows.json"] = json.dumps([row] * count).encode()
        for m, model in enumerate(view.MODELS):
            bank = io.BytesIO()
            prediction = np.repeat(original, count, axis=0)
            prediction[:, :-1] += m * 0.0002
            np.savez(bank, relative_log_close=prediction)
            files[f"{split}_{model}.npz"] = bank.getvalue()
    path = make_archive(tmp_path, monkeypatch, files)
    monkeypatch.setattr(view, "SPLITS", {"test": 2, "cross_research": 1})
    monkeypatch.setattr(
        view, "load_raw", lambda *args: ({"test": [window, window], "cross_research": [window]}, 1)
    )
    result = view.build_data(path, tmp_path)
    assert [c["id"] for c in result["cases"]] == ["test:0", "test:1", "cross_research:0"]
    assert result["receipt"]["windows"] == 3
    assert result["receipt"]["model_forwards"] == 0
    assert result["receipt"]["continuous_replay"] is False
    assert len(result["receipt"]["verified_members"]) == 6
    np.testing.assert_allclose(result["cases"][0]["errors"]["macro_s43"]["rmse_bps"], [2, 2, 2])


def test_csv_fingerprint_prevents_data_substitution(tmp_path):
    records, row, cutoff = raw_fixture()
    raw = tmp_path / "X"
    raw.mkdir()
    path = raw / "TEST_15m.csv"
    keys = list(records[0])
    path.write_text(
        ",".join(keys) + "\n" + "\n".join(",".join(str(r[k]) for k in keys) for r in records)
    )
    inventory = {"files": {row["key"]: view.sha256(path)}, "asof": cutoff}
    windows, count = view.load_raw(tmp_path, {"test": [row]}, inventory)
    assert count == 1 and windows["test"][0] == view.bar_window(records, row, cutoff)
    path.write_text(path.read_text().replace("300", "400"))
    with pytest.raises(ValueError, match="fingerprint"):
        view.load_raw(tmp_path, {"test": [row]}, inventory)


def test_json_cannot_break_out_of_script_and_output_cannot_overwrite_inputs(tmp_path):
    data = {"note": "</script><script>alert(1)</script>\u2028", "value": [1, 2]}
    before = copy.deepcopy(data)
    path = view.render(data, tmp_path / "view.html")
    payload = (
        path.read_text()
        .split('<script id="market-data" type="application/json">')[1]
        .split("</script>")[0]
    )
    assert "<" not in payload and json.loads(payload) == before and data == before
    with pytest.raises(ValueError):
        view.render({"bad": np.nan}, tmp_path / "bad.html")
    for out in [path, tmp_path / "data" / "view.html", Path(view.__file__).with_suffix(".html")]:
        with pytest.raises(ValueError, match="separate"):
            view.output_path(out, path, tmp_path / "data")
    assert view.output_path(tmp_path / "ok.html", path, tmp_path / "data") == tmp_path / "ok.html"
