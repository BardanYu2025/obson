"""CPU temporal selection only; fixtures contain no model outputs or raw prices."""

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from obson.babel import continuous_state_selection as pick


def series(symbol, period, suffix="1", n=1800):
    key = f"{symbol}/{period}/EX.{symbol}{suffix}"
    frame = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=n, freq=f"{period}min")})
    return SimpleNamespace(key=key, period=period, code=symbol, frame=frame)


def source_row(s, row):
    return {
        "key": s.key,
        "period": s.period,
        "symbol": s.code,
        "row": row,
        "end": str(s.frame.datetime.iloc[row]),
        "input_start": str(s.frame.datetime.iloc[row - 254]),
        "original_index": row,
    }


def fixture():
    sources, rows, masks = {}, {}, {}
    for split in pick.SPLITS:
        sources[split], rows[split], masks[split] = {}, [], {}
        for period in pick.PERIODS:
            for symbol, suffix in (("A", "1"), ("A", "2"), ("B", "1"), ("C", "1")):
                s = series(symbol, period, suffix)
                sources[split][s.key] = s
                masks[split][s.key] = np.ones(len(s.frame), bool)
                rows[split].extend([source_row(s, 600), source_row(s, 300)])
    extra = next(iter(sources["test"].values()))
    while len(rows["test"]) < 171:
        rows["test"].append(source_row(extra, 700 + len(rows["test"])))
    s = series("ag", 15, n=6300)
    s.key = pick.STRESS["key"]
    sources["test"][s.key] = s
    masks["test"][s.key] = np.ones(len(s.frame), bool)
    rows["test"].append(source_row(s, 5759))
    return sources, rows, masks


def test_exact_budget_distinct_symbols_deterministic_ranking_and_source_fields():
    sources, rows, masks = fixture()
    result = pick.select_segments(sources, rows, masks)
    assert len(result) == 13
    for split in pick.SPLITS:
        for period in pick.PERIODS:
            cell = [
                r
                for r in result
                if r["split"] == split and r["period"] == period and r["role"] == "coverage"
            ]
            assert len(cell) == 2 and len({r["symbol"] for r in cell}) == 2
            order = sorted(
                (hashlib.sha256(f"{split}/{k}".encode()).hexdigest(), k)
                for k, s in sources[split].items()
                if s.period == period
            )
            expected, symbols = [], set()
            for _, key in order:
                s = sources[split][key]
                if s.code not in symbols:
                    expected.append(key)
                    symbols.add(s.code)
                if len(expected) == 2:
                    break
            assert [r["key"] for r in cell] == expected
            for record in cell:
                expected_first = 5759 if record["key"] == pick.STRESS["key"] else 300
                assert record["first"] == expected_first
                assert record["last"] - record["first"] + 1 == 512
                assert record["source_offset"] == 0
                assert rows[split][record["source_index"]]["row"] == record["source_row"]
                assert len(record["oracle_offsets"]) == 16
    stress = result[-1]
    assert stress["role"] == "known_flat_stress"
    assert (stress["first"], stress["last"], stress["source_index"], stress["source_offset"]) == (
        5632,
        6143,
        171,
        127,
    )
    assert stress["source_row"] == 5759
    reordered = {split: dict(reversed(list(sources[split].items()))) for split in pick.SPLITS}
    assert pick.select_segments(reordered, rows, masks) == result
    assert tuple(result[0]["oracle_offsets"]) == (
        0,
        31,
        63,
        95,
        127,
        159,
        191,
        223,
        255,
        287,
        319,
        351,
        383,
        415,
        447,
        511,
    )


def test_skip_early_endpoint_that_cannot_cover512_without_weakening_its_context():
    sources, rows, masks = fixture()
    # The first candidate's early row lies in the split but its future512 span
    # crosses a split hole. Its later source row has a fully eligible context.
    key = next(
        r["key"]
        for r in pick.select_segments(sources, rows, masks)
        if r["split"] == "test" and r["period"] == 30
    )
    masks["test"][key][320] = False
    for i, row in enumerate(rows["test"]):
        if row["key"] == key and row["row"] == 600:
            rows["test"][i] = source_row(sources["test"][key], 700)
    result = pick.select_segments(sources, rows, masks)
    selected = [r for r in result if r["key"] == key]
    assert selected and selected[0]["first"] == 700


@pytest.mark.parametrize(
    "field,value",
    [
        ("row", -1),
        ("end", "2026-01-01"),
        ("input_start", "2026-01-01"),
        ("period", 60),
        ("symbol", "wrong"),
    ],
)
def test_source_identity_errors_fail_before_selection(field, value):
    sources, rows, masks = fixture()
    rows["test"][0][field] = value
    with pytest.raises(ValueError, match="identity/partition"):
        pick.select_segments(sources, rows, masks)


def test_original_full255_source_partition_is_not_relaxed_to128():
    sources, rows, masks = fixture()
    row = rows["test"][0]
    masks["test"][row["key"]][row["row"] - 254] = False
    with pytest.raises(ValueError, match="identity/partition"):
        pick.select_segments(sources, rows, masks)


def test_insufficient_distinct_symbols_or_length_fails_without_reduction():
    sources, rows, masks = fixture()
    rows["cross_research"] = [
        r for r in rows["cross_research"] if r["period"] != 60 or r["symbol"] == "A"
    ]
    with pytest.raises(ValueError, match="Insufficient512"):
        pick.select_segments(sources, rows, masks)
    sources, rows, masks = fixture()
    for s in sources["cross_research"].values():
        if s.period == 60:
            masks["cross_research"][s.key][801:] = False
    with pytest.raises(ValueError, match="Insufficient512"):
        pick.select_segments(sources, rows, masks)


def test_stress_identity_and_whole_history_mask_are_locked():
    sources, rows, masks = fixture()
    rows["test"][171] = source_row(sources["test"][pick.STRESS["key"]], 5760)
    with pytest.raises(ValueError, match="known-flat test:171"):
        pick.select_segments(sources, rows, masks)
    sources, rows, masks = fixture()
    masks["test"][pick.STRESS["key"]][6143] = False
    with pytest.raises(ValueError, match="stress context"):
        pick.select_segments(sources, rows, masks)


def test_nominal_closed_asof_and_bad_masks_or_timezone_rejected():
    sources, rows, masks = fixture()
    key = "A/15/EX.A1"
    masks["test"][key] = masks["test"][key].astype(int)
    with pytest.raises(ValueError, match="boolean"):
        pick.select_segments(sources, rows, masks)
    sources, rows, masks = fixture()
    sources["test"][key].frame["datetime"] = sources["test"][key].frame.datetime.dt.tz_localize(
        "Asia/Shanghai"
    )
    with pytest.raises(ValueError, match="Naive"):
        pick.select_segments(sources, rows, masks)
    dates = pd.date_range(end=pick.ASOF, periods=700, freq="15min")
    same = np.ones(len(dates), bool)
    assert not pick._eligible(188, dates, same, 15)  # Last bar has opened but not closed.
    assert pick._eligible(187, dates, same, 15)


def test_input_objects_are_not_modified():
    sources, rows, masks = fixture()
    old_rows = copy.deepcopy(rows)
    old_masks = {s: {k: v.copy() for k, v in m.items()} for s, m in masks.items()}
    pick.select_segments(sources, rows, masks)
    assert rows == old_rows
    for split, items in masks.items():
        for key, same in items.items():
            np.testing.assert_array_equal(same, old_masks[split][key])
