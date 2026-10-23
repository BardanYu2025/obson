"""Synthetic endpoint, matrix and immutable export tests; no real weights."""

import copy
import json

import numpy as np
import pytest
import torch

from obson.babel import multiscale_evaluate as inherited
from obson.babel import observed_history_export_v2 as corrected
from obson.babel import state_organization as model
from obson.babel import state_organization_evaluate as evaluation
from obson.babel import state_organization_export as export

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}
TINY = {"width": 16, "heads": 4, "ff": 32, "layers": 4}


@pytest.fixture(autouse=True)
def cpu_runtime():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def identities(n):
    return [
        {"key": f"TEST/15/c{i % 3}", "month": f"2025-{i % 6 + 1:02d}", "parent": i}
        for i in range(n)
    ]


def bank(n=120):
    rows = identities(n)
    keys = ["full/price/composite", *evaluation.protection_keys()]
    support_keys = [f"price/{b}" for b in ("near", "mid", "far", "composite")]
    support_keys += [f"nonprice/{f}/{b}" for f in evaluation.FIELDS for b in ("near", "mid", "far")]
    return {
        "errors": {k: np.ones(n) for k in keys},
        "support": {k: np.ones(n, np.int64) for k in keys},
        "contexts": {k: np.ones(n, np.int64) for k in keys if not k.startswith("overlap/")},
        "position_support": {k: np.ones((n, 128), np.int64) for k in support_keys},
        "mask_digest": "same-synthetic-masks",
        "rows_digest": evaluation._digest(rows),
        "data_digest": "same-synthetic-data",
    }, rows


def named_banks(factors):
    base, rows = bank()
    banks = {}
    for split in ("test", "cross_research"):
        banks[split] = {}
        for name in evaluation.RUNS:
            arm = name.rsplit("_s", 1)[0]
            item = copy.deepcopy(base)
            item["errors"] = {k: v * factors[arm] for k, v in item["errors"].items()}
            item.update(
                model_name=name,
                split=split,
                model_lock_digest="synthetic-complete-eight",
                model_contract={
                    "arm": evaluation.MODEL_ARMS[arm],
                    "attention_mode": "causal",
                    "public_output": "organized_final_state_only",
                    "private_memory_exposed": False,
                    "memory_write_sources": evaluation.WRITE_SOURCES[evaluation.MODEL_ARMS[arm]],
                },
            )
            banks[split][name] = item
    return banks, dict.fromkeys(banks, rows)


def test_physical_scoring_and_gate_functions_are_unchanged():
    for name in (
        "score_rows",
        "overlap_rows",
        "aggregate_rows",
        "comparison",
        "paired_intervals",
        "protection_keys",
    ):
        assert getattr(evaluation, name) is getattr(inherited, name)
    assert export._score_array is corrected._score_array


@pytest.mark.parametrize(
    "factors,primary,plain,supported",
    [
        (
            {"initial": 1.0, "plain": 1.0, "late_memory": 0.8, "staged_memory": 0.8},
            False,
            True,
            False,
        ),
        (
            {"initial": 1.0, "plain": 1.0, "late_memory": 1.2, "staged_memory": 1.1},
            True,
            False,
            False,
        ),
        (
            {"initial": 1.0, "plain": 1.0, "late_memory": 0.9, "staged_memory": 0.7},
            True,
            True,
            True,
        ),
    ],
)
def test_both_matched_and_plain_references_required(factors, primary, plain, supported):
    banks, rows = named_banks(factors)
    result = evaluation.decisions(banks, rows)
    assert result["limited_hypothesis_supported"] is supported
    assert result["comparison_groups"]["staged_vs_late"]["limited_hypothesis_supported"] is primary
    assert result["comparison_groups"]["staged_vs_plain"]["limited_hypothesis_supported"] is plain
    assert result["primary_comparison"] == "staged_vs_late"
    assert result["required_comparisons"] == ["staged_vs_late", "staged_vs_plain"]
    assert "budget_hypothesis_supported" not in result
    assert all(len(v["comparisons"]) == 4 for v in result["comparison_groups"].values())
    assert result["promoted"] is False and result["fit_count"] == 0
    assert result["f11"]["general_reuse"] == "unknown_not_tested"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_initial",
        "mislabeled_initial",
        "other_lock",
        "noncausal",
        "wrong_depth",
        "exposed_memory",
        "missing_cohort",
    ],
)
def test_full_eight_model_two_cohort_identity_required(mutation):
    banks, rows = named_banks(
        {"initial": 1.0, "plain": 1.0, "late_memory": 0.8, "staged_memory": 0.6}
    )
    if mutation == "missing_initial":
        del banks["test"]["initial_s42"]
    elif mutation == "mislabeled_initial":
        banks["test"]["initial_s42"]["model_contract"]["arm"] = "initial"
    elif mutation == "other_lock":
        banks["test"]["initial_s43"]["model_lock_digest"] = "different"
    elif mutation == "noncausal":
        banks["test"]["staged_memory_s42"]["model_contract"]["attention_mode"] = "full_observed"
    elif mutation == "wrong_depth":
        banks["test"]["staged_memory_s42"]["model_contract"]["memory_write_sources"] = [4, 4]
    elif mutation == "exposed_memory":
        banks["test"]["staged_memory_s42"]["model_contract"]["private_memory_exposed"] = True
    else:
        del banks["cross_research"]
    with pytest.raises(ValueError):
        evaluation.decisions(banks, rows)


def test_primary_gain_cannot_hide_activity_damage():
    banks, rows = named_banks(
        {"initial": 1.0, "plain": 1.0, "late_memory": 0.8, "staged_memory": 0.6}
    )
    banks["cross_research"]["staged_memory_s43"]["errors"]["full/nonprice/oi_per_volume_asinh/far"][
        :
    ] = 2.0
    result = evaluation.decisions(banks, rows)
    assert not result["limited_hypothesis_supported"]
    assert result["comparison_groups"]["late_vs_plain"]["limited_hypothesis_supported"]


def source(n=1):
    x = np.zeros((n, 255, 28), np.float32)
    x[..., 23:28] = 1
    x[..., 18:23] = 0.01
    raw = 100 * np.exp(np.random.default_rng(7).normal(0, 0.001, (n, 255)).cumsum(1))
    return x, raw


def extract(out, name="staged_memory_s42", *, pair=None, verifier=None):
    arm = evaluation.MODEL_ARMS[name.rsplit("_s", 1)[0]]
    if pair is None:
        pair = model.make_models(arm, 42, "cpu", TINY)
    if verifier is None:

        def verifier(*_):
            return {
                "model_lock": "synthetic-eight-models",
                "source": "synthetic",
                "raw_receipt": "synthetic",
            }

    x, raw = source()
    bank = evaluation.extract(
        {"statistics": STATS, "epochs": 60, "batch": 128, "micro": 8},
        out,
        name,
        *pair,
        x,
        raw,
        identities(1),
        "test",
        "cpu",
        verify_lock=verifier,
    )
    return bank, pair


def test_final_only_extraction_all128_fixed8_and_resume(tmp_path):
    pair = model.make_models("staged_memory", 42, "cpu", TINY)

    query_shapes = []
    pair[1].register_forward_pre_hook(
        lambda module, args: query_shapes.append(tuple(args[0].shape))
    )
    calls = []
    pair[0].register_forward_pre_hook(lambda *args: calls.append(1))
    bank, _ = extract(tmp_path, pair=pair)
    assert len(calls) == 26
    assert query_shapes and all(shape[1:] == (16,) for shape in query_shapes)
    receipt = json.loads((tmp_path / bank["receipt"]).read_text())
    assert receipt["states"] == 128 and receipt["updates"] == receipt["fits"] == 0
    with np.load(tmp_path / "results/test_staged_memory_s42_predictions.npz") as saved:
        assert saved["predictions"].shape == (1, 8, 127, 7)
        np.testing.assert_array_equal(saved["positions"], evaluation.SAMPLE_POSITIONS)
    with np.load(tmp_path / "evaluation/staged_memory_s42/test/position_rows.npz") as saved:
        assert saved["errors::price/near"].shape == (1, 128)
    extract(tmp_path, pair=pair)
    assert len(calls) == 26


def test_initial_maps_to_plain_and_partial_research_is_not_replayed(tmp_path):
    bank, pair = extract(tmp_path, "initial_s42")
    assert bank["model_contract"]["arm"] == "plain"
    path = tmp_path / "predictions/initial_s42/test/parents_00000_00001.npy"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        extract(tmp_path, "initial_s42", pair=pair)
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    pair = model.make_models("staged_memory", 42, "cpu", TINY)

    def fail(*_):
        raise RuntimeError("synthetic failure")

    hook = pair[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        extract(interrupted, pair=pair)
    hook.remove()
    with pytest.raises(ValueError, match="export-only"):
        extract(interrupted, pair=pair)


def test_eight_model_two_cohort_pipeline_keeps_every_score_bank(tmp_path):
    """Tiny synthetic neural execution is evidence plumbing, not market acceptance."""
    pairs = {
        name: model.make_models(
            evaluation.MODEL_ARMS[name.rsplit("_s", 1)[0]], int(name[-2:]), "cpu", TINY
        )
        for name in evaluation.RUNS
    }
    banks, rows_by_cohort, calls = {}, {}, []

    def verify(meta, out, name, split):
        assert out == tmp_path and name in pairs and len(pairs) == 8
        calls.append((name, split))
        return {
            "model_lock": "synthetic-eight-model-lock",
            "source": "synthetic",
            "raw_receipt": "synthetic",
        }

    for split in ("test", "cross_research"):
        x, raw = source()
        rows = identities(1)
        rows_by_cohort[split] = rows
        banks[split] = {
            name: evaluation.extract(
                {"statistics": STATS, "epochs": 60, "batch": 128, "micro": 8},
                tmp_path,
                name,
                *pair,
                x,
                raw,
                rows,
                split,
                "cpu",
                verify_lock=verify,
            )
            for name, pair in pairs.items()
        }
    assert len(calls) == 32  # before and after each of 16 complete extractions
    result = evaluation.decisions(banks, rows_by_cohort)
    assert not result["limited_hypothesis_supported"]
    assert all(
        value["checks"]["gain"]["status"] == "insufficient"
        for group in result["comparison_groups"].values()
        for value in group["comparisons"].values()
    )
    for name in evaluation.RUNS:
        for split in ("test", "cross_research"):
            root = tmp_path / "evaluation" / name / split
            assert export._score_array(root / "parent_rows.npz", False) == 1
            assert export._score_array(root / "position_rows.npz", True) == 1
