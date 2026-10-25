"""Synthetic CPU tests for factorial identification and all128 endpoint evidence."""

import json

import numpy as np
import pytest
import torch

from obson.babel import input_factorial as model
from obson.babel import input_factorial_evaluate as evaluation
from obson.babel import input_patch as historical
from obson.babel import multiscale_evaluate as inherited

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


def banks(factors=None):
    factors = factors or {
        "dense_random": 1.0,
        "hybrid_random": 0.8,
        "dense_endpoint": 0.8,
        "hybrid_endpoint": 0.6,
        "reference_bar": 0.9,
    }
    rows = identities(120)
    keys = ["full/price/composite", *evaluation.protection_keys()]
    positions = [f"price/{b}" for b in ("near", "mid", "far", "composite")]
    positions += [f"nonprice/{f}/{b}" for f in evaluation.FIELDS for b in ("near", "mid", "far")]
    result = {}
    for split in evaluation.COHORTS:
        result[split] = {}
        for name in evaluation.RUNS:
            arm = name.rsplit("_s", 1)[0]
            result[split][name] = {
                "errors": {k: np.full(120, factors[arm]) for k in keys},
                "support": {k: np.ones(120, np.int64) for k in keys},
                "contexts": {
                    k: np.ones(120, np.int64) for k in keys if not k.startswith("overlap/")
                },
                "position_support": {k: np.ones((120, 128), np.int64) for k in positions},
                "mask_digest": "same-mask",
                "rows_digest": evaluation._digest(rows),
                "data_digest": "same-data",
                "model_name": name,
                "split": split,
                "model_lock_digest": "complete-ten-models",
                "model_signature": name + "-last",
                "model_contract": historical.contract("bar", TINY)
                if arm == "reference_bar"
                else model.contract(arm, TINY),
            }
    return result, dict.fromkeys(result, rows)


def test_metric_formulas_are_unchanged_and_all_twenty_cases_are_required():
    for name in ("comparison", "score_rows", "overlap_rows", "paired_intervals", "aggregate_rows"):
        assert getattr(evaluation, name) is getattr(inherited, name)
    values, rows = banks()
    result = evaluation.decisions(values, rows)
    assert result["limited_hypothesis_supported"] and not result["promoted"]
    assert result["primary_candidate"] == "hybrid_endpoint"
    assert result["historical_anchor_passed"]
    cases = [
        v for group in result["comparison_groups"].values() for v in group["comparisons"].values()
    ]
    assert len(cases) == 20 and sum(len(c["checks"]) for c in cases) == 1580
    assert (
        result["comparison_groups"]["historical_bar_anchor"]["role"]
        == "historical_frozen_anchor_not_concurrent_control"
    )


def test_one_initialization_only_is_conditional_not_structure_independent():
    values, rows = banks(
        {
            "dense_random": 1.0,
            "hybrid_random": 1.0,
            "dense_endpoint": 1.0,
            "hybrid_endpoint": 0.8,
            "reference_bar": 0.7,
        }
    )
    result = evaluation.decisions(values, rows)
    assert result["conclusion"] == "initialization_dependent_conditional_structure_gain"
    assert not result["limited_hypothesis_supported"]
    assert not result["historical_anchor_passed"] and not result["promoted"]


def test_initialization_without_structure_and_no_gain_are_distinct():
    factors = {
        "dense_random": 1.0,
        "hybrid_random": 1.0,
        "dense_endpoint": 0.8,
        "hybrid_endpoint": 0.8,
        "reference_bar": 0.9,
    }
    values, rows = banks(factors)
    assert (
        evaluation.decisions(values, rows)["conclusion"]
        == "initialization_gain_without_accepted_structure_gain"
    )
    values, rows = banks(dict.fromkeys(factors, 1.0))
    result = evaluation.decisions(values, rows)
    assert result["conclusion"] == "fixed_32_4_120_recipe_closed_without_balanced_gain"
    assert not result["promoted"]


def test_interaction_is_resampled_as_one_paired_parent_difference():
    values, rows = banks()
    b = values["test"]
    n = len(rows["test"])
    common = np.linspace(0, 4, n)
    offsets = {
        "dense_random": 2.0,
        "hybrid_random": 1.8,
        "dense_endpoint": 2.2,
        "hybrid_endpoint": 1.7,
    }
    for arm, offset in offsets.items():
        for band in ("composite", "near", "mid", "far"):
            b[f"{arm}_s42"]["errors"][f"full/price/{band}"] = common + offset
    result = evaluation.interaction(b, rows["test"], 42)
    for metric in result["metrics"].values():
        assert metric["mean"] == pytest.approx(-0.3)
        for ci in metric["intervals"].values():
            assert ci["low"] == pytest.approx(-0.3) and ci["high"] == pytest.approx(-0.3)
        assert metric["direction"] == "hybrid_relative_benefit_larger_with_endpoint_initialization"
    assert not result["acceptance_gate"]


@pytest.mark.parametrize(
    "metric",
    ["endpoint/price/near", "full/nonprice/oi_per_volume_asinh/far", "overlap/shift64/shape_b_mse"],
)
def test_single_guard_failure_cannot_be_hidden_by_composite_or_other_seed(metric):
    values, rows = banks()
    values["cross_research"]["hybrid_random_s43"]["errors"][metric][:] = 1.4
    result = evaluation.decisions(values, rows)
    case = result["comparison_groups"]["structure_random"]["comparisons"]["cross_research/s43"]
    assert case["gain_passed"] and not case["protection_passed"]
    assert not result["limited_hypothesis_supported"]


@pytest.mark.parametrize(
    "damage", ["missing", "lock", "signature", "contract", "support", "reference"]
)
def test_matrix_semantics_and_reference_identity_fail_closed(damage):
    values, rows = banks()
    entry = values["cross_research"]["hybrid_endpoint_s43"]
    if damage == "missing":
        del values["test"]["dense_random_s42"]
    elif damage == "lock":
        entry["model_lock_digest"] = "other"
    elif damage == "signature":
        entry["model_signature"] = "other"
    elif damage == "contract":
        entry["model_contract"] = model.contract("dense_endpoint", TINY)
    elif damage == "reference":
        values["test"]["reference_bar_s42"]["model_contract"] = model.contract(
            "dense_endpoint", TINY
        )
    else:
        entry["position_support"]["price/near"][0, 0] += 1
    with pytest.raises(ValueError):
        evaluation.decisions(values, rows)


def source(n=2):
    x = np.zeros((n, 255, 28), np.float32)
    x[..., 23:28], x[..., 18:23] = 1, 0.01
    raw = 100 * np.exp(np.random.default_rng(2).normal(0, 0.001, (n, 255)).cumsum(1))
    return x, raw


def verifier(*_):
    return {
        "model_lock": "eight-final120-plus-two-reference",
        "source": "source",
        "raw_receipt": "raw",
    }


def extract(out, arm="hybrid_endpoint", pair=None, verify=verifier):
    pair = pair or (
        historical.make_models("bar", 42, "cpu", TINY)
        if arm == "reference_bar"
        else model.make_models(arm, 42, "cpu", TINY)
    )
    x, raw = source()
    result = evaluation.extract(
        {"statistics": STATS, "batch": 128, "micro": 8, "epochs": 120},
        out,
        f"{arm}_s42",
        *pair,
        x,
        raw,
        identities(2),
        "test",
        "cpu",
        verify_lock=verify,
    )
    return result, pair


@pytest.mark.parametrize("arm", (*model.ARMS, "reference_bar"))
def test_each_state_gets_own128_and_fixed8_are_exact_subset_and_resume_is_zero_forward(
    tmp_path, arm
):
    pair = (
        historical.make_models("bar", 42, "cpu", TINY)
        if arm == "reference_bar"
        else model.make_models(arm, 42, "cpu", TINY)
    )
    calls = []
    pair[0].register_forward_pre_hook(
        lambda _, args: calls.append((args[0].shape, torch.is_grad_enabled()))
    )
    result, _ = extract(tmp_path, arm, pair)
    assert sum(s[0] for s, _ in calls) == 256
    assert len(calls) == 26 and all(s[1:] == (128, 28) and not grad for s, grad in calls)
    with np.load(tmp_path / f"results/test_{arm}_s42_predictions.npz") as saved:
        assert saved["predictions"].shape == (2, 8, 127, 7)
        subset = evaluation.score_rows(
            saved["predictions"],
            saved["targets"],
            saved["masks"],
            saved["raw_closes"],
            STATS,
            saved["positions"],
        )
    with np.load(tmp_path / f"evaluation/{arm}_s42/test/position_rows.npz") as saved:
        for key, value in subset["errors"].items():
            np.testing.assert_allclose(
                value,
                saved["errors::" + key][:, np.array(evaluation.SAMPLE_POSITIONS) - 1],
                equal_nan=True,
            )
    receipt = json.loads((tmp_path / result["receipt"]).read_text())
    assert receipt["states"] == 256 and receipt["updates"] == receipt["fits"] == 0
    before = len(calls)
    again, _ = extract(tmp_path, arm, pair)
    assert len(calls) == before and again["model_signature"] == result["model_signature"]


def test_incomplete_extraction_is_export_only_and_missing_lock_never_forwards(tmp_path):
    pair = model.make_models("hybrid_endpoint", 42, "cpu", TINY)
    calls = []
    pair[0].register_forward_pre_hook(lambda *_: calls.append(1))
    with pytest.raises(ValueError, match="matrix"):
        extract(tmp_path, pair=pair, verify=lambda *_: {"source": "missing"})
    assert not calls

    def fail(*_):
        raise RuntimeError("synthetic interruption")

    hook = pair[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="interruption"):
        extract(tmp_path, pair=pair)
    hook.remove()
    with pytest.raises(ValueError, match="export-only"):
        extract(tmp_path, pair=pair)
