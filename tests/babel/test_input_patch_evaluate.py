"""Synthetic CPU matrix/contract and complete-endpoint extraction evidence."""

import json

import numpy as np
import pytest
import torch

from obson.babel import input_patch as model
from obson.babel import input_patch_evaluate as evaluation
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
    factors = factors or {"bar": 1.0, "local4": 0.9, "patch4": 0.7}
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
                "mask_digest": "matched-masks",
                "rows_digest": evaluation._digest(rows),
                "data_digest": "matched-data",
                "model_name": name,
                "split": split,
                "model_lock_digest": "complete-six-models",
                "model_signature": name + "-final",
                "model_contract": model.contract(arm, TINY),
            }
    return result, dict.fromkeys(result, rows)


def test_metric_formulas_are_explicit_unchanged_imports():
    for name in ("comparison", "score_rows", "overlap_rows", "paired_intervals", "aggregate_rows"):
        assert getattr(evaluation, name) is getattr(inherited, name)
    assert evaluation.metric_contract()["haar_fit"] is False


def test_both_required_comparisons_and_all948_interval_gates():
    values, rows = banks()
    result = evaluation.decisions(values, rows)
    assert result["limited_hypothesis_supported"] and result["promoted"] is False
    assert result["fit_count"] == result["f11"]["new_readout_fits"] == 0
    assert result["required_comparisons"] == ["patch4_vs_bar", "patch4_vs_local4"]
    assert result["secondary_comparison"] == "local4_vs_bar"
    cases = [
        v for group in result["comparison_groups"].values() for v in group["comparisons"].values()
    ]
    assert len(cases) == 12 and sum(len(c["checks"]) for c in cases) == 948
    assert all(c["passed"] for c in cases)


@pytest.mark.parametrize(
    "factors",
    [
        {"bar": 1.0, "local4": 0.7, "patch4": 0.7},
        {"bar": 0.7, "local4": 1.0, "patch4": 0.7},
    ],
)
def test_beating_only_one_required_reference_is_not_accepted(factors):
    values, rows = banks(factors)
    result = evaluation.decisions(values, rows)
    assert not result["limited_hypothesis_supported"]
    assert result["conclusion"] == "no_accepted_input_patch_gain"
    assert not result["promoted"]


@pytest.mark.parametrize(
    "metric",
    [
        "endpoint/price/near",
        "full/nonprice/oi_per_volume_asinh/far",
        "overlap/shift64/shape_b_mse",
    ],
)
def test_one_seed_cohort_guard_cannot_be_hidden_by_average_gain(metric):
    values, rows = banks()
    values["cross_research"]["patch4_s43"]["errors"][metric][:] = 1.3
    result = evaluation.decisions(values, rows)
    assert not result["limited_hypothesis_supported"]
    case = result["comparison_groups"]["patch4_vs_bar"]["comparisons"]["cross_research/s43"]
    assert case["gain_passed"] and not case["protection_passed"]


def test_tail_guard_is_required_despite_passing_mean_gain():
    values, rows = banks({"bar": 1.0, "local4": 1.0, "patch4": 0.1})
    values["test"]["patch4_s42"]["errors"]["full/price/composite"][:12] = 2.0
    result = evaluation.decisions(values, rows)
    case = result["comparison_groups"]["patch4_vs_bar"]["comparisons"]["test/s42"]
    assert case["gain_passed"] and not case["tail"]["passed"]
    assert not result["limited_hypothesis_supported"]


@pytest.mark.parametrize(
    "damage", ["missing_arm", "lock", "signature", "packing", "positions", "padding", "support"]
)
def test_matrix_and_same_tensor_different_semantics_are_rejected(damage):
    values, rows = banks()
    entry = values["cross_research"]["patch4_s43"]
    if damage == "missing_arm":
        del values["test"]["local4_s42"]
    elif damage == "lock":
        entry["model_lock_digest"] = "other"
    elif damage == "signature":
        entry["model_signature"] = "other"
    elif damage == "packing":
        entry["model_contract"]["input_order"] = "feature_first"
    elif damage == "positions":
        entry["model_contract"]["token_endpoints_zero_based"] = list(range(32))
    elif damage == "padding":
        entry["model_contract"]["padding"] = "read_before_window"
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
    return {"model_lock": "all-six-final120", "source": "data-only", "raw_receipt": "raw"}


def extract(out, arm="patch4", pair=None, verify=verifier):
    pair = pair or model.make_models(arm, 42, "cpu", TINY)
    x, closes = source()
    result = evaluation.extract(
        {"statistics": STATS, "batch": 128, "micro": 8, "epochs": 120},
        out,
        f"{arm}_s42",
        *pair,
        x,
        closes,
        identities(2),
        "test",
        "cpu",
        verify_lock=verify,
    )
    return result, pair


@pytest.mark.parametrize("arm", model.ARMS)
def test_all128_endpoint_states_fixed8_predictions_and_zero_update_resume(tmp_path, arm):
    pair = model.make_models(arm, 42, "cpu", TINY)
    calls = []
    pair[0].register_forward_pre_hook(
        lambda _, args: calls.append((args[0].shape, torch.is_grad_enabled()))
    )
    result, _ = extract(tmp_path, arm, pair)
    assert sum(shape[0] for shape, _ in calls) == 256
    assert len(calls) == 26 and all(shape[1:] == (128, 28) and not grad for shape, grad in calls)
    path = tmp_path / f"results/test_{arm}_s42_predictions.npz"
    with np.load(path) as saved:
        assert saved["predictions"].shape == (2, 8, 127, 7)
        np.testing.assert_array_equal(saved["positions"], evaluation.SAMPLE_POSITIONS)
        subset = evaluation.score_rows(
            saved["predictions"],
            saved["targets"],
            saved["masks"],
            saved["raw_closes"],
            STATS,
            saved["positions"],
        )
    with np.load(tmp_path / f"evaluation/{arm}_s42/test/position_rows.npz") as positions:
        for key, value in subset["errors"].items():
            np.testing.assert_allclose(
                value,
                positions["errors::" + key][:, np.array(evaluation.SAMPLE_POSITIONS) - 1],
                equal_nan=True,
            )
    receipt = json.loads((tmp_path / result["receipt"]).read_text())
    assert receipt["states"] == 256 and receipt["updates"] == receipt["fits"] == 0
    assert result["model_signature"] == receipt["binding"]["model_signature"]
    assert receipt["binding"]["model_contract"]["token_count"] == (32 if arm == "patch4" else 128)
    before = len(calls)
    again, _ = extract(tmp_path, arm, pair)
    assert len(calls) == before and again["model_signature"] == result["model_signature"]


def test_incomplete_extraction_is_never_replayed(tmp_path):
    pair = model.make_models("patch4", 42, "cpu", TINY)

    def fail(*_):
        raise RuntimeError("synthetic interruption")

    hook = pair[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="interruption"):
        extract(tmp_path, pair=pair)
    hook.remove()
    with pytest.raises(ValueError, match="export-only"):
        extract(tmp_path, pair=pair)
    ledger = json.loads((tmp_path / "evaluation_attempts.json").read_text())
    assert ledger["patch4_s42/test"]["status"] == "failed_or_stopped"


def test_no_forward_without_complete_global_lock_and_binding_change(tmp_path):
    pair = model.make_models("patch4", 42, "cpu", TINY)
    calls = []
    pair[0].register_forward_pre_hook(lambda *_: calls.append(1))
    with pytest.raises(ValueError, match="matrix"):
        extract(tmp_path, pair=pair, verify=lambda *_: {"source": "only"})
    assert not calls
    extract(tmp_path, pair=pair)
    before = len(calls)
    with pytest.raises(ValueError, match="binding"):
        extract(tmp_path, pair=pair, verify=lambda *_: verifier() | {"source": "changed"})
    assert len(calls) == before
