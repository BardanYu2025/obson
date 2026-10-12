"""Synthetic CPU evidence tests; no source checkpoints or real-weight execution."""

import copy
import json
import tarfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import multiscale_content as content
from obson.babel import multiscale_evaluate as evaluation
from obson.babel import multiscale_export as exporter
from obson.babel import multiscale_metrics as price
from obson.babel import state_coverage as coverage

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def cpu_runtime():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def identities(n):
    return [
        {"key": f"TEST/15/contract{i % 3}", "month": f"2025-{i % 6 + 1:02d}", "parent": i}
        for i in range(n)
    ]


def source(n=2):
    x = np.zeros((n, 255, 28), np.float32)
    x[..., 23:28] = 1
    x[..., 18:23] = 0.01
    closes = 100 * np.exp(np.random.default_rng(2).normal(0, 0.001, (n, 255)).cumsum(1))
    return x, closes


def labels(x, closes, positions):
    ps = torch.tensor(positions).expand(len(x), -1)
    target, mask = coverage.targets(torch.from_numpy(x), ps, STATS)
    target = content.replace_price_targets(target, torch.from_numpy(closes), ps, STATS)
    return target.numpy(), mask.numpy()


def bank(n=120):
    rows = identities(n)
    keys = ["full/price/composite", *evaluation.protection_keys()]
    support_keys = [f"price/{band}" for band in (*price.BANDS, "composite")]
    support_keys += [
        f"nonprice/{field}/{band}" for field in evaluation.FIELDS for band in price.BANDS
    ]
    return {
        "errors": {k: np.ones(n) for k in keys},
        "support": {k: np.ones(n, dtype=np.int64) for k in keys},
        "contexts": {k: np.ones(n, dtype=np.int64) for k in keys if not k.startswith("overlap/")},
        "position_support": {k: np.ones((n, 128), dtype=np.int64) for k in support_keys},
        "mask_digest": "synthetic-identical-masks",
        "rows_digest": evaluation._digest(rows),
        "data_digest": "synthetic-identical-feature-raw-statistics-lock",
    }, rows


def improved(reference, factor=0.8):
    candidate = copy.deepcopy(reference)
    candidate["errors"] = {k: v * factor for k, v in reference["errors"].items()}
    return candidate


def test_physical_price_units_and_each_nonprice_field_independent_numpy():
    x, closes = source()
    positions = [1, 64, 128]
    target, mask = labels(x, closes, positions)
    pred = target.copy()
    # Physical close error =0.2 log-percent =20 basis points, all ages.
    pred[..., 0] += 0.2 / (np.sqrt(np.arange(1, 128)) * STATS["delta_scale"][0])
    pred[..., 1:] += np.arange(1, 7)
    mask[0, 1, :16, 3] = False
    actual = evaluation.score_rows(pred, target, mask, closes, STATS, positions)
    for name in (*price.BANDS, "composite"):
        np.testing.assert_allclose(actual["errors"][f"price/{name}"], 0.04, atol=1e-14)
    assert actual["support"]["price/near"][0, 0] == 16
    assert actual["support"]["price/far"][0, 0] == 63
    for j, field in enumerate(evaluation.FIELDS):
        for band, (lo, hi) in price.BANDS.items():
            key = f"nonprice/{field}/{band}"
            for row in range(2):
                for p in range(3):
                    valid = mask[row, p, lo - 1 : hi, j + 1]
                    count = sum(bool(v) for v in valid)
                    assert actual["support"][key][row, p] == count
                    if count:
                        expected = (
                            sum(
                                (pred[row, p, a, j + 1] - target[row, p, a, j + 1]) ** 2
                                for a in range(lo - 1, hi)
                                if mask[row, p, a, j + 1]
                            )
                            / count
                        )
                        assert actual["errors"][key][row, p] == pytest.approx(expected)
                    else:
                        assert np.isnan(actual["errors"][key][row, p])


def test_aggregate_excludes_missing_states_and_keeps_endpoint_separate():
    values = np.full((2, 128), 4.0)
    counts = np.ones((2, 128), dtype=np.int64)
    values[0, :64], counts[0, :64] = np.nan, 0
    values[0, 127] = 20
    values[1, 127], counts[1, 127] = np.nan, 0
    actual = evaluation.aggregate_rows({"field": values}, {"field": counts})
    assert actual["errors"]["full/field"][0] == (63 * 4 + 20) / 64
    assert actual["errors"]["interior/field"][0] == 4
    assert actual["errors"]["endpoint/field"][0] == 20
    assert actual["contexts"]["full/field"][0] == 64
    assert np.isnan(actual["errors"]["endpoint/field"][1])
    assert actual["support"]["endpoint/field"][1] == 0


def test_agreeing_wrong_outputs_are_not_true_common_content_success():
    closes = np.full((2, 255), 100.0)
    pred = np.zeros((2, 8, 127, 7), np.float64)
    # The same sloped path has the same centered overlap shape under shifts,
    # while true constant closes contain no such slope.
    pred[..., 0] = -np.arange(1, 128) / (np.sqrt(np.arange(1, 128)) * 0.1)
    result = evaluation.overlap_rows(pred, closes, STATS)
    for shift in (1, 16, 64):
        prefix = f"overlap/shift{shift}/"
        np.testing.assert_allclose(result["errors"][prefix + "disagreement_mse"], 0, atol=1e-31)
        assert (result["errors"][prefix + "shape_a_mse"] > 0).all()
        assert (result["errors"][prefix + "shape_b_mse"] > 0).all()
        assert (result["support"][prefix + "shape_a_mse"] == 127 - shift).all()


def test_paired_bootstrap_matches_independent_cluster_draws():
    rows = identities(120)
    values = np.asarray([int(r["month"][-2:]) / 10 for r in rows])
    actual = evaluation.paired_intervals(values[:, None], rows)
    rng = np.random.default_rng(20261012)
    groups = sorted({r["month"] for r in rows})
    indices = rng.integers(0, len(groups), (2000, len(groups)))
    observations = {g: [values[i] for i, r in enumerate(rows) if r["month"] == g] for g in groups}
    draws = [
        np.mean([value for idx in sample for value in observations[groups[idx]]])
        for sample in indices
    ]
    expected = np.quantile(draws, [0.025, 0.975])
    assert actual["month"][0]["low"] == pytest.approx(expected[0])
    assert actual["month"][0]["high"] == pytest.approx(expected[1])
    assert actual["month"][0]["supported"]
    assert actual["month"][0]["valid_draws"] == 2000


def test_both_cohorts_seeds_required_for_limited_decision():
    control, rows = bank()
    candidate = improved(control)
    assert evaluation.comparison(candidate, control, rows)["passed"]
    models = {
        f"{arm}_s{seed}": candidate if arm == "multiscale" else control
        for arm in content.ARMS
        for seed in (42, 43)
    }
    result = evaluation.decisions(
        dict.fromkeys(("test", "cross_research"), models),
        dict.fromkeys(("test", "cross_research"), rows),
    )
    assert result["limited_hypothesis_supported"] and not result["promoted"]
    assert result["fit_count"] == 0 and len(result["comparisons"]) == 4
    with pytest.raises(ValueError, match="cohorts"):
        evaluation.decisions({"test": models}, {"test": rows})


@pytest.mark.parametrize(
    "damage",
    ["endpoint/price/near", "full/nonprice/oi_per_volume_asinh/far", "overlap/shift64/shape_b_mse"],
)
def test_total_gain_cannot_hide_endpoint_activity_or_common_content_damage(damage):
    control, rows = bank()
    candidate = improved(control)
    candidate["errors"][damage][:] = 1.3
    actual = evaluation.comparison(candidate, control, rows)
    assert actual["gain_passed"]
    assert not actual["protection_passed"] and not actual["passed"]
    assert actual["checks"]["retain/" + damage]["status"] == "fail"


def test_missing_class_field_support_is_insufficient_not_passed():
    control, rows = bank()
    key = "endpoint/nonprice/delta_log_volume/near"
    control["errors"][key][30:] = np.nan
    control["support"][key][30:] = 0
    control["contexts"][key][30:] = 0
    candidate = improved(control)
    result = evaluation.comparison(candidate, control, rows)
    check = result["checks"]["retain/" + key]
    assert check["status"] == "insufficient" and not result["passed"]
    assert check["intervals"]["month"]["parents"] == 30
    assert not evaluation.paired_intervals(np.ones((99, 1)), identities(99))["month"][0][
        "supported"
    ]
    few_groups = [r | {"month": "2025-01"} for r in rows]
    assert not evaluation.paired_intervals(np.ones((120, 1)), few_groups)["month"][0]["supported"]


@pytest.mark.parametrize(
    "changed", ["mask_digest", "rows_digest", "data_digest", "support", "position_support"]
)
def test_exact_support_and_population_mismatch_rejected(changed):
    control, rows = bank()
    candidate = improved(control)
    if changed.endswith("digest"):
        candidate[changed] = "different"
    else:
        candidate[changed][next(iter(candidate[changed]))].flat[0] += 1
    with pytest.raises(ValueError, match="mismatch|differs"):
        evaluation.comparison(candidate, control, rows)


def test_descriptive_p95_guard_cannot_hide_behind_improved_average():
    control, rows = bank()
    candidate = improved(control, 0.1)
    candidate["errors"]["full/price/composite"][:12] = 2
    result = evaluation.comparison(candidate, control, rows)
    assert result["gain_passed"]
    assert not result["tail"]["passed"] and not result["passed"]
    assert "not a confidence" in result["tail"]["scope"]


def test_already_perfect_control_cannot_claim_five_percent_gain():
    control, rows = bank()
    for values in control["errors"].values():
        values[:] = 0
    result = evaluation.comparison(copy.deepcopy(control), control, rows)
    assert result["checks"]["gain"]["status"] == "insufficient"
    assert not result["checks"]["gain"]["gain_denominator_defined"]
    assert result["protection_passed"] and not result["passed"]


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(28, 4)
        self.calls = []

    def forward(self, x):
        self.calls.append((tuple(x.shape), self.training, torch.is_grad_enabled()))
        return self.layer(x.cumsum(1) / torch.arange(1, 129, device=x.device)[None, :, None])


class TinyQuery(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 127 * 7)

    def forward(self, x):
        return self.layer(x).reshape(len(x), 127, 7)


def meta():
    return {"statistics": STATS, "scales": [1.0] * 8, "batch": 128, "micro": 8, "epochs": 120}


def verifier(*_):
    return {"model_lock": "fixed-full-four-model-matrix", "source": "fixed-data-lineage"}


def extract(tmp_path, *, encoder=None, query=None, callback=verifier):
    x, closes = source()
    encoder = TinyEncoder() if encoder is None else encoder
    query = TinyQuery() if query is None else query
    result = evaluation.extract(
        meta(),
        tmp_path,
        "control_s42",
        encoder,
        query,
        x,
        closes,
        identities(2),
        "test",
        "cpu",
        verify_lock=callback,
    )
    return result, encoder, query


def test_synthetic_extraction_all128_fixed8_shards_no_embeddings_and_resume(tmp_path):
    result, encoder, query = extract(tmp_path)
    assert len(encoder.calls) == 26
    assert sum(shape[0] for shape, _, _ in encoder.calls) == 256
    assert all(shape[0] <= 8 * 5 and not train and not grad for shape, train, grad in encoder.calls)
    assert result["errors"]["full/price/composite"].shape == (2,)
    path = tmp_path / "results/test_control_s42_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"predictions", "targets", "masks", "raw_closes", "positions"}
        np.testing.assert_array_equal(saved["positions"], evaluation.SAMPLE_POSITIONS)
        assert saved["predictions"].dtype == np.float32
        assert saved["targets"].dtype == saved["raw_closes"].dtype == np.float64
        assert saved["predictions"].shape == (2, 8, 127, 7)
        subset = evaluation.score_rows(
            saved["predictions"],
            saved["targets"],
            saved["masks"],
            saved["raw_closes"],
            STATS,
            saved["positions"],
        )
    with np.load(tmp_path / "evaluation/control_s42/test/position_rows.npz") as allrows:
        for key, values in subset["errors"].items():
            np.testing.assert_allclose(
                allrows["errors::" + key][:, np.asarray(evaluation.SAMPLE_POSITIONS) - 1],
                values,
                equal_nan=True,
            )
    shard = np.load(tmp_path / "predictions/control_s42/test/parents_00000_00002.npy")
    assert shard.shape == (2, 128, 127, 7) and shard.dtype == np.float32
    np.testing.assert_array_equal(
        shard[:, np.asarray(evaluation.SAMPLE_POSITIONS) - 1], np.load(path)["predictions"]
    )
    receipt = json.loads((tmp_path / result["receipt"]).read_text())
    assert receipt["states"] == 256 and receipt["updates"] == receipt["fits"] == 0
    assert receipt["embeddings_saved"] is False
    assert all(name.startswith("predictions/") for name in receipt["omitted_from_download"])
    before = len(encoder.calls)
    again, _, _ = extract(tmp_path, encoder=encoder, query=query)
    assert len(encoder.calls) == before
    np.testing.assert_array_equal(
        again["errors"]["full/price/composite"], result["errors"]["full/price/composite"]
    )
    json.dumps(evaluation.summarize(result), allow_nan=False)


def test_completed_receipt_repairs_interrupted_ledger_without_forwards(tmp_path):
    _, encoder, query = extract(tmp_path)
    path = tmp_path / "evaluation_attempts.json"
    ledger = json.loads(path.read_text())
    ledger["control_s42/test"]["status"] = "started"
    path.write_text(json.dumps(ledger))
    calls = len(encoder.calls)
    extract(tmp_path, encoder=encoder, query=query)
    assert len(encoder.calls) == calls
    repaired = json.loads(path.read_text())["control_s42/test"]
    assert repaired["status"] == "complete" and repaired["reconciled_from_verified_receipt"]
    assert repaired["previous_attempt"]["status"] == "started"


def test_interrupted_extraction_is_export_only_and_never_replayed(tmp_path):
    encoder, query = TinyEncoder(), TinyQuery()
    with (
        patch.object(coverage, "predict", side_effect=RuntimeError("synthetic interruption")),
        pytest.raises(RuntimeError),
    ):
        extract(tmp_path, encoder=encoder, query=query)
    attempt = json.loads((tmp_path / "evaluation_attempts.json").read_text())["control_s42/test"]
    assert attempt["status"] == "failed_or_stopped" and attempt["reserved_states"] == 256
    with pytest.raises(ValueError, match="export-only"):
        extract(tmp_path, encoder=encoder, query=query)
    assert not encoder.calls


def test_no_research_forward_before_global_lock_and_changed_source_rejected(tmp_path):
    encoder, query = TinyEncoder(), TinyQuery()
    with pytest.raises(ValueError, match="matrix"):
        extract(tmp_path, encoder=encoder, query=query, callback=lambda *_: {"source": "only"})
    assert not encoder.calls and not (tmp_path / "evaluation_attempts.json").exists()
    extract(tmp_path, encoder=encoder, query=query)
    before = len(encoder.calls)
    with pytest.raises(ValueError, match="binding"):
        extract(
            tmp_path,
            encoder=encoder,
            query=query,
            callback=lambda *_: {"source": "changed", "model_lock": "fixed-full-four-model-matrix"},
        )
    assert len(encoder.calls) == before


def test_full_remote_shard_hash_and_unsafe_output_protected(tmp_path):
    result, encoder, query = extract(tmp_path)
    shard = tmp_path / "predictions/control_s42/test/parents_00000_00002.npy"
    with shard.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        extract(tmp_path, encoder=encoder, query=query)
    other = tmp_path / "new"
    other.mkdir()
    (other / "evaluation").symlink_to(tmp_path / "evaluation", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        extract(other)
    assert result["receipt"].endswith("complete.json")


def test_malformed_coordinate_masks_rows_and_support_rejected():
    x, closes = source()
    y, m = labels(x, closes, [1, 128])
    altered = y.copy()
    altered[..., 0] += 1
    with pytest.raises(ValueError, match="target differs"):
        evaluation.score_rows(y, altered, m, closes, STATS, [1, 128])
    m[0, 0, 0, 0] = False
    with pytest.raises(ValueError, match="all prices"):
        evaluation.score_rows(y, y, m, closes, STATS, [1, 128])
    with pytest.raises(ValueError, match="positions"):
        evaluation.score_rows(y, y, m, closes, STATS, [128, 1])
    with pytest.raises(ValueError, match="month"):
        evaluation.paired_intervals(np.ones((1, 1)), [{"key": "x", "month": "bad"}])
    with pytest.raises(ValueError, match="support"):
        evaluation.aggregate_rows({"a": np.ones((2, 128))}, {"a": np.zeros((2, 128), int)})


def test_cpu_four_model_two_cohort_pipeline_decisions_and_export(tmp_path):
    """Actual tiny neural extraction and report packaging, not mocked scores.

    This synthetic integration does NOT simulate120 epochs or source-lineage
    acceptance. The callback supplies a clearly synthetic four-model lock.
    With fewer than100 parents the scientific decision must stay insufficient.
    """
    out = tmp_path / "synthetic_multiscale"
    out.mkdir()
    pairs = {
        f"{arm}_s{seed}": (TinyEncoder(), TinyQuery()) for arm in content.ARMS for seed in (42, 43)
    }
    calls = []

    def locked(meta, path, name, split):
        assert path == out and name in pairs and len(pairs) == 4
        calls.append((name, split))
        return {
            "model_lock": "synthetic-four-models-not-trained",
            "source": "synthetic-only",
            "names": sorted(pairs),
        }

    banks, row_bank = {}, {}
    for split, n in (("test", 3), ("cross_research", 2)):
        x, closes = source(n)
        rows = identities(n)
        row_bank[split] = rows
        banks[split] = {
            name: evaluation.extract(
                meta(), out, name, *pair, x, closes, rows, split, "cpu", verify_lock=locked
            )
            for name, pair in pairs.items()
        }
    assert len(calls) == 16  # Before forward and before each receipt commit.
    decision = evaluation.decisions(banks, row_bank)
    assert not decision["limited_hypothesis_supported"] and not decision["promoted"]
    assert all(
        v["checks"]["gain"]["status"] == "insufficient" for v in decision["comparisons"].values()
    )
    (out / "multiscale_metrics.json").write_text(json.dumps(decision, allow_nan=False))
    (out / "status.json").write_text(json.dumps({"status": "synthetic_component_complete"}))
    result = exporter.export_reports(out, tmp_path / "download")
    with tarfile.open(result["archive"], "r:gz") as archive:
        names = archive.getnames()
        assert not any(name.endswith((".pt", ".npy")) for name in names)
        assert sum(name.endswith("_predictions.npz") for name in names) == 8
        assert sum(name.endswith("position_rows.npz") for name in names) == 8
        manifest = json.load(archive.extractfile(out.name + "/export_manifest.json"))
        full_shards = [
            item for item in manifest["omitted"] if item["reason"] == "full128_prediction_shard"
        ]
        assert len(full_shards) == 8
        assert all(item["sha256"] and item["bytes"] > 0 for item in full_shards)
