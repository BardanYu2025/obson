"""CPU-only accounting and provenance tests; synthetic arrays, no real weights."""

import itertools
import json
import shutil
import sys

import numpy as np
import pytest

from obson.babel import state_value_audit as audit


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def seal(source, monkeypatch):
    """Pin a synthetic completion, never weaken production's reviewed hash."""
    files = {p.name: audit.sha256(p) for p in source.iterdir() if p.name != "completion.json"}
    write_json(
        source / "completion.json",
        {
            "files": files,
            "optimizer_updates": 0,
            "readout_fits": 0,
        },
    )
    monkeypatch.setattr(audit, "COMPLETION", audit.sha256(source / "completion.json"))


@pytest.fixture(scope="module")
def report_template(tmp_path_factory):
    source = tmp_path_factory.mktemp("value_source_template")
    segments = [
        {
            "id": f"segment_{i}",
            "key": f"X/15/CONTRACT{i}",
            "role": "coverage" if i < 12 else "stress",
        }
        for i in range(13)
    ]
    write_json(source / "segments.json", segments)
    write_json(source / "manifest.json", {"protocol": {"synthetic_test_only": True}})
    # All segments deliberately contain equal prices: identity, not a numerical
    # price discontinuity, must prevent cross-segment archive deduplication.
    raw = np.full((512, 128), 100.0, dtype=np.float64)
    times = np.arange(512).astype("timedelta64[m]") + np.datetime64("2026-01-01T09:00")
    for segment in segments:
        for seed, model in enumerate(audit.MODELS):
            name = f"{segment['id']}_{model}"
            tag = f"{name}_attempt1"
            arrays = {
                "raw_closes": raw,
                "embedding": np.full((512, 768), seed + 1, dtype=np.float32),
                "bar_times": times.astype(str),
                "online_seconds": np.linspace(0.001, 0.002, 512) * (seed + 1),
            }
            np.savez_compressed(source / f"{tag}.npz", **arrays)
            write_json(source / f"{tag}_runtime.json", {"device": "synthetic_gpu"})
            write_json(
                source / f"{name}.json",
                {
                    "segment": segment,
                    "model": model,
                    "attempt": 1,
                    "files": {f"{tag}.npz": audit.sha256(source / f"{tag}.npz")},
                },
            )
    return source


@pytest.fixture
def source(report_template, tmp_path, monkeypatch):
    result = tmp_path / "report"
    shutil.copytree(report_template, result)
    seal(result, monkeypatch)
    return result


def change_arrays(source, monkeypatch, model="macro_s42", **updates):
    tag = f"segment_0_{model}_attempt1"
    path = source / f"{tag}.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for name, transform in updates.items():
        arrays[name] = transform(arrays[name])
    np.savez_compressed(path, **arrays)
    receipt_path = source / f"segment_0_{model}.json"
    receipt = audit.read_json(receipt_path)
    receipt["files"][path.name] = audit.sha256(path)
    write_json(receipt_path, receipt)
    seal(source, monkeypatch)


@pytest.fixture
def fast_benchmark(monkeypatch):
    calls = []

    def fake(raw):
        calls.append(raw.copy())
        return [0.00001] * 96

    monkeypatch.setattr(audit, "benchmark_descriptors", fake)
    return calls


def test_complete_accounting_keeps_physical_segments_and_both_seed_timings(
    source,
    fast_benchmark,
):
    result = audit.audit(source)
    assert len(result["segments"]) == len(fast_benchmark) == 13
    assert result["totals"]["physical_endpoints"] == 13 * 512
    # Independent actual allocations, not the implementation's arithmetic formula.
    raw_streams = [np.full(639, 100.0, dtype=np.float64) for _ in range(13)]
    embeddings = [np.empty((512, 768), dtype=np.float32) for _ in range(13)]
    assert result["totals"]["raw_close_stream_bytes_separate_segments"] == sum(
        a.nbytes for a in raw_streams
    )
    assert result["totals"]["one_seed_state_bytes"] == sum(a.nbytes for a in embeddings)
    for seed, model in enumerate(audit.MODELS):
        measured = np.tile(np.linspace(0.001, 0.002, 512) * (seed + 1), 13)
        actual = result["imported_autodl_online_seconds"][model]
        assert actual == {
            "count": measured.size,
            "p50": np.median(measured),
            "p95": np.quantile(measured, 0.95),
        }
    assert result["direct_cpu_seconds"]["count"] == 13 * 96
    assert result["budget"]["measured_cpu_calls"] == 13 * 96
    assert result["budget"]["neural_forwards"] == 0
    assert result["decision"]["scientific_acceptance"] is False
    assert result["decision"]["promoted"] is False
    assert result["decision"]["next_gpu_jobs"] == []
    assert all(d["close_roundtrip_exact"] for d in result["segments"])
    assert result["imported_runtime_profiles"] == [{"device": "synthetic_gpu"}]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "target", ["completion.json", "manifest.json", "segment_0_macro_s42_attempt1.npz"]
)
def test_changed_bound_source_rejected_before_loading_arrays(source, monkeypatch, target):
    path = source / target
    path.write_bytes(path.read_bytes() + b" ")

    def no_load(*args, **kwargs):
        pytest.fail("Array loading must not precede completion/member verification")

    monkeypatch.setattr(audit.np, "load", no_load)
    with pytest.raises(ValueError, match="completion|fingerprint"):
        audit.audit(source)


def test_verify_regular_member_and_reject_tampering(tmp_path):
    path = tmp_path / "member.json"
    path.write_text("original")
    bound = {path.name: audit.sha256(path)}
    audit.verify_files(tmp_path, bound)
    path.write_text("modified")
    with pytest.raises(ValueError, match="fingerprint changed"):
        audit.verify_files(tmp_path, bound)


@pytest.mark.parametrize(
    "kind", ["absolute", "parent", "missing", "directory", "file_symlink", "directory_symlink"]
)
def test_verify_rejects_paths_and_symlinks_even_when_payload_hash_matches(tmp_path, kind):
    root = tmp_path / "source"
    root.mkdir()
    target = root / "actual"
    target.mkdir()
    member = target / "member.json"
    member.write_text("payload")
    digest = audit.sha256(member)
    if kind == "absolute":
        name = str(member)
    elif kind == "parent":
        name = "actual/../actual/member.json"
    elif kind == "missing":
        name = "missing.json"
    elif kind == "directory":
        name = "actual"
    elif kind == "file_symlink":
        (root / "alias.json").symlink_to(member)
        name = "alias.json"
    else:
        (root / "alias").symlink_to(target, target_is_directory=True)
        name = "alias/member.json"
    with pytest.raises(ValueError, match="Invalid report member"):
        audit.verify_files(root, {name: digest})


@pytest.mark.parametrize("n", [1, 7, 512])
def test_dedup_roundtrip_and_payloads_match_independent_array_allocations(n):
    prices = 100 + np.arange(n + 127, dtype=np.float64) / 17
    windows = np.lib.stride_tricks.sliding_window_view(prices, 128).copy()
    compact = audit.compact_closes(windows)
    assert not np.shares_memory(compact, windows)
    np.testing.assert_array_equal(compact, prices)
    np.testing.assert_array_equal(
        np.lib.stride_tricks.sliding_window_view(compact, 128),
        windows,
    )
    states = np.empty((n, 768), dtype=np.float32)
    anchors = windows[:, -1].copy()
    derived_rows = np.empty((len(prices), 28), dtype=np.float32)
    ledger = audit.payload_ledger(n)
    stream = ledger["consecutive_numeric_archive_bytes"]
    assert stream["close_float64"] == compact.nbytes
    assert stream["one_seed_dense_states_float32"] == states.nbytes
    assert stream["one_seed_states_plus_endpoint_anchors"] == states.nbytes + anchors.nbytes
    assert stream["derived_rows_float32"] == derived_rows.nbytes
    single = ledger["single_window_bytes"]
    assert single["close128_float64"] == windows[0].nbytes
    assert single["raw128x6_float32_reference_only"] == np.empty((128, 6), dtype=np.float32).nbytes
    assert single["derived128x28_float32"] == np.empty((128, 28), dtype=np.float32).nbytes
    assert single["close64_float64_for_fixed16_64_descriptors"] == windows[0, -64:].nbytes
    windows[0, 0] = -1
    assert compact[0] == prices[0]


@pytest.mark.parametrize("bad", [True, 0, -1, 1.0, np.int64(1)])
def test_payload_rejects_invalid_endpoint_count(bad):
    with pytest.raises(ValueError, match="Positive integer"):
        audit.payload_ledger(bad)


@pytest.mark.parametrize(
    "bad",
    [
        lambda a: a.astype(np.float32),
        lambda a: np.full_like(a, np.nan),
        lambda a: np.full_like(a, np.inf),
        lambda a: np.zeros_like(a),
        lambda a: -a,
        lambda a: a[:0],
        lambda a: a[:, :-1],
        lambda a: a[0],
        lambda a: a[[0, 2]],
    ],
)
def test_dedup_rejects_precision_loss_invalid_prices_and_gapped_windows(bad):
    prices = 100 + np.arange(132, dtype=np.float64)
    raw = np.lib.stride_tricks.sliding_window_view(prices, 128).copy()
    with pytest.raises(ValueError):
        audit.compact_closes(bad(raw))


@pytest.mark.parametrize("model", ["macro_s42", "macro_s43"])
def test_each_seed_must_preserve_original_float64_precision(
    source, monkeypatch, fast_benchmark, model
):
    change_arrays(source, monkeypatch, model, raw_closes=lambda a: a.astype(np.float32))
    with pytest.raises(ValueError, match="float64|raw windows|original"):
        audit.audit(source)


@pytest.mark.parametrize(
    "field,transform,error",
    [
        ("raw_closes", lambda a: np.full_like(a, np.nan), "finite"),
        ("raw_closes", lambda a: np.zeros_like(a), "Positive"),
        ("raw_closes", lambda a: np.where(np.indices(a.shape)[0] == 1, 101.0, a), "share127"),
        ("embedding", lambda a: a.astype(np.float64), "float32"),
        ("embedding", lambda a: np.full_like(a, np.nan), "archived output"),
        ("online_seconds", lambda a: -a, "nonnegative"),
        ("online_seconds", lambda a: np.full_like(a, np.nan), "nonnegative"),
        ("bar_times", lambda a: a[:-1], "archived output"),
    ],
)
def test_invalid_bound_array_semantics_are_rejected(
    source, monkeypatch, fast_benchmark, field, transform, error
):
    change_arrays(source, monkeypatch, **{field: transform})
    with pytest.raises(ValueError, match=error):
        audit.audit(source)


@pytest.mark.parametrize(
    "field,transform",
    [
        ("raw_closes", lambda a: a + 1),
        ("bar_times", lambda a: np.roll(a, 1)),
    ],
)
def test_seed_replicas_cannot_drift_to_different_physical_history(
    source, monkeypatch, fast_benchmark, field, transform
):
    change_arrays(source, monkeypatch, "macro_s43", **{field: transform})
    with pytest.raises(ValueError, match="identical physical history"):
        audit.audit(source)


def test_receipt_array_binding_is_not_replaced_by_completion_binding(
    source, monkeypatch, fast_benchmark
):
    path = source / "segment_0_macro_s42.json"
    receipt = audit.read_json(path)
    receipt["files"]["segment_0_macro_s42_attempt1.npz"] = "0" * 64
    write_json(path, receipt)
    seal(source, monkeypatch)
    with pytest.raises(ValueError, match="Receipt and completion disagree"):
        audit.audit(source)


def test_benchmark_calls_bound_truth_on_fixed_physical_windows_and_excludes_warmup(monkeypatch):
    prices = 100 * np.exp(np.arange(639) * 0.0001)
    raw = np.lib.stride_tricks.sliding_window_view(prices, 128).copy()
    calls = []
    real_truth = audit.truth.from_raw_closes

    def record(window):
        calls.append(window.copy())
        return real_truth(window)

    monkeypatch.setattr(audit.truth, "from_raw_closes", record)
    ticks = itertools.count(step=100)
    monkeypatch.setattr(audit.time, "perf_counter_ns", lambda: next(ticks))
    measured = audit.benchmark_descriptors(raw)
    assert measured == [100 / 1e9] * 96
    assert len(calls) == 97
    np.testing.assert_array_equal(calls[0], raw[:1])
    offsets = np.linspace(0, 511, 32, dtype=int)
    for actual, index in zip(calls[1:], np.tile(offsets, 3), strict=True):
        np.testing.assert_array_equal(actual, raw[index : index + 1])
        assert actual.dtype == np.float64


def test_direct_truth_uses_exact_flat_and_last16_last64_histories():
    flat = np.full((1, 128), 4382.5, dtype=np.float64)
    result = audit.truth.from_raw_closes(flat)
    np.testing.assert_array_equal(result["relative_log_close"], 0)
    np.testing.assert_array_equal(result["rms_log_return"], [[0, 0]])
    np.testing.assert_array_equal(result["direction"], [[1, 1]])
    # 48 falling returns followed by 15 rising returns: 16-close and
    # 64-close histories have opposite directions, independent of first64.
    changes = np.r_[np.zeros(64), np.full(48, -0.002), np.full(15, 0.001)]
    raw = (100 * np.exp(np.r_[0, np.cumsum(changes)]))[None, :]
    actual = audit.truth.from_raw_closes(raw)
    np.testing.assert_array_equal(actual["direction"], [[2, 0]])
    np.testing.assert_allclose(
        actual["rms_log_return"], [[0.001, np.sqrt((48 * 0.002**2 + 15 * 0.001**2) / 63)]]
    )
    altered_prefix = raw.copy()
    altered_prefix[:, :64] *= 2
    np.testing.assert_allclose(
        audit.truth.from_raw_closes(altered_prefix)["descriptors"],
        actual["descriptors"],
    )


@pytest.mark.parametrize("destination", ["inside", "existing", "symlink_inside"])
def test_cli_rejects_mutating_source_or_existing_evidence_before_audit(
    tmp_path, monkeypatch, destination
):
    source = tmp_path / "source"
    source.mkdir()
    out = source / "cost.json"
    if destination == "existing":
        out = tmp_path / "previous.json"
        out.write_text("retain this evidence")
    elif destination == "symlink_inside":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        out = alias / "cost.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--source", str(source), "--out", str(out)])
    monkeypatch.setattr(audit, "audit", lambda _: pytest.fail("Invalid output reached audit"))
    with pytest.raises(ValueError, match="new output file outside"):
        audit.main()
    if destination == "existing":
        assert out.read_text() == "retain this evidence"
    else:
        assert not out.exists()


def test_cli_exclusive_creation_preserves_concurrent_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "cost.json"

    def raced(_):
        out.write_text("another completed audit")
        return {"synthetic_test": True}

    monkeypatch.setattr(sys, "argv", ["audit", "--source", str(source), "--out", str(out)])
    monkeypatch.setattr(audit, "audit", raced)
    with pytest.raises(FileExistsError):
        audit.main()
    assert out.read_text() == "another completed audit"


def test_cli_writes_new_evidence_outside_source_without_source_mutation(
    source, monkeypatch, fast_benchmark, tmp_path
):
    before = {p.name: audit.sha256(p) for p in source.iterdir()}
    out = tmp_path / "evidence" / "cost.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--source", str(source), "--out", str(out)])
    audit.main()
    result = audit.read_json(out)
    assert result["source"]["completion_sha256"] == before["completion.json"]
    assert result["totals"]["physical_endpoints"] == 13 * 512
    assert before == {p.name: audit.sha256(p) for p in source.iterdir()}
