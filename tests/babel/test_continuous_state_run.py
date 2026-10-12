"""F11 runner accounting and fail-closed evidence using synthetic CPU readers only."""

import copy
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from obson.babel import closed_bar_state as stream
from obson.babel import continuous_state_run as run
from obson.babel.data import validate_frame

KEY = "X/15/EX.X123"
ORIGIN = "synthetic-full-contract-prefix"
MODEL = "macro_s42"
STATS = {"x_mean": [0.0] * 28, "x_scale": [1.0] * 28}


def fake_outputs(windows):
    """A deterministic row-wise function, with no neural parameters or fitting."""
    windows = np.asarray(windows, np.float64)
    embedding = np.column_stack((windows.mean((1, 2)), windows[:, -1, 1], windows[:, :, 1].mean(1)))
    path = (windows[:, :, 1] - windows[:, -1:, 1]) * 0.001
    return {
        "embedding": embedding,
        "query": windows[:, :-1, :7].copy(),
        "relative_log_close": path,
    }


class FakeReader:
    def __init__(self):
        self.statistics = copy.deepcopy(STATS)
        self.input_signature = stream.PreparedBarStream(KEY, 15, STATS, ORIGIN).input_signature
        self.calls = []
        self.batch_fault = None
        self.nonfinite_field = None

    def read_prepared(self, windows, endpoints, *, input_signature, audit_observed):
        assert input_signature == self.input_signature
        assert audit_observed is False
        assert len(windows) == len(endpoints)
        self.calls.append(copy.deepcopy(endpoints))
        outputs = fake_outputs(windows)
        if len(windows) > 1 and self.batch_fault is not None:
            outputs[self.batch_fault][:, 0] += 1.0
        if self.nonfinite_field is not None:
            outputs[self.nonfinite_field].flat[0] = np.nan
        return outputs


@pytest.fixture
def case(tmp_path):
    rng = np.random.default_rng(51)
    n = 700
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
    opening = np.r_[100.0, close[:-1]]
    times = pd.date_range("2026-01-01 09:00", periods=n, freq="15min").to_numpy(copy=True)
    times[260:] += np.timedelta64(2, "D")
    frame = validate_frame(
        pd.DataFrame(
            {
                "datetime": times,
                "open": opening,
                "high": np.maximum(opening, close) * 1.002,
                "low": np.minimum(opening, close) * 0.998,
                "close": close,
                "volume": rng.integers(0, 100, n).astype(float),
                "close_oi": rng.integers(900, 1100, n).astype(float),
            }
        )
    )
    series = SimpleNamespace(key=KEY, code="X", period=15, frame=frame)
    first, source_offset = 160, 255
    segment = {
        "id": "synthetic",
        "key": KEY,
        "split": "test",
        "first": first,
        "last": first + run.LENGTH - 1,
        "source_row": first + source_offset,
        "source_offset": source_offset,
        "source_index": 0,
    }
    source = segment["source_row"]
    # Independent vectorized feature formula supplies the original-source cache.
    vector = run.data.original.original.encode(frame.iloc[: source + 1], 15)
    cached = vector[source - 127 : source + 1].copy()
    expected = {
        key: value
        for key, value in fake_outputs(cached[None]).items()
        if key in ("embedding", "query")
    }
    binding = {"source": "synthetic-source", "segment": segment, "model": MODEL}
    return {
        "reader": FakeReader(),
        "s": series,
        "segment": segment,
        "stats": STATS,
        "cached_window": cached,
        "expected": expected,
        "origin": ORIGIN,
        "out": tmp_path,
        "model": MODEL,
        "binding": binding,
        "deadline": time.monotonic() + 100000,
        "remaining_endpoints": run.PER_SEGMENT,
    }


def read(path):
    return json.loads(path.read_text())


def name(case):
    return f"{case['segment']['id']}_{MODEL}"


def test_full_segment_closed_asof_snapshot_and_exact_forward_budget(case, monkeypatch):
    original_push = stream.PreparedBarStream.push
    observed = []

    def checked_push(self, bar, asof):
        assert pd.Timestamp(asof) == pd.Timestamp(bar.datetime) + pd.Timedelta(minutes=15)
        observed.append(pd.Timestamp(bar.datetime))
        return original_push(self, bar, asof)

    monkeypatch.setattr(stream.PreparedBarStream, "push", checked_push)
    receipt = run.run_segment(**case)
    reader, segment, out = case["reader"], case["segment"], case["out"]
    assert len(reader.calls) == 512 + 2
    assert [len(batch) for batch in reader.calls] == [1] * 512 + [8, 8]
    assert sum(map(len, reader.calls)) == 528
    assert len(observed) == segment["last"] + 1
    assert observed[-1] == case["s"].frame.datetime.iloc[segment["last"]]
    assert receipt["online_endpoints"] == 512 and receipt["oracle_endpoints"] == 16
    runtime_name = f"{name(case)}_attempt1_runtime.json"
    assert runtime_name in receipt["files"]
    assert read(out / runtime_name)["kind"] == "synthetic_injected_reader"
    assert read(out / "forward_attempts.json")[0]["reserved_endpoints"] == 528
    checkpoint = read(out / f"{name(case)}_attempt1_snapshot.json")["payload"]
    assert checkpoint["count"] == segment["first"] + 256
    assert checkpoint["last_time"] == str(case["s"].frame.datetime.iloc[segment["first"] + 255])
    assert reader.calls[256][0]["row"] == segment["first"] + 256
    assert reader.calls[256][0]["feature_first_bar"] == str(case["s"].frame.datetime.iloc[0])
    assert all(
        value["passed"] for value in read(out / f"{name(case)}_attempt1_numeric.json").values()
    )
    with np.load(out / f"{name(case)}_attempt1.npz", allow_pickle=False) as bank:
        assert bank["embedding"].shape == (512, 3)
        assert bank["query"].shape == (512, 127, 7)
        assert bank["relative_log_close"].shape == (512, 128)
        assert np.all(bank["relative_log_close"][:, -1] == 0)
        expected_raw = case["s"].frame.close.to_numpy(np.float64)
        expected_raw = np.lib.stride_tricks.sliding_window_view(expected_raw, 128)
        np.testing.assert_array_equal(
            bank["raw_closes"], expected_raw[segment["first"] - 127 : segment["last"] - 126]
        )
        for field in ("embedding", "query", "relative_log_close"):
            np.testing.assert_allclose(
                bank[field][list(run.OFFSETS)], bank[f"oracle/{field}"], atol=1e-6, rtol=2e-5
            )
    json.dumps(receipt, allow_nan=False)


def test_raw_preparation_does_not_read_beyond_last_closed_bar(case):
    def prepare(series):
        return run.raw_preparation(series, case["segment"], STATS, case["cached_window"], ORIGIN)

    before = prepare(case["s"])
    altered = copy.deepcopy(case["s"])
    future = altered.frame.index > case["segment"]["last"]
    altered.frame.loc[future, ["open", "high", "low", "close", "volume", "oi"]] *= 100
    after = prepare(altered)
    assert len(before[0]) == case["segment"]["last"] + 1
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]
    np.testing.assert_array_equal(before[3], after[3])
    assert before[4:] == after[4:]


def test_committed_resume_has_zero_new_calls_and_rejects_changed_binding(case):
    first = run.run_segment(**case)
    calls = len(case["reader"].calls)
    # The deadline can already be expired: a verified committed result is reused.
    case["deadline"] = -1
    assert run.run_segment(**case) == first
    assert len(case["reader"].calls) == calls
    assert len(read(case["out"] / "forward_attempts.json")) == 1
    case["binding"] = {**case["binding"], "source": "different"}
    with pytest.raises(ValueError, match="Committed segment binding changed"):
        run.run_segment(**case)
    assert len(case["reader"].calls) == calls


def test_committed_resume_rejects_changed_output_before_model_call(case):
    run.run_segment(**case)
    artifact = case["out"] / f"{name(case)}_attempt1_numeric.json"
    artifact.write_text("{}\n")
    calls = len(case["reader"].calls)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run.run_segment(**case)
    assert len(case["reader"].calls) == calls


def test_attempts_are_reserved_before_work_and_third_attempt_is_refused(tmp_path):
    binding = {"source": "one"}
    assert run.reserve(tmp_path, "segment_model", binding) == 1
    assert run.reserve(tmp_path, "segment_model", binding) == 2
    entries = read(tmp_path / "forward_attempts.json")
    assert [entry["attempt"] for entry in entries] == [1, 2]
    assert sum(entry["reserved_endpoints"] for entry in entries) == 1056
    with pytest.raises(ValueError, match="forward-attempt budget exhausted"):
        run.reserve(tmp_path, "segment_model", binding)
    assert read(tmp_path / "forward_attempts.json") == entries
    with pytest.raises(ValueError, match="Attempt binding changed"):
        run.reserve(tmp_path, "segment_model", {"source": "two"})
    assert read(tmp_path / "forward_attempts.json") == entries


def test_global_forward_budget_cannot_be_bypassed_with_a_different_segment(tmp_path):
    entries = [
        {
            "name": f"previous_{i}",
            "binding": {},
            "attempt": 1,
            "reserved_endpoints": run.PER_SEGMENT,
            "started_unix": 0,
        }
        for i in range(run.NOMINAL * run.MAX_ATTEMPTS // run.PER_SEGMENT)
    ]
    run.atomic_json(entries, tmp_path / "forward_attempts.json")
    with pytest.raises(ValueError, match="forward-attempt budget exhausted"):
        run.reserve(tmp_path, "new_segment", {})
    assert read(tmp_path / "forward_attempts.json") == entries


def test_wrong_original_cache_stops_before_forward_and_preserves_attempt_diagnostic(case):
    case["cached_window"] = case["cached_window"] + 1.0
    with pytest.raises(ValueError, match="numeric evidence saved"):
        run.run_segment(**case)
    assert not case["reader"].calls
    assert read(case["out"] / "forward_attempts.json")[0]["reserved_endpoints"] == 528
    evidence = read(case["out"] / f"{name(case)}_attempt1_input.json")
    assert evidence["original_cached_input"]["passed"] is False
    assert not (case["out"] / f"{name(case)}.json").exists()
    with pytest.raises(ValueError, match="numeric evidence saved"):
        run.run_segment(**case)
    assert read(case["out"] / f"{name(case)}_attempt1_input.json") == evidence
    assert read(case["out"] / f"{name(case)}_attempt2_input.json") == evidence
    with pytest.raises(ValueError, match="forward-attempt budget exhausted"):
        run.run_segment(**case)
    assert not case["reader"].calls


def test_wrong_original_source_output_saves_numeric_before_rejecting_commit(case):
    case["expected"]["query"] += 1.0
    with pytest.raises(ValueError, match="original_source/query"):
        run.run_segment(**case)
    evidence = read(case["out"] / f"{name(case)}_attempt1_numeric.json")
    assert evidence["original_source/query"]["passed"] is False
    assert evidence["batch_oracle/query"]["passed"] is True
    assert sum(map(len, case["reader"].calls)) == 528
    assert not (case["out"] / f"{name(case)}.json").exists()


@pytest.mark.parametrize("field", ["embedding", "query", "relative_log_close"])
def test_batch_oracle_discrepancy_saves_evidence_and_cannot_commit(case, field):
    case["reader"].batch_fault = field
    with pytest.raises(ValueError, match=f"batch_oracle/{field}"):
        run.run_segment(**case)
    evidence = read(case["out"] / f"{name(case)}_attempt1_numeric.json")
    assert evidence[f"batch_oracle/{field}"]["passed"] is False
    assert not (case["out"] / f"{name(case)}.json").exists()


def test_nonfinite_online_output_saves_numeric_before_any_further_calls(case):
    case["reader"].nonfinite_field = "relative_log_close"
    with pytest.raises(ValueError, match="relative_log_close"):
        run.run_segment(**case)
    evidence = read(case["out"] / f"{name(case)}_attempt1_numeric.json")
    assert evidence["relative_log_close"] == {"passed": False, "reason": "nonfinite"}
    assert len(case["reader"].calls) == 1
    assert read(case["out"] / "forward_attempts.json")[0]["reserved_endpoints"] == 528
    assert not (case["out"] / f"{name(case)}.json").exists()


def test_projection_stops_after_32_reused_online_outputs_without_oracle(case, monkeypatch):
    ticks = iter(range(100000))
    monkeypatch.setattr(
        run, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)), time=lambda: 0.0)
    )
    case["deadline"] = 1000
    case["remaining_endpoints"] = 10000
    with pytest.raises(run.ResourceLimitError, match="Measured projection exceeds"):
        run.run_segment(**case)
    assert len(case["reader"].calls) == 32
    assert all(len(batch) == 1 for batch in case["reader"].calls)
    evidence = read(case["out"] / f"{name(case)}_attempt1_resource.json")
    assert evidence["observed_endpoints"] == 32 and evidence["passed"] is False
    assert evidence["projected_remaining_seconds_including_margin"] > evidence["remaining_seconds"]
    assert read(case["out"] / "forward_attempts.json")[0]["reserved_endpoints"] == 528
    assert not (case["out"] / f"{name(case)}.json").exists()


def test_expired_session_stops_before_first_online_forward(case):
    case["deadline"] = -1
    with pytest.raises(TimeoutError, match="session time exhausted"):
        run.run_segment(**case)
    assert not case["reader"].calls
    assert read(case["out"] / "forward_attempts.json")[0]["reserved_endpoints"] == 528


def test_changed_model_signature_saves_diagnostic_and_never_commits(case):
    case["check_frozen"] = lambda: False
    with pytest.raises(ValueError, match="frozen_tensors"):
        run.run_segment(**case)
    evidence = read(case["out"] / f"{name(case)}_attempt1_numeric.json")
    assert evidence["frozen_tensors"]["passed"] is False
    assert sum(map(len, case["reader"].calls)) == 528
    assert not (case["out"] / f"{name(case)}.json").exists()


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    """Run real orchestration/I/O, substituting all CUDA and neural boundaries."""
    source, raw_root, out, coverage = [
        tmp_path / name for name in ("source", "raw", "out", "coverage")
    ]
    for path in (source, raw_root, out, coverage):
        path.mkdir()
    run.atomic_json({"synthetic": True}, source / "completion.json")
    monkeypatch.setattr(run.parent, "SOURCE_COMPLETION", run.sha256(source / "completion.json"))
    meta = {"statistics": copy.deepcopy(STATS), "source_meta": {"synthetic": True}}
    state = SimpleNamespace(
        source=source,
        raw_root=raw_root,
        out=out,
        coverage=coverage,
        meta=meta,
        rows={},
        arrays={},
        loaded={},
        selected=[],
        inventory={"files": {}},
        model_loads=[],
        readers=[],
        calls=[],
        verifies=[],
        masks=[],
        empty_calls=[],
        on_segment=lambda _: None,
        after_model=lambda _: None,
    )
    for split, size in (("test", 7), ("cross_research", 6)):
        state.rows[split], state.loaded[split] = [], []
        for index in range(size):
            symbol = f"{split[0].upper()}{index}"
            period = (15, 30, 60)[index % 3]
            key = f"{symbol}/{period}/EX.{symbol}"
            timestamps = pd.date_range("2026-01-01 09:00", periods=2, freq=f"{period}min")
            s = SimpleNamespace(
                key=key, code=symbol, period=period, frame=pd.DataFrame({"datetime": timestamps})
            )
            state.loaded[split].append(s)
            row = {"key": key, "row": 200 + index, "end": str(timestamps[-1])}
            state.rows[split].append(row)
            state.selected.append(
                {
                    "id": f"{split}_{index}",
                    "split": split,
                    "key": key,
                    "source_index": index,
                    "source_row": row["row"],
                    "source_offset": 40 + index,
                    "first": 160,
                    "last": 671,
                }
            )
            path = raw_root / symbol / f"EX.{symbol}_{period}m.csv"
            path.parent.mkdir()
            path.write_text(f"synthetic raw identity for {key}\n")
            state.inventory["files"][key] = run.sha256(path)
        run.atomic_json(state.rows[split], source / f"{split}_rows.json")
        windows = np.full((size, 255, 28), -999, dtype=np.float32)
        offset = 100 if split == "cross_research" else 0
        for index in range(size):
            windows[index, -128:] = offset + index
        state.arrays[split] = windows
        for model in run.MODELS:
            model_offset = 1000 if model == "macro_s43" else 0
            values = np.arange(size) + offset + model_offset
            np.savez(
                source / f"{split}_{model}.npz",
                embedding=np.repeat(values[:, None], 3, axis=1),
                query=np.broadcast_to(values[:, None, None], (size, 2, 7)).copy(),
            )

    def verified(requested_source, requested_out):
        assert requested_source == source and requested_out == out
        if run.sha256(source / "completion.json") != run.parent.SOURCE_COMPLETION:
            raise ValueError("Synthetic original source completion changed")
        state.verifies.append((requested_source, requested_out))
        return coverage, meta

    def cache(given_meta, given_coverage, split):
        assert given_meta == meta and given_coverage == coverage
        return state.arrays[split], copy.deepcopy(state.rows[split])

    def raw_loader(tm, cross):
        assert tm["raw_root"] == str(raw_root)
        split = "cross_research" if cross else "test"
        return state.loaded[split], f"bounds-{split}"

    def mask(series, bounds, kind):
        assert kind == "test"
        state.masks.append((series.key, bounds))
        return np.ones(len(series.frame), dtype=bool)

    def choose(series, rows, masks):
        for split in run.SPLITS:
            assert set(series[split]) == {s.key for s in state.loaded[split]}
            assert rows[split] == state.rows[split]
            assert set(masks[split]) == set(series[split])
        return copy.deepcopy(state.selected)

    def model_loader(given_meta, given_coverage, model, device):
        assert given_meta == meta and given_coverage == coverage and device == "cuda"
        state.model_loads.append(model)
        return SimpleNamespace(model=model), SimpleNamespace(model=model)

    def reader_factory(encoder, query, statistics, identity, device):
        assert encoder.model == query.model == identity["model"]
        assert identity["width"] == 768 and statistics == STATS and device == "cuda"
        state.readers.append(identity["model"])
        return SimpleNamespace(model=encoder.model)

    def segment_runner(
        reader,
        s,
        segment,
        statistics,
        cached_window,
        expected,
        origin,
        given_out,
        model,
        binding,
        deadline,
        remaining,
        sync,
        runtime,
        check_frozen,
    ):
        assert reader.model == model and s.key == segment["key"]
        assert given_out == out and statistics == STATS
        index, split = segment["source_index"], segment["split"]
        value = index + (100 if split == "cross_research" else 0)
        assert cached_window.shape == (128, 28) and np.all(cached_window == value)
        value += 1000 if model == "macro_s43" else 0
        assert expected["embedding"].shape == (1, 3)
        assert expected["query"].shape == (1, 2, 7)
        assert np.all(expected["embedding"] == value) and np.all(expected["query"] == value)
        assert binding["raw_sha256"] == state.inventory["files"][s.key]
        assert binding["manifest_sha256"] == run.sha256(out / "manifest.json")
        assert origin.startswith(f"raw:{binding['raw_sha256']};first:")
        assert origin.endswith(";complete_prefix") and runtime["runtime"]["synthetic"]
        assert deadline > time.monotonic()
        assert check_frozen()
        assert remaining == (26 - len(state.calls)) * 528
        state.calls.append((segment["id"], model, index, split))
        name = f"{segment['id']}_{model}"
        attempt = run.reserve(out, name, binding)
        proof = out / f"{name}_attempt{attempt}_synthetic.json"
        run.atomic_json({"synthetic": True, "expected_source_row": index}, proof)
        receipt = {
            "binding": binding,
            "segment": segment,
            "model": model,
            "attempt": attempt,
            "online_endpoints": 512,
            "oracle_endpoints": 16,
            "metrics": {"synthetic": True},
            "files": {proof.name: run.sha256(proof)},
        }
        run.atomic_json(receipt, out / f"{name}.json")
        state.on_segment(segment)
        return receipt

    def emptied():
        state.empty_calls.append(len(state.model_loads))
        state.after_model(len(state.model_loads))

    monkeypatch.setattr(run.parent, "verify_source", verified)
    monkeypatch.setattr(run.parent, "inventory", lambda: copy.deepcopy(state.inventory))
    monkeypatch.setattr(run.data, "cache", cache)
    monkeypatch.setattr(run.data.source.old.parent, "tm", lambda _: {"synthetic": True})
    monkeypatch.setattr(run.data.source.old.parent.xr.xd, "load_raw", raw_loader)
    monkeypatch.setattr(run.data.original.rp, "time_mask", mask)
    monkeypatch.setattr(run.selection, "select_segments", choose)
    monkeypatch.setattr(run, "SELECTION_SHA256", run.api.token(state.selected))
    monkeypatch.setattr(run.prior.ev, "load", model_loader)
    monkeypatch.setattr(run.prior.source_run, "signature", lambda e, q: (e.model, q.model))
    monkeypatch.setattr(run.api, "HistoryStateReader", reader_factory)
    monkeypatch.setattr(
        run.prior.r0, "replay_runtime", lambda mode: nullcontext({"synthetic": True})
    )
    monkeypatch.setattr(run, "run_segment", segment_runner)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    cuda = run.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "get_device_name", lambda: "synthetic CUDA boundary; no GPU used")
    monkeypatch.setattr(
        cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=12 * 1024**3)
    )
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(cuda, "empty_cache", emptied)
    run.atomic_json(
        {
            "source": str(source),
            "raw_root": str(raw_root),
            "output": str(out),
            "implementation": run.launch.implementation(),
        },
        out / "request.json",
    )
    state.execute = lambda: run.execute(source, raw_root, out, 7200)
    return state


def test_execute_complete_26_pairs_preserves_source_mapping_and_binds_ledger(orchestration):
    state = orchestration
    state.execute()
    assert state.model_loads == state.readers == list(run.MODELS)
    assert state.calls == [
        (segment["id"], model, segment["source_index"], segment["split"])
        for model in run.MODELS
        for segment in state.selected
    ]
    assert len(state.masks) == 13 and state.empty_calls == [1, 2]
    assert len(state.verifies) == 3  # Preflight, final source check, completion recheck.
    done = run.check_complete(state.out)
    assert done["completed_segment_models"] == 26
    assert done["planned_endpoint_forwards"] == done["reserved_endpoint_forwards"] == 13728
    assert "forward_attempts.json" in done["files"]
    assert done["optimizer_updates"] == done["readout_fits"] == 0
    assert done["quality_assured"] is False and done["promoted"] is False
    assert len(read(state.out / "continuous_report.json")["reports"]) == 26
    assert len(read(state.out / "raw_identity.json")) == 13
    # Supervisor accounting changes after a worker commit are intentionally mutable.
    for filename in ("status.json", "budget.json", "attempt_001.json"):
        run.atomic_json({"mutable": "supervisor updated after completion"}, state.out / filename)
    assert run.check_complete(state.out) == done
    assert len(state.model_loads) == 2 and len(state.calls) == 26


@pytest.mark.parametrize("field", ["source", "raw_root", "output", "implementation"])
def test_execute_rejects_changed_supervisor_request_before_loading(orchestration, field):
    state = orchestration
    request = read(state.out / "request.json")
    request[field] = {"changed": True} if field == "implementation" else "/different"
    run.atomic_json(request, state.out / "request.json")
    with pytest.raises(ValueError, match="matching bounded supervisor request"):
        state.execute()
    assert not state.verifies and not state.model_loads and not state.calls


def test_execute_cuda_guard_precedes_source_and_model_loading(orchestration, monkeypatch):
    state = orchestration
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="AutoDL CUDA"):
        state.execute()
    assert not state.verifies and not state.model_loads and not state.calls


def test_execute_rejects_cache_row_identity_drift_before_model_loading(orchestration):
    state = orchestration
    state.rows["test"][0]["row"] += 1
    with pytest.raises(ValueError, match="source endpoint identities changed"):
        state.execute()
    assert not state.model_loads and not state.calls


def test_execute_changed_selection_hash_stops_before_model_loading(orchestration):
    state = orchestration
    state.selected[0]["first"] += 1
    with pytest.raises(ValueError, match="Fixed13-segment inventory"):
        state.execute()
    assert not state.model_loads and not state.calls
    assert not (state.out / "segments.json").exists()
    diagnostic = read(state.out / "selection_check.json")
    assert diagnostic["expected_sha256"] != diagnostic["actual_sha256"]


def test_execute_changed_raw_before_load_stops_without_model_execution(orchestration):
    state = orchestration
    path, _ = run.source_file(state.raw_root, state.selected[0], state.inventory)
    path.write_text("raw file changed before inference\n")
    with pytest.raises(ValueError, match="Original raw fingerprint changed"):
        state.execute()
    assert not state.model_loads and not state.calls


def test_execute_raw_changed_during_segment_stops_after_one_call(orchestration):
    state = orchestration
    path, _ = run.source_file(state.raw_root, state.selected[0], state.inventory)
    state.on_segment = lambda _: path.write_text("raw file changed during inference\n")
    with pytest.raises(ValueError, match="Raw file changed during streaming"):
        state.execute()
    assert state.model_loads == ["macro_s42"] and len(state.calls) == 1
    assert not (state.out / "completion.json").exists()


def test_execute_rechecks_earlier_raw_after_all_segments_before_completion(orchestration):
    state = orchestration
    path, _ = run.source_file(state.raw_root, state.selected[0], state.inventory)
    state.after_model = lambda n: (
        path.write_text("raw file changed after last model\n") if n == 2 else None
    )
    with pytest.raises(ValueError, match="Raw identity changed before completion"):
        state.execute()
    assert len(state.calls) == 26 and not (state.out / "completion.json").exists()


def rewrite_bound_json(out, filename, value):
    """Expose structural checks independently of the first byte-fingerprint check."""
    run.atomic_json(value, out / filename)
    done = read(out / "completion.json")
    done["files"][filename] = run.sha256(out / filename)
    run.atomic_json(done, out / "completion.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_pair",
        "third_attempt",
        "wrong_units",
        "wrong_total",
        "missing_matrix",
        "wrong_completed_count",
        "wrong_status",
    ],
)
def test_completion_rejects_inconsistent_ledger_or_matrix_after_hash_checks(
    orchestration, mutation
):
    state = orchestration
    state.execute()
    out = state.out
    if mutation in ("missing_pair", "third_attempt", "wrong_units"):
        journal = read(out / "forward_attempts.json")
        if mutation == "missing_pair":
            journal.pop()
        elif mutation == "third_attempt":
            journal.extend([copy.deepcopy(journal[0]), copy.deepcopy(journal[0])])
        else:
            journal[0]["reserved_endpoints"] -= 1
        rewrite_bound_json(out, "forward_attempts.json", journal)
    elif mutation == "missing_matrix":
        matrix = read(out / "continuous_report.json")
        matrix["reports"].pop(next(iter(matrix["reports"])))
        rewrite_bound_json(out, "continuous_report.json", matrix)
    else:
        done = read(out / "completion.json")
        key = {
            "wrong_total": "reserved_endpoint_forwards",
            "wrong_completed_count": "completed_segment_models",
            "wrong_status": "status",
        }[mutation]
        done[key] = "wrong" if mutation == "wrong_status" else done[key] - 1
        run.atomic_json(done, out / "completion.json")
    with pytest.raises(ValueError, match="ledger/matrix changed|Incomplete trajectory matrix"):
        run.check_complete(out)


def test_completion_ledger_byte_change_is_rejected_without_rebinding(orchestration):
    state = orchestration
    state.execute()
    journal = read(state.out / "forward_attempts.json")
    journal[0]["started_unix"] += 1
    run.atomic_json(journal, state.out / "forward_attempts.json")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run.check_complete(state.out)


@pytest.mark.parametrize("changed", ["original_completion", "raw_file"])
def test_completion_rechecks_original_source_and_raw_after_success(orchestration, changed):
    state = orchestration
    state.execute()
    if changed == "original_completion":
        (state.source / "completion.json").write_text("{}\n")
    else:
        path, _ = run.source_file(state.raw_root, state.selected[0], state.inventory)
        path.write_text("changed after success\n")
    with pytest.raises(
        ValueError, match="Original source completion changed|Completed source raw file changed"
    ):
        run.check_complete(state.out)
