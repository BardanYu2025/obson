"""Synthetic alignment, gradients, fair budgets and recovery; no optimizer updates."""

import copy
import inspect
import os
import subprocess
import tarfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from test_capacity_growth import small
from test_cross_period_audit import coarse_bars, fine_bars
from test_overlap_diagnostic import fixture_data

from obson.babel import cross_period_consistency as xp
from obson.babel import cross_period_data as xd
from obson.babel import cross_period_evaluate as ev
from obson.babel import cross_period_run as r
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256


@pytest.fixture(autouse=True)
def no_neural_updates():
    torch.set_num_threads(1)
    with patch.object(
        torch.optim.AdamW, "step", side_effect=AssertionError("Local optimizer updates forbidden")
    ):
        yield


def raw_fixture():
    fine = fine_bars(4096)
    series = []
    specs = []
    originals = []
    offset = 0
    for i, period in enumerate((15, 30, 60)):
        frame = fine if period == 15 else coarse_bars(fine, period // 15)
        key = f"x/{period}/EX.x1"
        n = len(frame)
        series.append(
            SimpleNamespace(
                key=key,
                code="x",
                period=period,
                contract="EX.x1",
                frame=frame,
                ends=(frame.datetime + pd.Timedelta(minutes=period)).to_numpy(),
                main=np.ones(n, bool),
                sessions=frame.datetime.to_numpy(dtype="datetime64[D]"),
            )
        )
        originals.append(
            {
                "key": key,
                "symbol": "x",
                "period": period,
                "row": n - 1,
                "end": str(frame.datetime.iloc[-1]),
                "week": "w",
            }
        )
        specs.append(
            {"series": i, "lo": 0, "length": n, "offset": offset, "endpoints": [[n - 1, i]]}
        )
        offset += n
    anchor = originals[0] | {"row": 3071, "end": str(fine.datetime.iloc[3071])}
    return (
        series,
        {"train_until": "2025-12-31", "val_until": "2026-12-31", "test_until": "2027-12-31"},
        specs,
        originals,
        [anchor],
    )


def test_raw_pair_gates_and_held_frequency():
    series, bounds, specs, rows, anchors = raw_fixture()
    pairs, excluded = xd.plan_pairs(series, bounds, "train", specs, rows, anchors, True)
    assert {p["pair"] for p in pairs} == {"15->30", "15->60"}
    for p in pairs:
        assert all(max(link) < 127 for link in p["links"])
        fine, coarse = series[0], series[int(p["pair"].split("->")[1]) // 30]
        for f0, f1, c0, c1 in p["links"]:
            assert fine.ends[p["row"] - 127 + f0] == coarse.ends[p["coarse_row"] - 127 + c0]
            assert fine.ends[p["row"] - 127 + f1] == coarse.ends[p["coarse_row"] - 127 + c1]
    # 30->60 never enters training even if supplied as an anchor.
    coarse_anchor = rows[1] | {"row": 1535, "end": str(series[1].frame.datetime.iloc[1535])}
    pairs, _ = xd.plan_pairs(series, bounds, "train", specs, rows, [coarse_anchor], True)
    assert not pairs
    series[2].main[:] = False
    pairs, excluded = xd.plan_pairs(series, bounds, "train", specs, rows, anchors, True)
    assert [p["pair"] for p in pairs] == ["15->30"]
    assert excluded["main_contract_session_or_512_partition_warmup"] == 1


def test_pair_partition_warmup_and_staggered_close():
    series, bounds, specs, rows, anchors = raw_fixture()
    # Fine endpoint one bar earlier cannot be used with unfinished coarse bar.
    anchors[0]["row"] -= 1
    anchors[0]["end"] = str(series[0].frame.datetime.iloc[anchors[0]["row"]])
    result, excluded = xd.plan_pairs(series, bounds, "train", specs, rows, anchors, True)
    assert not result and excluded["no_simultaneous_close"] == 2
    series, bounds, specs, rows, anchors = raw_fixture()
    anchors[0].update(row=1023, end=str(series[0].frame.datetime.iloc[1023]))
    result, excluded = xd.plan_pairs(series, bounds, "train", specs, rows, anchors, True)
    assert {p["pair"] for p in result} == {"15->30"}  # 60m has only256 rows.
    assert excluded["main_contract_session_or_512_partition_warmup"] == 1


def test_strict_map_rejects_partial_and_zero_volume_price_disagreement():
    fine = fine_bars(40)
    coarse = coarse_bars(fine, 4)
    last, valid = xp.strict_map(fine, coarse, 15, 60)
    assert valid.all()
    np.testing.assert_array_equal(last, np.arange(3, 40, 4))
    fine.loc[0, "high"] += 10
    assert not xp.strict_map(fine, coarse, 15, 60)[1][0]
    fine = fine.drop(index=4).reset_index(drop=True)
    assert not xp.strict_map(fine, coarse, 15, 60)[1][1]


def test_cross_loss_cancels_offsets_and_has_both_branch_gradients():
    stats = {"y_scale": [2] * 7, "delta_scale": [0.5]}
    t = torch.arange(128, dtype=torch.float32)
    a = torch.zeros(2, 128, 7)
    b = a.clone()
    a[:, :, 0] = t
    b[:, :, 0] = 2 * t + 100
    links = [[2 * i, 2 * i + 2, i, i + 1] for i in range(40)]
    ix, mask = xp.padded_links([{"links": links}] * 2)
    ix = torch.tensor(ix)
    mask = torch.tensor(mask)
    assert torch.equal(xp.common_rows(a, b, ix, mask, stats)["gap"], torch.zeros(2))
    b[:, 20, 0] += 1
    a.requires_grad_()
    b.requires_grad_()
    result = xp.common_rows(a, b, ix, mask, stats)["gap"].sum()
    result.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert not a.grad[:, 127].any() and not b.grad[:, 127].any()
    with pytest.raises(ValueError, match="target price"):
        xp.common_rows(a, b, ix, mask, stats, a, b)
    bad = ix.clone()
    bad[0, 0, 1] = 127
    with pytest.raises(ValueError, match="current/future"):
        xp.common_rows(a, b, bad, mask, stats)


def test_loss_truth_and_gap_distinguish_joint_wrong_prediction():
    y = torch.zeros(1, 128, 7)
    a = y.clone()
    b = y.clone()
    a[:, :, 0] = torch.arange(128)
    b[:, :, 0] = torch.arange(128)
    ix, mask = xp.padded_links([{"links": [[i, i + 1, i, i + 1] for i in range(16)]}])
    z = xp.common_rows(
        a,
        b,
        torch.tensor(ix),
        torch.tensor(mask),
        {"y_scale": [1] * 7, "delta_scale": [1]},
        y,
        y,
        False,
    )
    assert z["gap"].item() == 0 and z["fine_error"].item() == 1 and z["coarse_error"].item() == 1


def synthetic_builder(stats):
    _, _, _, views = fixture_data(stats)

    class Builder:
        plan = list(range(8))

        def __init__(self, *args):
            pass

        def __call__(self, ids, ds):
            a = {
                k: np.stack(
                    [views["views"][int(d)][k][i % 3] for i, d in zip(ids, ds, strict=False)]
                )
                for k in ("x", "y", "mask")
            }
            b = {k: views["views"][0][k][np.asarray(ids) % 3] for k in ("x", "y", "mask")}
            ix, mask = xp.padded_links(
                [{"links": [[i, i + 1, i, i + 1] for i in range(16)]}] * len(ids)
            )
            return a, b, b, ix, mask

    return Builder


@pytest.mark.parametrize("count,batch,micros", [(3, 4, (1, 4)), (17, 16, (2, 16))])
def test_microbatch_gradient_accumulation_with_noop_steps(count, batch, micros):
    model, data, stats, local = small()
    builder = synthetic_builder(stats)()
    ids = np.arange(count)
    ds = np.array([1, 16])[ids % 2]
    ps = np.array([[32, 64, 96, 128]] * count)
    grads = []
    scores = []
    for micro in micros:
        m = copy.deepcopy(model)
        enc = torch.optim.AdamW(m.core.encoder.parameters())
        head = torch.optim.AdamW(m.local_head.parameters())
        captures = []

        def record_noop_step(captures=captures, m=m):
            captures.append(
                {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
            )

        with (
            patch.object(enc, "step", side_effect=record_noop_step),
            patch.object(head, "step", return_value=None),
        ):
            score, steps = xp.run_epoch(
                m, builder, ids, ds, ps, stats, local, batch, micro, "cpu", 0.1, enc, head
            )
        assert steps == (count + batch - 1) // batch
        assert np.isfinite(list(score.values())).all()
        assert len(captures) == steps
        grads.append(captures)
        scores.append(list(score.values()))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(m.state_dict()[k], v)
    for left, right in zip(grads[0], grads[1], strict=True):
        for k in left:
            torch.testing.assert_close(left[k], right[k], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-5, rtol=1e-4)


def worker_fixture(tmp_path):
    model, data, stats, local = small()
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    jobs = [
        {"name": f"{v}_s{s}", "seed": s, "variant": v, "layers": 4, "ff": 3072, "weight": w}
        for s in (42, 43)
        for v, w in [("control", 0.0), ("cross", 0.1)]
    ]
    parent = {"experiments": [{"name": f"deep_s{s}", "seed": s} for s in (42, 43)]}
    meta = {
        "schema": r.SCHEMA,
        "code_sha256": r.code_identity(),
        "source": str(source),
        "identity": {"manifest": parent},
        "experiments": jobs,
        "epochs": 6,
        "warmup": 5,
        "encoder_lr": 1e-5,
        "head_lr": 3e-5,
        "budget": 4,
        "batch": 2,
        "micro": 1,
        "evaluation_batch": 2,
        "state_width": 8,
    }
    atomic_json(meta, out / "manifest.json")
    atomic_json({"eligible": 8}, out / "train_cross_plan.json")
    return model, data, stats, local, meta, out


def test_four_workers_resume_selection_budget_and_frozen_decoder(tmp_path):
    model, data, stats, local, meta, out = worker_fixture(tmp_path)
    val = {"selection": 1.0, "primary": 0.5, "path": 0.2}
    original = xp.run_epoch

    def no_update(*args):
        result, _ = original(*args[:-2])
        return result, 2

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(r.ge, "load", side_effect=lambda *a: (copy.deepcopy(model), val))
        )
        stack.enter_context(patch.object(r, "validation", return_value=val))
        stack.enter_context(patch.object(r, "statistics", return_value=(stats, local)))
        stack.enter_context(patch.object(xd, "Builder", synthetic_builder(stats)))
        stack.enter_context(patch.object(xd, "verify_data"))
        calls = []

        def interrupt(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("interrupt")
            return no_update(*args)

        with (
            patch.object(xp, "run_epoch", side_effect=interrupt),
            pytest.raises(RuntimeError, match="interrupt"),
        ):
            r.worker(out, "control_s42", "cpu")
        with patch.object(xp, "run_epoch", side_effect=no_update):
            for job in meta["experiments"]:
                r.worker(out, job["name"], "cpu")
        lock = r.lock_selection(meta, out)
        for v in lock["trials"].values():
            assert (
                v["selected_epoch"] == 0
                and v["encoder_steps"] == 12
                and v["views"] == 72
                and v["decoder_unchanged"]
            )
        for seed in (42, 43):
            a = r.read_json(out / f"control_s{seed}/history.json")
            b = r.read_json(out / f"cross_s{seed}/history.json")
            assert [v["sampling"] for v in a] == [v["sampling"] for v in b]
        with patch.object(xp, "run_epoch", side_effect=AssertionError("No repeat")):
            r.worker(out, "cross_s43", "cpu")
        (out / "cross_s42/best.pt").write_bytes(b"tamper")
        with pytest.raises(ValueError):
            r.lock_selection(meta, out)


def test_schedule_balanced_cycles_no_held_prefixes_and_tamper(tmp_path):
    _, _, _, _, meta, _ = worker_fixture(tmp_path)
    meta["budget"] = 19
    a = r.plan(meta, meta["experiments"][0], 1, 8)
    b = r.plan(meta, meta["experiments"][1], 1, 8)
    for x, y in zip(a[:3], b[:3], strict=False):
        np.testing.assert_array_equal(x, y)
    assert len(set(a[0][:8])) == 8 and len(set(a[0][8:16])) == 8
    assert a[3]["absolute_epoch"] == 501 and a[3]["unique_triplets"] == 8
    assert not set(a[2].ravel()) & {48, 80, 112}


def decision_fixture():
    rows = [{"symbol": "A", "period": 15, "month": "m", "week": f"w{i // 10}"} for i in range(60)]

    def record(gap):
        errors = {
            t: {k: [0.1] * 60 for k in ("primary", "path", "changes", "body", "activity")}
            for t in ("global", "held", "recent")
        }
        utility = {
            "scores": {"targets": [{"r2": 0.95} for _ in range(13)]},
            "errors": {k: [0.03] * 60 for k in list(r.rr.up.NAMES) + list(r.rr.up.GROUPS)},
        }
        keys = (
            "combined_gap",
            "combined_error_a",
            "combined_error_b",
            "path_shape_gap_bps",
            "path_native_mae_a_bps",
            "path_native_mae_b_bps",
        )
        overlap = {
            str(d): {
                "per_pair": {k: [0.03] * 60 for k in keys},
                "valid": {k: [True] * 60 for k in keys},
            }
            for d in (1, 16, 64)
        }
        return {
            "reconstruction": {"errors": errors},
            "utility": utility,
            "overlap": overlap,
            "cross_period": {
                "per_pair": {
                    "gap": [gap] * 180,
                    "fine_error": [0.1] * 180,
                    "coarse_error": [0.1] * 180,
                }
            },
        }

    bank = {f"parent_s{s}": record(0.1) for s in (42, 43)}
    for s in (42, 43):
        for kind in ("best", "last"):
            for mode in ("control", "cross"):
                bank[f"{mode}_s{s}_{kind}"] = record(0.08 if mode == "cross" else 0.1)
    records = {s: copy.deepcopy(bank) for s in ("test", "cross_research")}
    rs = dict.fromkeys(records, rows)
    ms = {s: np.ones((60, 13), bool) for s in records}
    cross = {
        s: [r | {"pair": p} for p in ("15->30", "15->60", "30->60") for r in rows] for s in records
    }
    return records, rs, ms, cross


@pytest.mark.parametrize("failure", ["none", "truth", "utility", "last", "support", "held"])
def test_decision_rejects_gap_only_gain_and_partial_success(failure):
    records, rows, masks, cross = decision_fixture()
    target = records["test"]["cross_s43_last"]
    if failure == "truth":
        target["cross_period"]["per_pair"]["fine_error"] = [1.0] * 180
    if failure == "utility":
        target["utility"]["errors"]["utility"] = [1.0] * 60
    if failure == "last":
        target["cross_period"]["per_pair"]["gap"] = [0.1] * 180
    if failure == "support":
        cross["test"] = [r | {"week": "one"} for r in cross["test"]]
    if failure == "held":
        target["cross_period"]["per_pair"]["gap"][120:] = [1.0] * 60
    result = ev.decide({}, records, rows, rows, masks, cross)
    assert result["status"] == (
        "cross_period_stability_candidate" if failure == "none" else "no_cross_period_upgrade"
    )
    assert not result["automatic_promotion"]


def test_audit_manifest_tamper_and_missing_source(tmp_path):
    from test_cross_period_audit import write_contract

    from obson.babel import cross_period_audit

    raw = tmp_path / "raw"
    out = tmp_path / "audit"
    write_contract(raw, "EX.x1")
    cross_period_audit.run(raw, out)
    xd.audit_identity(out, raw)
    (raw / "x/EX.x1_15m.csv").write_text("changed")
    with pytest.raises(ValueError):
        xd.audit_identity(out, raw)


def test_full_evaluation_locks_eight_readouts_and_replays_frozen_parent(tmp_path):
    import test_consistency_readout as reader_tests

    oldmeta, parent, rows, banks = reader_tests.ReadoutTests().fixture(tmp_path)
    r.rr.fit(oldmeta, parent, "cpu")
    r.rr.evaluate(oldmeta, parent, rows)
    # Synthetic legacy growth artifacts: deep500 aliases with existing reader coefficients.
    fit = r.read_json(parent / "fit.json")
    metrics = r.read_json(parent / "readout_metrics.json")
    with np.load(parent / "heads.npz", allow_pickle=False) as f:
        heads = dict(f)
    for seed in (42, 43):
        old = f"consistent_s{seed}"
        new = f"deep_s{seed}_best"
        fit["heads"][new] = fit["heads"][old]
        for suffix in ("weights", "intercepts"):
            heads[new + "_" + suffix] = heads[old + "_" + suffix]
        for split in ("test", "cross_research"):
            metrics["datasets"][split]["scores"][new] = metrics["datasets"][split]["scores"][old]
    atomic_json(fit, parent / "fit.json")
    np.savez(parent / "heads.npz", **heads)
    atomic_json(metrics, parent / "readout_metrics.json")
    cm = copy.deepcopy(oldmeta["identity"]["manifest"])
    cm.update(source="unused", identity={}, packed={s: {} for s in ("test", "cross_research")})
    jobs = [{"name": f"{v}_s{s}", "seed": s} for s in (42, 43) for v in ("control", "cross")]
    gm = {"experiments": [{"name": f"deep_s{s}", "seed": s} for s in (42, 43)]}
    out = tmp_path / "trial"
    (out / "cache").mkdir(parents=True)
    meta = {
        "source": str(parent),
        "identity": {"manifest": gm},
        "experiments": jobs,
        "evaluation_batch": 128,
    }
    atomic_json(meta, out / "manifest.json")
    weights = {}
    for j in jobs:
        cell = out / j["name"]
        cell.mkdir()
        weights[j["name"]] = {}
        for kind in ("best", "last"):
            (cell / f"{kind}.pt").write_bytes(b"locked")
            weights[j["name"]][kind] = sha256(cell / f"{kind}.pt")
    atomic_json(
        {"manifest": meta, "trials": {j["name"]: {} for j in jobs}, "weights": weights},
        out / "model_selection_lock.json",
    )
    for split, bank in banks.items():
        values = {k: bank[k] for k in ("raw", "current", "targets", "mask")}
        for seed in (42, 43):
            values[f"parent_s{seed}"] = bank[f"consistent_s{seed}"]
        for name, j, _kind in ev.entries(meta):
            values[name] = bank[f"consistent_s{j['seed']}"]
        np.savez(out / f"cache/{split}.npz", **values)
        atomic_json(
            {
                "manifest_sha256": sha256(out / "manifest.json"),
                "files": {f"{split}.npz": sha256(out / f"cache/{split}.npz")},
            },
            out / f"cache/{split}_index.json",
        )
        atomic_json(rows[split], out / f"{split}_inventory.json")
    with pytest.raises(FileNotFoundError):
        ev.prepare(meta, out, "test", "cpu")
    ev.fit(meta, out, "cpu")
    assert r.read_json(out / "fit.json")["candidates"] == 520
    ev.check_readouts(out)
    with patch.object(ev.up, "fit_heads", side_effect=AssertionError("No repeated fit")):
        ev.fit(meta, out, "cpu")

    def model(label):
        m = torch.nn.Module()
        m.label = label
        return m

    def score(*args):
        err = {
            t: {k: np.full(72, 0.1) for k in ("primary", "path", "changes", "body", "activity")}
            for t in ("global", "held", "recent")
        }
        scores = {
            t: {"metrics": {k: float(v.mean()) for k, v in e.items()}} for t, e in err.items()
        }
        return scores, err, {"path": np.zeros(72)}, np.zeros((72, 128, 7))

    def overlap(*args):
        keys = (
            "combined_gap",
            "combined_error_a",
            "combined_error_b",
            "path_shape_gap_bps",
            "path_native_mae_a_bps",
            "path_native_mae_b_bps",
        )
        return {
            str(d): {
                "per_pair": {k: [0.03] * 72 for k in keys},
                "valid": {k: [True] * 72 for k in keys},
                "summary": {"all": {"combined_gap": {"mean": 0.03}}},
            }
            for d in (1, 16, 64)
        }

    for split in ("test", "cross_research"):
        for seed in (42, 43):
            atomic_json(
                {"reconstruction": {"scores": score()[0]}, "overlap": overlap()},
                parent / f"{split}_deep_s{seed}_best.json",
            )

    def cross_data(meta, out, split):
        return (
            [r | {"pair": p} for p in ("15->30", "15->60", "30->60") for r in rows[split]],
            [],
            None,
            None,
        )

    def cross_scores(m, *args):
        gap = 0.08 if m.label.startswith("cross") else 0.1
        return {
            "per_pair": {
                "gap": [gap] * 216,
                "fine_error": [0.1] * 216,
                "coarse_error": [0.1] * 216,
            },
            "groups": {},
        }

    with ExitStack() as stack:
        stack.enter_context(patch.object(r, "statistics", return_value=({}, {})))
        stack.enter_context(patch.object(r, "parent_meta", return_value=cm))
        stack.enter_context(patch.object(r.cr, "alignment", return_value={}))
        stack.enter_context(patch.object(r.cr.odr, "make_manifest", return_value={}))
        stack.enter_context(
            patch.object(
                r.cr.odr,
                "prepare",
                return_value=({s: {"rows": rows[s]} for s in ("test", "cross_research")}, {}),
            )
        )
        stack.enter_context(
            patch.object(
                r.ur.bb,
                "load_data",
                return_value={"y": np.zeros((72, 128, 7)), "mask": np.ones((72, 128, 7), bool)},
            )
        )
        stack.enter_context(
            patch.object(r.ge, "load", side_effect=lambda *a: (model("parent"), {}))
        )
        stack.enter_context(
            patch.object(
                ev, "load", side_effect=lambda meta, out, j, kind, device: (model(j["name"]), {})
            )
        )
        stack.enter_context(patch.object(r.ur.pf, "score", side_effect=score))
        stack.enter_context(patch.object(ev.ce, "overlap", side_effect=overlap))
        stack.enter_context(patch.object(xd, "evaluation_data", side_effect=cross_data))
        stack.enter_context(patch.object(ev, "cross_scores", side_effect=cross_scores))
        ev.evaluate(meta, out, "cpu")
        report = r.read_json(out / "cross_period_metrics.json")
        assert len(report["summary"]["test"]) == 10
        assert report["decision"]["status"] == "cross_period_stability_candidate"
        prior = r.read_json(parent / "readout_metrics.json")
        prior["datasets"]["test"]["scores"]["deep_s42_best"]["groups"]["utility"] += 1
        atomic_json(prior, parent / "readout_metrics.json")
        with pytest.raises(ValueError):
            ev.evaluate(meta, out, "cpu")
    (out / "cache/train.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError):
        ev.check_readouts(out)


def test_evaluation_anchors_are_fixed_dense_and_inside_packed_bounds():
    series, bounds, specs, rows, _ = raw_fixture()
    # Use train-labelled geometry just for synthetic cohort construction.
    anchors = xd.evaluation_anchors(series, bounds, "train", specs)
    assert anchors and {r["period"] for r in anchors} == {15, 30}
    for row in anchors:
        assert row["row"] >= 511 and (row["row"] - 127) % 16 == 0
    assert len(anchors) > len(rows)


def test_manager_failure_stops_other_worker(tmp_path):
    _, _, _, _, meta, out = worker_fixture(tmp_path)
    failure_path = out / "control_s42/run.log"
    failure_path.parent.mkdir()
    failure_path.write_text("old output\n" * 10000 + "torch.OutOfMemoryError: CUDA out of memory\n")
    failed = MagicMock()
    failed.poll.return_value = 1
    failed.returncode = 1
    running = MagicMock()
    running.poll.return_value = None
    with (
        patch.object(r.subprocess, "Popen", side_effect=[failed, running]),
        pytest.raises(RuntimeError, match="exit1") as error,
    ):
        r.run_jobs(out, 2)
    assert "torch.OutOfMemoryError: CUDA out of memory" in str(error.value)
    assert str(failure_path) in str(error.value)
    assert len(str(error.value).splitlines()) <= 81
    running.terminate.assert_called_once()
    running.wait.assert_called_once()


def test_failure_export_contains_reports_but_no_weights(tmp_path):
    cell = tmp_path / "run/cross_s42"
    cell.mkdir(parents=True)
    (cell / "best.pt").write_bytes(b"omit")
    atomic_json([{"epoch": 1}], cell / "history.json")
    env = os.environ | {
        "BABEL_CROSS_PERIOD_RUN": str(tmp_path / "run"),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "BABEL_CROSS_PERIOD_LOG": str(tmp_path / "missing"),
        "PYTHON_BIN": "/usr/bin/false",
    }
    result = subprocess.run(
        ["bash", "scripts/babel_cross_period768_autodl.sh", "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        assert "run/cross_s42/history.json" in tar.getnames()
        assert "run/cross_s42/best.pt" not in tar.getnames()
        assert "run_status=failed" in tar.extractfile("run/run_status.txt").read().decode()


def test_release_preflight_memory_before_launch(tmp_path):
    with (
        patch.object(r.gc, "collect") as collect,
        patch.object(r.torch.cuda, "device") as device,
        patch.object(r.torch.cuda, "empty_cache") as empty,
        patch.object(r.torch.cuda, "mem_get_info", return_value=(11 * 2**30, 12 * 2**30)),
        patch.object(r.torch.cuda, "get_device_name", return_value="Synthetic GPU"),
        patch.object(r.torch.cuda, "memory_allocated", return_value=0),
        patch.object(r.torch.cuda, "memory_reserved", return_value=0),
    ):
        r.release_preflight_cuda(tmp_path, 1, 16, "cuda")
        collect.assert_called_once()
        device.assert_called_once_with("cuda")
        empty.assert_called_once()
    report = r.read_json(tmp_path / "gpu_launch.json")
    assert report["jobs"] == 1 and report["micro_triplets"] == 16
    assert report["windows_per_forward"] == 48 and report["effective_batch"] == 128
    assert report["manager_reserved_bytes"] == 0


def test_conservative_defaults_and_shell_overrides(tmp_path):
    assert inspect.signature(r.run).parameters["jobs"].default == 1
    assert inspect.signature(r.run).parameters["micro"].default == 16
    assert inspect.signature(r.make_manifest).parameters["micro"].default == 16
    executable = tmp_path / "capture.sh"
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    executable.chmod(0o755)
    env = os.environ.copy()
    for name in ("BABEL_CROSS_PERIOD_JOBS", "BABEL_CROSS_PERIOD_MICRO"):
        env.pop(name, None)
    env.update(
        BABEL_CROSS_PERIOD_RUN=str(tmp_path / "run"),
        BABEL_CROSS_PERIOD_LOG=str(tmp_path / "missing.log"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN=str(executable),
        CAPTURE_ARGS=str(tmp_path / "args.txt"),
    )
    for overrides, expected in (
        ({}, ["1", "16"]),
        ({"BABEL_CROSS_PERIOD_JOBS": "2", "BABEL_CROSS_PERIOD_MICRO": "32"}, ["2", "32"]),
    ):
        subprocess.run(
            ["bash", "scripts/babel_cross_period768_autodl.sh", "all"],
            env=env | overrides,
            check=True,
            capture_output=True,
        )
        args = (tmp_path / "args.txt").read_text().splitlines()
        assert [args[args.index("--jobs") + 1], args[args.index("--micro") + 1]] == expected
