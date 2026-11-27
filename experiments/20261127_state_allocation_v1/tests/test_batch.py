"""Independent oracles for the real source, model, objective and persistence paths."""

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.assets import load_checkpoint, save_checkpoint
from src.common import QUOTAS, digest
from src.dataset import Corpus, build, make_shard, verify_cache
from src.model import StateModel, calibrate, geometry
from src.objective import example_loss, field_coeff, optimizer, query_weights, update
from src.scales import fit
from src.scope import select_sources
from src.views import U, blank_stats, sample, spans_for_mask


torch.set_num_threads(1)


def market():
    t = int(np.datetime64("2023-07-03T21:00", "m").astype(int)) + np.arange(330)
    k = np.arange(len(t))
    c = 100 + 0.008 * k + 0.05 * np.sin(k * 0.61)
    o = np.r_[c[0], c[:-1]] + 0.006 * np.cos(k)
    raw = np.column_stack(
        (
            o,
            np.maximum(o, c) + 0.02,
            np.minimum(o, c) - 0.03,
            c,
            k % 17,
            k + 400,
            k + 403,
        )
    )
    return t, raw


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    folder = tmp_path_factory.mktemp("corpus")
    times, raw = market()
    groups = make_shard(
        folder / "s0000",
        {
            "id": 0,
            "product": "ag",
            "contract": "SHFE.ag202312",
            "symbol": "SHFE.ag2312",
        },
        times,
        raw,
        times,
        np.unique(times // 1440),
    )
    (folder / "catalogue.json").write_text(json.dumps({"groups": groups}))
    return Corpus(folder)


def small(arm, stats, seed=42):
    torch.manual_seed(seed)
    return StateModel(arm, stats, width=168, heads=3, ffn=48, layers=2, film_width=24)


def test_official_size_without_materializing_weights():
    with torch.device("meta"):
        model = StateModel("P3", blank_stats())
    assert [
        sum(p.numel() for p in m.parameters())
        for m in (model.encoder, model.history, model.describe)
    ] == [42610308, 2107911, 100754]


def test_score_and_value_sources():
    g = torch.tensor(
        [
            [
                [-2.0, -1.0, -2.0, -1.0, 1.0],
                [-1.0, 0.0, -1.0, 0.0, 1.0],
                [-5.0, 0.0, -5.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    phi, legal = geometry(g, torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.bool))
    assert legal.sum(-1).tolist() == [[1, 3, 3, 0, 4]]
    torch.testing.assert_close(phi[0, 1, 2, 5], torch.tensor(0.2))
    assert phi[0, 2, 1, 6] == 1 and phi[0, 1, 2, 7] == 1
    model = small("P2", blank_stats())
    layer = model.encoder.layers[1]
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
        layer.out.weight.copy_(torch.eye(168))
        layer.qkv.bias[2 * 168 :] = 2.0
    first = torch.zeros(1, 3, 5, 56)
    first[0, 0, :, 0] = torch.tensor([4.0, 8.0, 12.0, 900.0, 999.0])
    y, _ = layer(torch.zeros(1, 5, 168), phi, legal, first)
    torch.testing.assert_close(y[0, :, 0], torch.tensor([3.0, 5.0, 5.0, 0.0, 4.25]))
    assert torch.isfinite(y).all()


def test_mask_closure_and_future_invariance(corpus, tmp_path):
    t, raw = market()
    s1 = corpus.shard(0)
    stats = blank_stats()
    end = 290
    holes = np.array([230, 255])
    x = sample(s1, end, stats, holes)
    changed = raw.copy()
    for m in holes:
        changed[m : m + 8, :4] *= 1.07
        changed[m : m + 8, 4:] += 7
    changed[end + 1 :] *= 2
    make_shard(
        tmp_path / "s0000",
        {"id": 0, "product": "ag", "symbol": "SHFE.ag2312", "contract": "c"},
        t,
        changed,
        t,
        np.unique(t // 1440),
    )
    from src.dataset import Shard

    y = sample(Shard(tmp_path / "s0000"), end, stats, holes)
    for key in (
        "x",
        "context",
        "geom",
        "period",
        "valid",
        "query",
        "facts",
        "block",
        "slot",
    ):
        np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
    assert np.any(x.target[x.hidden, 0] != y.target[y.hidden, 0])
    assert x.hidden.sum() == 16
    with torch.no_grad():
        m = small("P3", stats)
        assert torch.equal(m.encode(x), m.encode(y))
    only_future = raw.copy()
    only_future[end + 1 :] *= 1.4
    make_shard(
        tmp_path / "s0001",
        {"id": 1, "product": "ag", "symbol": "SHFE.ag2312", "contract": "c"},
        t,
        only_future,
        t,
        np.unique(t // 1440),
    )
    full = sample(corpus.shard(0), end, stats)
    replay = sample(Shard(tmp_path / "s0001"), end, stats)
    np.testing.assert_array_equal(full.x, replay.x)
    assert np.all(full.geom[:, 1] <= 0)


def test_missing_slot_and_partition_purge(tmp_path):
    from src.dataset import Shard

    t, raw = market()
    ix = np.arange(len(t)) != 282
    r = {"id": 0, "product": "ag", "symbol": "SHFE.ag2312", "contract": "c"}
    make_shard(tmp_path / "one", r, t[ix], raw[ix], t, np.unique(t // 1440))
    s = sample(Shard(tmp_path / "one"), 300, blank_stats())
    gap = (s.period == 0) & (s.age == 18)
    assert (
        gap.sum() == 1
        and not s.valid[gap][0]
        and not np.isfinite(s.target[gap, 0]).any()
    )
    assert not s.facts[s.block[gap][0], 4]
    assert not Shard(tmp_path / "one").flags[300, 3]
    extended = np.r_[t, t + 365 * 1440]
    make_shard(
        tmp_path / "two",
        r,
        extended,
        np.vstack((raw, raw)),
        extended,
        np.unique(extended // 1440),
    )
    other = Shard(tmp_path / "two")
    assert other.flags[len(t), 0] == 1 and not other.flags[len(t), 1:].any()


def test_blocks_prices_and_restore_order(corpus):
    stats = blank_stats()
    s = sample(corpus.shard(0), 300, stats)
    old = s.period == 0
    idx = np.flatnonzero((s.block == 0) & old)
    assert s.facts[0, 4] == 1 and np.array_equal(s.slot[idx], np.arange(16))
    np.testing.assert_allclose(
        s.facts[0, :2], np.log(s.physical[idx, 3]) @ U, atol=5e-8
    )
    np.testing.assert_allclose(
        np.expm1(s.facts[0, 2]), s.physical[idx, 4].sum(), rtol=2e-6
    )
    m = small("P3", stats)
    z = m.encode(s)
    order = idx[[13, 2, 13]]

    def read(indices):
        return m.read(z, s.query, s.period, s.current, s.block, s.slot, indices)[0]

    answer = read(order)
    torch.testing.assert_close(
        answer, torch.cat([read([i]) for i in order]), atol=2e-6, rtol=2e-5
    )
    fixed = read(idx).detach().numpy()
    np.testing.assert_allclose(fixed[:, 0] @ U, s.facts[0, :2], atol=1e-9)
    np.testing.assert_allclose(
        np.expm1(fixed[:, 4]).sum(), np.expm1(float(s.facts[0, 2])), atol=1e-7
    )
    np.testing.assert_allclose(
        np.expm1(fixed[-1, 6]) - np.expm1(fixed[0, 5]),
        np.sinh(float(s.facts[0, 3])),
        atol=1e-7,
    )


def test_projection_jacobian_and_state_gradient():
    y = torch.randn(1, 16, 7, dtype=torch.float64, requires_grad=True)
    facts = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    basis = torch.tensor(U)
    fixed = calibrate(y, facts, torch.ones(1), basis)
    g = torch.arange(16, dtype=torch.float64)
    grad = torch.autograd.grad((fixed[0, :, 0] * g).sum(), y)[0][0, :, 0]
    torch.testing.assert_close(grad, g - basis @ (basis.T @ g))
    model = small("P3", blank_stats())
    h = torch.randn(2, 168, requires_grad=True)
    after = model.encoder.norm(h)
    after.retain_grad()
    z = model.pack(after, torch.zeros(2, 32, 5, requires_grad=True))
    z[:, 160:].square().sum().backward()
    assert after.grad[:, 8:].abs().sum() == 0 and h.grad[:, 8:].abs().sum() > 0


def test_joint_update_and_exact_resume(corpus, tmp_path):
    stats = blank_stats()
    for arm in ("P2", "P3"):
        m = small(arm, stats)
        opt = optimizer(m)
        tr = update(m, opt, corpus, stats, 42, 1, 3, quotas=(2, 1, 1))
        assert [x[0] for x in tr["exposure"]] == [0, 1, 2, 0]
        assert min(tr["grad_norms_E_D_A"]) > 0
        assert {int(v["step"]) for v in opt.state.values()} == {1}
        save_checkpoint(tmp_path / arm, m, opt, 1, [tr], "run", True)
        next1 = update(m, opt, corpus, stats, 42, 2, 3, quotas=(2, 1, 1))
        recovered = small(arm, stats, 99)
        opt2 = optimizer(recovered)
        assert load_checkpoint(tmp_path / arm, "last", recovered, opt2, "run")[0] == 1
        next2 = update(recovered, opt2, corpus, stats, 42, 2, 3, quotas=(2, 1, 1))
        assert next1 == next2
        for a, b in zip(m.parameters(), recovered.parameters()):
            assert torch.equal(a, b)
    assert QUOTAS == (32, 16, 16)


def test_field_law_and_separate_training_routes(corpus):
    s = sample(corpus.shard(0), 290, blank_stats())
    w = query_weights(s)
    assert np.isclose(w.sum(), 1) and w[s.current].sum() == 0
    np.testing.assert_allclose(
        field_coeff(np.r_[1, 0, 0, 0, 0, 1, 0, 0, 0]), [0.75, 0, 0, 0, 0, 0.25, 0, 0, 0]
    )
    m = small("P3", blank_stats())
    for task in range(3):
        rr = np.random.default_rng(715)
        hidden = spans_for_mask(corpus.shard(0), 290, rr) if task == 1 else ()
        s = sample(corpus.shard(0), 290, blank_stats(), hidden)
        m.zero_grad(set_to_none=True)
        val, _ = example_loss(m, s, task, rr)
        val.backward()
        for module, want in (
            (m.encoder, True),
            (m.history, task < 2),
            (m.describe, task == 2),
        ):
            present = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
            )
            assert present == want


def test_raw_parquet_to_dataset_source_checks_and_train_only_fits(tmp_path):
    t, raw = market()
    root = tmp_path / "raw"
    for d in ("month_contracts/bars/ag", "main", "mapping"):
        (root / d).mkdir(parents=True)
    p = root / "month_contracts/bars/ag/c.parquet"
    from src.dataset import FIELDS

    table = pa.table(
        {
            "datetime_nano": (t - 480) * 60_000_000_000,
            "datetime": (t * 60).astype("datetime64[s]"),
            **dict(zip(FIELDS, raw.T)),
        }
    )
    pq.write_table(table, p)
    main = root / "main/ag.parquet"
    pq.write_table(table.select(["datetime", "datetime_nano"]), main)
    mapping = root / "mapping/ag.csv"
    mapping.write_text(
        "date,underlying\n2023-07-03,SHFE.ag2312\n2023-07-04,SHFE.ag2312\n"
    )
    rec = {
        "id": 0,
        "product": "ag",
        "symbol": "SHFE.ag2312",
        "contract": "SHFE.ag202312",
        "rows": len(t),
        "first": str(np.datetime64(int(t[0]), "m")),
        "last": str(np.datetime64(int(t[-1]), "m")),
        "path": p.relative_to(root).as_posix(),
        "bytes": p.stat().st_size,
        "sha256": digest(p),
    }
    man = {
        "files": [rec],
        "clocks": {"ag": {"main": digest(main), "mapping": digest(mapping)}},
    }
    cache = tmp_path / "work"
    build(root, cache, man)
    verify_cache(cache)
    scales = fit(Corpus(cache), 8)
    assert all(np.isfinite(value).all() for value in scales.values())
    assert scales["f_on"][0] == 1 and np.all(scales["f_sd"] >= 1e-6)
    with pytest.raises(ValueError, match="SHA256"):
        rec["sha256"] = "0" * 64
        build(root, tmp_path / "bad", man)
    with pytest.raises(ValueError, match="Prepared file changed"):
        (cache / "s0000/time.npy").write_bytes(b"corrupt")
        verify_cache(cache)


def test_batch_axis_permutation_and_two_gradients(corpus):
    from src.model import Encoder

    torch.manual_seed(7)
    net = Encoder(width=168, heads=3, ffn=48, layers=2)
    x, ctx = torch.rand(2, 5, 28), torch.rand(2, 5, 8)
    p = torch.tensor([[0, 1, 0, 2, 3]] * 2)
    g = torch.tensor(
        [
            [
                [-5.0, -4.0, -5.0, -4.0, 1.0],
                [-5.0, -1.0, -5.0, -1.0, 1.0],
                [-2.0, -1.0, -2.0, -1.0, 1.0],
                [-1.0, 0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
        * 2
    )
    good = torch.tensor([[1, 1, 1, 1, 0]] * 2, dtype=torch.bool)
    h = net(x, ctx, p, g, good)
    perm = torch.tensor([2, 0, 4, 1, 3])
    torch.testing.assert_close(
        net(x[:, perm], ctx[:, perm], p[:, perm], g[:, perm], good[:, perm]), h
    )
    changed = x.clone()
    changed[1] *= 3
    torch.testing.assert_close(net(changed, ctx, p, g, good)[0], h[0])
    h[:, 0].sum().backward()
    assert (net.layers[0].relation[0].weight.grad == 0).all()
    assert net.layers[0].relation[-1].weight.grad.abs().sum() > 0


def test_full_mixed_quota_and_independent_gradient_oracle(corpus):
    stats = blank_stats()
    m = small("P3", stats)
    reference = copy.deepcopy(m)
    o1, o2 = optimizer(m), optimizer(reference)
    expected = []
    gradients = []
    for task, count in enumerate((2, 1, 1)):
        rr = np.random.default_rng(np.random.SeedSequence([42, 1, task, 0]))
        sid, row = corpus.draw(0, task, rr)
        shard = corpus.shard(sid)
        hidden = spans_for_mask(shard, row, rr) if task == 1 else ()
        loss, _ = example_loss(reference, sample(shard, row, stats, hidden), task, rr)
        if task == 0:
            rr = np.random.default_rng(np.random.SeedSequence([42, 1, 0, 1]))
            sid, row = corpus.draw(0, 0, rr)
            other, _ = example_loss(
                reference, sample(corpus.shard(sid), row, stats), task, rr
            )
            loss = (loss + other) / 2
        weighted = loss * (1.0, 0.25, 0.1)[task]
        grad = torch.autograd.grad(
            weighted, list(reference.parameters()), retain_graph=True, allow_unused=True
        )
        gradients.append(
            torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(grad, reference.parameters())
                ]
            ).double()
        )
        expected.append(weighted)
    sum(expected).backward()
    for mod in (reference.encoder, reference.history, reference.describe):
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
    for g in o2.param_groups:
        from src.objective import rate

        g["lr"] = rate(1, 3)
    o2.step()
    actual = update(m, o1, corpus, stats, 42, 1, 3, (2, 1, 1), measure_gradients=True)
    gram = torch.stack(gradients) @ torch.stack(gradients).T
    np.testing.assert_allclose(
        actual["weighted_gradient_gram"], gram.numpy(), atol=2e-6, rtol=2e-5
    )
    for a, b in zip(m.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=2e-4)
    m2 = small("P2", stats)
    info = update(m2, optimizer(m2), corpus, stats, 43, 1, 3)
    assert np.bincount(np.array(info["exposure"])[:, 0]).tolist() == [32, 16, 16]
    assert np.isfinite(info["loss"]).all()


def test_baselines_and_natural_order(corpus, tmp_path):
    from src.frozen import (
        principal,
        ridge,
        historical_ridge,
        fit_description,
        order_truth,
    )

    rng = np.random.default_rng(32)
    x = rng.normal(size=(48, 7))
    y = x[:, 2:3] * 3 + 4
    w = rng.uniform(0.1, 2, len(x))
    coef, inter = ridge(x, y, w)
    z = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag([0] + [0.01] * 7)
    expect = np.linalg.solve((z.T * w) @ z / w.sum() + penalty, (z.T * w) @ y / w.sum())
    np.testing.assert_allclose(np.vstack((inter, coef)), expect, atol=2e-12)
    mu, sd, basis = principal(x, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-10)
    fields = np.stack((y[:, 0], 2 * y[:, 0]), axis=1)[:, None, :]
    fields[:8, 0, 1] = np.nan
    mh, sh, c, b, n = historical_ridge(x, fields)
    assert n == 2
    c0, b0 = ridge(((x - mh) / sh)[8:], fields[8:, 0, 1:], np.ones(40))
    np.testing.assert_allclose(c[:, 1], c0[:, 0])
    assert b[1] == pytest.approx(b0[0])
    labels = rng.normal(size=(48, 18))
    first, *_ = fit_description(x, labels, 4, "cpu", tmp_path / "head.pt")
    again, *_ = fit_description(x, labels, 4, "cpu", tmp_path / "head.pt")
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), again.parameters())
    )
    sid, row = corpus.draw(0, 3, rng)
    quotes = corpus.shard(sid).bars[0]["v"][row - 36 : row + 1, 3]
    assert order_truth(corpus, (sid, row))[0] == (
        1 if quotes.argmax() < quotes.argmin() else -1
    )


def test_package_and_independent_restore(tmp_path, monkeypatch):
    import src.assets as assets

    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "launch.py").write_text("print(1)")
    (exp / "release.json").write_text(
        json.dumps({"files": {"launch.py": digest(exp / "launch.py")}})
    )
    monkeypatch.setattr(assets, "ROOT", exp)
    folder = tmp_path / "preflight"
    for arm in ("P2", "P3"):
        leaf = folder / arm
        (leaf / "weights").mkdir(parents=True)
        (leaf / "weights/last.pt").write_bytes(arm.encode())
        (leaf / "checkpoints.json").write_text(
            json.dumps(
                {
                    "last": {
                        "path": "weights/last.pt",
                        "sha256": digest(leaf / "weights/last.pt"),
                    }
                }
            )
        )
    pkg = assets.export_run(folder, tmp_path / "download")
    assert assets.verify_package(pkg)["files"] == 6
    restored = assets.restore_package(pkg, tmp_path / "restore")
    assert (restored / "run/P3/weights/last.pt").read_bytes() == b"P3"
    with pytest.raises(FileExistsError):
        assets.restore_package(pkg, restored)
    pkg.write_bytes(pkg.read_bytes()[:-4])
    with pytest.raises(ValueError, match="Outer"):
        assets.verify_package(pkg)


def test_signed_source_scope_and_run_card(corpus):
    from src.run import PROTOCOL, USE_KEYS, train_limits

    card = {
        **PROTOCOL["study_defaults"],
        "physical_use_limits": {k: 1 for k in USE_KEYS},
        "ohlc_unit": "basis_points",
        "intended_use": "research",
        "support_and_near_zero_rule": "native limits",
    }
    train_limits(card)
    with pytest.raises(ValueError):
        train_limits({**card, "physical_use_limits": {}})
    meta = {
        "files": [
            {"id": 0, "product": "ag", "first": "2023-01-01", "last": "2023-12-30"}
        ],
        "clocks": {"ag": {}},
    }
    row = {
        "product": "ag",
        "first": "2023-01-01",
        "last": "2023-12-31",
        "calendar_evidence": "a",
        "volume_evidence": "b",
        "oi_evidence": "c",
        "contract_selection": "known source inventory",
        "volume_multiplier": 0.5,
        "oi_multiplier": 1.0,
    }
    assert (
        select_sources(meta, {"ranges": [row]})["files"][0]["volume_multiplier"] == 0.5
    )
    with pytest.raises(ValueError, match="Overlapping"):
        select_sources(meta, {"ranges": [row, row]})


def test_training_entry_failure_export_resume_and_dense_metrics(
    corpus, tmp_path, monkeypatch
):
    import src.assets as assets
    import src.run as runner
    from src.common import identity, json_file, read_json
    from src.metrics import evaluate_endpoint

    root = tmp_path / "test-release"
    root.mkdir()
    (root / "receipt.txt").write_text("synthetic exercise only")
    json_file(
        root / "release.json", {"files": {"receipt.txt": digest(root / "receipt.txt")}}
    )
    monkeypatch.setattr(assets, "ROOT", root)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "catalogue.json").write_bytes(
        (corpus.folder / "catalogue.json").read_bytes()
    )
    json_file(cache / "scales.json", {"scope": {"ranges": ["synthetic fixture"]}})
    rel = {"commit": "test", "release_sha256": "test"}
    pre = {
        "release": rel,
        "scales": digest(cache / "scales.json"),
        "arms": {"P2": {}, "P3": {}},
        "hardware": runner.environment(),
        "bundle": {"fixture": True},
    }
    json_file(cache / "preflight.json", pre)
    card = {
        **runner.PROTOCOL["study_defaults"],
        "updates": 2,
        "checkpoint_every": 1,
        "validation_every": 1,
        "validation_endpoints": 1,
        "release": rel,
        "preflight": identity(pre),
        "scales": pre["scales"],
        "intended_use": "synthetic",
        "ohlc_unit": "basis_points",
        "support_and_near_zero_rule": "absolute",
        "physical_use_limits": {k: 1 for k in runner.USE_KEYS},
    }
    json_file(cache / "card.json", card)
    runner.lock_study(cache, cache / "card.json", rel)
    stats = blank_stats()
    monkeypatch.setattr(runner, "device_setup", lambda: "cpu")
    monkeypatch.setattr(runner, "ready_data", lambda cache: (corpus, stats))
    monkeypatch.setattr(
        runner, "new_model", lambda arm, stats, seed, device: small(arm, stats, seed)
    )

    def fail_second(model, opt, c, st, sd, step, total):
        if step == 2:
            raise RuntimeError("test power interruption")
        return update(model, opt, c, st, sd, step, total, (2, 1, 1))

    monkeypatch.setattr(runner, "update", fail_second)
    downloads = tmp_path / "downloads"
    with pytest.raises(RuntimeError, match="test power"):
        runner.train(cache, downloads, "P3", 42, rel)
    run = next((cache / "runs").iterdir())
    assert read_json(run / "checkpoints.json")["last"]["step"] == 1
    assert not (run / "completed.json").exists()
    failed = next(downloads.glob("*.tar.gz"))
    assert assets.verify_package(failed)["files"] >= 7
    monkeypatch.setattr(
        runner,
        "update",
        lambda m, o, c, st, sd, step, total: update(
            m, o, c, st, sd, step, total, (2, 1, 1)
        ),
    )
    runner.train(cache, downloads, "P3", 42, rel, True)
    assert read_json(run / "completed.json")["step"] == 2
    assert len(list(downloads.glob("*.tar.gz"))) == 2
    model = small("P3", stats)
    load_checkpoint(run, "last", model, None, read_json(run / "run.json")["run"])
    for task in (0, 1, 2):
        result = evaluate_endpoint(
            model, corpus, stats, 0, 300, task, np.random.default_rng(52)
        )
        json.dumps(result, allow_nan=False)
        if task == 0:
            assert result["R_terms"]["deployed"][0] >= 0 and result["direction64"][
                "truth"
            ] in (-1, 0, 1)
            assert result["failed_price_queries"]["deployed"] == [0, 0]


def test_end_to_end_batched_states_and_distinct_addresses(corpus):
    stats = blank_stats()
    samples = [sample(corpus.shard(0), e, stats) for e in (280, 200)]
    requests = [[12, 2, 12], [1, 9, 5]]
    for arm in ("P2", "P3"):
        m = small(arm, stats)
        z = m.encode(samples)
        fields = [
            [getattr(s, k) for s in samples]
            for k in ("query", "period", "current", "block", "slot")
        ]
        result, _, _ = m.read(z, *fields, requests)
        assert result.shape == (2, 3, 7)
        for i, s in enumerate(samples):
            zs = m.encode(s)
            torch.testing.assert_close(z[i], zs[0], atol=2e-6, rtol=2e-5)
            single = m.read(
                zs, s.query, s.period, s.current, s.block, s.slot, requests[i]
            )[0]
            torch.testing.assert_close(result[i], single, atol=2e-6, rtol=2e-5)
