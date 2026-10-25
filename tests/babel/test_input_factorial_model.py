"""Synthetic-only organization, initialization and causal dependency checks."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import input_factorial as model
from obson.babel import uniform_context as original
from obson.babel.history_autoencoder import positions

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def cpu_runtime():
    threads, fastpath = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def synthetic(n=2):
    x = torch.tensor(np.random.default_rng(65).normal(0, 0.02, (n, 255, 28)), dtype=torch.float32)
    x[..., 23:28], x[..., 22] = 1, 0.1
    return x


@pytest.mark.parametrize("arm", model.ARMS)
def test_every_input_row_and_boundary_are_independently_mapped(arm):
    raw = np.arange(1, 2 * 128 * 28 + 1, dtype=np.float32).reshape(2, 128, 28)
    encoder = model.Encoder(arm, 42, CONFIG)
    ends = list(range(3, 96, 4)) if arm.startswith("hybrid") else list(range(96))
    expected = np.zeros((2, len(ends), 112), dtype=np.float32)
    for token, end in enumerate(ends):
        for slot, row in enumerate(range(end - 3, end + 1)):
            if row >= 0:
                expected[:, token, slot * 28 : (slot + 1) * 28] = raw[:, row]
    rows = encoder.input_rows(torch.tensor(raw))
    np.testing.assert_array_equal(rows["context"], expected)
    np.testing.assert_array_equal(rows["single"], raw[:, 96:])
    assert encoder.token_endpoints().tolist() == ends + list(range(96, 128))
    if arm.startswith("hybrid"):
        np.testing.assert_array_equal(rows["context"].reshape(2, 96, 28), raw[:, :96])
    else:
        assert rows["context"][0, 0, :84].count_nonzero() == 0
        assert rows["context"][0, 1, :56].count_nonzero() == 0
        assert rows["context"][0, 2, :28].count_nonzero() == 0
    # The last old pack ends95; recent96 begins with the single-row projection.
    np.testing.assert_array_equal(rows["context"][:, -1].reshape(2, 4, 28), raw[:, 92:96])
    np.testing.assert_array_equal(rows["single"][:, 0], raw[:, 96])


@pytest.mark.parametrize("initialization", ["random", "endpoint"])
def test_common_projection_tokens_and_uncompressed_positional_coordinates(initialization):
    dense = model.Encoder(f"dense_{initialization}", 42, CONFIG)
    hybrid = model.Encoder(f"hybrid_{initialization}", 42, CONFIG)
    x = synthetic()[:, :128]
    ends = hybrid.token_endpoints()
    torch.testing.assert_close(
        dense.input_tokens(x)[:, ends], hybrid.input_tokens(x), atol=1e-6, rtol=1e-6
    )
    expected = (
        hybrid.input_tokens(x, include_positions=False)
        + positions(128, 32, "cpu", torch.float32)[ends]
    )
    torch.testing.assert_close(hybrid.input_tokens(x), expected, atol=0, rtol=0)
    assert not torch.equal(
        positions(128, 32, "cpu", torch.float32)[ends], positions(56, 32, "cpu", torch.float32)
    )


@pytest.mark.parametrize("seed", model.SEEDS)
def test_common_initialization_parameter_count_rng_and_original_bar_parity(seed):
    rng = torch.random.get_rng_state().clone()
    with patch.object(torch, "load", side_effect=AssertionError("No real weights")):
        receipt = model.initialization_receipt(seed, "cpu", CONFIG)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert receipt["passed"], receipt
    assert receipt["hybrid_endpoint_original_bar_equivalent"] is False
    assert len({arm["parameter_count"] for arm in receipt["arms"].values()}) == 1
    assert all(
        arm["parameter_bytes"] == 4 * arm["parameter_count"] for arm in receipt["arms"].values()
    )
    json.dumps(receipt, allow_nan=False)
    pairs = {arm: model.make_models(arm, seed, "cpu", CONFIG) for arm in model.ARMS}
    prior = original.make_models("rolling", seed, "cpu", CONFIG)
    for pair in pairs.values():
        for name, tensor in prior[0].state_dict().items():
            torch.testing.assert_close(tensor, pair[0].state_dict()[name], atol=0, rtol=0)
        for name, tensor in prior[1].state_dict().items():
            torch.testing.assert_close(tensor, pair[1].state_dict()[name], atol=0, rtol=0)
    for left, right in zip(pairs["dense_endpoint"], pairs["hybrid_endpoint"], strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
            torch.testing.assert_close(p, q, atol=0, rtol=0)
    with torch.no_grad():
        pairs["hybrid_endpoint"][1].output.weight[0, 0] += 0.1
    assert not model.compare_initialized_pairs(pairs)["passed"]


@pytest.mark.parametrize("checkpointing", [False, True])
def test_dense_endpoint_function_and_input_gradient_match_original_not_parameter_update(
    checkpointing,
):
    encoder, query = model.make_models("dense_endpoint", 42, "cpu", CONFIG)
    reference, reference_query = original.make_models("rolling", 42, "cpu", CONFIG)
    encoder.checkpointing = reference.checkpointing = checkpointing
    x = synthetic()[:, :128].clone().requires_grad_()
    xx = x.detach().clone().requires_grad_()
    actual, expected = query(encoder(x)), reference_query(reference(xx)[:, -1])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=1e-7, rtol=2e-4)
    # Parameter updates intentionally differ: two independent projections.
    assert encoder.context_input.weight[:, :84].count_nonzero() == 0
    assert encoder.context_input.weight.grad[:, :84].abs().sum() > 0


@pytest.mark.parametrize("arm", model.ARMS)
def test_private_tokens_are_causal_and_complete_window_api_rejects_wrong_shapes(arm):
    encoder = model.Encoder(arm, 42, CONFIG).eval()
    x = synthetic(1)[:, :128]
    for boundary in (4, 64, 96, 127):
        hidden = []
        hook = encoder.layers[-1].register_forward_hook(
            lambda _m, _a, value, hidden=hidden: hidden.append(value.detach().clone())
        )
        changed = x.clone()
        changed[:, boundary:] += 0.7
        try:
            with torch.no_grad():
                encoder(x)
                encoder(changed)
        finally:
            hook.remove()
        keep = encoder.token_endpoints() < boundary
        torch.testing.assert_close(hidden[0][:, keep], hidden[1][:, keep], atol=0, rtol=0)
    for shape in ((0, 128, 28), (1, 127, 28), (1, 129, 28), (1, 255, 28), (1, 128, 27)):
        with pytest.raises(ValueError):
            encoder(torch.zeros(shape))


@pytest.mark.parametrize("arm", model.ARMS)
def test_public_window_context_and_labels_audit(arm):
    report = model.audit(*model.make_models(arm, 42, "cpu", CONFIG), synthetic(1), STATS)
    assert report["passed"], report
    json.dumps(report, allow_nan=False)


def test_hybrid_endpoint_initially_ignores_non_endpoint_old_rows_but_can_learn_them():
    encoder = model.Encoder("hybrid_endpoint", 42, CONFIG)
    x = synthetic(1)[:, :128]
    changed = x.clone()
    changed[:, 0] += 0.5
    torch.testing.assert_close(encoder(x), encoder(changed), atol=0, rtol=0)
    encoder(x).square().mean().backward()
    assert encoder.context_input.weight.grad[:, :84].abs().sum() > 0
    with torch.no_grad():
        encoder.context_input.weight -= 0.1 * encoder.context_input.weight.grad
    assert not torch.equal(encoder.input_tokens(x), encoder.input_tokens(changed))


def test_contract_rejects_mutated_semantics_and_invalid_dimensions():
    encoder = model.Encoder("dense_random", 42, CONFIG)
    encoder.token_stride = 4
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
    for arm, config in (
        ("bar", CONFIG),
        ("dense_random", CONFIG | {"layers": 2}),
        ("dense_random", CONFIG | {"width": True}),
    ):
        with pytest.raises(ValueError):
            model.contract(arm, config)
