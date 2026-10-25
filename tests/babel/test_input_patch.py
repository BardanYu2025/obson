"""Synthetic packing, original-baseline parity and endpoint dependency checks."""

import json

import numpy as np
import pytest
import torch

from obson.babel import input_patch as model
from obson.babel import uniform_context as old
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
    threads = torch.get_num_threads()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def synthetic(n=2):
    rng = np.random.default_rng(65)
    x = torch.tensor(rng.normal(0, 0.02, (n, 255, 28)), dtype=torch.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    return x


@pytest.mark.parametrize("checkpointing", [False, True])
def test_bar_outputs_inputs_parameter_gradients_and_state_match_original(checkpointing):
    encoder, query = model.make_models("bar", 42, "cpu", CONFIG)
    reference, reference_query = old.make_models("rolling", 42, "cpu", CONFIG)
    encoder.checkpointing = reference.checkpointing = checkpointing
    for a, b in ((encoder, reference), (query, reference_query)):
        assert a.state_dict().keys() == b.state_dict().keys()
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], atol=0, rtol=0)
    x = synthetic()[:, :128].clone().requires_grad_()
    xx = x.detach().clone().requires_grad_()
    actual, expected = encoder(x), reference(xx)[:, -1]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    query(actual).square().mean().backward()
    reference_query(expected).square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    for a, b in ((encoder, reference), (query, reference_query)):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_independent_ordered_packing_zero_padding_and_patch_covers_every_row_once():
    # Unique nonzero labels make a transposed feature/bar axis or dropped row visible.
    raw = np.arange(1, 2 * 128 * 28 + 1, dtype=np.float32).reshape(2, 128, 28)
    x = torch.tensor(raw)
    dense = model.Encoder("local4", 42, CONFIG)
    patch = model.Encoder("patch4", 42, CONFIG)
    expected = np.zeros((2, 128, 112), dtype=np.float32)
    for b in range(2):
        for end in range(128):
            pieces = [raw[b, row] if row >= 0 else np.zeros(28) for row in range(end - 3, end + 1)]
            expected[b, end] = np.concatenate(pieces)
    np.testing.assert_array_equal(dense.input_rows(x).numpy(), expected)
    packed = patch.input_rows(x).numpy()
    np.testing.assert_array_equal(packed, expected[:, 3::4])
    np.testing.assert_array_equal(packed.reshape(2, 128, 28), raw)
    assert dense.input_rows(x)[0, 0, :84].count_nonzero() == 0
    assert dense.input_rows(x)[0, 1, :56].count_nonzero() == 0
    assert dense.input_rows(x)[0, 2, :28].count_nonzero() == 0
    torch.testing.assert_close(model.Encoder("bar", 42, CONFIG).input_rows(x), x, atol=0, rtol=0)


def test_shared_projection_tokens_and_original_bar_positions_not_compressed_positions():
    dense = model.Encoder("local4", 42, CONFIG)
    patch = model.Encoder("patch4", 42, CONFIG)
    x = synthetic()[:, :128]
    for name, value in dense.state_dict().items():
        torch.testing.assert_close(value, patch.state_dict()[name], atol=0, rtol=0)
    ends = torch.arange(3, 128, 4)
    assert torch.equal(patch.token_endpoints(), ends)
    torch.testing.assert_close(
        dense.input_tokens(x)[:, ends], patch.input_tokens(x), atol=0, rtol=0
    )
    pe = positions(128, 32, "cpu", torch.float32)
    expected = patch.input_tokens(x, include_positions=False) + pe[ends]
    torch.testing.assert_close(patch.input_tokens(x), expected, atol=0, rtol=0)
    assert not torch.allclose(pe[ends], positions(32, 32, "cpu", torch.float32))
    # Ordered concat is not averaging: swapping two within-patch bars changes tokens.
    changed = x.clone()
    changed[:, [0, 1]] = changed[:, [1, 0]]
    assert not torch.allclose(patch.input_tokens(x)[:, 0], patch.input_tokens(changed)[:, 0])


@pytest.mark.parametrize("arm", model.ARMS)
def test_private_tokens_only_depend_on_rows_through_their_declared_endpoint(arm):
    encoder = model.Encoder(arm, 42, CONFIG).eval()
    x = synthetic(1)[:, :128]
    changed = x.clone()
    changed[:, 64:] += 0.7
    hidden = []
    handle = encoder.layers[-1].register_forward_hook(
        lambda _m, _args, output: hidden.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            encoder(x)
            encoder(changed)
    finally:
        handle.remove()
    keep = encoder.token_endpoints() < 64
    torch.testing.assert_close(hidden[0][:, keep], hidden[1][:, keep], atol=0, rtol=0)
    assert not torch.allclose(hidden[0][:, ~keep], hidden[1][:, ~keep])


@pytest.mark.parametrize("arm", model.ARMS)
def test_checkpoint_recomputation_matches_outputs_and_parameter_gradients(arm):
    a = model.Encoder(arm, 42, CONFIG)
    b = model.Encoder(arm, 42, CONFIG)
    a.checkpointing = False
    b.checkpointing = True
    x = synthetic(1)[:, :128]
    left, right = a(x), b(x)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    left.square().sum().backward()
    right.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_public_states_equal_independent_windows_and_have_no_external_dependency(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    encoder.eval()
    query.eval()
    x = synthetic(1).requires_grad_()
    ps = torch.tensor([[48]])
    z, pred = model.predict(encoder, query, x, ps)
    pred.square().sum().backward()
    assert x.grad[:, :47].count_nonzero() == 0
    assert x.grad[:, 175:].count_nonzero() == 0
    assert x.grad[:, 47:175].abs().sum() > 0
    changed = x.detach().clone()
    changed[:, :47] += 100
    changed[:, 175:] -= 100
    with torch.no_grad():
        other_z, other_pred = model.predict(encoder, query, changed, ps)
        single = encoder(x[:, 47:175])
        ages = torch.tensor([127, 20, 1, 64])
        selected = query(single, ages)
    torch.testing.assert_close(z, other_z, atol=0, rtol=0)
    torch.testing.assert_close(pred, other_pred, atol=0, rtol=0)
    torch.testing.assert_close(z[:, 0], single, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(selected, pred[:, 0, ages - 1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("arm", model.ARMS)
def test_report_audit_packing_graph_and_endpoint_boundaries(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    report = model.audit(encoder, query, synthetic(), STATS)
    assert report["passed"], report
    assert report["model_contract"] == model.contract(arm, CONFIG)
    assert report["attention_graph"]["allowed_edges_per_layer"] == (
        528 if arm == "patch4" else 8256
    )
    assert not report["model_contract"]["internal_tokens_are_public_bar_states"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "shape", [(1, 127, 28), (1, 129, 28), (1, 255, 28), (0, 128, 28), (128, 28), (1, 128, 27)]
)
def test_encoder_rejects_incomplete_extra_or_padded_public_history(shape):
    encoder = model.Encoder("patch4", 42, CONFIG)
    with pytest.raises(ValueError, match="128"):
        encoder(torch.zeros(shape))


@pytest.mark.parametrize(
    "ps", [torch.tensor([[0]]), torch.tensor([[129]]), torch.tensor([[1.0]]), torch.tensor([1])]
)
def test_invalid_endpoint_positions_rejected(ps):
    encoder = model.Encoder("patch4", 42, CONFIG)
    with pytest.raises(ValueError, match="endpoints"):
        model.read_states(encoder, synthetic(1), ps)


def test_arm_and_stride_must_be_bound_beyond_identical_weight_keys():
    encoder = model.Encoder("local4", 42, CONFIG)
    encoder.arm = "patch4"
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
    with pytest.raises(ValueError, match="declared arm"):
        model.Encoder("relative", 42, CONFIG)
    for config in (
        CONFIG | {"layers": 3},
        CONFIG | {"width": 24},
        CONFIG | {"ff": 0},
        CONFIG | {"heads": 3},
    ):
        with pytest.raises(ValueError):
            model.contract("bar", config)


def test_canonical_contract_can_be_checked_without_model_construction(monkeypatch):
    monkeypatch.setattr(
        model.Encoder, "__init__", lambda *a, **k: pytest.fail("No model construction")
    )
    a, b = model.contract("bar"), model.contract("patch4")
    assert a["additional_parameters_vs_bar"] == 0
    assert b["additional_parameters_vs_bar"] == 64512
    assert b["token_endpoints_zero_based"] == list(range(3, 128, 4))
