"""Synthetic graph and endpoint-causality tests; no market-weight execution."""

import json

import numpy as np
import pytest
import torch

from obson.babel import observed_history as model
from obson.babel import uniform_context as old

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
def test_control_forward_gradients_and_state_dict_match_original_rolling(checkpointing):
    encoder, query = model.make_models("control", 42, "cpu", CONFIG)
    reference, reference_query = old.make_models("rolling", 42, "cpu", CONFIG)
    encoder.checkpointing = reference.checkpointing = checkpointing
    for a, b in ((encoder, reference), (query, reference_query)):
        assert a.state_dict().keys() == b.state_dict().keys()
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], atol=0, rtol=0)
    x = synthetic()[:, :128].clone().requires_grad_()
    xx = x.detach().clone().requires_grad_()
    actual, expected = encoder(x), reference(xx)[:, -1]
    assert actual.shape == (2, CONFIG["width"])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    query(actual).square().mean().backward()
    reference_query(expected).square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    for a, b in ((encoder, reference), (query, reference_query)):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_only_graph_differs_first_layer_endpoint_same_but_internal_early_tokens_not():
    control, _ = model.make_models("control", 42, "cpu", CONFIG)
    full, _ = model.make_models("full_history", 42, "cpu", CONFIG)
    x = synthetic()[:, :128]
    hidden = {}
    handles = []
    for name, encoder in (("control", control), ("full_history", full)):
        encoder.eval()
        handles.append(
            encoder.layers[0].register_forward_hook(
                lambda _m, _args, output, name=name: hidden.update({name: output.detach().clone()})
            )
        )
    with torch.no_grad():
        c, f = control(x), full(x)
    for handle in handles:
        handle.remove()
    # The final row already reads all128 tokens in either graph in layer1.
    torch.testing.assert_close(
        hidden["control"][:, -1], hidden["full_history"][:, -1], atol=0, rtol=0
    )
    assert not torch.allclose(hidden["control"][:, 0], hidden["full_history"][:, 0])
    assert not torch.allclose(c, f)
    masks = [e.attention_mask(128, 33, len(x), x) for e in (control, full)]
    assert all(m.dtype == torch.bool and m.device == x.device for m in masks)
    assert masks[0].shape == masks[1].shape == (128, 128)
    assert [int((~m).sum()) for m in masks] == [8256, 16384]


@pytest.mark.parametrize("arm", model.ARMS)
def test_internal_early_tokens_have_only_the_declared_causality(arm):
    encoder = model.Encoder(arm, 42, CONFIG).eval()
    x = synthetic(1)[:, :128]
    changed = x.clone()
    changed[:, 64:] += 0.7
    hidden = []
    handle = encoder.layers[0].register_forward_hook(
        lambda _m, _args, output: hidden.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            encoder(x)
            encoder(changed)
    finally:
        handle.remove()
    if arm == "control":
        torch.testing.assert_close(hidden[0][:, :64], hidden[1][:, :64], atol=0, rtol=0)
    else:
        # These private internal tokens cannot be published as historical
        # bar states: they already depend on later observed bars in this window.
        assert not torch.allclose(hidden[0][:, :64], hidden[1][:, :64])


def test_full_attention_checkpoint_recomputation_preserves_outputs_and_gradients():
    a = model.Encoder("full_history", 42, CONFIG)
    b = model.Encoder("full_history", 42, CONFIG)
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
def test_batch_endpoints_equal_separate128_windows_and_query_subsets(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    encoder.eval()
    query.eval()
    x = synthetic()
    ps = torch.tensor([[1, 32, 128], [2, 64, 127]])
    with torch.no_grad():
        states, pred = model.predict(encoder, query, x, ps)
        for b in range(len(x)):
            for j, endpoint in enumerate(ps[b]):
                single = encoder(x[b : b + 1, endpoint - 1 : endpoint + 127])
                torch.testing.assert_close(states[b, j], single[0], atol=2e-6, rtol=2e-5)
                ages = torch.tensor([127, 20, 1, 64])
                q = query(single, ages)[0]
                torch.testing.assert_close(q, pred[b, j, ages - 1], atol=2e-6, rtol=2e-5)
    assert states.shape == (2, 3, 32) and pred.shape == (2, 3, 127, 7)


@pytest.mark.parametrize("arm", model.ARMS)
def test_selected_endpoint_has_zero_gradient_and_value_dependency_outside_its_window(arm):
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
    torch.testing.assert_close(z, other_z, atol=0, rtol=0)
    torch.testing.assert_close(pred, other_pred, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_report_audit_declares_actual_graph_and_semantic_limits(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    report = model.audit(encoder, query, synthetic(), STATS)
    assert report["passed"], report
    assert report["model_contract"]["endpoint_only"]
    assert report["model_contract"]["attention_mode"] == model.ATTENTION_MODES[arm]
    assert report["attention_graph"]["allowed_edges_per_layer"] == (
        8256 if arm == "control" else 16384
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "shape", [(1, 127, 28), (1, 129, 28), (1, 255, 28), (0, 128, 28), (128, 28), (1, 128, 27)]
)
def test_public_encoder_rejects_missing_history_padding_and_sequence_output_requests(shape):
    encoder = model.Encoder("full_history", 42, CONFIG)
    with pytest.raises(ValueError, match="128"):
        encoder(torch.zeros(shape))


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([[0]]),
        torch.tensor([[129]]),
        torch.tensor([[1.0]]),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([1]),
    ],
)
def test_invalid_positions_rejected_before_encoder(positions):
    encoder = model.Encoder("full_history", 42, CONFIG)
    with pytest.raises(ValueError, match="endpoints"):
        model.read_states(encoder, synthetic(1), positions)


def test_attention_metadata_is_bound_separately_from_identical_tensor_keys():
    with pytest.raises(ValueError, match="declared arm"):
        model.Encoder("relative", 42, CONFIG)
    encoder = model.Encoder("full_history", 42, CONFIG)
    contract = model.model_contract(encoder)
    assert not contract["internal_tokens_are_causal_bar_states"]
    encoder.attention_mode = "causal"
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
