"""Synthetic model paths only: no market weights, training or fitting."""

import json

import numpy as np
import pytest
import torch

from obson.babel import layer_supervision as model
from obson.babel import observed_history as old

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


def synthetic(n=1):
    rng = np.random.default_rng(65)
    x = torch.tensor(rng.normal(0, 0.02, (n, 255, 28)), dtype=torch.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    return x


@pytest.mark.parametrize("arm", model.ARMS)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_final_output_keys_parameter_order_and_gradients_match_source(arm, checkpointing):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    source, source_query = old.make_models("control", 42, "cpu", CONFIG)
    encoder.checkpointing = source.checkpointing = checkpointing
    for a, b in ((encoder, source), (query, source_query)):
        assert list(a.state_dict()) == list(b.state_dict())
        assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], atol=0, rtol=0)
        # Real restoration will use exactly these keys; no new head/statistic buffers.
        a.load_state_dict(b.state_dict(), strict=True)
    x = synthetic()[:, :128].clone().requires_grad_()
    xx = x.detach().clone().requires_grad_()
    final, middle = encoder.forward_levels(x)
    actual, expected = encoder(x), source(xx)
    assert actual.shape == middle.shape == (1, 32)
    torch.testing.assert_close(final, actual, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    query(final).square().mean().backward()
    source_query(expected).square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    for a, b in ((encoder, source), (query, source_query)):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_explicit_middle_path_matches_manual_first_two_blocks_without_hooks():
    encoder = model.Encoder("layer2", 42, CONFIG)
    encoder.checkpointing = False
    x = synthetic()[:, :128]
    final, middle = encoder.forward_levels(x)
    h = encoder.input(x)
    h = h + model.positional_encoding(128, h.shape[-1], h.device, h.dtype)
    for layer in encoder.layers[:2]:
        h = layer(h, src_mask=encoder.attention_mask(128, 33, len(x), h))
    expected = encoder.coordinates(h)[:, -1]
    torch.testing.assert_close(middle, expected, atol=0, rtol=0)
    assert not torch.allclose(final, middle)
    assert all(not m._forward_hooks for m in encoder.modules())


def test_checkpoint_recomputation_matches_both_level_gradients():
    a = model.Encoder("layer2", 42, CONFIG)
    b = model.Encoder("layer2", 42, CONFIG)
    a.checkpointing, b.checkpointing = False, True
    x, xx = synthetic()[:, :128], synthetic()[:, :128]
    za, ma = a.forward_levels(x)
    zb, mb = b.forward_levels(xx)
    torch.testing.assert_close(za, zb, atol=0, rtol=0)
    torch.testing.assert_close(ma, mb, atol=0, rtol=0)
    (0.8 * za.square().mean() + 0.2 * ma.square().mean()).backward()
    (0.8 * zb.square().mean() + 0.2 * mb.square().mean()).backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_both_levels_and_query_have_no_gradient_or_value_dependency_outside_own128(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    x = synthetic().requires_grad_()
    ps = torch.tensor([[48]])
    final, middle = model.read_state_levels(encoder, x, ps)
    (query(final[:, 0]).square().mean() + query(middle[:, 0]).square().mean()).backward()
    assert x.grad[:, :47].count_nonzero() == 0
    assert x.grad[:, 175:].count_nonzero() == 0
    assert x.grad[:, 47:175].abs().sum() > 0
    changed = x.detach().clone()
    changed[:, :47] += 100
    changed[:, 175:] -= 100
    with torch.no_grad():
        other = model.read_state_levels(encoder, changed, ps)
    for left, right in zip((final, middle), other, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_batched_states_are_separate_complete_windows_and_public_output_is_final(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    x = synthetic(2)
    ps = torch.tensor([[1, 32, 128], [2, 64, 127]])
    with torch.no_grad():
        final, middle = model.read_state_levels(encoder, x, ps)
        public, pred = model.predict(encoder, query, x, ps)
        torch.testing.assert_close(final, public, atol=0, rtol=0)
        for b in range(len(x)):
            for j, endpoint in enumerate(ps[b]):
                pair = encoder.forward_levels(x[b : b + 1, endpoint - 1 : endpoint + 127])
                for batched, single in zip((final, middle), pair, strict=True):
                    torch.testing.assert_close(batched[b, j], single[0], atol=2e-6, rtol=2e-5)
    assert pred.shape == (2, 3, 127, 7)


@pytest.mark.parametrize("arm", model.ARMS)
def test_audit_declares_causal_graph_shared_projection_and_final_only_output(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    report = model.audit(*pair, synthetic(), STATS)
    assert report["passed"], report
    assert report["attention_graph"]["allowed_edges_per_layer"] == 8256
    assert report["model_contract"]["additional_parameters"] == 0
    assert report["model_contract"]["public_output"] == "final_block_only"
    assert report["model_contract"]["internal_tokens_are_public_bar_states"] is False
    assert report["model_contract"]["internal_attention_is_causal"] is True
    assert report["model_contract"]["early_tokens_have_complete_own128"] is False
    assert "internal_tokens_are_causal_bar_states" not in report["model_contract"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("shape", [(1, 127, 28), (1, 255, 28), (0, 128, 28), (128, 28)])
def test_both_entrypoints_require_complete128(shape):
    encoder = model.Encoder("layer2", 42, CONFIG)
    for call in (encoder, encoder.forward_levels):
        with pytest.raises(ValueError, match="128"):
            call(torch.zeros(shape))


def test_graph_and_arm_metadata_cannot_silently_change():
    with pytest.raises(ValueError, match="declared arm"):
        model.Encoder("full_history", 42, CONFIG)
    encoder = model.Encoder("layer2", 42, CONFIG)
    encoder.attention_mode = "full_observed"
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
