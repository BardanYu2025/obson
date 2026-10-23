"""Synthetic organization paths only; no market weights or optimizer updates."""

import json

import numpy as np
import pytest
import torch

from obson.babel import observed_history as old
from obson.babel import state_organization as model

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


def enable_correction(encoder):
    if hasattr(encoder, "memory"):
        with torch.no_grad():
            encoder.memory.output.weight.copy_(0.03 * torch.eye(encoder.config["width"]))


@pytest.mark.parametrize("arm", model.ARMS)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_initial_public_function_and_inherited_parameters_gradients_match_source(
    arm, checkpointing
):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    source, source_query = old.make_models("control", 42, "cpu", CONFIG)
    encoder.checkpointing = source.checkpointing = checkpointing
    inherited = {k: v for k, v in encoder.state_dict().items() if not k.startswith("memory.")}
    assert list(inherited) == list(source.state_dict())
    assert [k for k, _ in encoder.named_parameters() if not k.startswith("memory.")] == [
        k for k, _ in source.named_parameters()
    ]
    for k, v in source.state_dict().items():
        torch.testing.assert_close(inherited[k], v, atol=0, rtol=0)
    for k, v in query.state_dict().items():
        torch.testing.assert_close(v, source_query.state_dict()[k], atol=0, rtol=0)
    x = synthetic()[:, :128].clone().requires_grad_()
    xx = x.detach().clone().requires_grad_()
    actual, expected = encoder(x), source(xx)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    query(actual).square().mean().backward()
    source_query(expected).square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=0, rtol=0)
    inherited_params = dict(encoder.named_parameters())
    for k, p in source.named_parameters():
        torch.testing.assert_close(inherited_params[k].grad, p.grad, atol=0, rtol=0)
    for p, q in zip(query.parameters(), source_query.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    if arm != "plain":
        assert encoder.memory.output.weight.grad.abs().sum() > 0
        # A zero final projection initially blocks the new internal paths only.
        for k, p in encoder.memory.named_parameters():
            if not k.startswith("output."):
                assert p.grad is not None and p.grad.count_nonzero() == 0, k


def test_candidate_parameters_match_exactly_and_write_steps_are_independent():
    late, query = model.make_models("late_memory", 42, "cpu", CONFIG)
    staged, other_query = model.make_models("staged_memory", 42, "cpu", CONFIG)
    assert list(late.state_dict()) == list(staged.state_dict())
    for left, right in ((late, staged), (query, other_query)):
        for k, v in left.state_dict().items():
            torch.testing.assert_close(v, right.state_dict()[k], atol=0, rtol=0)
    assert late.memory.writes[0] is not late.memory.writes[1]
    assert not torch.equal(
        late.memory.writes[0].cross.q_proj_weight, late.memory.writes[1].cross.q_proj_weight
    )
    for a, b in zip(
        late.memory.writes[0].parameters(), late.memory.writes[1].parameters(), strict=True
    ):
        assert a.data_ptr() != b.data_ptr()
    assert not any(isinstance(m, torch.nn.Dropout) and m.p for m in late.modules())
    assert model.model_contract(late)["memory_write_sources"] == [4, 4]
    assert model.model_contract(staged)["memory_write_sources"] == [2, 4]


@pytest.mark.parametrize("width", [32, 768])
def test_memory_parameter_count_has_no_hidden_public_capacity(width):
    memory = model.Memory(width, 42)
    # Two cross/self/FF write blocks, learned4 slots and final full-width projection.
    expected = (15 * width * width + 62 * width) // 4
    assert sum(p.numel() for p in memory.parameters()) == expected
    if width == 768:
        assert expected == 2_223_744
    assert memory.initial.shape == (4, width // 4)
    assert memory.output.in_features == memory.output.out_features == width


@pytest.mark.parametrize("arm", model.ARMS)
def test_backbone_never_reads_memory_and_depths_match_source(arm):
    encoder = model.Encoder(arm, 42, CONFIG)
    encoder.checkpointing = False
    enable_correction(encoder)
    x = synthetic()[:, :128]
    h2, h4 = encoder.depth_features(x)
    source = model.Encoder("plain", 42, CONFIG)
    source.checkpointing = False
    a, b = source.depth_features(x)
    torch.testing.assert_close(h2, a, atol=0, rtol=0)
    torch.testing.assert_close(h4, b, atol=0, rtol=0)
    changed = x.clone()
    changed[:, 64:] += 5
    c, d = encoder.depth_features(changed)
    # Internal tokens remain causal but do not each have a full trailing128.
    torch.testing.assert_close(h2[:, :64], c[:, :64], atol=0, rtol=0)
    torch.testing.assert_close(h4[:, :64], d[:, :64], atol=0, rtol=0)
    assert not torch.allclose(h4[:, -1], d[:, -1])


@pytest.mark.parametrize("arm,first_depth", [("late_memory", "h4"), ("staged_memory", "h2")])
def test_nonzero_memory_reads_declared_depth_and_carries_first_write(arm, first_depth):
    encoder = model.Encoder(arm, 42, CONFIG)
    enable_correction(encoder)
    parts = encoder.forward_parts(synthetic()[:, :128])
    m = encoder.memory
    initial = m.initial.expand(1, -1, -1)
    expected_first = m.writes[0](initial, parts[first_depth])
    expected_final = m.writes[1](expected_first, parts["h4"])
    torch.testing.assert_close(parts["memory_first"], expected_first, atol=0, rtol=0)
    torch.testing.assert_close(parts["memory_final"], expected_final, atol=0, rtol=0)
    torch.testing.assert_close(parts["state"], parts["base"] + m.output(expected_final.flatten(1)))
    reset = encoder.forward_parts(synthetic()[:, :128], reset_first=True)
    assert not torch.allclose(parts["state"], reset["state"])
    torch.testing.assert_close(parts["base"], reset["base"], atol=0, rtol=0)
    parts["memory_first"].retain_grad()
    parts["state"].square().sum().backward()
    assert parts["memory_first"].grad.abs().sum() > 0
    assert m.writes[0].cross.q_proj_weight.grad.abs().sum() > 0
    assert m.writes[1].cross.q_proj_weight.grad.abs().sum() > 0
    assert m.initial.grad.abs().sum() > 0


def test_middle_history_has_a_direct_gradient_via_first_write_without_final_history_path():
    encoder = model.Encoder("staged_memory", 42, CONFIG)
    enable_correction(encoder)
    h2, h4 = encoder.depth_features(synthetic()[:, :128])
    h2.retain_grad()
    # Stop the final-history route to isolate h2 -> first memory -> second -> z.
    encoder.memory(h2, h4.detach()).square().sum().backward()
    assert h2.grad.abs().sum() > 0
    assert encoder.layers[1].linear1.weight.grad.abs().sum() > 0
    assert encoder.layers[2].linear1.weight.grad is None
    encoder.zero_grad(set_to_none=True)
    h2, h4 = encoder.depth_features(synthetic()[:, :128])
    h2.retain_grad()
    encoder.memory.forward_parts(h2, h4.detach(), reset_first=True)[-1].square().sum().backward()
    assert h2.grad is None
    assert encoder.memory.writes[0].cross.q_proj_weight.grad is None


@pytest.mark.parametrize("arm", model.ARMS)
def test_nonzero_public_state_and_prediction_depend_only_on_own128_rows(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    enable_correction(encoder)
    x = synthetic().requires_grad_()
    ps = torch.tensor([[48]])
    z, pred = model.predict(encoder, query, x, ps)
    pred.square().mean().backward()
    assert x.grad[:, :47].count_nonzero() == 0
    assert x.grad[:, 175:].count_nonzero() == 0
    assert x.grad[:, 47:175].abs().sum() > 0
    changed = x.detach().clone()
    changed[:, :47] += 100
    changed[:, 175:] -= 100
    with torch.no_grad():
        zz, pp = model.predict(encoder, query, changed, ps)
    torch.testing.assert_close(z, zz, atol=0, rtol=0)
    torch.testing.assert_close(pred, pp, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_nonzero_batch_states_are_independent128_windows_and_query_ages_match(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    encoder.eval()
    query.eval()
    enable_correction(encoder)
    x = synthetic(2)
    ps = torch.tensor([[1, 32, 128], [2, 64, 127]])
    with torch.no_grad():
        states, pred = model.predict(encoder, query, x, ps)
        for b in range(len(x)):
            for j, endpoint in enumerate(ps[b]):
                single = encoder(x[b : b + 1, endpoint - 1 : endpoint + 127])
                torch.testing.assert_close(states[b, j], single[0], atol=2e-6, rtol=2e-5)
                ages = torch.tensor([127, 20, 1, 64])
                torch.testing.assert_close(
                    query(single, ages)[0], pred[b, j, ages - 1], atol=2e-6, rtol=2e-5
                )
        # Neither batch order nor prior unrelated call supplies persistent state.
        encoder(x[:1, :128] + 1)
        torch.testing.assert_close(model.read_states(encoder, x, ps), states, atol=0, rtol=0)
    assert states.shape == (2, 3, 32) and pred.shape == (2, 3, 127, 7)


@pytest.mark.parametrize("arm", ["late_memory", "staged_memory"])
def test_checkpointed_nonzero_paths_have_same_outputs_and_gradients(arm):
    a, b = [model.Encoder(arm, 42, CONFIG) for _ in range(2)]
    enable_correction(a)
    enable_correction(b)
    a.checkpointing, b.checkpointing = False, True
    left, right = a(synthetic()[:, :128]), b(synthetic()[:, :128])
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    left.square().sum().backward()
    right.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_public_query_receives_only_final_vector(arm):
    encoder, _ = model.make_models(arm, 42, "cpu", CONFIG)
    enable_correction(encoder)

    class StateOnly(torch.nn.Module):
        def forward(self, state):
            self.seen = state
            assert state.ndim == 2 and state.shape[-1] == 32
            return state[:, :7, None].transpose(1, 2).expand(-1, 127, -1)

    query = StateOnly()
    states, _ = model.predict(encoder, query, synthetic(), torch.tensor([[1, 128]]))
    torch.testing.assert_close(query.seen, states.flatten(0, 1), atol=0, rtol=0)


@pytest.mark.parametrize("arm", model.ARMS)
def test_audit_reports_actual_memory_scope_and_complete_window_causality(arm):
    encoder, query = model.make_models(arm, 42, "cpu", CONFIG)
    enable_correction(encoder)
    report = model.audit(encoder, query, synthetic(), STATS)
    assert report["passed"], report
    contract = report["model_contract"]
    assert contract["public_output"] == "organized_final_state_only"
    assert contract["internal_tokens_are_public_bar_states"] is False
    assert contract["private_memory_exposed"] is False
    assert contract["memory"]["backbone_reads_memory"] is False
    assert contract["memory"]["persistent_between_windows"] is False
    assert contract["additional_parameters"] == (0 if arm == "plain" else 4336)
    assert report["attention_graph"]["allowed_edges_per_layer"] == 8256
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "shape", [(1, 127, 28), (1, 129, 28), (1, 255, 28), (0, 128, 28), (128, 28), (1, 128, 27)]
)
def test_missing_or_extra_history_is_rejected(shape):
    encoder = model.Encoder("staged_memory", 42, CONFIG)
    for fn in (encoder, encoder.forward_parts, encoder.depth_features):
        with pytest.raises(ValueError, match="128"):
            fn(torch.zeros(shape))


@pytest.mark.parametrize(
    "ps",
    [
        torch.tensor([[0]]),
        torch.tensor([[129]]),
        torch.tensor([[1.0]]),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([1]),
    ],
)
def test_invalid_endpoint_positions_rejected(ps):
    encoder = model.Encoder("staged_memory", 42, CONFIG)
    with pytest.raises(ValueError, match="endpoints"):
        model.read_states(encoder, synthetic(), ps)


def test_arm_and_memory_metadata_cannot_silently_change():
    with pytest.raises(ValueError, match="declared arm"):
        model.Encoder("layer2", 42, CONFIG)
    encoder = model.Encoder("plain", 42, CONFIG)
    encoder.arm = "staged_memory"
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
    encoder = model.Encoder("staged_memory", 42, CONFIG)
    encoder.attention_mode = "full_observed"
    with pytest.raises(ValueError, match="semantics"):
        model.model_contract(encoder)
