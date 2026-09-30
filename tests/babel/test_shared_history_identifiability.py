"""Counterexamples distinguish noncollapse from historical query semantics."""

from unittest.mock import patch

import pytest
import torch

from obson.babel import shared_history as core
from obson.babel import shared_history_identifiability as audit


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_factorization_for_nonconstant_gate_and_masked_geometry():
    reader = core.IntervalReader(16, 42, 8).double().eval()
    generator = torch.Generator().manual_seed(19)
    state = torch.randn(6, 16, generator=generator, dtype=torch.float64)
    ages = torch.tensor([[[64, 48], [16, 8], [0, 0]]] * 6)
    mask = torch.tensor([[True, True, False]] * 6)
    torch.testing.assert_close(
        reader(state, ages, mask),
        audit.factorized(reader, state, ages, mask),
        atol=1e-14,
        rtol=1e-13,
    )


def test_perfect_auxiliary_objective_can_ignore_query_entirely():
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No optimization")):
        result = audit.counterexample()
    assert result["auxiliary_total"] < 1e-20
    assert result["query_change_max_abs"] == 0
    assert result["shared_diagnostics"]["noncollapsed"]
    assert result["shared_diagnostics"]["ratio"] == 0
    assert result["shared_diagnostics"]["effective_rank"] == pytest.approx(64)
    assert result["shared_diagnostics"]["within_pair_std"] == pytest.approx(1)
    assert result["shuffled_mean_distance"] == pytest.approx(2)
    assert not result["uses_market_data"] and not result["uses_learned_checkpoint"]


def test_euclidean_gradient_agreement_is_not_preconditioned_protection():
    result = audit.preconditioner_counterexample()
    assert result["euclidean_dot"] > 0
    assert result["preconditioned_dot"] < 0
    assert result["main_only_linear_loss_change"] == pytest.approx(-10.1)
    assert result["combined_linear_loss_change"] == pytest.approx(9.5)
