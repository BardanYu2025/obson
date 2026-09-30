"""CPU-only constructive audit: auxiliary success does not identify interval content.

No market data, learned checkpoint, fitting or optimizer is used. The constructed
states intentionally share an arbitrary nuisance across A/B/C, not bar history.
"""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import shared_history as core
from . import shared_history_evaluate as evaluate
from .ae_extend import atomic_json
from .dual_state import sha256


def factorized(reader, state, ages, mask):
    """Exact algebra of the existing reader, not a proposed new architecture."""
    safe = torch.where(mask[..., None], ages, 1).to(state.dtype) / 127
    gate = (reader.gate(safe) * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    return reader.output(torch.nn.functional.gelu(reader.content(state)) * gate)


@torch.inference_mode()
def counterexample():
    reader = core.IntervalReader(768, 42).double().eval()
    # H.T @ H = 64 I. Each adjacent nuisance-matched time pair has opposite codes.
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(6):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    state = torch.zeros(128, 768, dtype=torch.float64)
    state[::2, :64], state[1::2, :64] = h, -h
    for parameter in reader.parameters():
        parameter.zero_()
    reader.content.weight[:64, :64] = torch.eye(64, dtype=torch.float64)
    reader.output.weight[:, :64] = torch.eye(64, dtype=torch.float64) * (2 * math.sqrt(2))
    ages = torch.tensor([[[44, 42], [40, 38]], [[60, 58], [56, 54]], [[30, 29], [28, 27]]])
    ages = ages[:, None].expand(-1, 128, -1, -1)
    mask = torch.ones(128, 2, dtype=torch.bool)
    query = SimpleNamespace(
        shared=reader,
        state_mean=torch.zeros(768, dtype=torch.float64),
        state_scale=torch.ones(768, dtype=torch.float64),
    )
    features = core.reads(query, state.repeat(3, 1), ages, mask, "cross")
    loss, parts = core.auxiliary(features, torch.ones(128, dtype=torch.bool))
    changed = reader(state, ages[0] + 8, mask)
    score = evaluate.diagnostics(features[0].numpy(), features[1].numpy())
    return {
        "scope": "Constructed nuisance codes only; not measured trained-model behavior.",
        "source_sha256": {Path(m.__file__).name: sha256(m.__file__) for m in (core, evaluate)},
        "optimizer_steps": 0,
        "uses_market_data": False,
        "uses_learned_checkpoint": False,
        "auxiliary_total": float(loss),
        "components": {
            k: {n: float(v) if isinstance(v, torch.Tensor) else v for n, v in p.items()}
            for k, p in parts.items()
        },
        "query_change_max_abs": float((changed - features[0]).abs().max()),
        "factorization_max_abs": float(
            (factorized(reader, state, ages[0], mask) - features[0]).abs().max()
        ),
        "shared_diagnostics": {k: v for k, v in score.items() if k not in ("distance", "shuffled")},
        "shuffled_mean_distance": float(np.mean(score["shuffled"])),
        "conclusion": "Zero auxiliary loss and noncollapsed, discriminating outputs can coexist with complete query insensitivity. This does not pass or invalidate the original-state utility/retention gates.",
    }


def preconditioner_counterexample():
    """A Euclidean dot-product test alone cannot protect a preconditioned step.

    This is exact linear-loss arithmetic, not an Adam simulation or a claim
    against PCGrad under its own assumptions. No parameter is updated.
    """
    main = np.array([1.0, 1.0])
    auxiliary = np.array([4.0, -2.0])
    diagonal = np.array([0.1, 10.0])
    return {
        "euclidean_dot": float(main @ auxiliary),
        "preconditioned_dot": float(main @ (diagonal * auxiliary)),
        "main_only_linear_loss_change": float(-main @ (diagonal * main)),
        "combined_linear_loss_change": float(-main @ (diagonal * (main + auxiliary))),
        "scope": "Analytic positive-diagonal preconditioner; not a measured training trajectory.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = counterexample()
    result["preconditioner_example"] = preconditioner_counterexample()
    result["audit_code_sha256"] = sha256(__file__)
    atomic_json(result, args.out)
    print(f"Synthetic identifiability audit: {args.out}")


if __name__ == "__main__":
    main()
