"""V06: separate visible context from target support; preserve historical losses."""

from dataclasses import dataclass

import numpy as np
import torch

from . import context_transfer as old
from . import recovery_audit as audit
from .ae_extend import atomic_json


@dataclass(frozen=True)
class Policy:
    native_input: bool
    native_target: bool


ARMS = {
    "A_prefix_common": Policy(True, True),
    "B_full_common": Policy(False, True),
    "C_full_extended": Policy(False, False),
}


def targets(x, prefixes, stats, arm):
    return old.original.targets(x, prefixes, stats, ARMS[arm].native_target)


def forward(encoder, query, x, prefixes, stats, arm):
    policy = ARMS[arm]
    z, pred = old.predict(encoder, query, x, prefixes, policy.native_input)
    y, mask = targets(x, prefixes, stats, arm)
    return z, pred, y, mask


def objective(pred, y, mask, stats):
    # Addition and reduction order match the old actual training step.
    parts = old.original.loss_rows(pred, y, mask, stats, True)
    return {k: parts[k].mean(1) for k in ("query", "structure", "objective")}


def terms(encoder, query, x, prefixes, stats, arm, observer=None):
    z, pred, y, mask = forward(encoder, query, x, prefixes, stats, arm)
    result = objective(pred, y, mask, stats)
    if observer is not None:
        observer(arm, x, prefixes, pred, y, mask, result)
    return result


def control_audit(encoder, query, x, stats, path, prefixes=None):
    """Audit the actual loss route, including masks and loss Jacobians, without updates."""
    checks, captures = {}, {}
    ps = (
        torch.tensor(old.PREFIXES, device=x.device).expand(len(x), -1)
        if prefixes is None
        else prefixes
    )

    def capture(arm, data, prefixes, pred, y, mask, result):
        # Capture from terms(), not a separately reimplemented target pipeline.
        captures[arm] = (pred.detach(), y.detach(), mask.detach(), result)

    for arm in ARMS:
        with torch.no_grad():
            terms(encoder, query, x, ps, stats, arm, capture)
    for arm, (pred, y, mask, actual) in captures.items():
        expected = objective(pred, y, mask, stats)
        for key in expected:
            checks[f"actual_loss/{arm}/{key}"] = {
                "passed": bool(torch.equal(actual[key], expected[key]))
            }
    a, b, c = (captures[k] for k in ARMS)
    for label, actual, expected in (("AB_target", a[1], b[1]), ("AB_mask", a[2], b[2])):
        checks[label] = {"passed": bool(torch.equal(actual, expected))}
    checks["BC_prediction_same_input"] = {"passed": bool(torch.equal(b[0], c[0]))}
    # Probe equal effective reduction/weights with deliberately nonzero errors.
    sentinel = a[1] + torch.linspace(-0.7, 0.9, 127, device=x.device)[None, None, :, None]
    gradients, values = [], []
    for _, y, mask, _ in (a, b):
        pred = sentinel.clone().requires_grad_(True)
        loss = objective(pred, y, mask, stats)["objective"].sum()
        gradients.append(torch.autograd.grad(loss, pred)[0])
        values.append(loss.detach())
    checks["AB_effective_weight_jacobian"] = {"passed": bool(torch.equal(*gradients))}
    checks["AB_same_prediction_loss"] = {"passed": bool(torch.equal(*values))}
    checks["C_expands_only_target_support"] = {
        "passed": bool((a[2] <= c[2]).all() and (c[2].sum() > a[2].sum()))
    }
    checks["BC_common_label_values"] = {"passed": bool(torch.equal(b[1][b[2]], c[1][b[2]]))}
    # At position128 all policies coincide, including support and coordinates.
    for i, candidate in enumerate((b, c), 1):
        checks[f"endpoint_target_{i}"] = {
            "passed": bool(torch.equal(a[1][:, -1], candidate[1][:, -1]))
        }
        checks[f"endpoint_mask_{i}"] = {
            "passed": bool(torch.equal(a[2][:, -1], candidate[2][:, -1]))
        }
        checks[f"endpoint_prediction_{i}"] = audit.compare(
            a[0][:, -1].cpu().numpy(),
            candidate[0][:, -1].cpu().numpy(),
            audit.STATE_ATOL,
            audit.STATE_RTOL,
        )
    # Each arm's historical terms must ignore bars after the queried endpoint.
    for prefix in (32, 64, 96):
        selected = torch.full((len(x), 1), prefix, device=x.device, dtype=torch.long)
        changed = x.clone()
        changed[:, prefix + 127 :] += 0.375
        for arm in ARMS:
            with torch.no_grad():
                before = forward(encoder, query, x, selected, stats, arm)
                after = forward(encoder, query, changed, selected, stats, arm)
            for field, v, w in zip(
                ("state", "prediction", "target", "mask"), before, after, strict=True
            ):
                checks[f"causal/{arm}/{prefix}/{field}"] = (
                    audit.compare(
                        v.cpu().numpy(), w.cpu().numpy(), audit.STATE_ATOL, audit.STATE_RTOL
                    )
                    if field in ("state", "prediction")
                    else {"passed": bool(torch.equal(v, w))}
                )
    atomic_json(
        {
            "checks": checks,
            "optimizer_updates": 0,
            "prefixes": ps.detach().cpu().tolist(),
            "scope": "Actual objective route; no independent source scaler refit",
        },
        path,
    )
    audit.require(checks, "V06 actual training controls")
    return checks


def supports(mask):
    result = {
        "price": mask[..., :2].any(-1).any(-1),
        "activity": mask[..., 2:].any(-1).any(-1),
    }
    for name, (lo, hi) in zip(("near", "mid", "far"), old.original.hq.BANDS, strict=True):
        result[name] = mask[..., lo - 1 : hi, :2].any(-1).any(-1)
    # Same remote pools as historical structure metric; avoid scoring absent bands as zero.
    hs = old.original.hs
    valid = []
    for width in hs.WIDTHS:
        _, ok, _ = hs.pools(
            torch.zeros_like(mask[..., 0], dtype=torch.float32), mask[..., 0], width
        )
        valid.append(ok.any(-1))
    result["structure"] = torch.stack(valid, -1).any(-1)
    return result


@torch.no_grad()
def evaluate(encoder, query, x, stats, micro, device):
    """All models get the same three evaluation policies, with supported positions explicit."""
    encoder.eval()
    query.eval()
    arrays, valid = {}, {}
    for start in range(0, len(x), micro):
        batch = torch.as_tensor(x[start : start + micro], device=device)
        ps = torch.tensor(old.PREFIXES, device=device).expand(len(batch), -1)
        for arm in ARMS:
            _, pred, y, mask = forward(encoder, query, batch, ps, stats, arm)
            values = old.original.band_rows(pred, y, mask, stats)
            support = supports(mask)
            for metric in old.METRICS:
                key = arm + "/" + metric
                arrays.setdefault(key, []).append(values[metric].cpu().numpy())
                valid.setdefault(key, []).append(support[metric].cpu().numpy())
    return (
        {k: np.concatenate(v) for k, v in arrays.items()},
        {k: np.concatenate(v) for k, v in valid.items()},
    )
