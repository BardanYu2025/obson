"""F01/F11: full-context endpoint coverage and separately controlled remote loss."""

import numpy as np
import torch

from . import history_query as hq
from . import uniform_context as original
from . import utility_probe as up

CONFIG = original.CONFIG
ARMS = {
    "A": ("legacy", True),
    "B": ("uniform", True),
    "C": ("legacy", False),
    "D": ("uniform", False),
}
VAL_POSITIONS = (1, 16, 32, 48, 64, 80, 96, 112, 128)
make_models = original.make_models
rolling_windows = original.rolling_windows
band_rows = original.band_rows
learning_rate = original.learning_rate


def schedule(n, seed, epoch, arm):
    sampling, _ = ARMS[arm]
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261004])).permutation(n)
    if sampling == "legacy":
        ps = hq.sample_prefixes(n, seed, epoch)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261009]))
        ps = np.sort(np.argsort(rng.random((n, 128)), axis=1)[:, :5] + 1, axis=1).astype(np.int64)
    return order, ps


def read_states(encoder, x, ps):
    windows = rolling_windows(x, ps)
    return encoder(windows.flatten(0, 1))[:, -1].reshape(len(x), ps.shape[1], -1)


def predict(encoder, query, x, ps):
    z = read_states(encoder, x, ps)
    return z, query(z.flatten(0, 1)).reshape(len(x), ps.shape[1], 127, 7)


def targets(x, ps, statistics):
    """Canonical reconstruction masks reuse observed features; no future/current targets."""
    if (
        x.ndim != 3
        or x.shape[1:] != (255, 28)
        or ps.ndim != 2
        or len(ps) != len(x)
        or ps.shape[1] == 0
        or ps.dtype != torch.long
        or ps.device != x.device
        or not ((ps >= 1) & (ps <= 128)).all()
    ):
        raise ValueError("255 observed features and endpoints1..128 required")
    raw = x.double() * x.new_tensor(statistics["x_scale"], dtype=torch.float64) + x.new_tensor(
        statistics["x_mean"], dtype=torch.float64
    )
    for i in (20, 22):
        zero = np.float32(-statistics["x_mean"][i] / statistics["x_scale"][i])
        raw[..., i] = torch.where(x[..., i] == float(zero), 0.0, raw[..., i])
    geometry = raw[..., :2].sinh()
    physical = torch.cat(
        (geometry.sum(-1).cumsum(-1)[..., None], geometry[..., 1:2], raw[..., 18:23]), -1
    )
    valid = torch.ones_like(physical, dtype=torch.bool)
    valid[..., 3] = raw[..., 27] > 0
    valid[..., 4] = raw[..., 23] > 0
    valid[..., 5] = raw[..., 24] > 0
    valid[..., 6] = raw[..., 25] > 0
    suspect = ((raw[..., 23] > 0) & (raw[..., 22] == 0) & (raw[..., 20] != 0)) | (
        (raw[..., 24] > 0) & (raw[..., 21].abs() > np.arcsinh(1.0) + 1e-6)
    )
    valid[..., 4:6] &= ~suspect[..., None]
    ages = torch.arange(1, 128, device=x.device)
    index = ps[..., None] + 126 - ages
    rows = torch.arange(len(x), device=x.device)[:, None, None]
    value = physical[rows, index].clone()
    anchor = physical[torch.arange(len(x), device=x.device)[:, None], ps + 126, 0]
    value[..., 0] = (value[..., 0] - anchor[..., None]) / hq.price_scales(statistics, value)
    value[..., 1:] = (
        value[..., 1:] - value.new_tensor(statistics["y_mean"][1:])
    ) / value.new_tensor(statistics["y_scale"][1:])
    mask = valid[rows, index]
    return torch.where(mask, value, 0.0).to(x.dtype), mask


def terms(encoder, query, x, ps, stats, arm):
    _, pred = predict(encoder, query, x, ps)
    y, mask = targets(x, ps, stats)
    result = original.loss_rows(pred, y, mask, stats, True)
    result["objective"] = result["query"] + int(ARMS[arm][1]) * result["structure"]
    return {k: v.mean(1) for k, v in result.items()}


def oracle_targets(x, ps, stats):
    """Independent NumPy per-window calculation using reverse returns, not block cumsum."""
    raw = up.restore_raw(np.asarray(x), stats)
    yy = []
    mm = []
    ages = np.arange(1, 128)
    for b, positions in zip(raw, np.asarray(ps), strict=True):
        ys = []
        masks = []
        for p in positions:
            w = b[p - 1 : p + 127]
            value = np.zeros((127, 7))
            mask = np.ones((127, 7), bool)
            returns = np.sinh(w[:, :2]).sum(1)
            value[:, 0] = -np.cumsum(returns[:0:-1]) / (np.sqrt(ages) * stats["delta_scale"][0])
            observed = w[-2::-1]
            value[:, 1] = np.sinh(observed[:, 1])
            value[:, 2:] = observed[:, 18:23]
            value[:, 1:] = (value[:, 1:] - np.asarray(stats["y_mean"])[1:]) / np.asarray(
                stats["y_scale"]
            )[1:]
            for output, flag in ((3, 27), (4, 23), (5, 24), (6, 25)):
                mask[:, output] = observed[:, flag] > 0
            bad = ((observed[:, 23] > 0) & (observed[:, 22] == 0) & (observed[:, 20] != 0)) | (
                (observed[:, 24] > 0) & (np.abs(observed[:, 21]) > np.arcsinh(1) + 1e-6)
            )
            mask[bad, 4:6] = False
            ys.append(np.where(mask, value, 0))
            masks.append(mask)
        yy.append(ys)
        mm.append(masks)
    return np.asarray(yy, dtype=np.float32), np.asarray(mm)


@torch.no_grad()
def audit(encoder, query, x, stats):
    encoder.eval()
    query.eval()
    ps = torch.tensor([1, 2, 31, 32, 127, 128], device=x.device).expand(len(x), -1)
    y, m = targets(x, ps, stats)
    oy, om = oracle_targets(x.cpu().numpy(), ps.cpu().numpy(), stats)
    checks = {
        "independent_labels": bool(np.allclose(y.cpu().numpy(), oy, atol=1e-5, rtol=2e-5)),
        "independent_masks": bool(np.array_equal(m.cpu().numpy(), om)),
    }
    z, pred = predict(encoder, query, x, ps)
    for j, p in enumerate(ps[0].tolist()):
        single = encoder(x[:, p - 1 : p + 127])[:, -1]
        checks[f"rolling_{p}"] = bool(torch.allclose(z[:, j], single, atol=1e-4, rtol=2e-4))
        changed = x.clone()
        changed[:, p + 127 :] += 0.7
        zz = read_states(encoder, changed, ps[:, j : j + 1])[:, 0]
        cy, cm = targets(changed, ps[:, j : j + 1], stats)
        checks[f"future_{p}"] = bool(
            torch.allclose(single, zz, atol=1e-4, rtol=2e-4)
            and torch.equal(y[:, j : j + 1], cy)
            and torch.equal(m[:, j : j + 1], cm)
        )
        one = query(z[:, j])
        checks[f"query_batch_{p}"] = bool(torch.allclose(one, pred[:, j], atol=1e-4, rtol=2e-4))
    single_age = query(z[:, -1], torch.tensor([20], device=x.device))
    checks["single_age20"] = bool(
        torch.allclose(single_age, pred[:, -1, 19:20], atol=1e-4, rtol=2e-4)
    )
    # Historical compatible positions must retain the exact old target contract.
    oldy, oldm = original.targets(x, ps[:, 3:], stats)
    checks["old_targets"] = bool(torch.equal(y[:, 3:], oldy) and torch.equal(m[:, 3:], oldm))
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "label_max_abs": float(np.max(np.abs(y.cpu().numpy() - oy))),
    }
