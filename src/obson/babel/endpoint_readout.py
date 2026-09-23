"""Frozen endpoint readout comparisons; no fitted statistics or trainable additions."""
import copy

import numpy as np
import torch

from . import bar_alignment as ba

PREFIXES = (32, 48, 64, 80, 96, 112, 127, 128)
TOLERANCES = {'fp32': {'atol': 5e-5, 'rtol': 2e-4}, 'fp64': {'atol': 1e-9, 'rtol': 1e-8}}


def crop_prediction(prediction, original, local):
    """Rebase predicted rows111:127 on predicted row110 (zero-based).

    Only the model's full-window output enters this adapter. In particular, no
    observed price anchor or observed target mask is used to improve a prediction.
    """
    if prediction.ndim != 3 or prediction.shape[1:] != (128, 7):
        raise ValueError('Expected a full128-by7 predicted path')
    prefixes = torch.full((len(prediction), 1), 128, device=prediction.device, dtype=torch.long)
    observed = torch.ones_like(prediction, dtype=torch.bool)
    values, _ = ba.local_targets(prediction, observed, prefixes, original, local)
    return values[:, 0]


def comparison(a, b, tolerance):
    delta = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bound = tolerance['atol']+tolerance['rtol']*b.abs()
    return dict(passed=finite and bool((delta <= bound).all()),
                max_abs=float(delta.max()) if finite else None,
                rms=float((delta.square().mean()).sqrt()) if finite else None,
                max_tolerance_ratio=float((delta/bound).max()) if finite else None,
                **tolerance)


@torch.inference_mode()
def causal_checks(model, x, tolerance):
    """Fixed inputs; test trained states and local readouts, including h128."""
    model.eval()
    full = model.core.encoder(x)
    checks = {}
    for p in PREFIXES:
        state = full[:, p-1]
        head = model.local_head(state)
        truncated = model.core.encoder(x[:, :p])[:, -1]
        checks[f'p{p}/truncated_state'] = comparison(truncated, state, tolerance)
        checks[f'p{p}/truncated_head'] = comparison(model.local_head(truncated), head, tolerance)
        if p < 128:
            altered = x.clone()
            altered[:, p:] = altered[:, p:].flip(1)+.37
            changed = model.core.encoder(altered)[:, p-1]
            checks[f'p{p}/future_state'] = comparison(changed, state, tolerance)
            checks[f'p{p}/future_head'] = comparison(model.local_head(changed), head, tolerance)
    # h128 has no future inside its original window: append deterministic fake
    # future rows to check that its state does not change when they are present.
    longer = torch.cat((x, x[:, -16:].flip(1)+.37), dim=1)
    appended = model.core.encoder(longer)[:, 127]
    checks['p128/appended_future_state'] = comparison(appended, full[:, -1], tolerance)
    checks['p128/appended_future_head'] = comparison(model.local_head(appended), model.local_head(full[:, -1]), tolerance)
    checks['p128/appended_future_global'] = comparison(model.core.decoder(appended), model.core.decoder(full[:, -1]), tolerance)
    # A reset-window model must not retain hidden state between independent calls.
    model.core.encoder(x.flip(1))
    repeated = model.core.encoder(x)
    checks['independent_call_reset'] = comparison(repeated, full, tolerance)
    alone = model.core.encoder(x[:1])[:, -1]
    checks['batch_independence'] = comparison(alone, full[:1, -1], tolerance)
    suffix = model.core.encoder(x[:, -64:])[:, -1]
    # This is intentionally NOT an equality gate: context and positional origin
    # changed. It records the distinction between reset64 and reset128 semantics.
    context = dict(context128_vs_reset64_rms=float((suffix-full[:, -1]).square().mean().sqrt()),
                   equality_required=False, streaming_equivalence_claimed=False)
    return dict(passed=all(v['passed'] for v in checks.values()), checks=checks, reset_context=context)


@torch.inference_mode()
def trained_causality(model, x):
    result = dict(fp32=causal_checks(model, x, TOLERANCES['fp32']), fp64=None)
    if result['fp32']['passed']:
        result['status'] = 'passed'
    else:
        # Do not silently enlarge FP32 tolerances. A same-weights double-precision
        # CPU replay distinguishes fused-kernel rounding from a structural leak.
        reference = copy.deepcopy(model).to(device='cpu', dtype=torch.float64)
        result['fp64'] = causal_checks(reference, x.to(device='cpu', dtype=torch.float64), TOLERANCES['fp64'])
        result['status'] = 'passed_with_fp32_drift' if result['fp64']['passed'] else 'failed'
    return result


@torch.inference_mode()
def predict(model, data, original, local, batch, device):
    model.eval()
    arrays = {k: [] for k in ('global', 'local_head', 'global_crop', 'target', 'mask', 'anchor_error_bps')}
    for start in range(0, len(data['x']), batch):
        b = {k: torch.tensor(np.asarray(v[start:start+batch]), device=device) for k, v in data.items()}
        state = model.core.encoder(b['x'])[:, -1]
        global_prediction = model.core.decoder(state)
        local_prediction = model.local_head(state).reshape(-1, 16, 7)
        prefixes = torch.full((len(state), 1), 128, device=device, dtype=torch.long)
        target, mask = ba.local_targets(b['y'], b['mask'], prefixes, original, local)
        rows = dict(global_=global_prediction, local_head=local_prediction,
                    global_crop=crop_prediction(global_prediction, original, local),
                    target=target[:, 0], mask=mask[:, 0])
        rows['global'] = rows.pop('global_')
        # Diagnostic only: the observed anchor never enters global_crop.
        rows['anchor_error_bps'] = (global_prediction[:, 110, 0]-b['y'][:, 110, 0])*original['y_scale'][0]*100
        for key, value in rows.items():
            if key != 'mask' and not torch.isfinite(value).all():
                raise ValueError('Nonfinite frozen readout')
            arrays[key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in arrays.items()}
