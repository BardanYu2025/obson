"""F11 source-bound historical state API; estimates and observed audits stay separate."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import state_readability as reading
from . import state_readability_run as prior
from .holdout_audit import read_json

SCHEMA = "babel-history-state-api-v1"
SOURCE_COMPLETION = "cd2b0fe45a4faa4a10abe66b75d56f8ca91c0df2ea3e91d337fc95ffad21e477"
MODELS = ("macro_s42", "macro_s43", "D_s42", "D_s43")
FLOOR = 1e-8
LOW_RMS = 1e-4


def token(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def path_outputs(path):
    values = reading.descriptors(path)
    rms = np.column_stack(
        [np.sqrt(np.mean(np.diff(path[:, -h:], axis=1) ** 2, axis=1)) for h in (16, 64)]
    )
    directions = np.where(values[:, [0, 3]] < -0.2, 0, np.where(values[:, [0, 3]] > 0.2, 2, 1))
    return {
        "descriptors": values,
        "direction": directions,
        "rms_log_return": rms,
        "rms_bps": rms * 10000,
        "estimated_low_motion": rms <= LOW_RMS,
    }


def observed_audit(predicted_path, normalized_window, statistics):
    """Optional scoring of already observed history, never an input to the query."""
    x = np.asarray(normalized_window)
    if (
        x.ndim != 3
        or x.shape[1:] != (128, 28)
        or len(x) != len(predicted_path)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Matched finite normalized128x28 observations required")
    raw = prior.ev.up.restore_raw(x, statistics)
    truth = reading.true_path(raw)
    actual = path_outputs(truth)
    guessed = path_outputs(predicted_path)
    rms = actual["rms_log_return"]
    return {
        "source": "observed_history_not_model_output",
        "true_path": truth,
        "true_descriptors": actual["descriptors"],
        "true_rms": rms,
        "true_direction": actual["direction"],
        "stratum": np.where(rms <= FLOOR, 0, np.where(rms <= LOW_RMS, 1, 2)),
        "descriptor_error": guessed["descriptors"] - actual["descriptors"],
        "rms_error": guessed["rms_log_return"] - rms,
        "path_rmse": np.column_stack(
            [
                np.sqrt(np.mean((predicted_path[:, -h:] - truth[:, -h:]) ** 2, axis=1))
                for h in (16, 64)
            ]
        ),
    }


def verify_source(root):
    root = Path(root).resolve()
    if prior.sha256(root / "completion.json") != SOURCE_COMPLETION:
        raise ValueError("Expected reviewed readability source completion")
    done = read_json(root / "completion.json")
    prior.data.source.old.verify_files(root, done["files"])
    report = read_json(root / "manifest.json")
    if report["implementation"] != prior.implementation() or report["protocol"] != prior.protocol():
        raise ValueError("Readability implementation/protocol changed")
    coverage = Path(report["source"]).resolve()
    if prior.sha256(coverage / "completion.json") != prior.COMPLETION:
        raise ValueError("Original coverage completion changed")
    prior.data.source.old.verify_files(coverage, read_json(coverage / "completion.json")["files"])
    meta = read_json(coverage / "manifest.json")
    if meta["implementation"] != prior.source_run.implementation():
        raise ValueError("Original coverage implementation changed")
    for filename, key in (
        ("manifest.json", "source_manifest_sha256"),
        ("model_selection_lock.json", "model_lock_sha256"),
        ("readout_lock.json", "readout_lock_sha256"),
    ):
        if prior.sha256(coverage / filename) != report[key]:
            raise ValueError("Source lock changed")
    return coverage, meta


class HistoryStateReader:
    """Consumes prepared features or explicitly identified states; no raw-price bypass.

    Direct construction supports synthetic tests. Use open_reader for real weights.
    This is a research feature-window API, not a raw CSV or live-bar adapter.
    """

    def __init__(self, encoder, query, statistics, identity, device):
        self.encoder = encoder.to(device).eval().requires_grad_(False)
        self.query = query.to(device).eval().requires_grad_(False)
        self.statistics = json.loads(json.dumps(statistics))
        self.identity = dict(identity)
        self.device = torch.device(device)
        for name in ("x_mean", "x_scale"):
            a = np.asarray(statistics[name])
            if (
                a.shape != (28,)
                or not np.isfinite(a).all()
                or (name.endswith("scale") and (a <= 0).any())
            ):
                raise ValueError("Invalid source feature statistics")
        self.input_signature = token(
            {
                "schema": SCHEMA,
                "statistics": self.statistics,
                "features": "causal28_ema8_32_activity_v1",
            }
        )
        self.state_signature = token({"identity": self.identity, "input": self.input_signature})

    def _check_frozen(self):
        if any(
            m.training or any(p.requires_grad for p in m.parameters())
            for m in (self.encoder, self.query)
        ):
            raise ValueError("Source models must remain eval and frozen")

    @torch.no_grad()
    def decode_state(self, states, *, state_signature):
        self._check_frozen()
        z = np.asarray(states)
        if state_signature != self.state_signature:
            raise ValueError("State belongs to a different model/normalization identity")
        if (
            z.ndim != 2
            or z.shape[1] != self.identity["width"]
            or not len(z)
            or z.dtype != np.float32
            or not np.isfinite(z).all()
        ):
            raise ValueError("Expected finite float32 states with declared width")
        with prior.r0.replay_runtime("context"):
            q = self.query(torch.as_tensor(z, device=self.device)).cpu().numpy()
        path = reading.close_path(q, self.statistics)
        return {
            "query": q,
            "relative_log_close": path,
            **path_outputs(path),
            "metadata": {
                "schema": SCHEMA,
                "identity": dict(self.identity),
                "state_signature": self.state_signature,
                "input_signature": self.input_signature,
                "horizons": [16, 64],
                "direction_classes": ["down", "mixed", "up"],
                "scope": "past-only endpoint history; not a forecast or trading instruction",
                "current_zero_is_coordinate": True,
                "low_motion_is_estimated_not_reliability": True,
                "confidence_calibrated": False,
                "quality_assured": False,
            },
        }

    @torch.no_grad()
    def read_prepared(self, windows, endpoints, *, input_signature, audit_observed=False):
        self._check_frozen()
        x = np.asarray(windows)
        if input_signature != self.input_signature:
            raise ValueError("Prepared features require the exact source normalization signature")
        if (
            x.ndim != 3
            or x.shape[1:] != (128, 28)
            or not len(x)
            or x.dtype != np.float32
            or not np.isfinite(x).all()
        ):
            raise ValueError("Expected finite float32 normalized N x128x28 features, not raw bars")
        if len(endpoints) != len(x):
            raise ValueError("Every window requires its own closed endpoint identity")
        for row in endpoints:
            parts = row["key"].split("/")
            if (
                len(parts) != 3
                or not all(parts)
                or row["period"] not in (15, 30, 60)
                or parts[1] != str(row["period"])
                or row.get("closed") is not True
                or not row.get("feature_history_origin")
            ):
                raise ValueError(
                    "Explicit single contract, supported period, closed bar and feature origin required"
                )
            datetime.fromisoformat(row["end"])
        with prior.r0.replay_runtime("context"):
            z = self.encoder(torch.as_tensor(x, device=self.device))[:, -1].cpu().numpy()
        result = self.decode_state(z, state_signature=self.state_signature)
        result["embedding"] = z
        result["metadata"]["endpoints"] = [dict(r) for r in endpoints]
        result["metadata"]["feature_context"] = (
            "128 rows; causal EMA may carry earlier history; caller supplies prepared features"
        )
        if audit_observed:
            result["observed_audit"] = observed_audit(
                result["relative_log_close"], x, self.statistics
            )
        return result


def open_reader(source, model, device="cuda"):
    if model not in MODELS:
        raise ValueError(
            "Explicit retained Macro/D and seed42/43 required; no best-model selection"
        )
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Real-weight execution belongs on AutoDL CUDA")
    coverage, meta = verify_source(source)
    e, q = prior.ev.load(meta, coverage, model, device)
    return HistoryStateReader(
        e,
        q,
        meta["statistics"],
        {"source_completion": SOURCE_COMPLETION, "model": model, "width": 768},
        device,
    )


def boundary_report(path, audit, rows):
    """Descriptive strata; no removal, recalibration, new model ranking or confidence fit."""
    estimated = path_outputs(path)
    report = {}
    for j, h in enumerate((16, 64)):
        pieces = {}
        for group, name in enumerate(("floor", "low_to_1bp", "above_1bp")):
            ids = np.flatnonzero(audit["stratum"][:, j] == group)
            months = len({rows[i]["month"] for i in ids})
            if not len(ids):
                pieces[name] = {"support": 0, "months": 0, "evidence_sufficient": False}
                continue
            delta = audit["descriptor_error"][ids, 3 * j + 2]
            pieces[name] = {
                "support": len(ids),
                "months": months,
                "evidence_sufficient": len(ids) >= 100 and months >= 6,
                "log_rms_mse": float(np.mean(delta**2)),
                "physical_rms_mae_bps": float(np.abs(audit["rms_error"][ids, j]).mean() * 10000),
                "path_rmse_bps_mean": float(audit["path_rmse"][ids, j].mean() * 10000),
                "direction": prior.ev.up.probe.classification(
                    estimated["descriptors"][ids, 3 * j], audit["true_descriptors"][ids, 3 * j]
                ),
            }
        actual_low = audit["true_rms"][:, j] <= LOW_RMS
        predicted_low = estimated["estimated_low_motion"][:, j]
        matrix = [
            [int(np.sum((actual_low == a) & (predicted_low == p))) for p in (False, True)]
            for a in (False, True)
        ]
        report[str(h)] = {
            "strata": pieces,
            "estimated_low_motion_confusion": matrix,
            "confusion_rows": ["observed_above_1bp", "observed_at_most_1bp"],
            "confusion_columns": ["estimated_above_1bp", "estimated_at_most_1bp"],
            "all_windows_retained": len(rows),
            "quality_certified": False,
        }
    return report
