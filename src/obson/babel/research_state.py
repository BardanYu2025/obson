"""Pinned control600 research interface; rolling endpoints, never a forecast."""

import copy
from pathlib import Path

import numpy as np
import torch

from . import bar_alignment as ba
from . import local_warmstart_run as warm
from . import utility_probe as up
from . import window_state as ws
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json

SCHEMA = "babel-control600-state-v1"


def code_identity():
    return (
        warm.code_identity()
        | ws.code_identity()
        | {
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("research_state_cli.py"))
        }
    )


def restore_model(checkpoint):
    config = checkpoint["config"]
    state = checkpoint["model"]
    pca = {
        "components": state["core.decoder.basis"].numpy(),
        "mean": state["core.decoder.target_mean"].numpy(),
    }
    scales = {
        "mean": state["core.decoder.coordinate_mean"].tolist(),
        "scale": state["core.decoder.coordinate_scale"].tolist(),
    }
    model = ba.AlignedStudent(
        config, checkpoint["seed"], pca, scales, hidden=state["local_head.0.weight"].shape[0]
    )
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


class ResearchState(ws.WindowState):
    def __init__(self, model, statistics, local, utility, identity, key, period, device="cpu"):
        super().__init__(model.core, statistics, local, identity, key, period, device)
        self.aligned = model.to(device).eval().requires_grad_(False)
        self.utility = copy.deepcopy(utility)

    def _metadata(self, rows, count):
        result = super()._metadata(rows, count)
        result.update(
            schema=SCHEMA,
            quality_scope="control600 research candidate; full utility protocol not passed; rolling-bar quality not newly established",
            embedding_width=self.model.encoder.coordinates.out_features,
            reconstruction="Parallel historical decoding; current bar excluded, no autoregressive forecast",
            automatic_promotion=False,
        )
        return result

    @torch.inference_mode()
    def _result(self, rows, count):
        result = super()._result(rows, count)
        if not result["metadata"]["ready"]:
            return dict(result, original_local=None, readouts=None)
        z = torch.tensor([result["embedding"]], device=self.device)
        local = self.aligned.local_head(z).reshape(16, 7).cpu().numpy()
        values = local * np.asarray(self.local["scale"]) + np.asarray(self.local["mean"])
        u = self.utility
        readouts = up.predict(
            u["heads"],
            np.asarray(u["weights"]),
            np.asarray(u["intercepts"]),
            z.cpu().numpy(),
            u["target_stats"],
        )[0]
        if not np.isfinite(values).all() or not np.isfinite(readouts).all():
            raise ValueError("Nonfinite original local/readout output")
        # Local head has only relative paths; no observed anchor is supplied to it.
        result["original_local"] = {
            "channels": list(ws.ar.NAMES),
            "values": values.tolist(),
            "bar_starts": result["recent"]["bar_starts"],
            "valid_mask": result["recent"]["valid_mask"],
            "price_semantics": "Relative log-percent from the close preceding these16 bars; no absolute-price prediction",
            "current_bar_included": False,
        }
        result["readouts"] = {
            "values": dict(zip(up.NAMES, readouts.tolist(), strict=True)),
            "semantics": "Frozen linear estimates of observed history/current descriptors, not future returns or trading signals",
            "full_utility_protocol_passed": False,
        }
        return result


def load_bundle(bundle, seed, key, period, device="cpu", *, _audit=False):
    if torch.device(device).type == "cuda":
        ws.ea.bb.ab.configure_runtime()
    bundle = Path(bundle)
    index = read_json(bundle / "index.json")
    if index["schema"] != SCHEMA or index["code_sha256"] != code_identity():
        raise ValueError("Research bundle implementation differs")
    verify_files(bundle, index["files"])
    if not _audit:
        cert = read_json(bundle / "validation.json")
        if cert["status"] != "passed" or cert["index_sha256"] != sha256(bundle / "index.json"):
            raise ValueError("Unvalidated research bundle")
    if str(seed) not in index["models"]:
        raise ValueError("Explicit retained seed42 or43 required")
    name = index["models"][str(seed)]
    ck = torch.load(bundle / name, map_location="cpu", weights_only=True)
    if ck["schema"] != SCHEMA or ck["seed"] != seed:
        raise ValueError("Research checkpoint identity differs")
    identity = dict(
        bundle_index=sha256(bundle / "index.json"),
        model_sha256=index["files"][name],
        **ck["identity"],
    )
    return ResearchState(
        restore_model(ck),
        ck["statistics"],
        ck["local"],
        ck["utility"],
        identity,
        key,
        period,
        device,
    )


def certify(bundle):
    atomic_json(
        {
            "status": "passed",
            "index_sha256": sha256(bundle / "index.json"),
            "scope": "Source endpoint replay plus fixed raw rolling sequences; not a model promotion",
        },
        bundle / "validation.json",
    )
