"""Fresh matched-backbone capacity experiments; no forecast or objective changes."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .ae_context import feature_names
from .ae_diagnostics import fingerprint
from .ae_extend import atomic_save, publish, restore_rng, rng_state, run_epochs, validation_report
from .data import manifest
from .history_autoencoder import CHANNELS, SCHEMA, WEIGHTS, HistoryAE, HistoryWindows
from .progress import progress


def config(latent):
    if latent not in (128, 256):
        raise ValueError("Capacity latent must be 128 or 256")
    return dict(hidden=256, layers=6, heads=8, latent=latent, window=128, dropout=.1, context="ema8_32")


def build_model(latent, seed):
    """Both arms share exact initial input/encoder/decoder weights and subsequent RNG."""
    torch.manual_seed(seed)
    reference = HistoryAE(**config(128))
    after = torch.get_rng_state()
    if latent == 128:
        return reference
    model = HistoryAE(**config(latent))
    for name in ("input", "encoder", "decoder", "output", "context_input"):
        getattr(model, name).load_state_dict(getattr(reference, name).state_dict())
    torch.set_rng_state(after)
    return model


def check_metadata(metadata, data_manifest, latent, epochs, batch_size, seed):
    if (metadata.get("training_family") != "capacity" or metadata["schema"] != SCHEMA
            or metadata["config"] != config(latent) or metadata["manifest"] != data_manifest
            or metadata["epochs"] != epochs or metadata["batch_size"] != batch_size
            or metadata["seed"] != seed or metadata["stride"] != 16
            or tuple(metadata["weights"]) != WEIGHTS or metadata["delta_loss_weight"] != .5
            or list(metadata["features"]) != list(feature_names("ema8_32"))):
        raise ValueError("Capacity resume data/config/budget/objective mismatch")


def train_capacity(series, encoded, bounds, directory, epochs, batch_size, seed, device,
                   latent, baseline_run=None, resume=False):
    directory = Path(directory)
    data_manifest = manifest(series, bounds)
    if resume:
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if state.get("resume_schema") != "babel-ae-resume-v1":
            raise ValueError("Full last.pt required")
        metadata = state["metadata"]
        check_metadata(metadata, data_manifest, latent, epochs, batch_size, seed)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Capacity run directory must be empty")
        if baseline_run is None:
            raise ValueError("Capacity experiment requires the frozen 200-epoch baseline run")
        baseline_run = Path(baseline_run)
        baseline = json.loads((baseline_run / "manifest.json").read_text())
        if (baseline["schema"] != SCHEMA or baseline["config"] != HistoryAE(context="ema8_32").config
                or baseline["manifest"] != data_manifest or baseline["epochs"] != epochs
                or baseline["batch_size"] != batch_size or baseline["seed"] != seed
                or baseline["stride"] != 16 or tuple(baseline["weights"]) != WEIGHTS
                or baseline["delta_loss_weight"] != .5):
            raise ValueError("Capacity baseline data/config/budget/objective mismatch")
        history = [json.loads(line) for line in (baseline_run / "history.jsonl").read_text().splitlines()]
        if max(row["epoch"] for row in history) != epochs:
            raise ValueError("Baseline budget is incomplete")
        metadata = dict(schema=SCHEMA, training_family="capacity", training_mode="from_scratch",
                        config=config(latent), context="ema8_32", features=feature_names("ema8_32"),
                        channels=CHANNELS, weights=WEIGHTS, delta_loss_weight=.5,
                        manifest=data_manifest, epochs=epochs, batch_size=batch_size, seed=seed, stride=16,
                        optimizer={"name":"AdamW", "lr":3e-4, "weight_decay":.01},
                        baseline_artifacts_sha256={name:fingerprint(baseline_run / name) for name in
                                                  ("manifest.json", "history.jsonl", "ae_metrics.json", "best.pt")},
                        selection="original validation reconstruction loss only; test after training",
                        side_information="preceding-window price anchor for physical decode only",
                        comparison="two fresh wide models; old small baseline had a warm-start reset at budget epoch 31",
                        timing="per-epoch train and validation seconds; excludes checkpoint I/O, setup and final evaluation",
                        runtime={"torch":str(torch.__version__), "device":str(device),
                                 "gpu":torch.cuda.get_device_name() if str(device).startswith("cuda") else None})
    random.seed(seed)
    np.random.seed(seed)
    model = build_model(latent, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    tr, va = (HistoryWindows(series, encoded, bounds, split) for split in ("train", "val"))
    if resume:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        publish(state, directory)
    else:
        metadata.update(parameters=sum(p.numel() for p in model.parameters()), train_windows=len(tr), val_windows=len(va))
        initial = validation_report(model, va, device, batch_size)
        state = dict(resume_schema="babel-ae-resume-v1", metadata=metadata, epoch=0,
                     model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
                     initial_validation=initial, history=[], best_loss=initial["loss"], best_epoch=0,
                     best_model=copy.deepcopy(model.state_dict()))
        directory.mkdir(parents=True, exist_ok=True)
        atomic_save(state, directory / "last.pt")
        publish(state, directory)
    progress(f"Capacity latent={latent}; parameters={metadata['parameters']:,}; epoch={state['epoch']}/{epochs}")
    run_epochs(model, optimizer, tr, va, device, batch_size, state, directory, epochs)
