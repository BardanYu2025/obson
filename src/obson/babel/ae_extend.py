"""Fixed-budget AE warm start and epoch-boundary recovery; validation only in training."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ae_context import feature_names
from .ae_diagnostics import fingerprint, metrics, summarize
from .data import manifest
from .history_autoencoder import HistoryAE, HistoryWindows, SCHEMA, WEIGHTS, reconstruction_loss
from .progress import progress


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def atomic_json(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4])}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


@torch.no_grad()
def validation_report(model, dataset, device, batch_size):
    model.eval()
    losses, rows = [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        truth = batch["y"].to(device)
        pred = model(batch["x"].to(device))["reconstruction"]
        losses.extend(reconstruction_loss(pred, truth, batch["mask"].to(device)).cpu().tolist())
        rows.extend(metrics(y, p) for y, p in zip(truth.cpu().numpy(), pred.cpu().numpy(), strict=True))
    result = {"loss": float(np.mean(losses)), **summarize({"all": rows})["all"]}
    if not np.isfinite(result["loss"]):
        raise ValueError("Nonfinite validation loss")
    return result


def check_source(ck, data_manifest, context, batch_size, seed):
    expected = HistoryAE(context=context).config
    if (ck["schema"] != SCHEMA or ck["manifest"] != data_manifest
            or ck["config"] != expected or list(ck["features"]) != list(feature_names(context))
            or tuple(ck["weights"]) != WEIGHTS or ck["delta_loss_weight"] != .5
            or ck["batch_size"] != batch_size or ck["seed"] != seed or ck["stride"] != 16):
        raise ValueError("Extension checkpoint data/config/objective/batch/seed mismatch")


def publish(state, directory):
    """last.pt is authoritative; reconstruct derivative files after an interrupted write."""
    atomic_json(state["metadata"], directory / "manifest.json")
    atomic_json(state["initial_validation"], directory / "warm_start_validation.json")
    atomic_save({**state["metadata"], "epoch": state["best_epoch"], "model": state["best_model"]}, directory / "best.pt")
    path = directory / "history.jsonl"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in state["history"]))
    temp.replace(path)


def extend(series, encoded, bounds, directory, epochs, batch_size, seed, device, context,
           warm_start=None, resume=False):
    if bool(warm_start) == bool(resume):
        raise ValueError("Choose exactly one of warm-start or resume")
    directory = Path(directory)
    data_manifest = manifest(series, bounds)
    if resume:
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if state.get("resume_schema") != "babel-ae-resume-v1":
            raise ValueError("Not a full AE recovery checkpoint")
        metadata = state["metadata"]
        check_source(metadata, data_manifest, context, batch_size, seed)
        if metadata["epochs"] != epochs:
            raise ValueError("Resume must preserve the predeclared epoch budget")
    else:
        source = Path(warm_start)
        ck = torch.load(source, map_location="cpu", weights_only=True)
        check_source(ck, data_manifest, context, batch_size, seed)
        history_path = source.parent / "history.jsonl"
        old_history = [json.loads(line) for line in history_path.read_text().splitlines()]
        completed = max(row["epoch"] for row in old_history)
        if completed != ck["epochs"] or not (ck["epoch"] <= completed < epochs):
            raise ValueError("Warm start requires completed source run and a larger total budget")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("New warm-start directory must be empty")
        metadata = {k: v for k, v in ck.items() if k not in ("model", "epoch")}
        metadata.update(epochs=epochs, training_mode="warm_start_reset_adamw_then_epoch_resume",
                        source_checkpoint_epoch=ck["epoch"], source_completed_epochs=completed,
                        additional_epochs=epochs - completed,
                        source_artifacts_sha256={name: fingerprint(source.parent / name) for name in
                                                  (source.name, "manifest.json", "history.jsonl", "ae_metrics.json")},
                        optimizer={"name": "AdamW", "lr": 3e-4, "weight_decay": .01},
                        validation_diagnostics="every epoch; scales 1/4/16; best selected only by original validation loss")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = (HistoryWindows(series, encoded, bounds, name) for name in ("train", "val"))
    model = HistoryAE(**metadata["config"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    if resume:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        publish(state, directory)
        # publish does not consume random numbers.
    else:
        model.load_state_dict(ck["model"])
        initial = validation_report(model, va, device, batch_size)
        directory.mkdir(parents=True, exist_ok=True)
        state = {"resume_schema": "babel-ae-resume-v1", "metadata": metadata,
                 "epoch": completed, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": rng_state(), "initial_validation": initial, "history": [],
                 "best_loss": initial["loss"], "best_epoch": ck["epoch"],
                 "best_model": copy.deepcopy(model.state_dict())}
        atomic_save(state, directory / "last.pt")
        publish(state, directory)
        progress(f"Warm start from checkpoint epoch {ck['epoch']}; source spent {completed} epochs; "
                 f"{epochs - completed} additional epochs; AdamW reset; initial val={initial['loss']:.6f}")
    for epoch in range(state["epoch"] + 1, epochs + 1):
        model.train()
        total, count = 0., 0
        loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        for step, batch in enumerate(loader, 1):
            pred = model(batch["x"].to(device))["reconstruction"]
            loss = reconstruction_loss(pred, batch["y"].to(device), batch["mask"].to(device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            size = len(batch["x"])
            total += loss.item() * size
            count += size
            if step == 1 or step % 25 == 0 or step == len(loader):
                progress(f"epoch={epoch}/{epochs} batch={step}/{len(loader)} loss={total/count:.6f}")
        validation = validation_report(model, va, device, batch_size)
        if validation["loss"] < state["best_loss"]:
            state.update(best_loss=validation["loss"], best_epoch=epoch,
                         best_model=copy.deepcopy(model.state_dict()))
        state["history"].append({"epoch": epoch, "train_loss": total/count,
                                  "validation_loss": validation["loss"], "validation": validation})
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
        atomic_save(state, directory / "last.pt")
        publish(state, directory)
        progress(f"epoch={epoch} validation={validation['loss']:.6f} best_epoch={state['best_epoch']}")
