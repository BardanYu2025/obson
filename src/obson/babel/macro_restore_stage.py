"""First retained Macro ancestor: original scratch768 joint recipe, CUDA entrypoint.

The other twelve inherited stages remain explicitly unimplemented in this runner.
This stage never claims to have restored final Macro or completed its acceptance.
"""

import argparse
import fcntl
import signal
import time
from pathlib import Path

import numpy as np
import torch

from . import bar_alignment as ba
from . import macro_restore_data as data
from . import window_sampling as ws
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .architecture_benchmark import cpu_state
from .coverage_audit import ndarray_hash
from .progress import progress
from .raw_snapshot import digest

STAGE = "s01_bar_alignment"
SCHEMA = "babel-macro-restore-stage1-v1"


def recipe(source, micro=16):
    if micro not in (8, 16, 32, 64):
        raise ValueError("Micro must be8/16/32/64; effective batch remains128")
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "task_ids": ["F11", "F01"],
        "seeds": [42, 43],
        "epochs": 200,
        "budget": 4789,
        "pool_size": 38549,
        "batch": 128,
        "micro": micro,
        "evaluation_micro": 64,
        "config": source["config"],
        "local_hidden": source["local_hidden"],
        "encoder_lr": 3e-4,
        "head_lr": 1e-3,
        "encoder_decay": 0.01,
        "head_decay": 1e-4,
        "clip": 1.0,
        "objective": "original global SmoothL1 +0.25 original local SmoothL1; joint encoder gradients",
        "sampling": "original38549 dense pool,4789 without replacement; original4 prefix plan; original second minibatch shuffle",
        "selection": "minimum validation global MSE+0.25 local MSE at32/64/96/128 including epoch0; no research selection",
        "decoder": "fixed train-fitted PCA768; residual disabled; no target input shortcut",
        "per_seed": {"updates": 7600, "windows": 957800, "local_targets": 3831200},
        "precision": "fp32_tf32_disabled",
        "continuation_check": "one discarded effective-batch update from last checkpoint in training and independent CUDA processes; never retained or selected",
        "session_cap_seconds": 12 * 3600,
        "minimum_free_bytes": 12 * 1024**3,
        "acceptance": "recipe and engineering recovery only; no performance improvement claim",
        "stopping": "exact source gate, nonfinite/corruption,12h cumulative supervisor budget or200epochs; no automatic extension",
        "next_stage": "s02_path_control_not_implemented_here",
        "final_macro_restored": False,
    }


def optimizers(model, card):
    return (
        torch.optim.AdamW(
            [p for p in model.core.parameters() if p.requires_grad],
            lr=card["encoder_lr"],
            weight_decay=card["encoder_decay"],
        ),
        torch.optim.AdamW(
            model.local_head.parameters(), lr=card["head_lr"], weight_decay=card["head_decay"]
        ),
    )


def build(card, seed, root, device):
    with np.load(root / "cache/pca768.npz", allow_pickle=False) as f:
        pca = dict(f)
    return ba.AlignedStudent(
        card["config"],
        seed,
        pca,
        data.read(root / "cache/coordinates768.json"),
        card["local_hidden"],
    ).to(device)


def validate_model(model, val, stats, local, card, device):
    ps = np.tile(ba.VAL_PREFIXES, (len(val["x"]), 1)).astype(np.int64)
    return ba.run_epoch(
        model, val, stats, local, card["batch"], card["evaluation_micro"], device, True, ps
    )[0]


def replay(actual, expected, diagnostic=None):
    if actual.keys() != expected.keys() or any(
        not np.isclose(actual[k], expected[k], atol=1e-6, rtol=2e-5) for k in actual
    ):
        if diagnostic is not None:
            atomic_json(
                {"actual": actual, "expected": expected, "atol": 1e-6, "rtol": 2e-5}, diagnostic
            )
        raise ValueError("Restored validation differs; no update allowed")


def publish(state, path):
    # last is authoritative; best and history can be reconstructed if interrupted between writes.
    atomic_save(state, path / "last.pt")
    atomic_save(
        {
            "binding": state["binding"],
            "epoch": state["best_epoch"],
            "model": state["best_model"],
            "validation": state["best_validation"],
        },
        path / "best.pt",
    )
    atomic_json(state["history"], path / "history.json")


def verify_state(state, binding, seed, card):
    if (
        state["binding"] != binding
        or state["epoch"] != len(state["history"])
        or not 0 <= state["epoch"] <= card["epochs"]
    ):
        raise ValueError("Resume identity/epoch mismatch")
    steps = (card["budget"] + card["batch"] - 1) // card["batch"]
    for epoch, row in enumerate(state["history"], 1):
        ids = ws.sample_ids(card["pool_size"], card["budget"], seed, epoch)
        positions = ba.position_plan(len(ids), seed, epoch)
        expected = {
            "epoch": epoch,
            "ids_sha256": ndarray_hash(ids),
            "positions_sha256": ndarray_hash(positions),
            "encoder_steps": steps,
            "head_steps": steps,
            "encoder_lr": ba.ar.learning_rate(epoch, card["epochs"], card["encoder_lr"]),
            "head_lr": ba.ar.learning_rate(epoch, card["epochs"], card["head_lr"]),
        }
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("Resume sampling/rates/update budget changed")
    choices = [
        (0, state["initial_validation"]),
        *[(r["epoch"], r["validation"]) for r in state["history"]],
    ]
    selected = min(choices, key=lambda item: item[1]["selection"])
    if selected != (state["best_epoch"], state["best_validation"]):
        raise ValueError("Resume validation selection changed")


def train(model, pool, val, stats, local, card, binding, seed, path, device, *, stop_after=None):
    """CPU use is only for synthetic tests; public worker enforces CUDA before construction."""
    path.mkdir(parents=True, exist_ok=True)
    enc, head = optimizers(model, card)
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        verify_state(state, binding, seed, card)
        model.load_state_dict(state["model"], strict=True)
        enc.load_state_dict(state["encoder_optimizer"])
        head.load_state_dict(state["head_optimizer"])
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        replay(
            validate_model(model, val, stats, local, card, device),
            expected,
            path / "resume_validation_failure.json",
        )
        atomic_json(
            {"epoch": state["epoch"], "validation_passed": True}, path / "resume_check.json"
        )
    else:
        initial = validate_model(model, val, stats, local, card, device)
        state = {
            "binding": binding,
            "epoch": 0,
            "history": [],
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": cpu_state(model),
        }
    state.update(
        model=cpu_state(model),
        encoder_optimizer=enc.state_dict(),
        head_optimizer=head.state_dict(),
        rng=rng_state(),
    )
    publish(state, path)
    for epoch in range(state["epoch"] + 1, card["epochs"] + 1):
        start = time.monotonic()
        rates = {
            k: ba.ar.learning_rate(epoch, card["epochs"], card[k])
            for k in ("encoder_lr", "head_lr")
        }
        for opt, key in ((enc, "encoder_lr"), (head, "head_lr")):
            for group in opt.param_groups:
                group["lr"] = rates[key]
        ids = ws.sample_ids(len(pool["x"]), card["budget"], seed, epoch)
        positions = ba.position_plan(len(ids), seed, epoch)
        measured, steps = ba.run_epoch(
            model,
            ws.subset(pool, ids),
            stats,
            local,
            card["batch"],
            card["micro"],
            device,
            True,
            positions,
            enc,
            head,
            seed + epoch * 1009,
        )
        if steps != (card["budget"] + card["batch"] - 1) // card["batch"]:
            raise ValueError("Update budget mismatch")
        val_result = validate_model(model, val, stats, local, card, device)
        if val_result["selection"] < state["best_validation"]["selection"]:
            state.update(best_epoch=epoch, best_validation=val_result, best_model=cpu_state(model))
        state["history"].append(
            dict(
                epoch=epoch,
                train=measured,
                validation=val_result,
                **rates,
                encoder_steps=steps,
                head_steps=steps,
                ids_sha256=ndarray_hash(ids),
                positions_sha256=ndarray_hash(positions),
                seconds=time.monotonic() - start,
            )
        )
        state.update(
            epoch=epoch,
            model=cpu_state(model),
            encoder_optimizer=enc.state_dict(),
            head_optimizer=head.state_dict(),
            rng=rng_state(),
        )
        publish(state, path)
        progress(
            f"{STAGE}/s{seed}: {epoch}/{card['epochs']}; global={val_result['primary']:.6f}, local={val_result['local_primary']:.6f}, best={state['best_epoch']}, {state['history'][-1]['seconds']:.1f}s"
        )
        if stop_after is not None and epoch >= stop_after:
            return state
    verify_state(state, binding, seed, card)
    return state


@torch.inference_mode()
def sample_outputs(model, inputs, prefixes):
    model.eval()
    glob, local = model(inputs, prefixes, True)
    state = model.core.encoder(inputs)
    return {
        "global": glob.cpu().numpy(),
        "local": local.cpu().numpy(),
        "state": state.cpu().numpy(),
    }


def check_optimizer(opt):
    count = 0
    for group in opt.param_groups:
        for param in group["params"]:
            state = opt.state.get(param)
            if not state:
                continue
            count += 1
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].shape != param.shape or not torch.isfinite(state[key]).all():
                    raise ValueError("Invalid restored Adam state")
            if not torch.isfinite(state["step"]) or state["step"] <= 0:
                raise ValueError("Invalid restored Adam step")
    if not count:
        raise ValueError("No trained optimizer state")
    return count


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def continuation_probe(model, pool, stats, local, card, state, seed, device):
    """Discarded update on train data, at saved LR; leaves published state unchanged."""
    before_rng = rng_state()
    try:
        model.load_state_dict(state["model"], strict=True)
        enc, head = optimizers(model, card)
        enc.load_state_dict(cpu_tree(state["encoder_optimizer"]))
        head.load_state_dict(cpu_tree(state["head_optimizer"]))
        restore_rng(state["rng"])
        epoch = state["epoch"] + 1
        ids = ws.sample_ids(len(pool["x"]), card["budget"], seed, epoch)[: card["batch"]]
        positions = ba.position_plan(len(ids), seed, epoch)
        _, steps = ba.run_epoch(
            model,
            ws.subset(pool, ids),
            stats,
            local,
            card["batch"],
            card["micro"],
            device,
            True,
            positions,
            enc,
            head,
            seed + epoch * 1009,
        )
        if steps != 1:
            raise ValueError("Continuation diagnostic must execute exactly one discarded update")
        return {
            "model": cpu_state(model),
            "encoder_optimizer": cpu_tree(enc.state_dict()),
            "head_optimizer": cpu_tree(head.state_dict()),
            "ids_sha256": ndarray_hash(ids),
            "positions_sha256": ndarray_hash(positions),
            "discarded_updates": 1,
        }
    finally:
        model.load_state_dict(state["model"], strict=True)
        restore_rng(before_rng)


def compare_continuation(actual, expected, path="root"):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5, msg=path)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Continuation keys differ: {path}")
        for key in expected:
            compare_continuation(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"Continuation lengths differ: {path}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_continuation(a, e, f"{path}/{i}")
    elif actual != expected:
        raise ValueError(f"Continuation metadata differs: {path}")


def execute_worker(out, seed, check_only=False):
    from .macro_restore_run import verify_binding

    if not torch.cuda.is_available():
        raise ValueError("Real model construction/execution requires AutoDL CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(True)  # Historical stage01 default.
    meta = data.read(out / "manifest.json")
    verify_binding(meta)
    data.verify_prepared(out / "data")
    card = meta["stage"]
    if card != recipe(data.source_contract(), card["micro"]):
        raise ValueError("Fixed stage01 recipe changed")
    if seed not in card["seeds"]:
        raise ValueError("Unregistered seed")
    path = out / "stages" / STAGE / f"s{seed}"
    binding = {
        "run_manifest": digest(out / "manifest.json"),
        "data_receipt": digest(out / "data/data_receipt.json"),
        "seed": seed,
    }
    torch.manual_seed(seed)
    model = build(card, seed, out / "data", "cuda")
    val = data.arrays(out / "data", "val")
    stats = data.read(out / "data/cache/statistics.json")
    local = data.read(out / "data/cache/local_scales.json")
    if check_only:
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        verify_state(state, binding, seed, card)
        if state["epoch"] != card["epochs"]:
            raise ValueError("Stage budget incomplete")
        for kind in ("best", "last"):
            ck = torch.load(path / f"{kind}.pt", map_location="cpu", weights_only=True)
            if ck["binding"] != binding or (kind == "best" and ck["epoch"] != state["best_epoch"]):
                raise ValueError("Checkpoint identity/selection changed")
            model.load_state_dict(ck["model"], strict=True)
            expected = (
                state["best_validation"] if kind == "best" else state["history"][-1]["validation"]
            )
            replay(
                validate_model(model, val, stats, local, card, "cuda"),
                expected,
                path / f"{kind}_validation_failure.json",
            )
            if kind == "best" and any(
                not torch.equal(v, ck["model"][k]) for k, v in state["best_model"].items()
            ):
                raise ValueError("Published best differs from authoritative last snapshot")
            enc, head = optimizers(model, card)
            if kind == "last":
                enc.load_state_dict(ck["encoder_optimizer"])
                head.load_state_dict(ck["head_optimizer"])
                check_optimizer(enc)
                check_optimizer(head)
                restore_rng(ck["rng"])
            with np.load(path / "examples.npz", allow_pickle=False) as example:
                actual = sample_outputs(
                    model,
                    torch.tensor(example["x"], device="cuda"),
                    torch.tensor(example["prefixes"], device="cuda"),
                )
                if any(
                    not np.allclose(actual[k], example[kind + "_" + k], atol=1e-5, rtol=2e-5)
                    for k in actual
                ):
                    raise ValueError("Frozen example outputs do not replay")
        expected_probe = torch.load(path / "continuation.pt", map_location="cpu", weights_only=True)
        actual_probe = continuation_probe(
            model, data.arrays(out / "data", "pool"), stats, local, card, state, seed, "cuda"
        )
        try:
            compare_continuation(actual_probe, expected_probe)
        except (ValueError, AssertionError) as exc:
            atomic_json(
                {"error": str(exc), "atol": 1e-6, "rtol": 2e-5}, path / "continuation_failure.json"
            )
            raise
        atomic_json(
            {
                "status": "passed",
                "binding": binding,
                "process": "independent_cuda_process",
                "best_and_last_full_validation_replayed": True,
                "optimizer_groups_shapes_finite_and_rng_restored": True,
                "discarded_one_update_replay_passed": True,
                "scientific_acceptance": False,
                "files": {
                    n: digest(path / n)
                    for n in (
                        "best.pt",
                        "last.pt",
                        "history.json",
                        "examples.npz",
                        "continuation.pt",
                    )
                },
            },
            path / "completion.json",
        )
        return
    path.mkdir(parents=True, exist_ok=True)
    if not (path / "example_inputs.npz").exists():
        np.savez(
            path / "example_inputs.tmp.npz",
            x=np.asarray(val["x"][:4]),
            prefixes=np.tile(ba.VAL_PREFIXES, (4, 1)),
        )
        (path / "example_inputs.tmp.npz").replace(path / "example_inputs.npz")
    state = train(
        model,
        data.arrays(out / "data", "pool"),
        val,
        stats,
        local,
        card,
        binding,
        seed,
        path,
        "cuda",
    )
    examples = {"x": np.asarray(val["x"][:4]), "prefixes": np.tile(ba.VAL_PREFIXES, (4, 1))}
    for kind, weights in (("best", state["best_model"]), ("last", state["model"])):
        model.load_state_dict(weights, strict=True)
        values = sample_outputs(
            model,
            torch.tensor(examples["x"], device="cuda"),
            torch.tensor(examples["prefixes"], device="cuda"),
        )
        examples.update({kind + "_" + k: v for k, v in values.items()})
    np.savez(path / "examples.tmp.npz", **examples)
    (path / "examples.tmp.npz").replace(path / "examples.npz")
    probe = continuation_probe(
        model, data.arrays(out / "data", "pool"), stats, local, card, state, seed, "cuda"
    )
    atomic_save(probe, path / "continuation.pt")


def worker(out, seed, check_only=False):
    root = out / "stages" / STAGE
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_worker(out, seed, check_only)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()

    def stopped(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}; resume from last atomic epoch")

    signal.signal(signal.SIGTERM, stopped)
    worker(a.out, a.seed, a.check_only)


if __name__ == "__main__":
    main()
