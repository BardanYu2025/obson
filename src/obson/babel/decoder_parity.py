"""Frozen comparisons separating readout choice, prefix position and context length."""

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from . import history_query_run as upstream
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-decoder-parity768-v2"
PREFIXES = (32, 48, 64, 80, 96, 112, 128)
LENGTHS = (32, 64, 80, 128)
SPLITS = ("test", "cross_research")
ur, ba = upstream.ur, upstream.hq.ba


def code_identity():
    return upstream.code_identity() | {Path(__file__).name: sha256(__file__)}


def make_manifest(source, identity):
    return {
        "schema": SCHEMA,
        "source": str(source),
        "identity": identity,
        "code_sha256": code_identity(),
        "batch": 64,
        "numeric_protocol": "Original joint7-prefix local-head GEMM and target conversion; predicted path crops on CPU as in path_feature_benchmark.score. Original replay tolerances unchanged.",
        "seeds": [42, 43],
        "prefixes": list(PREFIXES),
        "lengths": list(LENGTHS),
        "encoder_updates": 0,
        "decoder_updates": 0,
        "reader_fits": 0,
        "statistics_fits": 0,
        "prefix_protocol": "Same cached128 input and same h[p-1], compare existing PCA decoder crop vs existing local16 head on identical past16 target/mask/scales. Native PCA scores use only first p-1 observed targets; do not compare absolute difficulty across positions.",
        "context_protocol": "Same original endpoint and identical previous16 target across right-aligned lengths32/64/80/128. Recompute encoder with retained normalized inputs and positional indices starting0. Carried causal EMA/features remain; not an isolated-history experiment. Position and context change together, not a position-only causal attribution.",
        "decoder": "One unchanged original PCA decoder at every state; own predicted price anchor only. Existing local head is a frozen comparison, never a newly fitted head. No query head or cross-period loss.",
        "slot_hypothesis": "Fixed128 outputs interpreted in input-window coordinates, input start at slot0; prefix p reads past16 slots before p-1. This untrained interior-slot interpretation is an explicit reuse hypothesis, not an established semantic property. No search across mappings.",
        "decision": "Report both seeds/sets/all scenarios without model selection. Frozen PCA recent16 must retain5% vs local head in every prefix, family10%, weekly support50/5, to qualify as a directly reusable shared readout in this limited scope. Context comparisons are paired diagnostics, not success gates. No automatic promotion or follow-up training.",
        "scope": "Existing research cohorts, not fresh holdouts. Zero updates. This audit does not test cross-resolution semantic alignment or prove information absent when a readout fails.",
    }


def source_meta(meta):
    return meta["identity"]["manifest"]


def data(meta, split):
    return upstream.delivery.data_for(source_meta(meta), split)


def load_engine(meta, seed, device):
    return upstream.rs.load_bundle(
        Path(meta["source"]) / "bundle", seed, "MA/15/CZCE.MA601", 15, device
    )


def shared_crop(prediction, prefix, statistics, local):
    """Use the decoder's own history anchor; no observed history enters this adapter."""
    if prediction.ndim != 3 or prediction.shape[1:] != (128, 7) or not 18 <= prefix <= 128:
        raise ValueError("A128x7 decoded path and valid past16 context required")
    ps = torch.full((len(prediction), 1), prefix, dtype=torch.long, device=prediction.device)
    values, _ = ba.local_targets(
        prediction, torch.ones_like(prediction, dtype=torch.bool), ps, statistics, local
    )
    return values[:, 0]


def observed_targets(batch, prefix, statistics, local):
    ps = torch.full((len(batch["y"]), 1), prefix, dtype=torch.long, device=batch["y"].device)
    y, mask = ba.local_targets(batch["y"], batch["mask"], ps, statistics, local)
    return y[:, 0], mask[:, 0]


def native_rows(prediction, batch, prefix, statistics):
    mask = batch["mask"].clone()
    mask[:, prefix - 1 :] = False  # Exclude the state's current bar AND its future.
    return ba.ar.error_rows(
        prediction.double().cpu(), batch["y"].double().cpu(), mask.cpu(), statistics
    )


@torch.no_grad()
def evaluate_model(engine, d, batch_size, device):
    model, statistics, local = engine.aligned, engine.statistics, engine.local
    model.eval().requires_grad_(False)
    before = ur.bb.state_signature(model)
    banks = {
        f"{kind}{p}": {"shared": [], "local": [], "target": [], "mask": [], "native": {}}
        for kind, ps in [("prefix", PREFIXES), ("context", LENGTHS)]
        for p in ps
    }
    for left in range(0, len(d["x"]), batch_size):
        b = {
            k: torch.tensor(np.asarray(d[k][left : left + batch_size]), device=device)
            for k in ("x", "y", "mask")
        }
        full = model.core.encoder(b["x"])
        # Preserve the source's Bx7xD head call, rather than seven BxD calls.
        # Mathematically equivalent GEMMs need not select identical CUDA kernels.
        ps = torch.tensor(PREFIXES, device=device)[None].expand(len(full), -1)
        chosen_states = full[torch.arange(len(full), device=device)[:, None], ps - 1]
        local_predictions = model.local_head(chosen_states).reshape(len(full), len(PREFIXES), 16, 7)
        targets, masks = ba.local_targets(b["y"], b["mask"], ps, statistics, local)
        for kind, positions in [("prefix", PREFIXES), ("context", LENGTHS)]:
            for p in positions:
                if kind == "prefix":
                    z = full[:, p - 1]
                else:
                    z = full[:, -1] if p == 128 else model.core.encoder(b["x"][:, -p:])[:, -1]
                decoded = model.core.decoder(z)
                # The pinned source rebases predictions on CPU after inference.
                shared = shared_crop(decoded.cpu(), p, statistics, local)
                index = PREFIXES.index(p) if kind == "prefix" else len(PREFIXES) - 1
                conventional = (
                    local_predictions[:, index]
                    if kind == "prefix" or p == 128
                    else model.local_head(z).reshape(-1, 16, 7)
                )
                target, mask = targets[:, index], masks[:, index]
                row = banks[f"{kind}{p}"]
                for k, v in [
                    ("shared", shared),
                    ("local", conventional),
                    ("target", target),
                    ("mask", mask),
                ]:
                    row[k].append(v.cpu().numpy())
                if kind == "prefix":
                    for k, v in native_rows(decoded, b, p, statistics).items():
                        row["native"].setdefault(k, []).append(v.cpu().numpy())
    records = {}
    chosen = np.linspace(0, len(d["x"]) - 1, min(4, len(d["x"])), dtype=int)
    for name, row in banks.items():
        arrays = {k: np.concatenate(row[k]) for k in ("shared", "local", "target", "mask")}
        result = {
            "target_sha256": ur.bb.cov.ndarray_hash(arrays["target"]),
            "mask_sha256": ur.bb.cov.ndarray_hash(arrays["mask"]),
            "scores": {},
            "errors": {},
            "native": {},
            "examples": [],
        }
        for head in ("shared", "local"):
            result["scores"][head], errors = ur.bb.pr.measure(
                arrays[head], arrays["target"], arrays["mask"], local
            )
            result["errors"][head] = {k: v.tolist() for k, v in errors.items()}
        if row["native"]:
            errors = {k: np.concatenate(v) for k, v in row["native"].items()}
            result["native"] = {
                "scores": {k: float(v.mean()) for k, v in errors.items()},
                "errors": {k: v.tolist() for k, v in errors.items()},
                "scope": "Original PCA output slots for this prefix only; different prefixes have different history and difficulty.",
            }
        for i in chosen:
            result["examples"].append(
                dict(index=int(i), **{k: arrays[k][i].tolist() for k in arrays})
            )
        records[name] = result
    # The long-context endpoint is exactly the same computation in both protocols.
    if records["prefix128"] != records["context128"] | {"native": records["prefix128"]["native"]}:
        raise ValueError("Endpoint protocols differ")
    for p in LENGTHS:
        for field in ("target_sha256", "mask_sha256"):
            if records[f"context{p}"][field] != records["prefix128"][field]:
                raise ValueError("Context experiment changed the target instead of only its inputs")
    if before != ur.bb.state_signature(model):
        raise ValueError("Frozen audit changed source model")
    return records


def replay(records, expected, out, label):
    actual, reference = {}, {}
    for p in PREFIXES:
        actual[f"p{p}"] = records[f"prefix{p}"]["errors"]["local"]
        reference[f"p{p}"] = expected["reconstruction"]["errors"][f"p{p}"]
    actual["recent"] = records["prefix128"]["errors"]["shared"]
    reference["recent"] = expected["reconstruction"]["errors"]["recent"]
    actual["global"] = records["prefix128"]["native"]["errors"]
    reference["global"] = {
        k: expected["reconstruction"]["errors"]["global"][k] for k in actual["global"]
    }
    differences = upstream.warm.recheck.mismatches(actual, reference)
    atomic_json(
        {"passed": not differences, "differences": differences}, out / f"{label}_replay.json"
    )
    if differences:
        first = differences[0]
        raise ValueError(
            f"{label}: original source replay differs ({len(differences)} differences); "
            f"first={first}; see {out / (label + '_replay.json')}"
        )


def decide(records, inventories):
    expected_names = {f"{s}/s{seed}" for s in SPLITS for seed in (42, 43)}
    if set(records) != expected_names:
        raise ValueError("Both seeds and both research cohorts required")
    checks, contexts = [], []
    for label, record in records.items():
        split = label.split("/")[0]
        rows = inventories[split]
        for p in PREFIXES:
            entry = record[f"prefix{p}"]["errors"]
            for field in ("primary", "path", "body", "activity", "change1"):
                a, b = np.array(entry["shared"][field]), np.array(entry["local"][field])
                if (
                    a.shape != b.shape
                    or a.shape != (len(rows),)
                    or not np.isfinite(a).all()
                    or not np.isfinite(b).all()
                ):
                    raise ValueError("Invalid paired readout scores")
                factor = 1.05 if field == "primary" else 1.10
                ci = ur.interval(a, factor * b, rows)
                checks.append(
                    {
                        "dataset": label,
                        "prefix": p,
                        "metric": field,
                        "factor": factor,
                        "interval": ci,
                        "passed": bool(
                            ci["supported"] and ci["high"] is not None and ci["high"] <= 0
                        ),
                    }
                )
        for p in LENGTHS[:-1]:
            for head in ("shared", "local"):
                a = record[f"context{p}"]["errors"][head]["primary"]
                b = record["context128"]["errors"][head]["primary"]
                ci = ur.interval(a, b, rows)
                contexts.append(
                    {
                        "dataset": label,
                        "context": p,
                        "head": head,
                        "interval": ci,
                        "interpretation": "Same endpoint/target; shorter direct context AND shifted positional indices. Prior causal features retained. Not a position-only or total-information intervention.",
                    }
                )
    ready = all(c["passed"] for c in checks)
    return {
        "status": "frozen_shared_readout_within_scope"
        if ready
        else "frozen_shared_readout_not_established",
        "checks": checks,
        "context_comparisons": contexts,
        "automatic_promotion": False,
        "training_updates": 0,
        "reader_fits": 0,
        "scope": "One finite readout audit; failure does not prove missing information or inability to train a shared decoder. No automatic follow-up training.",
    }


def write_summary(records, decision, out):
    lines = [
        "# Same-state, same-target decoder comparison",
        "",
        decision["status"],
        "",
        "Zero training / zero fitting. Lower is better. Prefix rows have different targets; compare heads within each row.",
        "",
        "| Dataset/seed | Prefix | Original PCA shared crop | Original local head |",
        "|---|---:|---:|---:|",
    ]
    for label, record in records.items():
        for p in PREFIXES:
            row = record[f"prefix{p}"]["scores"]
            lines.append(
                f"| {label} | {p} | {row['shared']['metrics']['primary']:.6f} | {row['local']['metrics']['primary']:.6f} |"
            )
    lines += [
        "",
        "Context records compare identical endpoints/targets with different direct context lengths; they do not isolate position encoding.",
        "This is not cross-resolution alignment or a new model. Do not turn an unsupported gate into a claim that an embedding cannot remember history.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def run(source, out, device="cuda"):
    source, out = source.resolve(), out.resolve()
    progress("Verifying control600 lineage; zero training, zero new heads and zero fitting")
    identity = upstream.source_identity(source)
    upstream.warm.check_output(source, out, identity["manifest"]["identity"])
    meta = make_manifest(source, identity)
    source_runtime = read_json(Path(identity["manifest"]["source"]) / "runtime.json")
    if (
        source_runtime["torch"] != str(torch.__version__)
        or source_runtime["numpy"] != np.__version__
    ):
        raise ValueError("Retain original Torch/NumPy environment")
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != meta:
        raise ValueError("Different source/config requires a fresh output")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete" or not done["source_unchanged"]:
            raise ValueError("Invalid completion record")
        upstream.verify_files(out, done["files"])
        progress("Completed frozen audit verified; no repeated inference")
        return
    started = time.monotonic()
    atomic_json(
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "encoder_updates": 0,
            "decoder_updates": 0,
            "reader_fits": 0,
        },
        out / "runtime.json",
    )
    records, inventories = {}, {}
    training = Path(source_meta(meta)["identity"]["training_source"])
    reference = Path(source_meta(meta)["source"])
    for split in SPLITS:
        inventories[split] = read_json(training / f"{split}_inventory.json")
        atomic_json(inventories[split], out / f"{split}_inventory.json")
        d = data(meta, split)
        if len(d["x"]) != len(inventories[split]):
            raise ValueError("Cached input/inventory row count differs")
        for seed in meta["seeds"]:
            progress(f"{split}/s{seed}: same-state readouts and matched-endpoint contexts")
            engine = load_engine(meta, seed, device)
            causal = ur.pf.er.trained_causality(
                engine.aligned, torch.tensor(np.asarray(d["x"][:4]), device=device)
            )
            atomic_json(causal, out / f"{split}_s{seed}_causality.json")
            if causal["status"] == "failed":
                raise ValueError("Source causality failed")
            record = evaluate_model(engine, d, meta["batch"], device)
            # Keep complete evidence even when the strict source replay rejects it.
            atomic_json(record, out / f"{split}_s{seed}.json")
            replay(
                record,
                read_json(reference / f"{split}_control_s{seed}_best.json"),
                out,
                f"{split}_s{seed}",
            )
            records[f"{split}/s{seed}"] = record
            del engine
    result = decide(records, inventories)
    atomic_json(result, out / "decision.json")
    write_summary(records, result, out)
    if upstream.source_identity(source) != identity:
        raise ValueError("Original source changed")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "seconds": time.monotonic() - started,
            "encoder_updates": 0,
            "decoder_updates": 0,
            "reader_fits": 0,
            "statistics_fits": 0,
            "files": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p.name not in ("completion.json", "run_status.txt")
                and p.suffix not in (".log", ".tmp")
            },
        },
        out / "completion.json",
    )
    progress("Frozen decoder comparison complete; no new model or training scheduled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="checkpoints/babel_control600_v2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Real model evaluation runs on AutoDL CUDA")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Decoder audit already running") from None
        run(Path(args.source), out)


if __name__ == "__main__":
    main()
