"""Reuse immutable completed255 input cache, never previous experiment weights."""

import shutil
from pathlib import Path

from . import uniform_context_data as original
from . import uniform_context_run as previous
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json

cache = original.cache
SPLITS = original.SPLITS


def source_identity(source, macro_identity):
    source = source.resolve()
    meta = read_json(source / "manifest.json")
    done = read_json(source / "completion.json")
    if (
        meta["schema"] != previous.SCHEMA
        or meta["code_sha256"] != previous.code_identity()
        or done["status"] != "complete"
        or not done["source_unchanged"]
        or meta["task_source"] != macro_identity
    ):
        raise ValueError(
            "Completed original Uniform Context cache and identical Macro ancestry required"
        )
    names = ["statistics.json", "model_selection_lock.json", "readout_lock.json"]
    for split in SPLITS:
        names += [
            f"cache/{split}{suffix}"
            for suffix in (".npz", "_rows.json", "_audit.json", "_index.json")
        ]
    files = {n: done["files"][n] for n in names}
    verify_files(source, files)
    files.update({n: sha256(source / n) for n in ("manifest.json", "completion.json")})
    return {"source": str(source), "files": files}


def prepare(meta, out, splits):
    if any(s not in SPLITS for s in splits):
        raise ValueError("Unknown input split")
    if any(s in SPLITS[2:] for s in splits):
        from .context_transfer_evaluate import check_readouts

        check_readouts(out)
    source = Path(meta["cache_source"]["source"])
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    for split in splits:
        if (folder / f"{split}_index.json").exists():
            cache(out, split)
            continue
        names = [f"{split}{suffix}" for suffix in (".npz", "_rows.json", "_audit.json")]
        verify_files(
            source, {f"cache/{n}": meta["cache_source"]["files"][f"cache/{n}"] for n in names}
        )
        for n in names:
            temp = folder / (n + ".tmp")
            shutil.copyfile(source / "cache" / n, temp)
            temp.replace(folder / n)
        atomic_json(
            {
                "manifest_sha256": sha256(out / "manifest.json"),
                "source_index_sha256": meta["cache_source"]["files"][f"cache/{split}_index.json"],
                "files": {n: sha256(folder / n) for n in names},
            },
            folder / f"{split}_index.json",
        )
