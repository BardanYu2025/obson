"""Run from the experiment directory via scripts/autodl.sh."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .assets import export_run, release_id, restore_package, verify_package
from .common import exclusive, identity, read_json
from .run import PROTOCOL, evaluate, lock_study, preflight, prepare, train


def main():
    p = argparse.ArgumentParser(
        description="Babel P2/P3 joint training on uploaded raw months"
    )
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "preflight",
            "lock",
            "train",
            "resume",
            "matrix",
            "frozen",
            "evaluate",
            "export",
            "verify-package",
            "restore",
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("BABEL_DATA_ROOT", "/root/autodl-tmp/tqdata_2015")),
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("BABEL_CACHE", "/root/autodl-tmp/babel_state_v1")),
    )
    p.add_argument("--download", type=Path, default=Path("/root/autodl-tmp/download"))
    p.add_argument(
        "--scope",
        type=Path,
        help="Dated unit/calendar source file; consumed at preparation",
    )
    p.add_argument("--card", type=Path)
    p.add_argument("--arm", choices=["P2", "P3"])
    p.add_argument("--seed", type=int, choices=[42, 43])
    p.add_argument(
        "--split", choices=["validation", "development"], default="development"
    )
    p.add_argument("--run", type=Path)
    p.add_argument("--package", type=Path)
    p.add_argument("--to", type=Path, help="New directory for a verified clean restore")
    args = p.parse_args()
    if args.stage in ("verify-package", "restore"):
        if args.package is None or (args.stage == "restore" and args.to is None):
            p.error("Supply --package, and --to for restore")
        print(
            verify_package(args.package)
            if args.stage == "verify-package"
            else restore_package(args.package, args.to)
        )
        return
    from importlib.metadata import version
    from .common import ROOT

    for item in (ROOT / "requirements.lock").read_text().splitlines():
        name, expected = item.split("==")
        if name != "pytest" and version(name).split("+")[0] != expected:
            raise RuntimeError(f"Install {name}=={expected} for this release")
    release = release_id()
    if args.stage in ("prepare", "preflight"):
        if args.stage == "prepare" or not (args.cache / "scales.json").exists():
            prepare(
                args.data_root,
                args.cache,
                read_json(args.scope) if args.scope else None,
            )
        elif (
            args.scope
            and read_json(args.scope) != read_json(args.cache / "scales.json")["scope"]
        ):
            p.error("Use a new cache for a different dated scope")
        if args.stage == "preflight":
            if (args.cache / "study.lock.json").exists():
                p.error("This comparison already has a preflight and locked card")
            preflight(args.cache, args.download, release)
        else:
            print("Prepared sample coverage:", args.cache / "coverage.json")
    elif args.stage == "lock":
        if not args.card:
            p.error("Supply --card with the use limits and budgets")
        print("Study", identity(lock_study(args.cache, args.card, release)))
    elif args.stage == "frozen":
        from .frozen import run_frozen

        print(run_frozen(args.cache, args.download, release))
    elif args.stage == "matrix":
        card = lock_study(args.cache, args.cache / "study.lock.json", release)
        for seed in PROTOCOL["seeds"]:
            for arm in PROTOCOL["arms"] if seed == 42 else reversed(PROTOCOL["arms"]):
                rid = identity([identity(card), arm, seed])
                run = args.cache / "runs" / f"{arm}-{seed}-{rid[:12]}"
                if (run / "completed.json").exists():
                    if read_json(run / "completed.json")["step"] != card["updates"]:
                        raise ValueError("Inconsistent completed-step receipt")
                    if not (run / "export-receipt.json").exists():
                        from .common import json_file

                        json_file(
                            run / "export-receipt.json",
                            verify_package(export_run(run, args.download)),
                        )
                    continue
                train(
                    args.cache,
                    args.download,
                    arm,
                    seed,
                    release,
                    (run / "checkpoints.json").exists(),
                )
    elif args.stage in ("train", "resume"):
        if not args.arm or args.seed is None:
            p.error("Supply --arm and --seed")
        train(
            args.cache,
            args.download,
            args.arm,
            args.seed,
            release,
            args.stage == "resume",
        )
    else:
        if not args.run:
            p.error("Supply --run")
        if args.stage == "evaluate":
            print(
                evaluate(
                    args.cache,
                    args.run,
                    release,
                    1 if args.split == "validation" else 2,
                )
            )
        else:
            with exclusive(
                args.cache / (".run-" + read_json(args.run / "run.json")["run"])
            ):
                print(verify_package(export_run(args.run, args.download)))


if __name__ == "__main__":
    main()
