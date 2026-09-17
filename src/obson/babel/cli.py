"""Local Babel command line; no credentials, orders or external services."""

import argparse
import json
from pathlib import Path

from .data import load_series, manifest, split_boundaries


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text)


def engine_from_args(args):
    import numpy as np

    from .retrieval import Engine

    with np.load(args.index, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
    keys = [r["key"].split("/") for r in meta["manifest"]["sources"]]
    series, notes = load_series(
        args.root, sorted({k[0] for k in keys}), sorted({int(k[1]) for k in keys})
    )
    return Engine(series, args.index, args.checkpoint, notes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="babel", description="结构看盘、严格历史回放、可比较检索与因果表示学习"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train", "index"):
        p = sub.add_parser(command)
        p.add_argument("--root", default="data/contracts")
        p.add_argument("--symbols", nargs="+")
        p.add_argument("--periods", nargs="+", type=int, default=[60, 30])
        p.add_argument("--out", required=True)
        if command != "audit":
            p.add_argument("--window", type=int, default=128)
            p.add_argument("--stride", type=int, default=16)
            p.add_argument("--device", default="cpu")
        if command == "index":
            p.add_argument("--checkpoint")
        if command == "train":
            p.add_argument("--epochs", type=int, default=10)
            p.add_argument("--hidden", type=int, default=64)
            p.add_argument("--layers", type=int, default=2)
            p.add_argument("--warmup", type=int, default=32)
            p.add_argument("--batch-size", type=int, default=64)
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument(
                "--boundaries",
                help="JSON with train_until/val_until/test_until, shared across runs",
            )
    for command in ("query", "serve", "benchmark", "blind", "blind-pairs"):
        p = sub.add_parser(command)
        p.add_argument("--root", default="data/contracts")
        p.add_argument("--index", required=True)
        p.add_argument("--checkpoint")
        if command == "query":
            p.add_argument("--code", default="rb")
            p.add_argument("--period", type=int, default=60)
            p.add_argument("--asof", help="Asia/Shanghai cutoff, only completed bars are eligible")
            p.add_argument("--method", choices=["rule", "path", "model"], default="rule")
            p.add_argument("--topk", type=int, default=8)
            p.add_argument("--out")
        elif command == "serve":
            p.add_argument("--port", type=int, default=8765)
        elif command == "blind-pairs":
            p.add_argument("--count", type=int, default=6, help="One query per sampled week")
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--seed", type=int, default=20260918)
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--count", type=int, default=50 if command == "benchmark" else 20)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate", help="Explicit held-out endpoint teacher-fidelity evaluation")
    p.add_argument("--root", default="data/contracts")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", required=True)
    p = sub.add_parser("audit-pairs", help="Trace blind charts to verified raw contract records; CPU only")
    p.add_argument("--root", default="data/contracts")
    p.add_argument("--index", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--answer-key", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score-pairs", help="Score dimensional preferences and repeat stability")
    p.add_argument("--ratings", required=True)
    p.add_argument("--answer-key", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "score-blind", help="Evaluate human relevance ratings without inventing labels"
    )
    p.add_argument("--ratings", required=True)
    p.add_argument("--answer-key", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "audit-pairs":
        from .quality import audit_pairs

        write_json(args.out, audit_pairs(args.root, args.index, args.pairs, args.answer_key))
    elif args.command in ("audit", "train", "index"):
        series, notes = load_series(args.root, args.symbols, args.periods)
        if args.command == "audit":
            result = manifest(series, split_boundaries(series))
            result["diagnostics"] = notes
            result["eligible_bars"] = sum(int(s.main.sum()) for s in series)
            write_json(args.out, result)
        elif args.command == "train":
            from .model import Config, train

            cfg = Config(
                hidden=args.hidden, layers=args.layers, window=args.window, warmup=args.warmup
            )
            bounds = json.loads(Path(args.boundaries).read_text()) if args.boundaries else None
            result = train(
                series,
                args.out,
                cfg,
                args.epochs,
                args.stride,
                args.batch_size,
                args.lr,
                args.seed,
                args.device,
                bounds,
            )
            write_json(None, result)
        else:
            from .retrieval import save_index

            write_json(
                None,
                save_index(
                    series, args.out, args.window, args.stride, args.checkpoint, args.device
                ),
            )
    elif args.command == "evaluate":
        from .model import Windows, evaluate, load_model

        model, ck = load_model(args.checkpoint, args.device)
        keys = [s["key"].split("/") for s in ck["manifest"]["sources"]]
        series, _ = load_series(
            args.root, sorted({k[0] for k in keys}), sorted({int(k[1]) for k in keys})
        )
        if manifest(series)["sources"] != ck["manifest"]["sources"]:
            raise ValueError("Evaluation data differs from the frozen run manifest")
        ds = Windows(series, ck["manifest"]["boundaries"], args.split, model.cfg, args.stride)
        result = evaluate(model, ds, args.device)
        result.update(split=args.split, stride=args.stride, checkpoint=args.checkpoint)
        if args.stride != 1:
            result["sampling_note"] = (
                "Events evaluated only at sampled endpoints, not all stream events."
            )
        write_json(args.out, result)
    elif args.command == "score-pairs":
        from .pair_review import score_pairs

        write_json(args.out, score_pairs(args.ratings, args.answer_key))
    elif args.command == "score-blind":
        from .retrieval import score_blind

        write_json(args.out, score_blind(args.ratings, args.answer_key))
    else:
        engine = engine_from_args(args)
        if args.command == "query":
            write_json(
                args.out, engine.query(args.code, args.period, args.asof, args.method, args.topk)
            )
        elif args.command == "serve":
            from .server import serve

            serve(engine, args.port)
        elif args.command == "benchmark":
            from .retrieval import evaluate_retrieval

            write_json(args.out, evaluate_retrieval(engine, args.count, args.seed))
        elif args.command == "blind-pairs":
            from .pair_review import export_pairs

            write_json(None, export_pairs(engine, args.out, args.count, args.repeats, args.seed))
        else:
            from .retrieval import blind_packet

            packet, key = blind_packet(engine, args.count, args.seed)
            folder = Path(args.out)
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "cases.json", packet)
            write_json(folder / "answer_key.json", key)
            template = (Path(__file__).parent / "blind.html").read_text()
            (folder / "blind.html").write_text(
                template.replace(
                    "__CASES__", json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c")
                )
            )
            (folder / "ratings.csv").write_text(
                "case_id,alternative,relevance_0_1_2,notes\n"
                + "".join(
                    f"{case['id']},{alt['id']},,\n"
                    for case in packet
                    for alt in case["alternatives"]
                )
            )
            print(f"Blind cases: {folder}/cases.json; keep answer_key.json away from raters")


if __name__ == "__main__":
    main()
