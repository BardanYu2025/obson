"""F11 offline market-state viewer from reviewed outputs. CPU only, no models."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np

from . import history_price_truth as truth

ARCHIVE_SHA256 = "c41cb6abeb04c9103b100abc55159fb22312efb0106aa4dd4680413c97184ba8"
COMPLETION_SHA256 = "a192bf5f0e733375bc36bb2af26e250561f1ddb664202b2dcb1107156d25b7d0"
INVENTORY_SHA256 = "007e788425787f60c115ff559796a36448908d0800e53c23986e7a04b5b2b0f1"
PREFIX = "babel_history_state768/"
SPLITS = {"test": 984, "cross_research": 439}
MODELS = ("macro_s42", "macro_s43")
SCHEMA = "babel-current-market-view-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(data):
    return json.loads(data.decode("utf-8"))


class ReviewedArchive:
    """No extraction or execution. Both archive identity and consumed members bound."""

    def __init__(self, path):
        if sha256(path) != ARCHIVE_SHA256:
            raise ValueError(
                "Expected the reviewed history_state768 archive; no source substitution"
            )
        self.resources = ExitStack()
        self.tar = self.resources.enter_context(tarfile.open(path))  # noqa: SIM115 -- lifetime owned by close()/failure cleanup
        try:
            members = self.tar.getmembers()
            names = set()
            for member in members:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or member.name in names
                    or not (member.isfile() or member.isdir())
                    or not name.parts
                    or name.parts[0] != PREFIX.rstrip("/")
                ):
                    raise ValueError("Unsafe or duplicate archive member")
                names.add(member.name)
            if sum(m.size for m in members) > 1024**3:
                raise ValueError("Unexpected report size")
            done = self._read("completion.json")
            if hashlib.sha256(done).hexdigest() != COMPLETION_SHA256:
                raise ValueError("Source completion changed")
            self.done = json_bytes(done)
            if not self.done["source_unchanged"]:
                raise ValueError("Source was modified")
            self.verified = []
        except Exception:
            self.resources.close()
            raise

    def _read(self, name):
        member = self.tar.getmember(PREFIX + name)
        if not member.isfile():
            raise ValueError("Expected a regular report file")
        with self.tar.extractfile(member) as file:
            return file.read()

    def read(self, name):
        data = self._read(name)
        if hashlib.sha256(data).hexdigest() != self.done["files"][name]:
            raise ValueError(f"Report fingerprint changed: {name}")
        self.verified.append(name)
        return data

    def close(self):
        self.resources.close()


def bar_window(records, row, asof):
    """Exact physical128 context; times are local bar starts, not close times."""
    key = row["key"]
    symbol, period, _ = key.split("/")
    if period not in ("15", "30", "60") or row["symbol"] != symbol or row["period"] != int(period):
        raise ValueError("Single-contract period identity mismatch")
    index = row["row"]
    if type(index) is not int or not 127 <= index < len(records):
        raise ValueError("Expected an exact endpoint with128 historical bars")
    window = records[index - 127 : index + 1]
    times = [datetime.fromisoformat(r["datetime"]) for r in window]
    if any(t.tzinfo is not None for t in times) or any(
        a >= b for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("Expected strictly increasing naive local bar starts")
    if times[-1] != datetime.fromisoformat(row["end"]):
        raise ValueError("Endpoint timestamp does not match raw CSV")
    closed_at = times[-1] + timedelta(minutes=int(period))
    cutoff = datetime.fromisoformat(asof)
    if cutoff.tzinfo is not None:
        raise ValueError("Expected a naive local as-of")
    if closed_at > cutoff:
        raise ValueError("Endpoint is not closed at the locked as-of")
    prices = np.asarray([[float(r[k]) for k in ("open", "high", "low", "close")] for r in window])
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid OHLC prices")
    if (prices[:, 1] < prices.max(1)).any() or (prices[:, 2] > prices.min(1)).any():
        raise ValueError("Raw OHLC ordering invalid; do not silently repair")

    def optional(field):
        result = []
        for record in window:
            try:
                value = float(record.get(field, ""))
            except (ValueError, TypeError):
                value = float("nan")
            result.append(value if np.isfinite(value) and value >= 0 else None)
        return result

    return {
        "time": [t.isoformat(sep=" ") for t in times],
        "closed_at": closed_at.isoformat(sep=" "),
        "ohlc": prices.tolist(),
        "volume": optional("volume"),
        "open_interest": optional("close_oi"),
    }


def describe(path):
    result = truth.summarize_path(path)
    return {
        "efficiency": result["descriptors"][:, [0, 3]],
        "linearity": result["descriptors"][:, [1, 4]],
        "rms_bps": result["rms_bps"],
        "direction": result["direction"],
    }


def score(raw, model_paths):
    """Current close is a coordinate, excluded from all scored price positions."""
    actual = truth.from_raw_closes(raw)
    summaries = {"raw": describe(actual["relative_log_close"])}
    errors = {}
    for model, values in model_paths.items():
        predicted = np.asarray(values, dtype=float)
        if predicted.shape != actual["relative_log_close"].shape:
            raise ValueError("Matched128-position predictions required")
        summaries[model] = describe(predicted)
        error = (predicted - actual["relative_log_close"]) * 10000
        # A displayed horizon includes current, with only h-1 nontrivial prices.
        errors[model] = {
            "rmse_bps": np.column_stack(
                [np.sqrt(np.mean(error[:, -h:-1] ** 2, axis=1)) for h in (16, 64, 128)]
            ),
            "max_abs_bps": np.max(np.abs(error[:, :-1]), axis=1),
        }
    return actual["relative_log_close"], summaries, errors


def cohort_summary(summaries, errors):
    result = {}
    true = summaries["raw"]["direction"]
    for model in MODELS:
        rms = errors[model]["rmse_bps"][:, 2]
        directions = summaries[model]["direction"]
        classifications = []
        for h in range(2):
            matrix = np.zeros((3, 3), dtype=int)
            np.add.at(matrix, (true[:, h], directions[:, h]), 1)
            support = matrix.sum(1)
            recall = np.divide(matrix.diagonal(), support, out=np.zeros(3), where=support > 0)
            classifications.append(
                {
                    "accuracy": float(np.mean(true[:, h] == directions[:, h])),
                    "balanced_accuracy": float(recall[support > 0].mean()),
                    "confusion": matrix.tolist(),
                }
            )
        result[model] = {
            "windows": len(rms),
            "mean_rmse_bps": float(rms.mean()),
            "median_rmse_bps": float(np.median(rms)),
            "p95_rmse_bps": float(np.quantile(rms, 0.95)),
            "classification": classifications,
        }
    return result


def load_raw(root, rows, inventory):
    grouped = {}
    for split, cohort in rows.items():
        for i, row in enumerate(cohort):
            grouped.setdefault(row["key"], []).append((split, i, row))
    windows = {split: [None] * len(cohort) for split, cohort in rows.items()}
    root = Path(root).resolve()
    for key, items in grouped.items():
        symbol, period, contract = key.split("/")
        path = (root / symbol / f"{contract}_{period}m.csv").resolve()
        if not path.is_relative_to(root) or key not in inventory["files"]:
            raise ValueError("CSV outside locked single-contract inventory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != inventory["files"][key]:
            raise ValueError(f"Raw CSV fingerprint mismatch: {key}")
        records = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        for split, i, row in items:
            windows[split][i] = bar_window(records, row, inventory["asof"])
    return windows, len(grouped)


def build_data(archive, raw_root):
    inventory_path = Path(__file__).with_name("overlap_content_inventory.json")
    if sha256(inventory_path) != INVENTORY_SHA256:
        raise ValueError("Pinned raw inventory changed")
    inventory = json.loads(inventory_path.read_text())
    report = ReviewedArchive(archive)
    try:
        rows = {split: json_bytes(report.read(split + "_rows.json")) for split in SPLITS}
        if any(len(rows[split]) != n for split, n in SPLITS.items()):
            raise ValueError("Original cohort sizes changed")
        raw, file_count = load_raw(raw_root, rows, inventory)
        cases, aggregate = [], {}
        for split in SPLITS:
            paths = {}
            for model in MODELS:
                with np.load(
                    io.BytesIO(report.read(f"{split}_{model}.npz")), allow_pickle=False
                ) as bank:
                    paths[model] = bank["relative_log_close"].copy()
            prices = np.asarray([np.asarray(w["ohlc"])[:, 3] for w in raw[split]])
            actual, summaries, errors = score(prices, paths)
            aggregate[split] = cohort_summary(summaries, errors)
            for i, row in enumerate(rows[split]):
                summary = {
                    k: {field: value[i].tolist() for field, value in v.items()}
                    for k, v in summaries.items()
                }
                cases.append(
                    {
                        "id": f"{split}:{i}",
                        "split": split,
                        "index": i,
                        "row": row,
                        **raw[split][i],
                        "actual_log": actual[i].tolist(),
                        "model_log": {k: v[i].tolist() for k, v in paths.items()},
                        "summary": summary,
                        "errors": {
                            k: {f: v[i].tolist() for f, v in e.items()} for k, e in errors.items()
                        },
                        "direction_disagreement": any(
                            summary[m]["direction"] != summary["raw"]["direction"] for m in MODELS
                        ),
                        "observed_low_motion": any(
                            r <= truth.LOW_RMS * 10000 for r in summary["raw"]["rms_bps"]
                        ),
                    }
                )
        return {
            "schema": SCHEMA,
            "source_archive_sha256": ARCHIVE_SHA256,
            "source_completion_sha256": COMPLETION_SHA256,
            "inventory_sha256": INVENTORY_SHA256,
            "asof": inventory["asof"],
            "source": "Macro control768/4/8; s42 best97 / s43 best94",
            "cases": cases,
            "cohorts": aggregate,
            "receipt": {
                "verified_members": report.verified,
                "verified_raw_files": file_count,
                "windows": len(cases),
                "models": list(MODELS),
                "model_forwards": 0,
                "optimizer_updates": 0,
                "fits": 0,
                "quality_assured": False,
                "continuous_replay": False,
                "repeated_research_data": True,
            },
        }
    finally:
        report.close()


def render(data, destination):
    path = Path(destination)
    if path.suffix.lower() != ".html":
        raise ValueError("Expected an HTML destination")
    template = Path(__file__).with_suffix(".html").read_text()
    if template.count("__MARKET_DATA__") != 1:
        raise ValueError("Expected exactly one data slot")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # JSON lives inside a script raw-text element: escape tag openers, not HTML entities.
    payload = (
        payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = template.replace("__MARKET_DATA__", payload)
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path


def output_path(out, archive, raw_root):
    path = Path(out).resolve()
    if path.suffix.lower() != ".html":
        raise ValueError("Expected an HTML destination")
    if (
        path == Path(archive).resolve()
        or path.is_relative_to(Path(raw_root).resolve())
        or path.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("Output must be separate from report, raw data and source package")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out, args.archive, args.raw_root)
    data = build_data(args.archive, args.raw_root)
    render(data, out)
    receipt = {
        **{k: v for k, v in data.items() if k != "cases"},
        "html_sha256": sha256(out),
        "html_bytes": out.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(f"F11 historical snapshot viewer: {out}; {len(data['cases'])} windows; no GPU/training")


if __name__ == "__main__":
    main()
