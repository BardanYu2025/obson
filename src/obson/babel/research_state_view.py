"""Offline inspection of existing control600 reports; no model execution."""

import argparse
import json
from pathlib import Path

import numpy as np

from .dual_state import sha256, verify_files
from .holdout_audit import read_json


def report_data(source):
    source = Path(source)
    done = read_json(source / "completion.json")
    if done["status"] != "complete" or not done["source_unchanged"]:
        raise ValueError("Completed unchanged source report required")
    # The evaluation archive contains all its own reports; upstream binary
    # checkpoint hashes are intentionally outside this report-only verification.
    verify_files(source, done["files"])
    stats = read_json(source / "evaluation_scales.json")["reconstruction"]
    mean, scale = np.asarray(stats["y_mean"]), np.asarray(stats["y_scale"])
    cases = []
    summary = []
    for split in ("test", "cross_research"):
        rows = (
            read_json(source / f"training_source/{split}_inventory.json")
            if (source / "training_source").exists()
            else read_json(
                Path(read_json(source / "manifest.json")["source"]) / f"{split}_inventory.json"
            )
        )
        for seed in (42, 43):
            record = read_json(source / f"{split}_control_s{seed}_best.json")
            summary.append(
                {"split": split, "seed": seed, "scores": record["reconstruction"]["scores"]}
            )
            for ex in record["examples"]:
                pred = np.asarray(ex["prediction"]) * scale + mean
                true = np.asarray(ex["target"]) * scale + mean
                mask = np.asarray(ex["mask"], bool)
                if pred.shape != (128, 7) or true.shape != pred.shape or mask.shape != pred.shape:
                    raise ValueError("Expected complete historical example")
                # Only127 scored historical bars, no unscored current-bar output.
                cases.append(
                    {
                        "split": split,
                        "seed": seed,
                        "index": ex["index"],
                        "row": rows[ex["index"]],
                        "pred": pred[:127].tolist(),
                        "true": true[:127].tolist(),
                        "mask": mask[:127].tolist(),
                    }
                )
    return {
        "cases": cases,
        "summary": summary,
        "source_manifest_sha256": sha256(source / "manifest.json"),
    }


def render_source(source, destination):
    data = report_data(source)
    template = Path(__file__).with_name("research_state_view.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__REPORT_DATA__", encoded))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(render_source(args.source, args.out))


if __name__ == "__main__":
    main()
