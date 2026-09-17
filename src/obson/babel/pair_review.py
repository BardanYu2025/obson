"""Exploratory method-blind pair comparisons; repeats measure rating stability.

Keep three dimensions separate. Neither human preferences nor repeat agreement
establish an objective ground truth. Repeats never enter method scores twice.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import block_interval
from .progress import progress

SCHEMA = "babel-pairs-v1"
DIMENSIONS = ("trend", "turns", "endpoint")
CHOICES = {"left", "right", "tie", "uncertain", ""}
FIELDS = ("packet_id", "case_id", *DIMENSIONS)


def _bars(snapshot):
    # No future path, scores, symbol, timestamps or method name in the review page.
    return [{k: bar[k] for k in ("open", "high", "low", "close")} for bar in snapshot["bars"]]


def pair_packet(engine, count=6, repeats=3, seed=20260918):
    if not {"model", "rule", "path"} <= set(engine.methods):
        raise ValueError(
            "Pair review requires a matching model checkpoint and all three index methods"
        )
    if count < 2 or repeats < 0 or repeats > count:
        raise ValueError("Use count>=2 and 0<=repeats<=count")
    cutoff = np.datetime64(engine.meta["model_available_after"])
    weeks = {}
    for j, (i, row) in enumerate(zip(engine.sid, engine.rows, strict=True)):
        day = engine.series[i].sessions[row]
        if day > cutoff:
            block = str(pd.Timestamp(day).to_period("W"))
            weeks.setdefault(block, []).append(j)
    if len(weeks) < count:
        raise ValueError(f"Only {len(weeks)} post-validation weeks available, need {count}")
    rng = np.random.default_rng(seed)
    selected_weeks = rng.choice(sorted(weeks), count, replace=False)
    primary = []
    for number, block in enumerate(selected_weeks):
        j = int(rng.choice(weeks[block]))
        i, row = int(engine.sid[j]), int(engine.rows[j])
        query = _bars(engine.snapshot(i, row))
        matches, hit_ids = {}, {}
        for method in ("model", "rule", "path"):
            hits = engine.retrieve(i, row, method, 1)
            if not hits:
                raise ValueError(
                    "A selected query has no legal historical hit; inspect index coverage"
                )
            h, _ = hits[0]
            matches[method] = _bars(engine.snapshot(int(engine.sid[h]), int(engine.rows[h])))
            hit_ids[method] = int(h)
        for baseline in ("rule", "path"):
            primary.append(
                {
                    "query": query,
                    "matches": matches,
                    "baseline": baseline,
                    "block": str(block),
                    "pair_id": f"Q{number + 1:03d}-{baseline}",
                    "query_index": j,
                    "hit_ids": hit_ids,
                    "repeat_of": None,
                }
            )
        progress(f"pair review: retrieved {number + 1}/{count} queries")
    rng.shuffle(primary)
    repeated = rng.choice(count, repeats, replace=False)
    # Originals first, repeats at the end: minimum spacing=count primary cards.
    # Balanced sampling avoids putting a repeated card immediately after its original.
    selected = sorted(int(v) for v in repeated)
    schedule = primary + [{**primary[j], "repeat_of": j} for j in selected]
    cases, hidden = [], {}
    original_sides = {}
    for position, item in enumerate(schedule):
        case_id = f"P{position + 1:03d}"
        methods = ["model", item["baseline"]]
        if item["repeat_of"] is None:
            rng.shuffle(methods)
            original_sides[position] = methods.copy()
        else:
            # Reversing sides makes repeat agreement insensitive to left/right position.
            methods = list(reversed(original_sides[item["repeat_of"]]))
        cases.append(
            {
                "id": case_id,
                "query": item["query"],
                "left": item["matches"][methods[0]],
                "right": item["matches"][methods[1]],
            }
        )
        hidden[case_id] = {
            "pair_id": item["pair_id"],
            "block": item["block"],
            "left": methods[0],
            "right": methods[1],
            "repeat_of": f"P{item['repeat_of'] + 1:03d}" if item["repeat_of"] is not None else None,
            "query_index": item["query_index"],
            "hit_ids": item["hit_ids"],
        }
    identity = json.dumps(
        {
            "schema": SCHEMA,
            "seed": seed,
            "cases": cases,
            "key": hidden,
            "checkpoint": engine.meta.get("checkpoint"),
        },
        sort_keys=True,
    )
    packet_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    packet = {"schema": SCHEMA, "packet_id": packet_id, "cases": cases}
    key = {
        "schema": SCHEMA,
        "packet_id": packet_id,
        "seed": seed,
        "query_count": count,
        "repeat_count": repeats,
        "cases": hidden,
        "checkpoint_sha256": engine.meta.get("checkpoint"),
        "interpretation": "Exploratory same-rater preference and stability, not independent model graduation.",
    }
    return packet, key


def export_pairs(engine, directory, count=6, repeats=3, seed=20260918):
    directory = Path(directory)
    # Never silently replace an existing rating session or its key.
    if directory.exists():
        raise FileExistsError(f"Review directory already exists: {directory}")
    packet, key = pair_packet(engine, count, repeats, seed)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "pairs.json").write_text(json.dumps(packet, ensure_ascii=False))
    (directory / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2))
    template = (Path(__file__).parent / "pair_review.html").read_text()
    encoded = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c")
    (directory / "review.html").write_text(template.replace("__PACKET__", encoded))
    progress(f"pair review: saved {len(packet['cases'])} cards → {directory}/review.html")
    return {
        "directory": str(directory),
        "cards": len(packet["cases"]),
        "packet_id": packet["packet_id"],
    }


def _vote(row, key, dimension):
    choice = row[dimension]
    if choice in ("", "uncertain"):
        return None
    if choice == "tie":
        return 0
    return 1 if key[choice] == "model" else -1


def score_pairs(ratings_path, answer_key_path):
    key = json.loads(Path(answer_key_path).read_text())
    if key.get("schema") != SCHEMA:
        raise ValueError(
            "Use the answer key from the same pair-review packet, not the old 0/1/2 review"
        )
    with Path(ratings_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or set(reader.fieldnames) != set(FIELDS):
            raise ValueError(f"Expected pair-review CSV columns: {FIELDS}")
        rows = {}
        for row in reader:
            if row["packet_id"] != key["packet_id"]:
                raise ValueError("Ratings and answer key belong to different packets")
            cid = row["case_id"]
            if cid not in key["cases"] or cid in rows:
                raise ValueError(f"Unknown or duplicate case: {cid}")
            if any(row[d] not in CHOICES for d in DIMENSIONS):
                raise ValueError("Choices must be left/right/tie/uncertain or empty")
            rows[cid] = row
    coverage, results, stability = {}, {}, {}
    for dimension in DIMENSIONS:
        results[dimension] = {}
        totals = dict.fromkeys(("left", "right", "tie", "uncertain", "missing"), 0)
        for cid, info in key["cases"].items():
            if info["repeat_of"] is None:
                value = rows.get(cid, {}).get(dimension, "")
                totals[value or "missing"] += 1
        coverage[dimension] = totals
        for baseline in ("rule", "path"):
            scores, blocks = [], []
            for cid, row in rows.items():
                info = key["cases"][cid]
                if info["repeat_of"] is not None or baseline not in (info["left"], info["right"]):
                    continue
                value = _vote(row, info, dimension)
                if value is not None:
                    scores.append(value)
                    blocks.append(info["block"])
            results[dimension][f"model_vs_{baseline}"] = {
                "wins": scores.count(1),
                "ties": scores.count(0),
                "losses": scores.count(-1),
                "net_preference": block_interval(scores, blocks),
            }
        comparable = agree = switches = ties_changed = skipped = 0
        for cid, info in key["cases"].items():
            original = info["repeat_of"]
            if original is None:
                continue
            if cid not in rows or original not in rows:
                skipped += 1
                continue
            a = _vote(rows[original], key["cases"][original], dimension)
            b = _vote(rows[cid], info, dimension)
            if a is None or b is None:
                skipped += 1
                continue
            comparable += 1
            agree += a == b
            switches += a * b == -1
            ties_changed += (a == 0) != (b == 0)
        stability[dimension] = {
            "comparable_repeats": comparable,
            "same_judgment": agree,
            "opposite_preference": switches,
            "tie_changed": ties_changed,
            "missing_or_uncertain": skipped,
            "agreement": agree / comparable if comparable else None,
        }
    return {
        "schema": SCHEMA,
        "packet_id": key["packet_id"],
        "coverage_primary_only": coverage,
        "dimensions": results,
        "repeat_stability": stability,
        "interpretation": "Pilot only. +1 prefers model, 0 ties, -1 prefers baseline; uncertainty/missing excluded and reported. Repeats excluded from method scores. Few queries and one rater do not establish graduation; bootstrap intervals do not include rating error.",
    }
