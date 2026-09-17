"""Pair review tests use synthetic charts only; no model training/inference."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from obson.babel.pair_review import DIMENSIONS, FIELDS, export_pairs, pair_packet, score_pairs


class FakeEngine:
    methods = ["model", "rule", "path"]
    meta = {"model_available_after": "2025-01-01", "checkpoint": "synthetic-test-only"}

    def __init__(self):
        self.sid = np.zeros(16, dtype=int)
        self.rows = np.arange(16)
        self.series = [
            SimpleNamespace(
                sessions=np.array(
                    [np.datetime64("2025-02-03") + np.timedelta64(7 * i, "D") for i in range(16)]
                )
            )
        ]

    def snapshot(self, i, row):
        return {
            "bars": [
                {
                    "open": 100 + np.sin(t * 0.3 + row),
                    "high": 102 + np.sin(t * 0.3 + row),
                    "low": 98 + np.sin(t * 0.3 + row),
                    "close": 100 + np.cos(t * 0.3 + row),
                }
                for t in range(40)
            ]
        }

    def retrieve(self, i, row, method, topk):
        return [((row + self.methods.index(method) + 1) % 16, 0.9)]


class PairReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.packet, self.key = pair_packet(FakeEngine())
        self.key_path = self.folder / "key.json"
        self.key_path.write_text(json.dumps(self.key))

    def tearDown(self):
        self.temp.cleanup()

    def write_ratings(self, repeat_preference="model", blank=False):
        rows = []
        for cid, info in self.key["cases"].items():
            target = "model" if info["repeat_of"] is None else repeat_preference
            if target == "baseline":
                target = info["right"] if info["left"] == "model" else info["left"]
            vote = "left" if info["left"] == target else "right"
            if blank:
                vote = ""
            rows.append(
                {
                    "packet_id": self.key["packet_id"],
                    "case_id": cid,
                    **dict.fromkeys(DIMENSIONS, vote),
                }
            )
        path = self.folder / "ratings.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_packet_is_blind_balanced_and_repeats_are_separated(self):
        self.assertEqual(len(self.packet["cases"]), 15)
        self.assertEqual(len({c["block"] for c in self.key["cases"].values()}), 6)
        public = json.dumps(self.packet)
        for token in ('"model"', '"rule"', '"path"', '"repeat_of"', '"checkpoint"'):
            self.assertNotIn(token, public)
        for index, case in enumerate(self.packet["cases"]):
            info = self.key["cases"][case["id"]]
            original = info["repeat_of"]
            if original is not None:
                original_index = int(original[1:]) - 1
                self.assertGreaterEqual(index - original_index, 6)
                old = self.packet["cases"][original_index]
                self.assertEqual(old["query"], case["query"])
                self.assertEqual(old["left"], case["right"])
                self.assertEqual(old["right"], case["left"])

    def test_packet_reproducible(self):
        other, _ = pair_packet(FakeEngine())
        self.assertEqual(self.packet, other)
        changed, _ = pair_packet(FakeEngine(), seed=7)
        self.assertNotEqual(self.packet["packet_id"], changed["packet_id"])

    def test_scores_correct_side_and_repeats_do_not_inflate_count(self):
        report = score_pairs(self.write_ratings(), self.key_path)
        for d in DIMENSIONS:
            self.assertEqual(report["repeat_stability"][d]["same_judgment"], 3)
            for baseline in ("rule", "path"):
                r = report["dimensions"][d][f"model_vs_{baseline}"]
                self.assertEqual(r["wins"], 6)
                self.assertEqual(r["net_preference"]["mean"], 1)

    def test_repeat_disagreement_is_reported_without_changing_primary_scores(self):
        report = score_pairs(self.write_ratings(repeat_preference="baseline"), self.key_path)
        for d in DIMENSIONS:
            self.assertEqual(report["repeat_stability"][d]["opposite_preference"], 3)
            self.assertEqual(report["dimensions"][d]["model_vs_path"]["wins"], 6)

    def test_missing_and_uncertain_are_not_ties(self):
        path = self.write_ratings(blank=True)
        text = path.read_text().replace(",,,\n", ",uncertain,,\n")
        path.write_text(text)
        report = score_pairs(path, self.key_path)
        self.assertEqual(report["coverage_primary_only"]["trend"]["uncertain"], 12)
        self.assertEqual(report["coverage_primary_only"]["turns"]["missing"], 12)
        self.assertEqual(report["dimensions"]["trend"]["model_vs_path"]["ties"], 0)
        self.assertIsNone(report["dimensions"]["trend"]["model_vs_path"]["net_preference"]["mean"])

    def test_wrong_packet_and_duplicate_rejected(self):
        path = self.write_ratings()
        original = path.read_text()
        path.write_text(original.replace(self.key["packet_id"], "other-packet"))
        with self.assertRaisesRegex(ValueError, "different packets"):
            score_pairs(path, self.key_path)
        path.write_text(original + original.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_pairs(path, self.key_path)

    def test_export_preserves_existing_packet_and_contains_no_key(self):
        folder = self.folder / "review"
        export_pairs(FakeEngine(), folder)
        html = (folder / "review.html").read_text()
        self.assertNotIn("__PACKET__", html)
        self.assertNotIn("synthetic-test-only", html)
        self.assertIn("无法判断", html)
        with self.assertRaises(FileExistsError):
            export_pairs(FakeEngine(), folder)


if __name__ == "__main__":
    unittest.main()
