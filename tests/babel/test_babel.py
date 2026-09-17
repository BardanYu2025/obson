"""Run: PYTHONPATH=src python -m unittest discover -s tests/babel -v"""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

from obson.babel.data import (
    Series,
    features,
    load_series,
    session_dates,
    split_boundaries,
    split_mask,
    validate_frame,
)
from obson.babel.metrics import block_interval, event_counts, event_f1
from obson.babel.model import Config, Encoder, Windows, load_model, train
from obson.babel.retrieval import Engine, blind_packet, evaluate_retrieval, save_index
from obson.babel.server import handler_for
from obson.babel.structure import annotate, describe


def frame(n=240, start="2025-01-01 09:00", shift=0):
    t = np.arange(n)
    close = 100 + 0.02 * t + 3 * np.sin(t * 0.32) + shift
    dates = pd.bdate_range(start[:10], periods=(n + 3) // 4)
    dt = [d + pd.Timedelta(hours=9 + i) for d in dates for i in range(4)][:n]
    return validate_frame(
        pd.DataFrame(
            {
                "datetime": dt,
                "open": close - 0.15,
                "high": close + 0.6,
                "low": close - 0.7,
                "close": close,
                "volume": 100 + t % 17,
                "close_oi": 1000 + t * 3,
            }
        )
    )


def series(n=400):
    result = []
    for code, shift in (("rb", 0), ("sr", 50)):
        f = frame(n, shift=shift)
        labels = annotate(f)
        result.append(
            Series(
                code,
                60,
                f"{code}2505",
                f,
                labels,
                features(f, labels["atr"], 60),
                np.ones(n, bool),
                f.datetime.to_numpy("datetime64[D]"),
                code,
            )
        )
    return result


class CausalityTests(unittest.TestCase):
    def test_future_changes_cannot_change_prefix_features_or_targets(self):
        f = frame()
        changed = f.copy()
        changed.loc[120:, ["open", "high", "low", "close"]] += 50
        changed.loc[120:, "volume"] *= 10
        changed.loc[120:, "oi"] *= 5
        a, b = annotate(f), annotate(changed)
        np.testing.assert_array_equal(
            features(f, a["atr"], 60)[:120], features(changed, b["atr"], 60)[:120]
        )
        for k in ("direction", "age", "amplitude", "event", "state", "levels", "compression"):
            np.testing.assert_array_equal(a[k][:120], b[k][:120])
        truncated = annotate(f.iloc[:120])
        self.assertEqual(describe(f, a, 119), describe(f.iloc[:120], truncated, 119))

    def test_encoder_causal_even_with_longer_suffix(self):
        torch.manual_seed(3)
        model = Encoder(Config(hidden=16, layers=1, window=32, warmup=8, dropout=0)).eval()
        x = torch.tensor(features(frame(40), annotate(frame(40))["atr"], 60)).unsqueeze(0)
        changed = x.clone()
        changed[:, 20:] += 7
        with torch.no_grad():
            h = model(x)["h"][:, :20]
            np.testing.assert_allclose(h, model(changed)["h"][:, :20], atol=1e-6)
            np.testing.assert_allclose(h, model(x[:, :20])["h"], atol=2e-6)

    def test_oi_is_used_and_absence_is_explicit(self):
        f = frame()
        x = features(f, annotate(f)["atr"], 60)
        self.assertGreater(float(np.std(x[:, 6])), 0)
        self.assertTrue((x[:, 7] == 1).all())
        missing = validate_frame(f.drop(columns=["close_oi", "oi", "oi_available"]))
        self.assertFalse(missing.oi_available.any())
        self.assertTrue((features(missing, annotate(missing)["atr"], 60)[:, 7] == 0).all())

    def test_confirmations_are_not_backfilled_and_amplitude_is_signed(self):
        f = frame()
        labels = annotate(f)
        for p in labels["pivots"]:
            self.assertGreater(p.confirmed, p.event)
            self.assertEqual(labels["event"][p.confirmed, p.scale], 1 if p.kind == 1 else 2)
            self.assertNotIn(p, annotate(f.iloc[: p.confirmed])["pivots"])
        self.assertTrue((labels["amplitude"][labels["direction"] == 0] < 0).any())
        self.assertTrue((labels["amplitude"][labels["direction"] == 2] > 0).any())

    def test_weekend_and_holiday_sessions(self):
        dt = pd.to_datetime(
            ["2025-09-30 21:00", "2025-10-01 01:00", "2025-10-09 09:00", "2025-10-10 21:00"]
        )
        got = session_dates(dt, ["2025-09-30", "2025-10-09", "2025-10-10", "2025-10-13"])
        np.testing.assert_array_equal(
            got,
            np.array(
                ["2025-10-09", "2025-10-09", "2025-10-09", "2025-10-13"], dtype="datetime64[D]"
            ),
        )

    def test_lagged_contract_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rb"
            root.mkdir()
            a, b = frame(80), frame(80, shift=10)
            a["volume"], b["volume"] = 100, 10
            a.to_csv(root / "a_60m.csv", index=False)
            b.to_csv(root / "b_60m.csv", index=False)
            before, _ = load_series(d, ["rb"], [60])
            # Change this session's eventual leader: its own selection must not change.
            b.loc[40:43, "volume"] = 10000
            b.to_csv(root / "b_60m.csv", index=False)
            after, _ = load_series(d, ["rb"], [60])
            np.testing.assert_array_equal(before[0].main[:44], after[0].main[:44])
            self.assertFalse(after[0].main[44])
            self.assertTrue(after[1].main[44])

    def test_global_splits_and_loss_masks(self):
        ss = series()
        bounds = split_boundaries(ss)
        for s in ss:
            masks = [split_mask(s, bounds, split) for split in ("train", "val", "test")]
            np.testing.assert_array_equal(np.sum(masks, axis=0), s.main.astype(int))
        cfg = Config(window=32, warmup=8)
        ds = Windows(ss, bounds, "val", cfg, 1)
        sample = ds[0]
        self.assertFalse(sample["valid"][:8].any())
        i, j = ds.items[0]
        self.assertTrue(sample["valid"][-1])
        self.assertFalse(
            sample["valid"][
                ss[i].sessions[j - 31 : j + 1] <= np.datetime64(bounds["train_until"])
            ].any()
        )

    def test_invalid_data_fails_loudly(self):
        f = frame(40)
        with self.assertRaises(ValueError):
            validate_frame(pd.concat([f, f.iloc[-1:]]))
        f.loc[3, "low"] = f.loc[3, "high"] + 10
        with self.assertRaises(ValueError):
            validate_frame(f)


class EvaluationTests(unittest.TestCase):
    def test_event_matching_does_not_reward_duplicate_predictions(self):
        self.assertEqual(event_counts(range(12, 19), [15], 3), (1, 6, 0))
        self.assertEqual(event_counts([10], [9, 11], 2), (1, 0, 1))

    def test_events_never_match_across_samples(self):
        result = event_f1([([], [11]), ([0], [])], tolerance=3)
        self.assertEqual(result, {"f1": 0, "tp": 0, "fp": 1, "fn": 1})

    def test_block_uncertainty_requires_blocks(self):
        self.assertEqual(
            block_interval([1] * 100, ["same-week"] * 100)["status"], "insufficient_blocks"
        )


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ss = series()
        self.path = Path(self.temp.name) / "index.npz"
        save_index(self.ss, self.path, window=32, stride=4)
        self.engine = Engine(self.ss, self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_completed_bar_and_query_future_exclusion(self):
        s = self.ss[0]
        asof = s.frame.datetime.iloc[300] + pd.Timedelta(minutes=30)
        i, r = self.engine.locate("rb", 60, str(asof))
        self.assertEqual(r, 299)
        hits = self.engine.retrieve(i, r, topk=8)
        start = s.frame.datetime.iloc[r - 31].to_datetime64()
        intervals = []
        for h, _ in hits:
            self.assertLess(self.engine.end[h], start)
            for lo, hi in intervals:
                self.assertFalse(self.engine.start[h] <= hi and self.engine.end[h] >= lo)
            intervals.append((self.engine.start[h], self.engine.end[h]))

    def test_outcomes_must_be_available_and_stay_in_contract(self):
        cutoff = self.ss[0].ends[303]
        snap = self.engine.snapshot(0, 300, cutoff, True)
        self.assertTrue(all(v is None for v in snap["outcomes"].values()))
        self.assertEqual(snap["bars"][-1]["row"], 303)
        snap = self.engine.snapshot(0, 398, np.datetime64("2030-01-01"), True)
        self.assertTrue(all(v is None for v in snap["outcomes"].values()))
        self.assertEqual(snap["bars"][-1]["row"], 399)

    def test_snapshot_does_not_expose_unconfirmed_pivots(self):
        snap = self.engine.snapshot(0, 250)
        self.assertTrue(all(p["confirmed"] <= 250 for p in snap["structure"]["pivots"]))
        self.assertEqual(snap["bars"][-1]["row"], 250)

    def test_data_version_mismatch(self):
        self.ss[0].source_hash = "changed"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            Engine(self.ss, self.path)

    def test_benchmark_and_blind_export(self):
        result = evaluate_retrieval(self.engine, 10)
        self.assertEqual(result["queries"], 10)
        self.assertEqual(set(result["summary"]), {"rule", "path"})
        packet, key = blind_packet(self.engine, 3)
        self.assertEqual(len(packet), 3)
        self.assertEqual(set(key), {r["id"] for r in packet})
        self.assertNotIn('"method"', json.dumps(packet))

    def test_local_http_api_and_ui(self):
        server = HTTPServer(("127.0.0.1", 0), handler_for(self.engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = f"http://127.0.0.1:{server.server_port}"
            with urlopen(root) as r:
                self.assertIn(b"BABEL", r.read())
            with urlopen(root + "/api/query?code=rb&period=60") as r:
                value = json.load(r)
                self.assertEqual(value["query"]["code"], "rb")
                self.assertTrue(value["matches"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class TrainingTests(unittest.TestCase):
    def test_train_save_load_and_model_retrieval_roundtrip(self):
        torch.set_num_threads(2)
        ss = series(160)
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(hidden=16, layers=1, window=32, warmup=8, dropout=0)
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(ss, d, cfg, epochs=1, stride=8, batch_size=16)
            m, ck = load_model(result["checkpoint"])
            self.assertEqual(ck["schema"], "babel-v1")
            self.assertIn("amplitude", ck["validation"])
            self.assertTrue(Path(d, "history.jsonl").exists())
            path = Path(d) / "model.npz"
            save_index(ss, path, window=32, stride=4, checkpoint=result["checkpoint"])
            engine = Engine(ss, path, result["checkpoint"])
            self.assertEqual(len(engine.vector(0, 159, "model")), 32)
            with self.assertRaisesRegex(ValueError, "historical time"):
                engine.vector(0, 40, "model")
            snapshot = engine.query("rb", 60, method="model")
            self.assertIsNotNone(snapshot["model_estimates"])
            with self.assertRaisesRegex(ValueError, "already"):
                train(ss, d, cfg, epochs=1)


if __name__ == "__main__":
    unittest.main()
